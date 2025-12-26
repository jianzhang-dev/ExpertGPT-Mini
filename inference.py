#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
import random
import time
import sys
import threading
from tqdm import tqdm
from transformers import BertTokenizer
from collections import deque
import re

# ==================== 模型定义（与训练时保持一致）====================
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=256, base_min=2000.0, base_max=100000.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.current_seq_len_cached = 0
        num_head_groups = num_heads // 2
        group_bases = torch.logspace(
            start=math.log10(base_min),
            end=math.log10(base_max),
            steps=num_head_groups
        )
        base_list = [base for base in group_bases for _ in range(2)]
        self.base_list = torch.tensor(base_list)
        self.register_buffer("base_cache", self.base_list, persistent=True)
        self._update_freqs(seq_len=max_seq_len)
    
    def _update_freqs(self, seq_len, device='cpu'):
        if seq_len > self.max_seq_len:
            alpha = (seq_len / self.max_seq_len)
            current_bases = self.base_cache * (alpha ** (self.dim / (self.dim - 2)))
        else:
            current_bases = self.base_cache
        current_bases = current_bases.to(device)
        inv_freq_list = [1.0 / (base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)) for base in current_bases]
        inv_freq = torch.stack(inv_freq_list)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum('i,hj->hij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
        self.current_seq_len_cached = seq_len
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if self.cos_cached.device != x.device or seq_len > self.current_seq_len_cached:
            self._update_freqs(seq_len, device=x.device)
        cos = self.cos_cached[:, :seq_len, ...].unsqueeze(0)
        sin = self.sin_cached[:, :seq_len, ...].unsqueeze(0)
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

class SingleAttentionExpert(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, x, attention_mask=None, rope=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)
        
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

class ParallelExpertAttention(nn.Module):
    def __init__(self, hidden_size=768, num_experts=2, num_heads=12, window_size=32, is_global=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.window_size = window_size
        self.is_global = is_global
        
        self.experts = nn.ModuleList(
            [SingleAttentionExpert(hidden_size, num_heads, window_size) for _ in range(num_experts)]
        )
        fused_dim = num_experts * hidden_size
        self.gate_proj = nn.Linear(fused_dim, hidden_size, bias=False)
        self.up_proj = nn.Linear(fused_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("sliding_window_mask", None, persistent=False)
        self.last_mask_len = 0

    def _get_mask(self, seq_len, device):
        if self.last_mask_len != seq_len or self.sliding_window_mask is None:
            if self.is_global:
                mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            else:
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
                dists = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
                sliding_window_mask = (dists >= 0) & (dists < self.window_size)
                mask = sliding_window_mask
            self.sliding_window_mask = mask.to(device)
            self.last_mask_len = seq_len
        return self.sliding_window_mask

    def forward(self, x, rope=None):
        batch_size, seq_len, _ = x.shape
        attention_mask = self._get_mask(seq_len, x.device)
        expert_outputs = []
        for expert in self.experts:
            output = expert(x, attention_mask=attention_mask, rope=rope)
            expert_outputs.append(output)
        fused_output = torch.cat(expert_outputs, dim=-1)
        gate = F.silu(self.gate_proj(fused_output))
        value = self.up_proj(fused_output)
        fused_result = self.dropout(gate * value)
        return fused_result

class GeGLUFeedForward(nn.Module):
    def __init__(self, hidden_size, expansion_ratio=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size * expansion_ratio)
        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x

class ExpertTransformerLayer(nn.Module):
    def __init__(self, hidden_size=768, num_experts=2, num_heads=12, window_size=32, is_global=False):
        super().__init__()
        self.self_attn = ParallelExpertAttention(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_heads=num_heads,
            window_size=window_size,
            is_global=is_global
        )
        self.ffn = GeGLUFeedForward(hidden_size)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)

    def forward(self, x, rope=None):
        attn_output = self.self_attn(self.norm1(x), rope=rope)
        x = x + attn_output
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x

class ExpertGPTModel(nn.Module):
    def __init__(self, vocab_size=21128, hidden_size=768, num_layers=12,
                 num_experts=2, num_heads=12, window_size=32, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.window_size = window_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(head_dim, num_heads, max_seq_len=max_seq_len)
        
        global_layer_indices = {num_layers // 3, (num_layers * 2) // 3}
        layers = []
        for i in range(num_layers):
            is_global = i in global_layer_indices
            layers.append(
                ExpertTransformerLayer(
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    num_heads=num_heads,
                    window_size=window_size,
                    is_global=is_global
                )
            )
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, rope=self.rope)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids, max_length=256, temperature=0.8, top_p=0.9, repetition_penalty=1.2):
        self.eval()
        generated = input_ids
        appeared_tokens = set(generated[0].tolist())
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated)[:, -1, :]
                if repetition_penalty != 1.0:
                    for token in appeared_tokens:
                        logits[0, token] /= repetition_penalty
                if temperature > 0:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        probs[..., indices_to_remove] = 0
                        if probs.sum() > 0:
                            probs = probs / probs.sum()
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                next_token_item = next_token.item()
                appeared_tokens.add(next_token_item)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() in [102, 0]:
                    break
        self.train()
        return generated

# ==================== 实用工具函数 ====================
def clear_screen():
    """跨平台清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def typing_effect(text, delay=0.03, color_code="\033[94m"):
    """模拟打字效果"""
    print(color_code, end="", flush=True)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print("\033[0m", end="", flush=True)

def loading_animation(stop_event, message="思考中"):
    """思考加载动画"""
    animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\r\033[93m{message} {animation[idx % len(animation)]}\033[0m")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 10) + "\r")
    sys.stdout.flush()

def generate_text(model, tokenizer, prompt, max_length=256, temperature=0.8, top_p=0.9, repetition_penalty=1.2, device='cpu'):
    """生成文本并显示思考动画"""
    stop_event = threading.Event()
    animation_thread = threading.Thread(target=loading_animation, args=(stop_event, "模型思考中"))
    animation_thread.start()
    
    try:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_tensor,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
        
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        # 移除输入部分，只保留生成的回答
        response = generated_text[len(prompt):].strip()
        return response
    finally:
        stop_event.set()
        animation_thread.join()

# ==================== 交互式对话系统 ====================
class ChatInterface:
    def __init__(self, model_path="expert_gpt_model_finetuned_final.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 正在加载模型到 {self.device}...")
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        print(f"🔤 Tokenizer加载完成 (词汇表大小: {self.tokenizer.vocab_size})")
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 初始化对话历史
        self.history = deque(maxlen=10)  # 保留最近10轮对话
        self.generation_params = {
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "max_length": 1000
        }
        
        # 主题设置
        self.themes = {
            "default": ("\033[94m", "\033[92m"),  # 用户蓝，AI绿
            "dark": ("\033[96m", "\033[95m"),     # 用户青，AI紫
            "retro": ("\033[93m", "\033[91m"),    # 用户黄，AI红
            "nature": ("\033[92m", "\033[93m")    # 用户绿，AI黄
        }
        self.current_theme = "default"
        self.user_color, self.ai_color = self.themes[self.current_theme]
        
        # 欢迎消息
        self.show_welcome()
    
    def load_model(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 从checkpoint加载配置
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        
        # 创建模型
        model = ExpertGPTModel(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_experts=config['num_experts'],
            num_heads=config['num_heads'],
            window_size=config['window_size'],
            max_seq_len=config['max_seq_len']
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"🧠 模型加载成功! 配置: {config['hidden_size']}维, {config['num_layers']}层, {config['num_experts']}专家")
        return model
    
    def show_welcome(self):
        """显示欢迎界面"""
        clear_screen()
        art = r"""
        ╔══════════════════════════════════════════════════════════════╗
        ║  🤖 欢迎使用 ExpertGPT 交互式对话系统!                      ║
        ║                                                              ║
        ║  ✨ 特色功能:                                                ║
        ║     • 实时思考动画与打字效果                                  ║
        ║     • 动态调整生成参数 (温度/top_p/重复惩罚)                  ║
        ║     • 多主题视觉切换                                          ║
        ║     • 对话历史管理与导出                                      ║
        ║     • 指令微调优化的中文对话能力                              ║
        ║                                                              ║
        ║  🎮 快捷指令:                                                ║
        ║     /help   - 显示帮助菜单                                    ║
        ║     /params - 调整生成参数                                    ║
        ║     /theme  - 切换显示主题                                    ║
        ║     /history- 查看对话历史                                    ║
        ║     /save   - 保存对话到文件                                  ║
        ║     /clear  - 清空对话历史                                    ║
        ║     /exit   - 退出对话                                        ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        print("\033[1;96m" + art + "\033[0m")
        typing_effect("系统初始化完成! 请输入您的问题开始对话...", delay=0.02, color_code="\033[1;93m")
        print("\n" + "="*60)
    
    def show_help(self):
        """显示帮助菜单"""
        help_text = """
        📚 帮助菜单:
        
        🎚️  参数调整:
          /params temp=0.7 top_p=0.95 rep=1.5 max=80
          - temp: 生成随机性 (0.1-2.0, 默认0.8)
          - top_p: 核采样阈值 (0.1-1.0, 默认0.9)
          - rep: 重复惩罚 (1.0-2.0, 默认1.2)
          - max: 最大生成长度 (10-200, 默认60)
        
        🎨 主题切换:
          /theme [default|dark|retro|nature]
          - default: 标准蓝绿配色
          - dark: 深色青紫配色
          - retro: 复古黄红配色
          - nature: 自然绿黄配色
        
        📜 历史管理:
          /history - 查看最近10轮对话
          /save [文件名] - 保存对话 (默认: chat_history.txt)
          /clear - 清空对话历史
        
        ⚡ 其他:
          /exit - 退出对话
        """
        print("\033[1;95m" + help_text + "\033[0m")
    
    def adjust_params(self, command):
        """调整生成参数"""
        try:
            # 解析命令: /params temp=0.7 top_p=0.95 rep=1.5 max=80
            parts = command.split()[1:]
            for part in parts:
                key, value = part.split('=')
                key = key.strip()
                value = float(value.strip())
                
                if key in ["temp", "temperature"]:
                    if 0.1 <= value <= 2.0:
                        self.generation_params["temperature"] = value
                    else:
                        raise ValueError("温度应在0.1-2.0之间")
                elif key in ["top_p"]:
                    if 0.1 <= value <= 1.0:
                        self.generation_params["top_p"] = value
                    else:
                        raise ValueError("top_p应在0.1-1.0之间")
                elif key in ["rep", "repetition_penalty"]:
                    if 1.0 <= value <= 2.0:
                        self.generation_params["repetition_penalty"] = value
                    else:
                        raise ValueError("重复惩罚应在1.0-2.0之间")
                elif key in ["max", "max_length"]:
                    if 1 <= value <= 1000:
                        self.generation_params["max_length"] = int(value)
                    else:
                        raise ValueError("最大长度应在1-256之间")
                else:
                    raise ValueError(f"未知参数: {key}")
            
            # 显示更新后的参数
            params_str = ", ".join([f"{k}={v}" for k, v in self.generation_params.items()])
            typing_effect(f"✅ 生成参数已更新: {params_str}", color_code="\033[1;92m")
        except Exception as e:
            typing_effect(f"❌ 参数调整失败: {str(e)}", color_code="\033[1;91m")
    
    def change_theme(self, command):
        """切换显示主题"""
        try:
            theme_name = command.split()[1] if len(command.split()) > 1 else "default"
            if theme_name in self.themes:
                self.current_theme = theme_name
                self.user_color, self.ai_color = self.themes[theme_name]
                typing_effect(f"🎨 已切换到 {theme_name} 主题", color_code="\033[1;93m")
            else:
                valid_themes = ", ".join(self.themes.keys())
                typing_effect(f"⚠️ 无效主题. 可用主题: {valid_themes}", color_code="\033[1;91m")
        except IndexError:
            typing_effect("💡 用法: /theme [主题名]", color_code="\033[1;93m")
    
    def show_history(self):
        """显示对话历史"""
        if not self.history:
            typing_effect("📭 对话历史为空", color_code="\033[1;93m")
            return
        
        print("\n\033[1;94m" + "="*30 + " 对话历史 " + "="*30 + "\033[0m")
        for i, (user_msg, ai_msg) in enumerate(self.history, 1):
            print(f"\033[1;96m[{i}] 用户:\033[0m {user_msg}")
            print(f"\033[1;92m[{i}] AI:\033[0m {ai_msg}")
            print("-"*65)
        print("\033[1;93m提示: 使用 /clear 清空历史, /save 保存历史\033[0m")
    
    def save_history(self, command):
        """保存对话历史到文件"""
        filename = command.split()[1] if len(command.split()) > 1 else "chat_history.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"ExpertGPT 对话记录 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                for i, (user_msg, ai_msg) in enumerate(self.history, 1):
                    f.write(f"[{i}] 用户: {user_msg}\n")
                    f.write(f"[{i}] AI: {ai_msg}\n")
                    f.write("-"*40 + "\n\n")
            
            typing_effect(f"💾 对话历史已保存到 {filename}", color_code="\033[1;92m")
        except Exception as e:
            typing_effect(f"❌ 保存失败: {str(e)}", color_code="\033[1;91m")
    
    def process_command(self, command):
        """处理特殊命令"""
        cmd = command.strip().lower()
        
        if cmd.startswith("/help"):
            self.show_help()
            return True
            
        elif cmd.startswith("/params"):
            self.adjust_params(command)
            return True
            
        elif cmd.startswith("/theme"):
            self.change_theme(command)
            return True
            
        elif cmd.startswith("/history"):
            self.show_history()
            return True
            
        elif cmd.startswith("/save"):
            self.save_history(command)
            return True
            
        elif cmd.startswith("/clear"):
            self.history.clear()
            typing_effect("🧹 对话历史已清空", color_code="\033[1;93m")
            return True
            
        elif cmd.startswith("/exit"):
            self.exit_chat()
            return False
            
        return False  # 不是命令，继续对话
    
    def exit_chat(self):
        """退出对话"""
        if self.history:
            print("\n\033[1;93m" + "="*30 + " 对话总结 " + "="*30 + "\033[0m")
            # 将deque转换为列表后再切片
            for i, (user_msg, ai_msg) in enumerate(list(self.history)[-3:], 1):
                print(f"\033[1;96m最后[{i}] 用户:\033[0m {user_msg}")
                print(f"\033[1;92m最后[{i}] AI:\033[0m {ai_msg}")
                print("-"*65)
            
            if input("\033[1;93m要保存对话历史吗? (y/n): \033[0m").lower().strip() == 'y':
                self.save_history("/save")
        
        goodbye_art = r"""
        ╔═══════════════════════════════════════════════════════╗
        ║                                                       ║
        ║   ██████╗  ██████╗  ██████╗ ██╗   ██╗███████╗██████╗  ║
        ║  ██╔════╝ ██╔═══██╗██╔═══██╗██║   ██║██╔════╝██╔══██╗ ║
        ║  ██║  ███╗██║   ██║██║   ██║██║   ██║█████╗  ██████╔╝ ║
        ║  ██║   ██║██║   ██║██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗ ║
        ║  ╚██████╔╝╚██████╔╝╚██████╔╝ ╚████╔╝ ███████╗██║  ██║ ║
        ║   ╚═════╝  ╚═════╝  ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝ ║
        ║                                                       ║
        ╚═══════════════════════════════════════════════════════╝
        """
        print("\033[1;96m" + goodbye_art + "\033[0m")
        typing_effect("感谢使用 ExpertGPT! 愿智慧与你同在 🌟", delay=0.05, color_code="\033[1;95m")
        sys.exit(0)
    
    def run(self):
        """运行对话循环"""
        while True:
            try:
                # 用户输入
                print("\n" + "="*60)
                user_input = input(f"\n{self.user_color}👤 你: \033[0m").strip()
                
                # 处理空输入
                if not user_input:
                    continue
                
                # 处理命令
                if self.process_command(user_input):
                    continue
                
                # 生成响应
                response = generate_text(
                    self.model,
                    self.tokenizer,
                    user_input,
                    max_length=self.generation_params["max_length"],
                    temperature=self.generation_params["temperature"],
                    top_p=self.generation_params["top_p"],
                    repetition_penalty=self.generation_params["repetition_penalty"],
                    device=self.device
                )
                
                # 显示AI响应 (带打字效果)
                print(f"\n{self.ai_color}🤖 AI: ", end="", flush=True)
                typing_effect(response, delay=0.03, color_code=self.ai_color)
                
                # 保存到历史
                self.history.append((user_input, response))
                
            except KeyboardInterrupt:
                self.exit_chat()
            except Exception as e:
                typing_effect(f"❌ 发生错误: {str(e)}", color_code="\033[1;91m")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    try:
        # 检查模型文件
        model_path = "expert_gpt_model_finetuned_final.pth"
        if not os.path.exists(model_path):
            print(f"\033[1;91m❌ 模型文件不存在: {model_path}\033[0m")
            print("\033[1;93m💡 请确保训练好的模型文件在当前目录下\033[0m")
            sys.exit(1)
        
        # 启动对话系统
        chat = ChatInterface(model_path)
        chat.run()
        
    except Exception as e:
        print(f"\033[1;91m❌ 启动失败: {str(e)}\033[0m")
        import traceback
        traceback.print_exc()
        sys.exit(1)