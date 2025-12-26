import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
import random
import time
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

# ====================【新增：余弦退火调度器】====================
class CosineLRScheduler:
    @staticmethod
    def get_cosine_schedule(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# ====================【新增结束】====================

# 首先，我们需要一个辅助函数来创建因果滑动窗口掩码
def create_causal_sliding_window_mask(seq_len, window_size):
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    dists = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
    sliding_window_mask = (dists >= 0) & (dists < window_size)
    return sliding_window_mask

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
    def __init__(self, hidden_size=384, num_experts=4, num_heads=6, window_size=16, is_global=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.window_size = window_size
        self.is_global = is_global  # 【新增】保存是否为全局注意力层
        
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
                # 【新增】全局模式：标准的下三角因果掩码 (Full Causal Mask)
                # 允许看到所有之前的 token，打破 window_size 限制
                mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            else:
                # 【原有】局部模式：滑动窗口掩码
                mask = create_causal_sliding_window_mask(seq_len, self.window_size)
            
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
    def __init__(self, hidden_size=384, num_experts=4, num_heads=6, window_size=16, is_global=False):
        super().__init__()
        # 【修改】将 is_global 参数传递给 ParallelExpertAttention
        self.self_attn = ParallelExpertAttention(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_heads=num_heads,
            window_size=window_size,
            is_global=is_global  # 传递标志位
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

class ExpertGPTModel(nn.Module):
    def __init__(self, vocab_size=21128, hidden_size=384, num_layers=4,
                 num_experts=4, num_heads=6, window_size=16, max_seq_len=256):
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
        
        # 【修改】计算全局注意力层的索引 (1/3 和 2/3 处)
        # 使用集合处理，防止层数极少时索引重复
        global_layer_indices = {num_layers // 3, (num_layers * 2) // 3}
        
        # 【修改】动态构建层列表，传入 is_global 参数
        layers = []
        for i in range(num_layers):
            is_global = i in global_layer_indices
            if is_global:
                print(f"  -> Layer {i}: 设置为全局注意力层 (Global Attention)")
            
            layers.append(
                ExpertTransformerLayer(
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    num_heads=num_heads,
                    window_size=window_size,
                    is_global=is_global  # 传入当前层是否为全局的标志
                )
            )
        self.layers = nn.ModuleList(layers)

        self.final_norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, rope=self.rope)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids, max_length=50, temperature=0.8, top_p=0.9, repetition_penalty=1.2):
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

# --- 复用您代码库中的数据处理和训练组件 ---
class OpenSourceTokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.vocab_size = self.tokenizer.vocab_size
        print(f"✅ 使用开源分词器，词汇表大小: {self.vocab_size}")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=700):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        tokens = tokens + [102]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {'input_ids': input_ids, 'labels': input_ids.clone()}

class DataManager:
    def __init__(self):
        self.tokenizer = OpenSourceTokenizer()
        self.max_length = 700

    def load_datasets(self):
        print("⏳ 正在加载预训练数据集...")
        all_texts = []
        data_files = ["data/raw/1.json", "data/raw/2.json", "data/raw/3.json",
                      "data/raw/4.jsonl", "data/raw/5.jsonl", "data/raw/6.json",'data/raw/7.json']
        for file_path in data_files:
            if os.path.exists(file_path):
                print(f"  -> 正在加载: {file_path}")
                if file_path.endswith('.jsonl'):
                    if "5.jsonl" in file_path:
                        file_texts = self._load_qa_jsonl_data(file_path)
                    else:
                        file_texts = self._load_jsonl_data(file_path)
                else:
                    if "6.json" in file_path:
                        file_texts = self._load_conversation_data(file_path)
                    else:
                        file_texts = self._load_pretrain_data(file_path)
                all_texts.extend(file_texts)
                print(f"     已加载 {len(file_texts)} 条文本")
            else:
                print(f"⚠️ 警告: 数据文件不存在 {file_path}")
        if not all_texts:
            print("🛑 错误: 未找到任何数据文件，将使用少量备用数据进行演示。")
            all_texts = self._generate_backup_data()
        else:
            print(f"✅ 总共加载了 {len(all_texts)} 条原始文本")
        cleaned_texts = self._clean_texts(all_texts)
        print(f"✅ 数据清洗完成，剩余有效文本: {len(cleaned_texts)} 条")
        return cleaned_texts, []

    def _load_conversation_data(self, file_path):
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"   解析对话数据", unit="行"):
                try:
                    item = json.loads(line.strip())
                    conv_text = "".join([turn.get('value', '') for turn in item.get('conversations', [])])
                    if conv_text: texts.append(conv_text)
                except (json.JSONDecodeError, AttributeError):
                    continue
        return texts

    def _load_qa_jsonl_data(self, file_path):
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"   解析QA数据", unit="行"):
                try:
                    item = json.loads(line.strip())
                    combined_text = f"{item.get('question', '')} {item.get('answer', '')}"
                    if len(combined_text) > 2: texts.append(combined_text)
                except (json.JSONDecodeError, AttributeError):
                    continue
        return texts

    def _load_pretrain_data(self, file_path):
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in tqdm(data, desc=f"   解析JSON数据", unit="条"):
            text = item.get('text', '')
            if text: texts.append(text)
        return texts

    def _load_jsonl_data(self, file_path):
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"   解析JSONL数据", unit="行"):
                try:
                    item = json.loads(line.strip())
                    text = item.get('text', '')
                    if text: texts.append(text)
                except (json.JSONDecodeError, AttributeError):
                    continue
        return texts

    def _clean_texts(self, texts):
        cleaned = [text.strip() for text in texts if isinstance(text, str) and 4 < len(text.strip()) < 2000]
        return cleaned

    def _generate_backup_data(self):
        return ["这是一个备用句子。", "另一个用于演示的句子。"] * 100

    def create_dataloaders(self, batch_size=16):
        train_texts, _ = self.load_datasets()
        random.shuffle(train_texts)
        train_dataset = TextDataset(train_texts, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"✅ DataLoader创建完成，共 {len(train_loader)} 个批次。")
        return train_loader, None

def causal_loss(logits, labels, ignore_index=0):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    return loss

# ====================【核心修改：Trainer 集成调度器 + weight decay 分组 + AMP】====================
class Trainer:
    def __init__(self, model, train_loader, device, tokenizer, learning_rate=1.5e-4,
                 warmup_ratio=0.0, min_lr_ratio=0.1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio

        # --- 混合精度 ---
        self.use_amp = (device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        print(f"✅ 训练器初始化完成，混合精度训练: {'启用' if self.use_amp else '禁用'}")

        # --- Weight decay 分组 ---
        no_decay = ["bias", "LayerNorm.weight", "RMSNorm.weight", "norm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.98))

        # --- 固定测试用例 ---
        self.fixed_prompts = [
            "你好。", "中国的首都是哪里？", "请给我讲一个关于人工智能的笑话", "你最喜欢什么水果？",'扮演一个女仆'
        ]

    def _run_generation_test(self, step):
        print(f"\n---  przeprowadzenie testu generowania w kroku {step} ---")
        self.model.eval()
        for prompt in self.fixed_prompts:
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            generated_ids = self.model.generate(
                input_tensor,
                max_length=50,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2
            )
            generated_text = self.tokenizer.decode(generated_ids[0].tolist())
            print(f"  Prompt    : {prompt}")
            print(f"  Generated : {generated_text}")
            print("-" * 20)
        self.model.train()
        print(f"--- Zakończono test generowania, wznowienie treningu ---\n")

    def save_checkpoint(self, epoch, lr_history=None, is_final=False):
        import gc
        print("正在释放内存并准备保存模型...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        config_dict = {
            'vocab_size': self.model.vocab_size,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'num_heads': self.model.num_heads,
            'max_seq_len': self.model.max_seq_len,
            'num_experts': getattr(self.model, 'num_experts', 4),
            'window_size': getattr(self.model, 'window_size', 16),
        }
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'config': config_dict,
            'lr_history': lr_history or [],
        }
        if is_final:
            filename = "expert_gpt_model_final.pth"
        else:
            filename = f"expert_gpt_checkpoint_epoch{epoch+1}.pth"
        torch.save(checkpoint, filename, _use_new_zipfile_serialization=False)
        print(f"✅ 检查点已保存: {filename}")

    def train_epoch(self, epoch, scheduler, lr_history):
        self.model.train()
        total_loss = 0
        total_steps = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} 训练中', unit='batch')
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(input_ids)
                loss = causal_loss(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ 检测到Loss为NaN/Inf，跳过本次更新！")
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{total_loss / (progress_bar.n + 1):.4f}', 'lr': f'{current_lr:.2e}'})

            if (batch_idx + 1) % 400 == 0:
                self._run_generation_test(step=batch_idx + 1)

        avg_loss = total_loss / len(self.train_loader)
        print(f"✅ Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        return avg_loss

    def train(self, epochs):
        self.epochs = epochs
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(len(self.train_loader) * self.warmup_ratio)
        scheduler = CosineLRScheduler.get_cosine_schedule(
            self.optimizer, warmup_steps, total_steps, self.min_lr_ratio
        )
        lr_history = []

        print(f"🚀 开始训练您的 ExpertGPTModel，共 {epochs} 个 Epoch...")
        for epoch in range(epochs):
            self.train_epoch(epoch, scheduler, lr_history)
        print("🎉 训练完成！")
        self.save_checkpoint(epoch=epochs-1, lr_history=lr_history, is_final=True)
# ====================【修改结束】====================

def generate_text(model, tokenizer, prompt, max_length=50, device='cpu', temperature=0.8, top_p=0.9, repetition_penalty=1.2):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated_ids = model.generate(
        input_tensor,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    return tokenizer.decode(generated_ids[0].tolist())

def main():
    config = {
        'vocab_size': 21128, 'hidden_size': 768, 'num_layers': 12,
        'num_experts': 2, 'num_heads': 12, 'window_size': 32,
        'max_seq_len': 700, 'batch_size': 3, 'epochs': 1, 'learning_rate': 1.4e-4,
        'warmup_ratio': 0.03, 'min_lr_ratio': 0.1  # 新增调度器参数
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌍 使用设备: {device}")

    data_manager = DataManager()
    train_loader, _ = data_manager.create_dataloaders(batch_size=config['batch_size'])
    config['vocab_size'] = data_manager.tokenizer.vocab_size

    print("🧩 正在创建 ExpertGPTModel...")
    model = ExpertGPTModel(
        vocab_size=config['vocab_size'], hidden_size=config['hidden_size'],
        num_layers=config['num_layers'], num_experts=config['num_experts'],
        num_heads=config['num_heads'], window_size=config['window_size'],
        max_seq_len=config['max_seq_len']
    )
    model.lm_head.weight = model.token_embedding.weight

    # 参数统计（略，保持原样）
    print("\n📊 模型参数统计:")
    print("-" * 50)
    total_params = 0
    trainable_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)) or isinstance(module, nn.Parameter):
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                total_params += module_params
                trainable_params += module_params
                if "token_embedding" in name or "lm_head" in name or "expert" in name or "proj" in name:
                    print(f"  {name:30s}: {module_params:>10,} 参数")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            total_params += param.numel()
    print("-" * 50)
    print(f"  🎯 总计可训练参数: {trainable_params:,}")
    print(f"  📈 模型总参数量: {total_params:,}")
    print(f"    ≈ {total_params / 1e6:.2f}M 参数")
    print(f"    ≈ {total_params / 1e9:.3f}B 参数")
    print("-" * 50)

    param_types = {}
    for name, param in model.named_parameters():
        param_type = name.split('.')[-2] if len(name.split('.')) >= 2 else 'other'
        if 'weight' in name or 'bias' in name:
            param_type = name.split('.')[-1]
        if param_type not in param_types:
            param_types[param_type] = 0
        param_types[param_type] += param.numel()
    for ptype, count in sorted(param_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_params * 100
        print(f"  {ptype:15s}: {count:>12,}  ({percentage:.1f}%)")

    print(f"\n🏗️  模型架构:")
    print(f"  隐藏维度: {config['hidden_size']}")
    print(f"  层数: {config['num_layers']}")
    print(f"  注意力头数: {config['num_heads']}")
    print(f"  专家数: {config['num_experts']}")
    print(f"  词汇表大小: {config['vocab_size']:,}")
    print(f"  最大序列长度: {config['max_seq_len']}")

    # ====================【关键：Trainer 初始化增加调度器参数】====================
    trainer = Trainer(
        model, train_loader, device, data_manager.tokenizer,
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'],
        min_lr_ratio=config['min_lr_ratio']
    )
    trainer.train(epochs=config['epochs'])

    print("💾 模型已保存到 expert_gpt_model_final.pth")

    print("\n--- 🤖 进入推理模式 ---")
    prompt = "中国的首都是哪里？"
    generated = generate_text(model, data_manager.tokenizer, prompt, device=device)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    prompt = "请给我讲一个关于人工智能的笑话"
    generated = generate_text(model, data_manager.tokenizer, prompt, device=device)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")

if __name__ == '__main__':
    seed = int(time.time() * 1000) % 2**32
    torch.manual_seed(seed)
    random.seed(seed)
    main()