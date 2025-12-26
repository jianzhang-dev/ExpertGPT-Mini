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
    def __init__(self, texts, tokenizer, max_length=256):
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

# ====================【替换为微调数据集加载逻辑】====================
class DataManager:
    def __init__(self):
        # 添加初始化 tokenizer
        self.tokenizer = OpenSourceTokenizer()
        self.max_length = 700
        
    def load_datasets(self):
        """加载训练数据集 - 包括新增的 7.json"""
        print("正在加载数据集...")
        all_texts = []
        
        # 1. 加载 Why 问答数据
        why_path = "data/raw/why.json"  
        if os.path.exists(why_path):
            print("加载Why问答数据...")
            why_texts = self._load_why_data(why_path)
            all_texts.extend(why_texts)
            print(f"Why数据: {len(why_texts)} 条")
        
        # 2. 加载 Alpaca 数据
        alpaca_path = "data/raw/alpaca_gpt4_data_zh.json"
        if os.path.exists(alpaca_path):
            print("加载Alpaca指令数据...")
            alpaca_texts = self._load_alpaca_data(alpaca_path)
            all_texts.extend(alpaca_texts)
            print(f"Alpaca数据: {len(alpaca_texts)} 条")
        
        # 3. 加载 Firefly 数据
        firefly_path = "data/raw/firefly-train-1.1M.jsonl"
        if os.path.exists(firefly_path):
            print("加载Firefly数据...")
            firefly_texts = self._load_firefly_data(firefly_path)
            all_texts.extend(firefly_texts)
            print(f"Firefly数据: {len(firefly_texts)} 条")
        
        # 4. 【新增】加载 7.json（与上述数据集同路径）
        json7_path = "data/raw/7.json"
        if os.path.exists(json7_path):
            print("加载7.json数据...")
            json7_texts = self._load_json7_data(json7_path)
            all_texts.extend(json7_texts)
            print(f"7.json数据: {len(json7_texts)} 条")
        
        # 5. 备用数据（如果没有找到任何文件）
        if not all_texts:
            print("使用自动生成的备用数据...")
            all_texts = self._generate_backup_data()
        else:
            print(f"总共加载了 {len(all_texts)} 条文本")
        
        # 数据清洗
        cleaned_texts = self._clean_texts(all_texts)
        print(f"数据清洗完成，有效文本: {len(cleaned_texts)} 条")
        
        # 所有数据都作为训练集
        train_texts = cleaned_texts
        print(f"训练集: {len(train_texts)} 条")
        return train_texts, []

    def _load_alpaca_data(self, alpaca_path):
        """加载Alpaca数据"""
        alpaca_texts = []
        try:
            with open(alpaca_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in tqdm(data, desc="处理Alpaca数据"):
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                if instruction and output_text:
                    if input_text and input_text.strip():
                        text = f"指令：{instruction}\n输入：{input_text}\n回答：{output_text}"
                    else:
                        text = f"指令：{instruction}\n回答：{output_text}"
                    if len(text) > 3:
                        alpaca_texts.append(text)
        except Exception as e:
            print(f"Alpaca数据加载错误: {e}")
        return alpaca_texts

    def _load_firefly_data(self, firefly_path):
        """加载Firefly数据，过滤指定类别"""
        firefly_texts = []
        filtered_categories = ["MusicComment", "ClassicalChinese", "Cot", "Translation","ProductDesc"]
        kept_count = 0
        filtered_count = 0
        try:
            with open(firefly_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="处理Firefly数据"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        input_text = item.get('input', '')
                        target_text = item.get('target', '')
                        kind = item.get('kind', '')
                        # 过滤指定类别
                        if kind in filtered_categories:
                            filtered_count += 1
                            continue
                        if input_text and target_text:
                            # Firefly数据格式转换为指令格式
                            text = f"指令：完成以下任务\n输入：{input_text}\n回答：{target_text}"
                            if len(text) > 4:
                                firefly_texts.append(text)
                                kept_count += 1
                    except json.JSONDecodeError:
                        continue
            print(f"Firefly数据: 保留 {kept_count} 条，过滤 {filtered_count} 条")
        except Exception as e:
            print(f"Firefly数据加载错误: {e}")
        return firefly_texts

    def _load_why_data(self, why_path):
        """加载Why问答数据 - 流式读取大文件"""
        why_texts = []
        try:
            print(f"正在流式读取大文件: {why_path}")
            # 使用ijson流式解析大JSON文件
            try:
                import ijson
            except ImportError:
                print("请先安装ijson: pip install ijson")
                return why_texts
            
            count = 0
            with open(why_path, 'r', encoding='utf-8') as f:
                # 流式解析JSON数组中的每个对象
                parser = ijson.parse(f)
                current_item = {}
                current_key = None
                in_item = False
                for prefix, event, value in parser:
                    if prefix == 'item' and event == 'start_map':
                        in_item = True
                        current_item = {}
                    elif in_item and event == 'map_key':
                        current_key = value
                    elif in_item and event in ['string', 'number']:
                        if current_key:
                            current_item[current_key] = value
                    elif prefix == 'item' and event == 'end_map':
                        in_item = False
                        count += 1
                        # 处理当前项
                        prompt = current_item.get('prompt', '')
                        response = current_item.get('response', '')
                        if prompt and response:
                            text = f"指令：{prompt}\n回答：{response}"
                            if len(text) > 4:
                                why_texts.append(text)
                        # 每处理1000条显示进度
                        if count % 1000 == 0:
                            print(f"已处理 {count} 条数据，当前有效: {len(why_texts)} 条")
                        current_item = {}
            print(f"✅ Why数据流式读取完成: 总共{count}条，有效{len(why_texts)}条")
        except Exception as e:
            print(f"❌ Why数据加载错误: {e}")
            import traceback
            traceback.print_exc()
        return why_texts
    def _load_json7_data(self, json7_path):
        """加载 7.json 数据（每行为独立 JSON 对象）"""
        texts = []
        try:
            with open(json7_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # 分割为多行 JSON 对象（容忍末尾逗号）
                json_objects = [line.strip().rstrip(',') for line in content.split('},\n{')]
                # 修复首尾缺失的大括号
                if len(json_objects) == 1:
                    # 整个文件是一个数组
                    data = json.loads(content)
                    for item in data:
                        if 'text' in item:
                            texts.append(item['text'])
                else:
                    # 多行独立 JSON 对象
                    if not content.startswith('{'):
                        json_objects[0] = '{' + json_objects[0]
                    if not content.endswith('}'):
                        json_objects[-1] = json_objects[-1] + '}'
                    for obj_str in json_objects:
                        try:
                            item = json.loads(obj_str)
                            if 'text' in item:
                                texts.append(item['text'])
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"7.json数据加载错误: {e}")
        return texts

    def _clean_texts(self, texts):
        """清洗文本数据"""
        cleaned = []
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < 4 or len(text) > 2000:
                continue
            if any(bad in text for bad in ['人民政府','政治局','有关部门','教研','教师','党员','入党','检查机关','监察机关','监督机关','乡村振兴','中医药','党的十','中国特色','机关人员','检察机关','中国式','共同体','中华民族','党委','副部','党委书记','党中央','秘书长','党组织','党校','学习强国','抗日','HTTP','HTML','^','支书','cos','sin','公式','地方政府','\\','\\\\','-----','C++','Python','Java','国务院','质数','$',',,','。。', 'http://', 'https://', 'Copyright','中国共产党','哈马斯','国民党','乌克兰','家国情怀','习近平','台湾是中国','法轮功','国家政策','哈萨克斯坦','党的领导','改革开放','社会主义','一国两制','中共中央','中央集权','国家安全','民族团结','政治制度']):
                continue
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            if chinese_chars / len(text) < 0.3:
                continue
            cleaned.append(text)
        print(f"过滤后保留 {len(cleaned)} 条文本")
        return cleaned

    def _generate_backup_data(self):
        """生成备用训练数据"""
        backup_texts = [
            "今天天气很好，阳光明媚，适合出去散步。",
            "人工智能是计算机科学的一个分支，旨在创造能够执行智能任务的机器。",
            "中国的首都是北京，它是一座历史悠久的城市。",
            "学习编程需要耐心和实践，多写代码才能提高技能。",
            "健康的生活方式包括均衡饮食、适量运动和充足睡眠。",
            "机器学习是人工智能的重要分支，它让计算机能够从数据中学习。",
            "深度学习通过神经网络模拟人脑的工作方式，实现复杂模式识别。",
            "自然语言处理让计算机能够理解和生成人类语言。",
            "计算机视觉技术使机器能够识别和理解图像内容。",
            "强化学习通过试错机制让智能体学习最优决策策略。",
        ] * 10
        return backup_texts

    def create_dataloaders(self, batch_size=16):
        """创建数据加载器 - 只返回训练集"""
        train_texts, _ = self.load_datasets()
        random.shuffle(train_texts)
        train_dataset = TextDataset(train_texts, self.tokenizer, self.max_length)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        print(f"✅ DataLoader创建完成，共 {len(train_loader)} 个批次。")
        return train_loader, None
# ====================【替换结束】====================

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
            "你好。", "中国的首都是哪里？", "请给我讲一个关于人工智能的笑话", "你最喜欢什么水果？"
        ]
    def _run_generation_test(self, step):
        print(f"\n--- 进行生成测试，步骤 {step} ---")
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
        print(f"--- 生成测试完成，继续训练 ---\n")
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
            filename = "expert_gpt_model_finetuned_final1.pth"
        else:
            filename = f"expert_gpt_finetuned_epoch{epoch+1}.pth"
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
        print(f"🚀 开始微调训练 ExpertGPTModel，共 {epochs} 个 Epoch...")
        for epoch in range(epochs):
            self.train_epoch(epoch, scheduler, lr_history)
        print("🎉 微调训练完成！")
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
        'max_seq_len': 700, 'batch_size': 3, 'epochs': 1, 'learning_rate': 3e-5,  # 微调通常使用更小的学习率
        'warmup_ratio': 0.03, 'min_lr_ratio': 0.1  # 新增调度器参数
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌍 使用设备: {device}")
    
    data_manager = DataManager()
    train_loader, _ = data_manager.create_dataloaders(batch_size=config['batch_size'])
    config['vocab_size'] = data_manager.tokenizer.vocab_size
    
    # 创建模型
    print("🧩 正在创建 ExpertGPTModel...")
    model = ExpertGPTModel(
        vocab_size=config['vocab_size'], hidden_size=config['hidden_size'],
        num_layers=config['num_layers'], num_experts=config['num_experts'],
        num_heads=config['num_heads'], window_size=config['window_size'],
        max_seq_len=config['max_seq_len']
    )
    model.lm_head.weight = model.token_embedding.weight
    
    # ====================【关键：加载预训练模型权重进行微调】====================
    pretrained_path = "./expert_gpt_model_final.pth"  # 预训练模型路径
    if os.path.exists(pretrained_path):
        print(f"📂 加载预训练模型权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
        
        # 提取模型状态字典
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # 尝试加载权重
        try:
            # strict=False 允许部分权重不匹配
            model.load_state_dict(state_dict, strict=False)
            print("✅ 预训练权重加载成功！")
        except RuntimeError as e:
            print(f"⚠️ 警告：部分权重加载失败，将使用随机初始化: {e}")
            # 尝试部分加载
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            print(f"✅ 部分权重加载成功 ({len(pretrained_dict)}/{len(model_dict)} 层)")
    else:
        print(f"❌ 预训练模型文件不存在: {pretrained_path}")
        print("⚠️ 警告：将从头开始训练模型")
    # ====================【加载结束】====================
    
    # 参数统计
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
    print(f"  学习率: {config['learning_rate']}")
    print(f"  微调轮数: {config['epochs']}")
    
    # ====================【关键：Trainer 初始化增加调度器参数】====================
    trainer = Trainer(
        model, train_loader, device, data_manager.tokenizer,
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'],
        min_lr_ratio=config['min_lr_ratio']
    )
    trainer.train(epochs=config['epochs'])
    print("💾 微调模型已保存到 expert_gpt_model_finetuned_final1.pth")
    
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