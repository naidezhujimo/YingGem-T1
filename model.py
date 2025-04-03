import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


class YingGemConfig:
    def __init__(self):
        self.vocab_size = 50257           # 词汇表大小
        self.hidden_size = 768         # 隐藏层维度
        self.num_hidden_layers = 12      # Transformer层
        self.num_attention_heads = 12    # 注意力头数
        self.head_dim = self.hidden_size // self.num_attention_heads # 每个注意力头的维度
        self.intermediate_size = 1024   # FFN的中间层维度
        self.max_position_embeddings = 1024 # 最大序列长度
        self.rms_norm_eps = 1e-6        # RMSNorm的epsilon值
        self.rope_theta = 10000.0       # RoPE的theta值
        self.pad_token_id = 0           # 填充符的ID
        self.device = 'cuda'
        self.attention_window = 1024     # 滑动窗口大小
        self.use_flash_attn = False      # 是否使用FlashAttention
        self.attention_dropout = 0.2    # 注意力dropout

# 替换LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # 每个样本的平方均值的倒数平方根并应用缩放
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x)
    
# 旋转位置嵌入
class RoPE(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # 计算频率
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
    
    def _apply_rotary(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, self.freqs) # 计算时间步和频率的外积
        freqs = torch.cat((freqs, freqs), dim=-1) # 将频率扩展到与输入维度一致
        return x * freqs.cos() + self._rotate_half(x) * freqs.sin()

    # 旋转
    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # 对查询（q）和键（k）应用RoPE
    def forward(self, q, k):
        seq_len = q.size(2)
        q = self._apply_rotary(q, seq_len)
        k = self._apply_rotary(k, seq_len)
        return q, k
    
class YingGemAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 查询、键和值的投影
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 输出投影
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.rope = RoPE(config.head_dim, config.rope_theta)
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def _create_sliding_mask(self, seq_len, device):
        """创建滑动窗口注意力掩码"""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            start = max(0, i - self.config.attention_window + 1)
            mask[i, :start] = False
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
    
    def forward(self, x, mask=None, kv_cache=None):
        B, T, C = x.size()
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 重塑为多头形式
        q = q.view(B, T, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(B, T, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        
        q, k = self.rope(q, k)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            # 滑动窗口截断
            if k_prev.size(2) >= self.config.attention_window:
                k_prev = k_prev[:, :, -self.config.attention_window+1:, :]
                v_prev = v_prev[:, :, -self.config.attention_window+1:, :]
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 应用滑动窗口掩码
        window_mask = self._create_sliding_mask(T, x.device)
        attn = attn.masked_fill(~window_mask, float('-inf'))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn_output = attn @ v
        # 重塑输出形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  
        return self.o_proj(attn_output), (k.detach(), v.detach())
    
# GeGLU激活函数
class GeGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.dim_out = dim_out

    # 将输出分为两部分，一部分作为激活门，另一部分作为激活值
    def forward(self, x):
        x_proj = self.proj(x)
        x, gate = x_proj.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)
    
class YingGemBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = YingGemAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = nn.Sequential(
            GeGLU(config.hidden_size, config.intermediate_size),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )

    
    def forward(self, x, mask=None, kv_cache=None):
        attn_out, kv_cache = self.attn(self.attn_norm(x), mask, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, kv_cache
    
class YingGem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.block = nn.ModuleList([YingGemBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.config.num_hidden_layers))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None, mask=None):
        B, T = x.size()
        x = self.embed(x)

        kv_caches = []
        for block in self.block:
            x, kv_cache = block(x, mask)
            kv_caches.append(kv_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_token_id
            )
        return logits, loss
    

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.2):  # 新增top_p参数
        self.eval()
        generated = input_ids.to(self.config.device)
        past_kv = None

        with torch.no_grad():
            for _ in range(max_length):
                if past_kv is None:
                    outputs, past_kv = self(generated)
                else:
                    outputs, new_kv = self(generated[:, -1:], kv_cache=past_kv)
                    # 修改缓存合并逻辑（应用滑动窗口）
                    new_past_kv = []
                    for (k_prev, v_prev), (k_new, v_new) in zip(past_kv, new_kv):
                        if k_prev.size(2) >= self.config.attention_window:
                            k_prev = k_prev[:, :, -self.config.attention_window+1:, :]
                            v_prev = v_prev[:, :, -self.config.attention_window+1:, :]
                        k = torch.cat([k_prev, k_new], dim=2)
                        v = torch.cat([v_prev, v_new], dim=2)
                        new_past_kv.append((k, v))
                    past_kv = new_past_kv
                    
                logits = outputs[:, -1, :] / temperature
                # 在logits计算后添加重复惩罚
                if repetition_penalty != 1.0:
                    for token in generated[0]:
                        logits[0, token] /= repetition_penalty  # 降低已生成token的概率

                # 先应用top_k过滤
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # 再应用top_p过滤
                if top_p is not None and top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 总是保留第一个token以防所有token都被移除
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # 将需要移除的token设为负无穷
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, -float('Inf'))

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)

        return self.tokenizer.decode(generated[0].cpu().tolist())

import tiktoken

class ShakespeareDataset(Dataset):
    def __init__(self, file_path, block_size=1024, mode='train', split_ratio=0.8):
        # 使用tiktoken的GPT-2编码器
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 编码文本（注意：tiktoken返回的是整数列表）
        encoded = self.tokenizer.encode_ordinary(text)
        
        # 分割数据集
        split_idx = int(len(encoded) * split_ratio)
        if mode == 'train':
            self.data = encoded[:split_idx]
        else:
            self.data = encoded[split_idx:]
        
        self.block_size = block_size
        self.vocab_size = self.tokenizer.n_vocab  # 获取词汇表大小

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self.data[start:end]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
def train():
    config = YingGemConfig()
    config.vocab_size = 128  # 根据实际数据集调整

    # 创建训练集和验证集
    train_dataset = ShakespeareDataset("shakespeare.txt", mode='train')
    val_dataset = ShakespeareDataset("shakespeare.txt", mode='val')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    config.vocab_size = train_dataset.vocab_size  # 动态获取词汇表大小
    model = YingGem(config).to(config.device)
    model.tokenizer = train_dataset.tokenizer  # 将tokenizer绑定到模型
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # 使用Adafactor优化器（Gemma推荐）
    from transformers.optimization import Adafactor
    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=True,
        relative_step=True,
        warmup_init=True
    )

    # 混合精度训练
    scaler = torch.amp.GradScaler()

    patience = 3     # 允许验证损失连续上升的轮次
    epochs_no_improve = 0
    best_val_loss = float('inf')
    train_losses = []  # 记录训练集损失
    val_losses = []    # 记录验证集损失

    for epoch in tqdm(range(100), desc="Training", unit="epoch"):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                x, y = x.to(config.device), y.to(config.device)
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                _, loss = model(x, y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        # 早停判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "yinggem_best.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch}")
                break

    iterations = list(range(len(train_losses)))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, label="Train Loss", color="blue")
    plt.plot(iterations, val_losses, label="Validation Loss", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")  # 保存图像
    plt.show()  # 显示图像

    print("Training completed!")

import argparse
if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="Sparse MoE Language Model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate text using the trained model")
    args = parser.parse_args()

    if args.train:
        # 训练模型
        train()

    if args.generate:
        # Load the tokenizer and attach it to the model
        tokenizer = tiktoken.get_encoding("gpt2")
        config = YingGemConfig()
        model = YingGem(config).to(config.device)
        model.tokenizer = tokenizer  # Attach the tokenizer
        
        model.load_state_dict(torch.load("yinggem_best.pth", weights_only=True))  # Add weights_only=True for security
        
        # Generate text
        start_text = "\n"
        input_ids = torch.tensor(
            [model.tokenizer.encode_ordinary(start_text)], 
            dtype=torch.long
        ).to(config.device)

        generated_text = model.generate(input_ids, max_length=100, temperature=0.9, top_p=0.85)
        print(generated_text)