# 2.5 Multi-head Latent Attention (MLA)

多头潜变量注意力机制，配合 KV Cache 使用，将 KV 压缩为潜变量存储，需要计算时再恢复原 KV 表示，从而降低显存占用。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, latent_dim):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_down_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.k_up_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.v_up_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None, past_latent_kv=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, past_len + seq_len)
        past_latent_kv: None或(batch, past_len+seq_len, latent_dim)
        """
        # 获取x相关维度
        batch, seq_len, _ = x.shape
        
        # 计算QKV
        Q = self.q_proj(x)  # Q: (batch, seq_len, d_model)
        latent_kv = self.kv_down_proj(x)  # K: (batch, seq_len, latent_dim)
        
        # 使用KV Cache
        past_len = past_latent_kv.size(-2) if past_latent_kv is not None else 0
        if past_latent_kv is not None:
            # 新旧latent_kv在seq_len维度拼接: (batch, past_len+seq_len, latent_dim)
            latent_kv = torch.cat([past_latent_kv, latent_kv], dim=-2)
        new_latent_kv = latent_kv
        
        # 潜变量还原
        K = self.k_up_proj(latent_kv)  # K: (batch, past_len+seq_len, d_model)
        V = self.v_up_proj(latent_kv)  # V: (batch, past_len+seq_len, d_model)
        
        # QKV分头, Q: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # KV: (batch, num_heads, past_len + seq_len, head_dim)
        K = K.view(batch, past_len + seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, past_len + seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力得分并添加掩码, scores: (batch, num_heads, seq_len, past_len + seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 计算注意力权重和输出
        attn_weights = F.softmax(scores, dim=-1)  # attn_weights: (batch, num_heads, seq_len, past_len + seq_len)
        out = torch.matmul(attn_weights, V)  # out: (batch, num_heads, seq_len, head_dim)
        
        # 合并多头结果
        out = out.transpose(1, 2)  # out: (batch, seq_len, num_heads, head_dim)
        out = out.contiguous().view(batch, seq_len, self.d_model)
        
        # return: (batch, seq_len, d_model), (batch, past_len + seq_len, latent_dim)
        return self.o_proj(out), new_latent_kv
```

