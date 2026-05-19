# 2.2 MHA with KV Cache

带 KV Cache 的多头注意力机制。

使用 KV Cache 的自回归生成中，每步只计算当前 token 的 Q/K/V，历史 token 的 K/V 保存在缓存中复用，从而避免重复计算加速推理。

![kv_cache](../../assets/kv_cache.png)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,  num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None, past_kv=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, past_len+seq_len), True=屏蔽, broadcast到batch和num_heads维度
        past_kv: tuple, (past_k, past_v)
        past_k/past_v: (batch, num_heads, past_len, head_dim)
        """
        # 获取x相关维度
        batch, seq_len, _ = x.shape
        
        # 计算QKV
        Q = self.q_proj(x)  # Q: (batch, seq_len, d_model)
        K = self.k_proj(x)  # K: (batch, seq_len, d_model)
        V = self.v_proj(x)  # V: (batch, seq_len, d_model)
        
        # QKV分头: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用KV Cache
        if past_kv is not None:
            past_k, past_v = past_kv
            # 新旧KV在seq_len维度拼接: (batch, num_heads, past_len+seq_len, head_dim)
            K = torch.cat([past_k, K], dim=-2)
            V = torch.cat([past_v, V], dim=-2)
        new_kv = (K, V)  # 更新KV Cache
        
        # 计算注意力得分并添加掩码, scores:(batch, num_heads, seq_len, past_len+seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 计算注意力权重和输出
        attn_weights = F.softmax(scores, dim=-1)  # attn_weights: (batch, num_heads, seq_len, past_len+seq_len)
        out = torch.matmul(attn_weights, V)  # out: (batch, num_heads, seq_len, head_dim)
        
        # 合并多头结果
        out = out.transpose(1, 2)  # out: (batch, seq_len, num_heads, head_dim)
        out = out.contiguous().view(batch, seq_len, self.d_model)  # out: (batch, seq_len, d_model)
        
        # return: (batch, seq_len, d_model), KV Cache
        return self.o_proj(out), new_kv
```

外部调用：

```python
batch = 2
d_model = 8
num_heads = 2
steps = 5

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
past_kv = None

for t in range(steps):
  # 自回归推理时，每一步只输入当前新 token
  x = torch.randn(batch, 1, d_model)
  out, past_kv = mha(x, past_kv=past_kv)

  print(f"step {t + 1}")
  print("out:", out.shape)
  print("cache_k:", past_kv[0].shape)
  print("cache_v:", past_kv[1].shape)
```

注意循环中不断传入 `past_kv`，结果输出大致如下：

```
step 1
out:     torch.Size([2, 1, 8])
cache_k: torch.Size([2, 2, 1, 4])
cache_v: torch.Size([2, 2, 1, 4])

step 2
out:     torch.Size([2, 1, 8])
cache_k: torch.Size([2, 2, 2, 4])
cache_v: torch.Size([2, 2, 2, 4])

step 3
out:     torch.Size([2, 1, 8])
cache_k: torch.Size([2, 2, 3, 4])
cache_v: torch.Size([2, 2, 3, 4])
```

从代码上，`past_kv` 保存了所有过去 token 的 KV。如果不使用 KV Cache，每次传入的 `x` 的 `seq_len` 会越来越长，因为每次 x 都得带上之前已经生成的 token 前缀，然后重新计算它们的 KV；使用 KV Cache 后，每次传入的 `x` 维度为 `(batch, 1, d_model)`。
