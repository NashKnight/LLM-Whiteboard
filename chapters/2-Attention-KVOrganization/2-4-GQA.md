# 2.4 Grouped-Query Attention (GQA)

分组查询注意力机制。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        super().__init__()
        assert d_model % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.num_groups = num_q_heads // num_kv_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # kv的线性层映射到 head_dim * num_kv_heads 维
        self.k_proj = nn.Linear(d_model, self.head_dim * self.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * self.num_kv_heads, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len), True=屏蔽, broadcast到batch和num_q_heads维度
        """
        # 获取x相关维度
        batch, seq_len, _ = x.shape
        
        # 计算QKV
        Q = self.q_proj(x)  # Q: (batch, seq_len, d_model)
        K = self.k_proj(x)  # K: (batch, seq_len, head_dim*num_kv_heads)
        V = self.v_proj(x)  # V: (batch, seq_len, head_dim*num_kv_heads)
        
        # Q分头, Q: (batch, num_q_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        # KV分头，KV: (batch, num_kv_heads, seq_len, head_dim)
        K = K.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 扩展KV: (batch, num_q_heads, seq_len, head_dim)
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)
        
        # 计算注意力得分并添加掩码, scores: (batch, num_q_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 计算注意力权重和输出
        attn_weights = F.softmax(scores, dim=-1)  # attn_weights: (batch, num_q_heads, seq_len, seq_len)
        out = torch.matmul(attn_weights, V)  # out: (batch, num_q_heads, seq_len, head_dim)
        
        # 合并多头结果
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch, seq_len, self.d_model)
        
        return self.o_proj(out)  # return: (batch, seq_len, d_model)
```

外部调用：

```python
batch = 2
seq_len = 4
d_model = 8
num_q_heads = 4
num_kv_heads = 2
x = torch.randn(batch, seq_len, d_model)

gqa = GroupedQueryAttention(
  d_model=d_model,
  num_q_heads=num_q_heads,
  num_kv_heads=num_kv_heads
)

mask = torch.triu(
  torch.ones(seq_len, seq_len, dtype=torch.bool),
  diagonal=1,
)

out = gqa(x, mask=mask)
```

**pytorch 相关基础**

- `K` 和 `V` 的 head 维度不是1，因此不能自动广播到 `Q` 的 head 维度。