# 1.2 Cross-Attention

交叉注意力机制。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x_q, x_kv, mask=None):
        """
        x_q: (batch, seq_q, d_model)
        x_kv: (batch, seq_kv, d_model)
        mask: (seq_q, seq_kv), True=屏蔽, broadcast 到batch维度
        """
        # 计算QKV
        Q = self.q_proj(x_q)   # Q: (batch, seq_q, d_model)
        K = self.k_proj(x_kv)  # K: (batch, seq_kv, d_model)
        V = self.v_proj(x_kv)  # V: (batch, seq_kv, d_model)
        
        # 计算注意力得分并添加掩码，scores: (batch, seq_q, seq_kv)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        
        # 计算注意力权重和输出
        attn_weights = F.softmax(scores, dim=-1)  # attn_weights: (batch, seq_q, seq_kv)
        out = torch.matmul(attn_weights, V)  # out: (batch, seq_q, d_model)
        
        return self.o_proj(out)  # return: (batch, seq_len, d_model)
```

外部调用，如果添加 `mask` 一般是 `padding mask`：

```python
batch = 2
seq_q = 3
seq_kv = 5
d_model = 8

x_q = torch.randn(batch, seq_q, d_model)
x_kv = torch.randn(batch, seq_kv, d_model)

cross_attn = CrossAttention(d_model)
out = cross_attn(x_q, x_kv)
```

## 