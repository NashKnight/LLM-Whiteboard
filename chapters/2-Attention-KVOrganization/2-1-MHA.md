# 2.1 Multi-Head Self-Attention (MHA)

多头自注意力机制。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len), True=屏蔽, broadcast到batch和num_heads维度
        """
        # 获取x的相关维度
        batch, seq_len, _ = x.shape
        
        # 计算QKV
        Q = self.q_proj(x)  # Q:(batch, seq_len, d_model)
        K = self.k_proj(x)  # K:(batch, seq_len, d_model)
        V = self.v_proj(x)  # V:(batch, seq_len, d_model)
        
        # QKV分头: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力得分并添加掩码, scores:(batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            
        # 计算注意力权重和输出
        attn_weights = F.softmax(scores, dim=-1)  # attn_weights: (batch, num_heads, seq_len, seq_len)
        out = torch.matmul(attn_weights, V)  # out: (batch, num_heads, seq_len, head_dim)
        
        # 合并多头结果
        out = out.transpose(1, 2)  # out: (batch, seq_len, num_heads, head_dim)
        out = out.contiguous().view(batch, seq_len, self.d_model)  # out: (batch, seq_len, d_model)
        
        return self.o_proj(out)  # return: (batch, seq_len, d_model)
```

外部调用：

```python
x = torch.randn(2, 4, 8)
mha = MultiHeadAttention(d_model=8, num_heads=2)
mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
out = mha(x, mask=mask)
```

**pytorch 相关基础**：

- 如果 `assert` 希望给用户提示，可以写 `assert d_model % num_heads == 0, "d_model must be a multiples of num_heads"`。

- 如果传入的 `mask` 是 `(batch, seq_len, seq_len)` 的形状（通过 `if mask.dim() == 3` 来判断），需要先执行 `mask.unsqueeze(1)` 变为 `(batch, 1, seq_len, seq_len)`，否则广播机制对齐会出错。