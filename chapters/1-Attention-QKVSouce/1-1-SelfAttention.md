# 1.1 Self-Attention

自注意力机制。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len), True=屏蔽, broadcast到batch维度
        """
        # 计算QKV
        Q = self.q_proj(x)  # Q: (batch, seq_len, d_model)
        K = self.k_proj(x)  # K: (batch, seq_len, d_model)
        V = self.v_proj(x)  # V: (batch, seq_len, d_model)
        
        # 计算注意力得分并添加掩码，scores: (batch, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        
        # 计算注意力权重和输出
        attn_weights = F.softmax(scores, dim=-1)  # attn_weights: (batch, seq_len, seq_len)
        out = torch.matmul(attn_weights, V)  # out: (batch, seq_len, d_model)
        return self.o_proj(out)  # return: (batch, seq_len, d_model)
```

外部调用 demo：

```python
import torch
batch = 2
seq_len = 4
d_model = 8

# 不传mask的版本
x = torch.randn(batch, seq_len, d_model)
attn = SelfAttention(d_model=d_model)
out = attn(x)

# 传mask的版本
x = torch.randn(batch, seq_len, d_model)
mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
out = attn(x, mask=mask)
```

**pytorch 相关基础**

- `super().__init__()` 调用所继承父类的初始化方法。
- 线性层 `self.W = nn.Linear(···)`  被 `self.W(x)` 调用时，实际会对 `x` 的最后一维执行 `x @ W^T + b`。
- `if mask is not None` 不能替换为 `if mask`，因为 `mask` 本身是布尔型张量，如果直接 `if mask`，pytorch 会直接报错：`Boolean value of Tensor with more than one value is ambiguous`。 
- `torch.triu()` 建立上三角矩阵，`diagonal=1` 保留主对角线上方（不含主对角线），`diagnal=0` 保留主对角线及其上方（包含主对角线），`digoanal=-1` 保留主对角线下方1条线及其上方。这里对于某个当前 token 不应该对自己加掩码，因此应设置 `mask=False`。