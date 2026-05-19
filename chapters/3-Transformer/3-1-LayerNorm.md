# 3.1 LayerNorm

层归一化。

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        # 初始化gamma=1,beta=0
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        #归一化并缩放平移
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.gamma * x + self.beta
        return x
```

**pytorch 相关基础**

- 对张量某一维特征计算统计量聚合时，`keepdim=True` 保证聚合后该维度为1，并不会消失而减少维度。