# 3.2 RMSNorm

Root Mean Square Layer Normalization，均方根归一化。

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        # 计算均方根
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化并缩放
        x = x / rms
        x = self.gamma * x
        return x
```

### 