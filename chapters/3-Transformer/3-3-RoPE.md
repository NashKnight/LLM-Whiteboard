# 3.3 RoPE

旋转位置编码。

对第 $m$ 个位置、某一对维度 $(2i,\,2i+1)$，RoPE 做二维旋转：
$$
\begin{pmatrix}
x'_{2i}\\
x'_{2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i)\\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
x_{2i}\\
x_{2i+1}
\end{pmatrix}
$$
其中
$$
\theta_i = 10000^{-2i/d}
$$
展开就是：
$$
x'_{2i}=x_{2i}\cos(m\theta_i)-x_{2i+1}\sin(m\theta_i)\\
x'_{2i+1}=x_{2i}\sin(m\theta_i)+x_{2i+1}\cos(m\theta_i)
$$
对 $Q,K$ 都这样旋转。其关键性质是：
$$
\langle R_m q,\; R_n k\rangle
=
\langle q,\; R_{n-m} k\rangle
$$
所以注意力里自然编码了相对位置 $n-m$。

```python
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, d_model, base=10000):
        super().__init__()
        assert d_model % 2 == 0
        
        self.d_model = d_model
        self.base = base
    
    def forward(self, x):
        """
        (x: batch, seq_len, d_model)
        """
        # 获取x相关信息
        seq_len = x.shape[-2]
        device = x.device
        
        # 生成频率, freq: (d_model / 2,)
        dim = torch.arange(0, self.d_model, 2, device=device)  # dim: (d_model / 2,)
        freq = self.base ** (-dim / self.d_model)
        
        # 生成旋转角, theta: (seq_len, d_model / 2)
        pos = torch.arange(seq_len, device=device)  # pos: (seq_len,)
        theta = torch.outer(pos, freq)
        
        # 计算正弦余弦值, cos/sin: (seq_len, d_model / 2)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        
        # x拆分为奇偶维, x_odd/x_even: (batch, seq_len, d_model / 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        # 计算奇偶维RoPE, out_odd, out_even: (batch, seq_len, d_model / 2)
        out_even = cos * x_even - sin * x_odd
        out_odd = sin * x_even + cos * x_odd
        
        # 合并奇偶维, out: (batch, seq_len, d_model)
        out = torch.zeros_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out
```

- 使用 `torch.arange, torch.randn, torch.ones, torch.zeros` 等创建新张量（例如本题中的 `dim` 和 `pos`）时，如果不指定 `device` 默认创建在 cpu，而一般传入的张量 `x` 会在 gpu 上，如果不指定 `device=device`，把新张量放在跟 `x` 同样的设备上，就会报错：`Expected all tensors to be on the same device`。如果使用 `torch.zeros_like(x)`，则会直接继承 `x` 的 `shape, device, dtype, layout`。
- `...` 可以省略部分维度，可以出现在开头、中间、结尾，但一个索引表达式里只能有一个。

自注意力机制调用 RoPE：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    """
    使用上面定义好的RoPE类，此处省略
    """

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0

        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(d_model)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len), True=屏蔽, broadcast到batch维度
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # RoPE 只作用在 Q 和 K 上
        Q = self.rope(Q)
        K = self.rope(K)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        return self.o_proj(out)
```

