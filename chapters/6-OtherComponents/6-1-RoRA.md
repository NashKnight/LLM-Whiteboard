# 6.1 LoRA

LoRA (Low-Rank Adaptation) 微调对原模型的权重矩阵 $W$ 添加两个可训练的低秩矩阵 $A$ 和 $B$，形成增量：
$$
W' = W + \Delta W = W + BA
$$
其中 $A \in \mathbb{R}^{r \times d}$，$B \in \mathbb{R}^{d \times r}$，$r \ll d$。

通过低秩矩阵学习特定任务，并保持原模型权重冻结，从而减少显存占用。LoRA 原文建议的初始化是 A 随机，B 为 0，这样让 LoRA 分支初始输出为 0，从而一开始完全不扰动原始预训练模型。

```python
class LoRALinear(nn.Module):
    def __init__(self, base_linear, r):
        """
        base_linear: nn.Linear(d_in, d_in)
        """
        super().__init__()

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_dim = base_linear.in_features
        out_dim = base_linear.out_features

        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)

        nn.init.zeros_(self.B.weight)  # A初始随机化，B初始为0

    def forward(self, x):
        return self.base(x) + self.B(self.A(x))
```

### 