# 4.1 SFT (Cross Entropy)

有监督微调 SFT 损失是 next-token prediction 的交叉熵损失：
$$
\mathcal{L}_{\text{SFT}}
=
-\frac{1}{N}
\sum_{t \in \text{response tokens}}
\log p_\theta(y_t \mid x, y_{<t})
$$
其中：

- $x$：prompt
- $y_t$：第 $t$ 个目标 token
- $y_{<t}$：前面已经生成的目标 token

直接使用 torch 中的 `cross_entropy`：

```python
import torch.nn.functional as F

# logits: (batch, seq_len, vocab_size)
# labels: (batch, seq_len)

# shift_logits: (batch, seq_len-1, vocab_size)，去掉最后一个token（结束符后面没有token了）
shift_logits = logits[:, :-1, :].contiguous() 
# shift_labels: (batch, seq_len-1)，去掉第一个token（起始不需要预测）
shift_labels = labels[:, 1:].contiguous()

loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)
```

自己实现 `cross_entropy`（一般不强制要求 `ignore_index` 参数）：

```python
import torch

def cross_entropy(logits, labels):
    """
    logits: (batch * (seq_len-1), vocab_size)
    labels: (batch * (seq_len-1),)
    """
    # 防止softmax溢出, logits: (batch * (seq_len-1), vocab_size)
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    # log_softmax: (batch * (seq_len-1), vocab_size)
    exp_logits = torch.exp(logits)
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
    log_probs = torch.log(probs)
    # 取对应标签id对应位置的负对数
    n = labels.size(0)
    loss = -log_probs[torch.arange(n, device=labels.device), labels]
    return loss.mean()

# logits: (batch, seq_len, vocab_size)
# labels: (batch, seq_len)

# shift_logits: (batch, seq_len-1, vocab_size)，去掉最后一个token（结束符后面没有token了）
shift_logits = logits[:, :-1, :].contiguous() 
# shift_labels: (batch, seq_len-1)，去掉第一个token（起始不需要预测）
shift_labels = labels[:, 1:].contiguous()

loss = cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
```

**pytorch 基础**：

- `torch.nn.functional.cross_entropy()` 中 `ignore_index=-100` 表示 `label = -100` 的 token 不需要计算 loss，例如 padding 的地方或者 prompt 就可以设置其 `label = -100`。