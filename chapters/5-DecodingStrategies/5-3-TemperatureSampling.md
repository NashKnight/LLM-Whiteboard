# 5.3 Temperature Sampling

温度采样。高温分布下概率更分散，低温分布下概率更集中。

```python
import torch
import torch.nn.functional as F

def temperature_sampling(logits, temperature=1.0):
    """
    logits: (batch, vocab_size)
    return: (batch, 1)
    """
    if temperature <= 0:  # temperature=0即贪心搜索
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    logits = logits / temperature  # logits: (batch, vocab_size)
    probs = F.softmax(logits, dim=-1)  # probs: (batch, vocab_size)
    next_token = torch.multinomial(probs, num_samples=1)  # next_token: (batch, 1)
    return next_token
```

### 