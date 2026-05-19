# 5.5 Top-p Sampling

Top-p 采样。每次从概率高到低选取累计概率至少覆盖 p 的 token 进行采样输出。

```python
import torch
import torch.nn.functional as F

def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    logits: (batch, vocab_size)
    """
    if temperature <= 0:  # temperature=0即贪心搜索
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # 温度采样和降序排序, sorted_logits/idx: (batch, vocab_size)
    logits = logits / temperature
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    
    # 计算累计概率, sorted_probs / cum_probs: (batch, vocab_size)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 大于p的位置使用掩码-inf, 并至少保留第一个超过p的token，使累计概率覆盖p
    sorted_mask = cum_probs > p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()  # 全体右移1位
    sorted_mask[:, 0] = False  # 第一个位置不掩码
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))
    
    # 只对top-p进行softmax归一化，并进行采样，映射回原始下标
    probs = F.softmax(sorted_logits, dim=-1)  # probs: (batch, vocab_size)
    sampled_idx = torch.multinomial(probs, num_samples=1)  # sampled_idx: (batch, 1)
    next_token = torch.gather(sorted_idx, dim=-1, index=sampled_idx)  # next_token: (batch, 1)
    return next_token
```

