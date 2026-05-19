# 5.1 Greedy Decoding

贪心搜索是确定性解码，只选取概率最高的 token 输出。

```python
import torch

def greedy_decode(logits):
    """
    logits: (batch, vocab_size)
    return: (batch, 1) 返回next_token
    """
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    return next_token
```

外部调用（后续 temperature sampling, top-k/p 等调用均类似）：

```python
logits = model(input_ids)  # logits: (batch, seq_len, vocab_size)

# 只取最后一个token预测
next_token_logits = logits[:, -1, :]  # next_token_logits: (batch, vocab_size)

next_token = greedy_decode(next_token_logits)  # next_token: (batch, 1)
```

自回归循环生成的版本：

```python
import torch

def greedy_search(model, input_ids, max_new_token, eos_token_id=None):
    """
    model: decoder-only LM
    input_ids: (batch, prompt_len) 用户输入的prompt
    max_new_token: 最多生成的新token数量
    eos_token_id: 结束符token id
    return: (batch, seq_len)
    """

    generated = input_ids
    for _ in range(max_new_token):
        # 不用KV Cache的话每步都输入完整序列
        logits = model(generated)  # logits: (batch, seq_len, vocab_size)
        # 只取最后一个token预测
        next_token_logits = logits[:, -1, :]  # next_token_logits: (batch, vocab_size)
        # 贪心解码
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # next_token: (batch, 1)
        # 在序列维度合并到已生成序列
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
    return generated
```

这里短样本生成 `<eos>` 以后还会继续生成浪费计算，一般实际工程中会再做处理（如维护 `finished` 记录已经生成结束的样本），但在面试手撕时可以简化。