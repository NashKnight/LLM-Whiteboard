# 5.2 Beam Search

集束搜索，每次保留多条累计分数最高的生成路径，最终返回分数最高的路径。以下为 `beam_size=2` 的示例。

![image-beam_search](../../assets/beam_search.png)

```python
import torch
import torch.nn.functional as F

def beam_search(model, input_ids, max_new_tokens, beam_size=4, eos_token_id=None):
    """
    model: decoder-only LM
    input_ids: (batch, prompt_len) 用户输入的prompt
    max_new_token: 最多生成的新token数量
    beam_size: 每步保留的候选路径数
    eos_token_id: 结束符token id
    return: (batch, seq_len) 得分最高的序列
    """
    batch, prompt_len = input_ids.shape
    # 每个样本复制beam_size份
    input_ids = input_ids.unsqueeze(1).repeat(1, beam_size, 1)  # input_ids: (batch, beam_size, prompt_len)
    input_ids = input_ids.view(batch * beam_size, prompt_len)  # input_ids: (batch * beam_size, prompt_len)
    # 初始化beam分数, beam_scores: (batch, beam_size)
    beam_scores = torch.zeros(batch, beam_size, device=input_ids.device)
    beam_scores[:, 1:] = -1e9  # 初始序列都是prompt，第一步生成一次即可，其他分数设为负无穷
    
    for _ in range(max_new_tokens):
        # step1: 生成对数概率
        logits = model(input_ids)  # logits: (batch * beam_size, seq_len, vocab_size)
        next_token_logits = logits[:, -1, :]  # next_token_logits: (batch * beam_size, vocab_size)
        logprobs = F.log_softmax(next_token_logits, dim=-1)  # logprobs: (batch * beam_size, vocab_size) 经softmax归一化以后的对数概率
        vocab_size = logits.size(-1)
        logprobs = logprobs.view(batch, beam_size, vocab_size)  # logprobs: (batch, beam_size, vocab_size)
        
        # step2: 计算各条路径得分
        scores = logprobs + beam_scores.unsqueeze(-1)  # scores: (batch, beam_size, vocab_size)
        scores = scores.view(batch, beam_size * vocab_size)  # scores:(batch, beam_size * vocab_size)
        
        # step3: 取出得分最高的beam_size条路径信息，并追溯beam_id, token_id
        # next_scores/ids/beam_ids/tokens: (batch, beam_size)
        next_scores, next_ids = torch.topk(scores, beam_size, dim=-1)
        next_beam_ids = next_ids // vocab_size  # 追溯来自哪条beam
        next_tokens = next_ids % vocab_size  # 追溯token id
        
        # step4: 取出得分最高的beam_size条上一轮序列
        input_ids = input_ids.view(batch, beam_size, -1)  # input_ids: (batch, beam_size, seq_len)
        # 为了能从input_ids中gather出next_beam_ids的对应beam，要保证它们维度相同
        gather_ids = next_beam_ids.unsqueeze(-1).expand(batch, beam_size, input_ids.size(-1))
        input_ids = torch.gather(input_ids, dim=1, index=gather_ids)  # input_ids: (batch, beam_size, seq_len)
        
        # step5: 拼接新token并更新beam_scores, 重新展平input_ids作为下一轮模型输入
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)  # input_ids: (batch, beam_size, seq_len+1)
        beam_scores = next_scores
        input_ids = input_ids.view(batch * beam_size, -1)  # input_ids: (batch * beam_size, seq_len)
        
        if eos_token_id is not None and (next_tokens == eos_token_id).all():
            break
    
    # 每条样本取分数最高的beam
    input_ids = input_ids.view(batch, beam_size, -1)
    best_seq = input_ids[:, 0, :]  # topk会降序排列，取第一个即分数高
    return best_seq
```

**pytorch 相关基础**：

- `x.repeat(1, beam_size, 1)` 即 `x` 的三个维度分别复制1次、`beam_size` 次、1次。`repeat()` 会将对应序列整体重复生成，例如 `repeat(2)` 将 `[1,2,3]` 重复2次变为 `[1,2,3,1,2,3]`；`repeat_interleave(2, dim=0)` 则会将第 `dim` 维逐元素重复2次，例如 `[1,2,3]` 重复2次变为 `[1,1,2,2,3,3]`。
- `x.expand()` 用于创建广播视图，效果上和 `x.repeat()` 类似，但并不会实际真的复制一份数据，因此更省显存。
