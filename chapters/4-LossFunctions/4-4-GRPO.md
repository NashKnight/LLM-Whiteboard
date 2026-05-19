# 4.4 GRPO

Group Relative Policy Optimization，组相对策略优化。

**（1）优势估计**

PPO 通常需要如下一系列模型，比较占用显存：

- policy model
- reference model
- reward model
- value model / critic

GRPO 希望节省 value model 的训练时间和显存占用。

对同一个 prompt $x$，一次采样 $G$ 个回答：
$$
\{y_1, y_2, \dots, y_G\}
$$
分别得到奖励：
$$
\{r_1, r_2, \dots, r_G\}
$$
然后在这一组内部做标准化：
$$
A_i=
\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}
{\operatorname{std}(r_1,\dots,r_G)}
$$
这个 $A_i$ 就作为第 $i$ 个回答的优势。

**（2）GRPO 损失**
$$
\text{loss}
=
\underbrace{
-\min\left(
r_tA_t,\ 
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
}_{\text{policy loss}}
+
\lambda_{kl}
\underbrace{
\operatorname{KL}\left(
\pi_\theta \,\|\, \pi_{\text{ref}}
\right)
}_{\text{kl loss}}
$$
其中 $A_t$ 由组内相对优势直接得到，不需要额外的 critic。

```python
import torch

def mask_mean(loss, mask=None):
    if mask is not None:
        return (loss * mask).sum() / mask.sum()
    else:
        return loss.mean()
    
def grpo_loss(
    new_logprobs,
    old_logprobs,
    rewards,
    group_size,
    mask=None,
    ref_logprobs=None,
    clip_eps=0.2,
    kl_coef=0.1,
    eps=1e-8
):
    """
    new_logprobs: (batch, seq_len) 每时间步下新策略模型输出的对数概率
    old_logprobs: (batch, seq_len) 每时间步下旧策略模型输出的对数概率
    rewards: (batch,) 和PPO不同，PPO中returns是逐token的，GRPO中rewards对整个sequence计算
    group_size: GRPO中一组的大小
    mask: (batch, seq_len)
    ref_logprobs: (batch, seq_len) 每时间步下参考模型输出的对数概率
    clip_eps是clip中的eps，eps是计算组间相对优势时标准差用到的eps
    返回的是整个batch一起算出的loss
    """
    # 获取组数
    batch = new_logprobs.shape[0]
    assert batch % group_size == 0
    num_group = batch // group_size
    
    # 1. 计算相对优势
    group_rewards = rewards.view(num_group, group_size)  # group_rewards: (num_group, group_size)
    group_mean = group_rewards.mean(dim=-1, keepdim=True)  # group_mean: (num_group, 1)
    group_std = group_rewards.std(dim=-1, keepdim=True, unbiased=False)  # group_std: (num_group, 1)
    advantages = (group_rewards - group_mean) / (group_std + eps)  # advantages: (num_group, group_size)
    advantages = advantages.view(batch, 1)  # advantages: (batch, 1)
    
    # 2. policy_loss
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, min=1.0-clip_eps, max=1.0+clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped)
    policy_loss = mask_mean(policy_loss, mask)
    
    # 3. kl_loss
    if ref_logprobs is not None:
        log_ratio = ref_logprobs - new_logprobs
        kl_loss = torch.exp(log_ratio) - 1.0 - log_ratio
        kl_loss = mask_mean(kl_loss, mask)
    else:
        kl_loss = 0.0
        
    # 4. total loss
    loss = policy_loss + kl_coef * kl_loss
    return loss
```

## 