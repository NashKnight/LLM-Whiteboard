# 4.2 DPO

Direct Preference Optimization，直接偏好优化。

**（1）DPO 数据格式**
$$
(x,\ y_w,\ y_l)
$$

- $x$：prompt
- $y_w$：chosen，偏好答案
- $y_l$：rejected，不偏好答案

**（2）DPO 损失函数（单样本）**
$$
\mathcal{L}_{\text{DPO}}
=
-\log \sigma
\left(
\beta
\left[
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-
\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right]
\right)
$$
其中：

- $\pi_\theta$：当前训练模型
- $\pi_{\text{ref}}$：冻结的参考模型，通常是 SFT 模型
- $\beta$：控制偏好优化强度
- $\sigma$：sigmoid

（3）从隐式 reward 看损失函数

先定义：
$$
r_\theta(x,y)
=
\beta
\left[
\log \pi_\theta(y|x)
-
\log \pi_{\text{ref}}(y|x)
\right]
$$
它可以看成 DPO 的隐式 reward。那么损失就是：
$$
\mathcal{L}_{\text{DPO}}
=
-\log \sigma
\left(
r_\theta(x,y_w)-r_\theta(x,y_l)
\right)
$$
即 `chosen 相对于参考模型被提高的水平 - rejected 相对于参考模型被提高的水平`。因此目标就是提高：
$$
r_\theta(x,y_w) > r_\theta(x,y_l)
$$
的概率，即让 chosen 的相对偏好分数高于 rejected。

```python
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1
):
    """
    logps: (batch,) 每个值代表每条整段response的对数概率。
    """
    # 计算policy和ref各自的chosen-rejected提升概率
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    # 计算loss
    logits = beta * (policy_logratios - ref_logratios)
    loss = -F.logsigmoid(logits)
    return loss.mean()
```

外部调用：

```python
import torch.nn.functional as F

def get_sequence_logps(logits, labels, response_mask):
    """
    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len)
    response_mask: (batch, seq_len)，1 表示 response token 参与计算
    """
    shift_logits = logits[:, :-1, :]  # shift_logits: (batch, seq_len-1, vocab_size)
    shift_labels = labels[:, 1:]  # shift_labels: (batch, seq_len-1)
    shift_mask = response_mask[:, 1:]  # shift_mask: (batch, seq_len-1)

    # 转换成对数softmax概率, log_probs: (batch, seq_len-1, vocab_size)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # 取出response token的对数概率, token_logps: (batch, seq_len-1, vocab_size)
    token_logps = log_probs.gather(
      dim=-1,
      index=shift_labels.unsqueeze(-1)  # (batch, seq_len-1, 1)，在vocab维度上gather
    ).squeeze(-1)
    
    # 过滤non-response token
    sequence_logps = (token_logps * shift_mask).sum(dim=-1)
    
    return sequence_logps

# 获得policy上chosen和rejected的对数概率得分（每个response token的对数概率求和）
policy_chosen_logps = get_sequence_logps(policy_chosen_logits, chosen_labels, chosen_mask)
policy_rejected_logps = get_sequence_logps(policy_rejected_logits, rejected_labels, rejected_mask)

# 获得ref上chosen和rejected的对数概率得分（每个response token的对数概率求和）
with torch.no_grad():
    ref_chosen_logps = get_sequence_logps(ref_chosen_logits, chosen_labels, chosen_mask)
    ref_rejected_logps = get_sequence_logps(ref_rejected_logits, rejected_labels, rejected_mask)

# 计算DPO损失
loss = dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1
)
```

### 