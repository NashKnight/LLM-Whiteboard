# 4.3 PPO

Proximal Policy Optimization，近端策略优化。

**（1）基础的策略梯度**

策略模型：
$$
\pi_\theta(a|s)
$$
表示在状态 $s$ 下选择动作 $a$ 的概率。希望高优势的动作概率变大：
$$
\mathcal{L}^{PG}(\theta)
=
\mathbb{E}_t
\left[
\log \pi_\theta(a_t|s_t) A_t
\right]
$$

- $A_t > 0$：这个动作比平均水平好，概率应提高
- $A_t < 0$：这个动作差，概率应降低

问题是：一次梯度更新可能把策略改太多，训练不稳定。

**（2）PPO 损失函数**
$$
\text{loss}
=
\underbrace{
-\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)
}_{\text{policy loss}}
+
\lambda_v
\underbrace{
(V_\phi(s_t)-R_t)^2
}_{\text{value loss}}
+
\lambda_{kl}
\underbrace{
\operatorname{KL}(\pi_\theta\|\pi_{\text{ref}})
}_{\text{kl loss}}
$$
其中：

- $r_t$：新旧策略概率比，表示新策略相比于旧策略把动作 $a_t$ 提高/降低的比例。

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

- $A_t$：优势函数，$A_t=Q_t(s_t,a_t)-V(s_t)$，表示当前动作 $a_t$ 比平均预期好多少。
- $V_{\phi}(st)$：value model 输出的价值，表示 critic 对状态 $s_t$ 的未来总回报估计。

```python
import torch

def mask_mean(loss, mask=None):
    if mask is not None:
        return (loss * mask).sum() / mask.sum()
    else:
        return loss.mean()

def ppo_loss(
    new_logprobs,
    old_logprobs,
    advantages,
    values,
    returns,
    mask=None,
    ref_logprobs=None,
    clip_eps=0.2,
    value_coef=0.5,
    kl_coef=0.1
):
    """
    new_logprobs: (batch, seq_len) 每时间步下新策略模型输出的对数概率
    old_logprobs: (batch, seq_len) 每时间步下旧策略模型输出的对数概率
    advantages: (batch, seq_len) 每时间步下优势估计
    values: (batch, seq_len) 每时间步下价值模型输出的预测价值
    returns: (batch, seq_len)  每时间步下的GAE回报
    mask: (batch, seq_len), 1 表示有效 token
    ref_logprobs: (batch, seq_len), 每时间步下参考模型输出的对数概率
    返回的是整个batch一起算出的loss
    """
    # 1.policy_loss
    ratio = torch.exp(new_logprobs - old_logprobs)  # 新旧策略概率比r_t，去对数处理
    
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, min=1.0-clip_eps, max=1.0+clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped)
    policy_loss = mask_mean(policy_loss, mask)
    
    # 2.value_loss
    value_loss = (values - returns) ** 2
    value_loss = mask_mean(value_loss, mask)

    # 3.kl_loss
    if ref_logprobs is not None:
        log_ratio = ref_logprobs - new_logprobs
        kl_loss = torch.exp(log_ratio) - 1.0 - log_ratio
        kl_loss = mask_mean(kl_loss, mask)
    else:
        kl_loss = 0.0
        
    # 4.total loss
    loss = policy_loss + value_coef * value_loss + kl_coef * kl_loss
    
    return loss
```

**（3）广义优势估计**

`advantages` 由广义优势估计 GAE 得到，核心思想就是用折扣 TD error 估计优势：
$$
\delta_t=R_t+\gamma\,V(s_{t+1})-V(s_t)
$$
而 $R_t+\gamma\,V(s_{t+1})$ 正是对 $Q_t$ 的一步 bootstrap 估计。

这里 `rewards` 是每一步 token 的奖励，`returns` 是从该步起的 GAE 回报。

由于 `advantages` 定义为 $A_t=Q_t(s_t,a_t)-V(s_t)$，其中 $Q_t$ 是折扣回报 $U_t$ 的期望（在这里还会包含 GAE 平滑，不是一般的折扣回报），因此移项可知：`returns` 可由 `advantages + values` 近似。

GAE 的实现如下：

```python
import torch

def advantage_estimate(
    rewards,
    values,
    dones,
    gamma=0.99,
    lam=0.95
):
    """
    rewards/values: (batch, seq_len)
    dones: (batch, seq_len), 1表示该轨迹结束，后续不再算values，0表示未结束
    gamma是折扣因子，lam是GAE平滑系数
    """
    # 获取rewards相关信息并初始化
    batch, seq_len = rewards.shape
    device = rewards.device
    advantages = torch.zeros_like(rewards)  # advantages: (batch, seq_len)
    last_gae_lam = torch.zeros(batch, device=device)  # last_gae_lam: (batch,)
    
    #
    for t in reversed(range(seq_len)):
        # 计算下一时刻起的values: V(s_{t+1})
        if t == seq_len - 1:
            next_values = torch.zeros(batch, device=device)
        else:
            next_values = values[:, t+1]
        next_non_terminal = 1.0 - dones[:, t]  # 是否达到该轨迹末端
        
        # TD ERROR: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_values * next_non_terminal - values[:, t]
        
        # 计算GAE
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        
        advantages[:, t] = last_gae_lam
        
    # 返回优势和折扣回报,advantages: (batch, seq_len), returns:(batch, seq_len)
    returns = advantages + values
    return advantages, returns
```

**（4）KL 散度的近似**

GRPO 损失中的 KL 散度：
$$
D_{KL}(\pi_\theta\|\pi_{\text{ref}})
=
\sum_a
\pi_\theta(a|s)
\log
\frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}
$$
即：
$$
D_{KL}(\pi_\theta\|\pi_{\text{ref}})
=
\mathbb{E}_{a\sim\pi_\theta}
\left[
\log
\frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}
\right]
$$
由于
$$
\text{log\_ratio}
=
\log\frac{\pi_{\text{ref}}}{\pi_\theta}
$$
所以：
$$
D_{KL}(\pi_\theta\|\pi_{\text{ref}})
=
\mathbb{E}_{a\sim\pi_\theta}
[-\text{log\_ratio}]
$$
但期望无法直接计算，因此考虑用蒙特卡洛近似，而近似值 `log_ratio` 不能保证非负性，因此需要构造一个新的 KL 散度项来代表 KL 损失。

由于：
$$
\mathbb{E}_{a\sim\pi_\theta}
\left[
\frac{\pi_{\text{ref}}(a|s)}{\pi_\theta(a|s)}
\right]
=
\sum_a \pi_\theta(a|s)
\frac{\pi_{\text{ref}}(a|s)}{\pi_\theta(a|s)}
=
\sum_a \pi_{\text{ref}}(a|s)
=
1
$$
所以：
$$
\mathbb{E}_{a\sim\pi_\theta}
\left[
\exp(\text{log\_ratio}) - 1
\right]
=
0
$$
因此在 KL 的估计量里加上它，期望不变：
$$
\mathbb{E}
\left[
-\text{log\_ratio}
\right]
=
\mathbb{E}
\left[
\exp(\text{log\_ratio}) - 1 - \text{log\_ratio}
\right]
$$
所以：
$$
D_{KL}(\pi_\theta\|\pi_{\text{ref}})
=
\mathbb{E}_{a\sim\pi_\theta}
\left[
\exp(\text{log\_ratio}) - 1 - \text{log\_ratio}
\right]
$$
就用 `torch.exp(log_ratio) - 1.0 - log_ratio` 来估计 KL 散度。