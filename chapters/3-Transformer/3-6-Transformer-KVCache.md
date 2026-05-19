### 3.6 Transformer with KV Cache

带 KV Cache 的 Transformer，需要使用 2.2 小节中的 `MultiHeadAttention`：

```python
import torch
import torch.nn as nn
# 自定义模块部分
from attention import MultiHeadAttention
from norm import RMSNorm
from ffn import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
    def forward(self, x, mask=None, past_kv=None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len), causal mask / padding mask
        """
        # Pre-LN + MHA + Residual Connection
        attn_out, new_kv = self.attn(self.norm1(x), mask=mask, past_kv=past_kv)
        x = x + attn_out
        # Pre-LN + FFN + Residual Connection
        x = x + self.ffn(self.norm2(x))
        return x, new_kv
    
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, d_model)  # 词嵌入
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # 可学习的位置编码
        
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )  # 注意力层
        
        self.norm = RMSNorm(d_model)  # 归一化层
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 输出头
        
    def forward(self, input_ids, mask=None, past_kv=None):
        """
        input_ids: (batch, seq_len)，带KV Cache的情形下一般seq_len=1
        mask: (seq_len, seq_len)
        past_kv: None或list, 长度为num_layers
        past_kv[i]: 第i层的(past_k, past_v)
        past_k/past_v: (batch, num_heads, past_len, head_dim)
        """
        # 获取input_ids相关信息
        seq_len = input_ids.shape[-1]
        device = input_ids.device
        past_len = past_kv[0][0].size(-2) if past_kv is not None else 0
        
        # 词嵌入并添加可学习位置编码，注意位置编码应是第past_len到past_len+seq_len之间的位置
        pos = torch.arange(past_len, past_len + seq_len, device=device)  # pos: (seq_len,)
        pos = pos.unsqueeze(0)  # pos: (1, seq_len), broadcast到batch维度
        x = self.token_emb(input_ids) + self.pos_emb(pos)  # x: (batch, seq_len, d_model)
        
        # 注意力层处理，处理过程中始终保持x: (batch, seq_len, d_model)
        new_past_kv = []
        for i, layer in enumerate(self.layers):
            layer_past_kv = None if past_kv is None else past_kv[i]
            x, layer_new_kv = layer(x, mask=mask, past_kv=layer_past_kv)
            new_past_kv.append(layer_new_kv)
        
        x = self.norm(x)  # 归一化层
        logits = self.lm_head(x)  # 输出头, logits: (batch, seq_len, vocab_size)
        
        return logits, new_past_kv
```

外部调用：

```python
batch = 2
vocab_size = 10000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
max_seq_len = 1024

model = DecoderOnlyTransformer(
  vocab_size=vocab_size,
  d_model=d_model,
  num_heads=num_heads,
  d_ff=d_ff,
  num_layers=num_layers,
  max_seq_len=max_seq_len
)

past_kv = None

# 假设逐 token 生成 5 步
for step in range(5):
  input_ids = torch.randint(0, vocab_size, (batch, 1))
  # input_ids: (batch, 1)

  logits, past_kv = model(
      input_ids,
      mask=None,
      past_kv=past_kv
  )
```

