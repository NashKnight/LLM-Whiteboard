# LLM-Whiteboard

大模型手撕代码复习笔记，主要面向算法岗/大模型相关面试中的 PyTorch live coding 准备。

## 内容说明

- `llm_pytorch_live_coding.md` 和 `llm_pytorch_live_coding.pdf` 是完整整合版 Markdown 以及对应的 PDF 版本。
- `chapters/` 目录下是各部分拆分后的单独内容，和整合版内容保持一致。
- 代码用于面试中的大模型手撕复习准备，和生产级实际实现会有差异；代码在保证准确性的前提下，尽量提高可读性。
- 仅包含 PyTorch 代码实现以供参考，部分内容会附公式，方便对照代码理解；默认读者已有一定大模型基础。
- 每部分可能会添加外部调用示例，帮助理解代码实现，并附带部分 PyTorch 代码说明。

## 整合版

- [llm_pytorch_live_coding.md](llm_pytorch_live_coding.md)
- [llm_pytorch_live_coding.pdf](llm_pytorch_live_coding.pdf)

## 目录

### 1. 按 QKV 来源划分的 Attention

- [1.1 Self-Attention](chapters/1-Attention-QKVSouce/1-1-SelfAttention.md)
- [1.2 Cross-Attention](chapters/1-Attention-QKVSouce/1-2-CrossAttention.md)

### 2. 按 KV 组织方式划分的 Attention

- [2.1 Multi-Head Self-Attention (MHA)](chapters/2-Attention-KVOrganization/2-1-MHA.md)
- [2.2 MHA with KV Cache](chapters/2-Attention-KVOrganization/2-2-MHA-KVCache.md)
- [2.3 Multi-Query Attention (MQA)](chapters/2-Attention-KVOrganization/2-3-MQA.md)
- [2.4 Grouped-Query Attention (GQA)](chapters/2-Attention-KVOrganization/2-4-GQA.md)
- [2.5 Multi-head Latent Attention (MLA)](chapters/2-Attention-KVOrganization/2-5-MLA.md)

### 3. Transformer

- [3.1 LayerNorm](chapters/3-Transformer/3-1-LayerNorm.md)
- [3.2 RMSNorm](chapters/3-Transformer/3-2-RMSNorm.md)
- [3.3 RoPE](chapters/3-Transformer/3-3-RoPE.md)
- [3.4 SwiGLU](chapters/3-Transformer/3-4-SwiGLU.md)
- [3.5 Transformer](chapters/3-Transformer/3-5-Transformer.md)
- [3.6 Transformer with KV Cache](chapters/3-Transformer/3-6-Transformer-KVCache.md)

### 4. 损失函数

- [4.1 SFT (Cross Entropy)](chapters/4-LossFunctions/4-1-SFT-CrossEntropy.md)
- [4.2 DPO](chapters/4-LossFunctions/4-2-DPO.md)
- [4.3 PPO](chapters/4-LossFunctions/4-3-PPO.md)
- [4.4 GRPO](chapters/4-LossFunctions/4-4-GRPO.md)

### 5. 解码策略

- [5.1 Greedy Decoding](chapters/5-DecodingStrategies/5-1-GreedyDecoding.md)
- [5.2 Beam Search](chapters/5-DecodingStrategies/5-2-BeamSearch.md)
- [5.3 Temperature Sampling](chapters/5-DecodingStrategies/5-3-TemperatureSampling.md)
- [5.4 Top-k Sampling](chapters/5-DecodingStrategies/5-4-TopkSampling.md)
- [5.5 Top-p Sampling](chapters/5-DecodingStrategies/5-5-ToppSampling.md)

### 6. 其他模块

- [6.1 LoRA](chapters/6-OtherComponents/6-1-RoRA.md)
- [6.2 MoE](chapters/6-OtherComponents/6-2-MoE.md)

## Chapters

```text
chapters/
|-- 1-Attention-QKVSouce/
|   |-- 1-1-SelfAttention.md
|   `-- 1-2-CrossAttention.md
|-- 2-Attention-KVOrganization/
|   |-- 2-1-MHA.md
|   |-- 2-2-MHA-KVCache.md
|   |-- 2-3-MQA.md
|   |-- 2-4-GQA.md
|   `-- 2-5-MLA.md
|-- 3-Transformer/
|   |-- 3-1-LayerNorm.md
|   |-- 3-2-RMSNorm.md
|   |-- 3-3-RoPE.md
|   |-- 3-4-SwiGLU.md
|   |-- 3-5-Transformer.md
|   `-- 3-6-Transformer-KVCache.md
|-- 4-LossFunctions/
|   |-- 4-1-SFT-CrossEntropy.md
|   |-- 4-2-DPO.md
|   |-- 4-3-PPO.md
|   `-- 4-4-GRPO.md
|-- 5-DecodingStrategies/
|   |-- 5-1-GreedyDecoding.md
|   |-- 5-2-BeamSearch.md
|   |-- 5-3-TemperatureSampling.md
|   |-- 5-4-TopkSampling.md
|   `-- 5-5-ToppSampling.md
`-- 6-OtherComponents/
    |-- 6-1-RoRA.md
    `-- 6-2-MoE.md
```
