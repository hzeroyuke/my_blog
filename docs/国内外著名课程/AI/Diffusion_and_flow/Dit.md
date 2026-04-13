
Dit 是现阶段生成模型最常见的模型架构，可以参考这些codebase

- Huggingface diffusers
- [lighting dit](https://github.com/hustvl/LightningDiT/blob/main/models/lightningdit.py)



## Wan

wan的架构是基本上市面上的开源的video model的参考对象，我们来看一下wan的dit架构

基本的流程如下

```bash
输入视频/图像
      ↓
  Patch Embedding (3D Conv)
      ↓
  Time Embedding (sinusoidal → MLP)
      ↓
  Text/Image Context Embedding
      ↓
  N × WanAttentionBlock
  (Self-Attn + Cross-Attn + FFN)
      ↓
  Head (反归一化 → Linear)
      ↓
  Unpatchify → 输出视频
```


**Embedding**

先对输入的内容进行Embedding

- patch embedding: Conv3d 模块，来做patchify
- time embedding: 用正弦位置编码
- text embedding: 用t5
- image embedding(image2video): 用Clip特征

```python

# embeddings
x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
grid_sizes = torch.stack(
	[torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
x = [u.flatten(2).transpose(1, 2) for u in x]
seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
assert seq_lens.max() <= seq_len
x = torch.cat([
	torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
			  dim=1) for u in x
])

# time embeddings
# with amp.autocast(dtype=torch.float32):
e = self.time_embedding(
	sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
e0 = self.time_projection(e).unflatten(1, (6, self.dim))
# assert e.dtype == torch.float32 and e0.dtype == torch.float32

# context
context_lens = None
context = self.text_embedding(
	torch.stack([
		torch.cat(
			[u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
		for u in context
	]))

if clip_fea is not None:
	context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
	context = torch.concat([context_clip, context], dim=1)
```

**3D Rope**

对于三个维度，长宽时间做Rope

```python
freqs = torch.cat([
    rope_params(1024, d - 4*(d//6)),  # 时间维度，占比最大
    rope_params(1024, 2*(d//6)),      # 高度维度
    rope_params(1024, 2*(d//6)),      # 宽度维度
], dim=1)
```

**WanAttentionBlock**

同时包含 Self-Attention + Cross-Attention + FFN 

```bash
输入 x
  │
  ├─ [1] Self-Attention（带 RoPE + QK-Norm）
  │     norm1(x) * (1 + e1) + e0 → Self-Attn → x + y * e2
  │
  ├─ [2] Cross-Attention（文本/图像条件注入）
  │     norm3(x) → Cross-Attn(context) → x + output
  │
  └─ [3] FFN（GELU激活）
        norm2(x) * (1 + e4) + e3 → Linear → GELU → Linear → x + y * e5
```


