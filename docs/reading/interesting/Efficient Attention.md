
现阶段做高效Attention的机制主要有两个，一个是低比特矩阵计算的Attention，另一种是稀疏的注意力机制

其实还存在另一种高效方案就是Linear Attention，包括是Kimi最近在做的一系列工作，但是这种方案还未能做到完全的工业场景的落地

在Context Length的增长中，Attention计算会变成一个巨大的瓶颈，在长文本，视频之类的领域尤其明显

![](asset/Pasted%20image%2020251201200000.png)

## 1. 低比特计算

这一系列的代表工作主要是SageAttention，在FP32的Full Attention的基线中引入了各项低比特机制

- SageAttention1 INT8 + FP16 inference
- SageAttention2 INT4/INT8 + FP8 inference
- SageAttention3 FP4 inference FP8 Training

SageAttention，已经集成在Diffusers，TensorRT等库中，很多视频生成产品都用了，可以在应用的时候直接替换FlashAttention，一行就可以实现加速，实现即插即用

SageAttention也是基于FlashAttention之上的

![](asset/Pasted%20image%2020251229160116.png)

### 1.1. SageAttention 1

量化是深度学习中很早期就有的工作，但是早期的量化集中在于FFN上面，对于Attention的量化研究很少，随着序列长度的增加，Attention的ON2的计算bound越发明显，在这个阶段，FlashAttention3已经实现FP8的量化，但是该种量化指标只能实现在N卡的Hopper的架构中

并且实验中发现，对于Attention直接进行量化效果非常糟糕，不论是对于生成还是LLM，主要面临两个问题

- K矩阵的离群值，这个和传统量化中遇到的现象很像，在channel维度上，存在一些channel上的数值远远大于其他channel，导致传统量化算法失效
- PV计算的时候，也就是softmax的结果和V的计算的时候，P的输出分布非常稀疏且不均匀，导致的结果也是单纯的量化失效

![](asset/Pasted%20image%2020251229161255.png)

本文提出的SageAttention，将Attention中的矩阵量化为INT8，INT8的加速对于不同硬件的支持更好，在3090，4090等硬件上也能有效加速，并且实现了即插即用，无需微调，并且应对上述的几个挑战，SageAttention也提出了应对方案

- Smooth K，来平滑K矩阵
- 对于PV计算，保持这两个矩阵为FP16，并且使用low-precision FP16 accumulator

在Smooth K的过程中，将K矩阵都减去K矩阵的均值，在Channel维度上做归一化，在保证Softmax是无损的情况下，平滑K矩阵

在低比特量化的时候，由于两个单位的上下界不同，因此一般在低比特量化中需要做一些缩放，量化完成的矩阵还会带一个小标量，就是一个缩放因子，可以看下图中的 $s_q$ 

![](asset/Pasted%20image%2020251229165252.png)

也是建立在FlashAttention的基础上，对于分块好的矩阵先进行量化，在 $QK^T$ 这个计算中使用INT8 在最后和 V 矩阵的相乘中使用FP16（实验发现这一块如果进行量化，效果会差很多，因此就使用FP16），但是这部分也做了一些优化，引入了FP16累加器，因为传统的矩阵乘法在累加器这一块用的仍然是32位，但是同时P矩阵是Softmax的结果，它的归一化是一，因此在累加中很少会溢出FP16的结果，因此FP32的累加是不太必要的，因此这里用上了FP16的累加器，可以做到性能没有什么变化但是速度变快了

但是量化本身发生在token维度上，一个序列的token的attention值其实差异波动是很大的，在这种情况下，量化的效果会比较糟糕，因为他是基于同一个缩放因子去进行缩放的

在这里引入了一个新的方案叫做Smooth K，其用于平均化序列中的异常值，其实就是减去每列的均值，并且这个操作不影响softmax

![](asset/Pasted%20image%2020251229170600.png)

### 1.2. SageAttention 2

SageAttention2 中引入了Per thread的量化，相对于之前的Per Tensor or Per Block的范式，粒度更细，粒度更细的结果就是量化的范围内离群值的更少，效果就更好

在SageAttention1中其实没有动Q矩阵，因为Q矩阵的离群值相对比较少，但是在引入了INT4的量化的时候，即便是Q矩阵也面临了很高的精度损失的风险。对于Q矩阵的量化相对更加麻烦一点

![](asset/Pasted%20image%2020251201202812.png)
## 2. 稀疏注意力

![](asset/Pasted%20image%2020251201222231.png)

这部分的主要工作是Sparge Attention等一系列的工作，并且他们都是建立在Saged Attention之上的

### 2.1 Pyramid Sparse Attention

[PSA](https://arxiv.org/abs/2512.04025) 这篇是稀疏注意力的一个进阶版本，Pyramid Sparse Attention，金字塔形的稀疏注意力机制

![](asset/Pasted%20image%2020251224152252.png)

我们来仔细分析一下它的实现，也过一下实现一个Sparse Attention机制的流程。一开始也和我们自定义任何一个Attention一样继承一个Module，然后实现forward

```python
class PyramidSparseAttention(nn.Module):
    def __init__(self, config: AttentionConfig = None, inference_num=50, layer_num=42, model_type="wan"):
        super().__init__()
        # If config not provided, use default config
        if config is None:
            config = AttentionConfig()
        self.config = config
        ...

    def forward(self, q, k, v, layer_idx):
        ...
```

我们来看它的核心部分，也就是forward过程，首先经过的rearrange的判断，这个过程可以将语义相近的token的排序在一起，提高局部性

```python
if self.use_rearrange:
	if self.config.rearrange_method == 'Gilbert':
		q_r, k_r, v_r = self.gilbert_rearranger.rearrange(q, k, v)
		q_sorted_indices = None
	elif self.config.rearrange_method == 'SemanticAware':
		q_r, k_r, v_r, q_sorted_indices = self.semantic_aware_rearranger_list[layer_idx].semantic_aware_permutation(q, k, v)
	elif self.config.rearrange_method == 'STA':
		q_r, k_r, v_r = self.STARearranger.rearrange(q, k, v)
		q_sorted_indices = None
	elif self.config.rearrange_method == 'Hybrid':
		q_r, k_r, v_r, q_sorted_indices = self.hybrid_rearranger_list[layer_idx].rearrange(q, k, v)
	else:
		raise ValueError(f"Unknown rearrange_method: {self.config.rearrange_method}")
else:
	q_r = q
	k_r = k
	v_r = v
	q_sorted_indices = None

```

经过rerange之后我们就进入核心部分，也就是Attention层的计算，这里的`adaptive_block_sparse_attn`是整个流程的核心

```python
if is_warmup:
	out_r = torch.nn.functional.scaled_dot_product_attention(q_r, k_r, v_r)
	sparsity = 0.0
	per_head_density = [1.0] * q_r.shape[1] if compute_stats else []
	sim_mask = None
else:
	out_r, sparsity, per_head_density, sim_mask = adaptive_block_sparse_attn(
		q_r, k_r, v_r, self.config, self.sparse_attention_fn, compute_stats=compute_stats
	)
	# Update sparsity statistics only if computing stats
	if compute_stats:
		self.sparsity_acc += sparsity
```

我们来看这个function的实现，这个function有三个主要的Step

- Pooling
- Mask
- Attention Compute

```python
# Disable gradient tracking for pooling and mask operations
with torch.no_grad():
	# STEP 1
	pooling, sim_mask = efficient_attn_with_pooling(q, k, v, config, num_keep_m=block_size_m//4, num_keep_n=block_size_n//4)
	# Support both attn_impl and mask_mode methods
	# STEP2
	if config.attn_impl == "old_mask_type":
		mask = transfer_attn_to_mask(pooling, config.mask_ratios, config.text_length, mode=config.mask_mode, blocksize=config.block_n, compute_tile=config.tile_n)
	elif config.attn_impl == "new_mask_type":
		# Use mask_mode parameter, supports topk and thresholdbound
		mode_map = {
			'topk': 'topk_newtype',
			'thresholdbound': 'thresholdbound_newtype'
		}
		mode = mode_map.get(config.mask_mode, 'topk_newtype')
		mask = transfer_attn_to_mask(pooling, config.mask_ratios, config.text_length, mode=mode, blocksize=config.block_n, compute_tile=config.tile_n)
	else:
		raise ValueError(f"Unknown attn_impl: {config.attn_impl}")
use_sim_mask = getattr(config, "use_sim_mask", True)
if use_sim_mask and sim_mask is not None:
	if sim_mask.dtype != mask.dtype:
		sim_mask = sim_mask.to(mask.dtype)
	fixed_mask = torch.min(sim_mask, mask)
else:
	fixed_mask = mask
# STEP3
out = sparse_attention_fn(q.contiguous(), k.contiguous(), v.contiguous(), fixed_mask, None)
```

那么首先我们来看Pooling，这个核心就干了以下内容，将QK矩阵划分Block，从每个Block中随机采样几个Token，以这几个Token为代表计算Attention Score，最后返回的是以Block为维度的Attention Matrix

```bash
  原始Q, K: [B, H, 1024, 64]
             ↓
  [Pad to block_size的倍数]
             ↓
  Q_padded: [B, H, 1024, 64]  (假设已对齐)
  K_padded: [B, H, 1024, 64]
             ↓
  [随机采样：每块128个token采32个]
             ↓
  sampled_Q: [B, H, 256, 64]  (8块 × 32采样)
  sampled_K: [B, H, 256, 64]
             ↓
  [Triton kernel计算注意力并块内pooling]
             ↓
  pooling: [B, H, 8, 8]  # 8×8的块级重要性矩阵!
           ↓
  [配合sim_mask生成稀疏mask]
```

在Pooling中还使用了另一个机制，生成了一个sim_mask这个东西是衡量了K矩阵block内部的相似度，block内部的相似度衡量了这个block应该被压缩的程度，如果这里面的token都很相似的话，就能被很好的压缩，这也与前面提到的rerange的过程相互呼应

其次我们来看Mask的过程，Mask过程利用上述Pooling的结果，将稀疏计算需要的信息传递给下一步真正的Attention计算，计算Mask有两种方式，TopK和ThresholdBound

```python
def transfer_attn_to_mask(
    attn: torch.Tensor,
    mask_ratios: Optional[Dict[int, Tuple[float, float]]] = None,
    text_length: int = 226,
    mode: str = "topk",
    min_full_attn_ratio: float = 0.06,
    blocksize=32,
    compute_tile=32
) -> torch.Tensor:
    """
    Convert attention weights to multi-level pooling mask matrix.

    Args:
        attn (torch.Tensor): Attention weight matrix, shape [batch, head, seq, seq]
        mask_ratios (dict): Mask value to percentage range mapping, format {mask_value: (start_ratio, end_ratio)}
                           Default is {1: (0.0, 0.05), 2: (0.05, 0.15), 4: (0.15, 0.55), 8: (0.55, 1.0)}
                           Other positions have mask=0 (skip)
        text_length (int): Text sequence length, used to calculate special token positions
        mode (str): Mask generation mode, 'topk' or 'thresholdbound'
                   - 'topk': Generate mask based on sorted position range
                   - 'thresholdbound': Generate mask based on cumulative energy percentage
                   - 'topk_newtype': topk new format mask
                   - 'thresholdbound_newtype': thresholdbound new format mask
        min_full_attn_ratio (float): Minimum interval ratio when mask_value=1, default 0.05 (5%)
                                     Ensures full attention interval occupies at least this ratio

    Returns:
        torch.Tensor: Multi-level mask matrix, same shape as input
        - 0: skip (no attention computation)
        - 1: full attention (default top 5%)
        - 2: 2x pooling (default 5%-15%)
        - 4: 4x pooling (default 15%-55%)
        - 8: 8x pooling (default 55%-100%)
    """
```

该Function最后返回给我们一个Mask矩阵，这个函数的实现非常的长，目前还在迭代中，因此采用了多种方案


最后我们来看Attention计算，这里是一个Triton算子了，之前在Pooling中其实也用到了Triton算子，它里面也有一个计算Attention形成block size的Attention Score的过程



### 2.2. Recitified Sparse Attention

https://arxiv.org/abs/2511.19835

矫正的稀疏注意力机制


### why sparse attention

从现有的工作来看，attention机制的计算是有很大的冗余的，不论是在video任务上还是文本任务上。从Transformer结构上来看，核心的两大块就是Attention+FFN

两层各司其职，Attention层用于捕获输入之间的关系，但是显然这里有很多的冗余，比如文本，不是每个token都和其他的token那么有关系，这和人类的阅读习惯是一致的，video & image 就更是如此了，本身就有很强的局部关联性，高分辨率场景下冗余更多，因此对于Attention做稀疏计算和KV cache压缩，效果显著。

而对于FFN，它承载着模型的世界知识，模型对于世界知识的记忆和模型的FFN参数成正相关，因此如果要对FFN做稀疏，相当于让模型忘记一些知识，但是在各种任务中，我们往往不知道哪些知识是可以被忘记，更重要的是，我们不知道去掉这个connection或者神经元，会损失哪些知识，因此对于FFN做稀疏计算的工作都难以生效

并且通过Moe这个做法我们也可以看出来这一点，随着人们对于FFN层的参数做扩展，模型能够掌握的知识越来越多，尤其是在大规模预训练阶段，Moe做的相当得好

现阶段的理解是这样，也可以看一些对于FFN做Sparse的论文来了解一下真实的场景，关于FFN那边的Efficience的工作有很多是围绕着activation进行的

- Spark Transformer


## 3. Linear Attention

Linear Attention的主流做法是，$softmax(QK^T)V$ 这个计算转换成，$Qsim(K^TV)$ ，也就是使用某些计算手段，来先计算KV这一块，这样子可以使得其计算复杂度变为一个线性状态，因为KV的计算结果是一个dxd的矩阵

![](asset/Pasted%20image%2020251215212145.png)

但是Linear attention的效果之所以还是无法逼近Full Attention，是因为LinearAttention计算出来的矩阵的低秩性

### 3.1. SLA

这是SageAttention和SpargeAttention的系列工作之一，在这些基础上加上了Linear Attention，达到更高的性能。论文分析发现，Attention计算的权重中，天然可以分为两个部分

- 高权重且high-rank
- low rank权重

自然的，我们考虑将Sparse Accelerate给high-rank权重加速，用low-rank方案（Linear方案）给low-rank权重加速，于是就有了对应的工作过，Sparse-Linear Attention，主要是针对的Dit，也就是视频生成这个领域，不仅支持Forward也支持Backward，这是一个trainable的方案

在流程中SLA将Attention权重分为了三个部分，分别采用N2的full attention和N1的Linear Attention以及丢弃的操作。

**Baseline**

SLA基于Block Sparse Attention(按照Block计算Attention Score并且进行drop)以及Linear Attention（先计算KV，再归一化）

![](asset/Pasted%20image%2020260104211915.png)

**Motivation**

SLA的推进基于两个发现

- Softmax矩阵的稀疏性
- Full Attention矩阵的low rank性质

![](asset/Pasted%20image%2020260104212458.png)

![](asset/Pasted%20image%2020260104212509.png)

**SLA Method**

![](asset/Pasted%20image%2020260104212554.png)


