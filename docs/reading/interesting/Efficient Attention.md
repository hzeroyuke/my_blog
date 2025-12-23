
现阶段做高效Attention的机制主要有两个，一个是低比特矩阵计算的Attention，另一种是稀疏的注意力机制

其实还存在另一种高效方案就是Linear Attention，包括是Kimi最近在做的一系列工作，但是这种方案还未能做到完全的工业场景的落地

在Context Length的增长中，Attention计算会变成一个巨大的瓶颈，在长文本，视频之类的领域尤其明显

![](asset/Pasted%20image%2020251201200000.png)

## 1. 低比特计算

这一系列的代表工作主要是SageAttention，在FP32的Full Attention的基线中引入了各项低比特机制

- SageAttention1 INT8 + FP16 inference
- SageAttention2 INT4/INT8 + FP8 inference
- SageAttention3 FP4 inference FP8 Training

SageAttention，已经集成在Diffusers，TensorRT等库中，很多视频生成产品都用了

在应用的时候直接替换FlashAttention，一行就可以实现加速

SageAttention也是基于FlashAttention之上的

![](asset/Pasted%20image%2020251201201334.png)

在低比特量化的时候，由于两个单位的上下界不同，因此一般在低比特量化中需要做一些缩放，量化完成的矩阵还会带一个小标量，就是一个缩放因子，可以看下图中的 $s_q$ 

![](asset/Pasted%20image%2020251201201733.png)

也是建立在FlashAttention的基础上，对于分块好的矩阵先进行量化，在 $QK^T$ 这个计算中使用INT8 在最后和 V 矩阵的相乘中使用FP16

但是量化本身发生在token维度上，一个序列的token的attention值其实差异波动是很大的，在这种情况下，量化的效果会比较糟糕，因为他是基于同一个缩放因子去进行缩放的

在这里引入了一个新的方案叫做Smooth K，其用于平均化序列中的异常值，其实就是减去每列的均值，并且这个操作不影响softmax

![](asset/Pasted%20image%2020251201202440.png)

在SageAttention1中其实没有动最后和V矩阵相乘这一块，但是在SageAttention2中也量化这一部分，随之也引入了SmoothQ

![](asset/Pasted%20image%2020251201202812.png)
## 2. 稀疏注意力

![](asset/Pasted%20image%2020251201222231.png)

这部分的主要工作是Sparge Attention等一系列的工作，并且他们都是建立在Saged Attention之上的

### 2.1 PSA

PSA 这篇是稀疏注意力的一个进阶版本，Pyramid Sparse Attention，金字塔形的稀疏注意力机制


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

