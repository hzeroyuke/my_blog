
Attention Sink现象是在量化，位置编码，以及对于Attention内容的改进中都很重要的现象

StreamLLM，如果不考虑KV的情况，Attention Sink指向的是一些特殊的token并且在位置编码为0的情况下，可以占据巨大的注意力内容

Attention Sink的起因的解读有很多，可以简单概括为这个描述，因为模型有时不需要所有的层来完成推理任务，因此Attention Block需要有输出不变的能力，在带有残差流的时候表现为输出为0。因此模型学到的范式是将前几个token在必要的时候，Attention Score计算得巨大，同时其Value又很小，导致整体的Attention输出趋近于0，不影响残差流。

这个现象可以通过Gated Attention这种方案来解决

## 1. Outlier & Attention Sink & Rope

这是一篇来自微信公众号的[文章](https://mp.weixin.qq.com/s/RdL9HwbhSyGJ_BGaT42lQA?poc_token=HKrZZWmjC1cSfIKieqplBrd-rFEmnfcLQ0PlBolt)个人认为讲得不错，在做量化的时候，人们往往会关注到，LLM中存在一些离群值，它们往往出现在前几个token。并且这些token本身的数值不重要，重要的是它们的量级要高于其他数值，2000和500区别不大，但是2000到10这个量级就会有问题

![](asset/Pasted%20image%2020260116161224.png)

正如上文中所说的一样，这些离群值充当注意力的汇点，并且有很低的value，最终的结果就是形成了注意力的空操作。我们通过一个4B的Qwen的例子来看看Attention Sink怎么构成

我们可以看到首个token占据了大量的注意力汇点

但是这样子仍然不能解释为什么这样子会形成 Attention 的空操作，因此我们要关注到Attention Sink的value值，这个值足够小，就可以让sink接受大量的softmax并且输出极小，对于残差连接来说也即为0

但是实际上，我们其实并不需要channel-wise outliers来形成注意力汇点，只需要让汇点token的key数值和普通token有足够大的差异即可，也就是说一种方案是靠channel维度里的几个很大的数值，另一种方案是依靠KV矩阵的学习到一些特征向量，可以把某些token和其他token正交开

![](asset/Pasted%20image%2020260116162107.png)

那么为什么两种路径中，LLM选择了channel-wise outlier这种做法呢，原因在于Rope这种位置编码方案

首先我们需要分析LLM对于长上下文的一些信息，原始的Attention操作，从理论上来说不相关文本越多，其困惑度自然会提升；但是在训练中，Attention架构自然演化出了方案来解决这种现象，Anthropic发表的论文中发现，多头注意力中有一些头（归纳头），专门用来关注局部信息（可以理解为关注topK的注意力权重的内容）；另一部分头（召回头），负责抓取长下文中于该token相关的信息，使其拥有更高的注意力权重

但是在Rope中，我们却要为远程的token分配更小的attention score，也就是说，模型不得不分配更大量级的Attention Score，来对抗Rope的衰减，因此大多数的离群值都出现在召回头上

![](asset/Pasted%20image%2020260116163017.png)

因此后续也衍生了一些操作，在部分Attention Head上禁用Rope，会拥有更好的效果

## 2. A Unified View of Attention and Residual Sinks

- https://mp.weixin.qq.com/s/9FEC995KPMb5SGHeWOKnmA

这篇[论文](https://arxiv.org/pdf/2601.22966)算是Qwen团队中对于Gated Attention的论文的续作，重新审视了Attention Sink现象，以及其中的异常值产生的原因以及其在训练中发挥的作用。以及GatedNorm等reScale方案如何对于这些异常值产生作用

在之前的工作中，人们发现了LLM中两种异常值

- Attention Sink: 最初的几个token产生的极大的注意力分数
- Massive Actiavtion: Attention Sink相关的token在某个特征上产生的极大的激活值

但是在这篇论文中，作者提出了Residual Sink这个现象，指向在大多数token中，固定维度上会出现一批异常值。Attention Sink现象是针对Softmax机制存在的，作者认为Residual Sink现象是针对RMSnorm机制存在的，所有的异常值都是模型尝试隐式地对于softmax/rmsnorm进行缩放

因此作者认为我们可以显示地引入缩放因子，除了Gated Attention以外，还可以引入Gated Norm，在Loss不变的情况下，大幅削减的异常值的出现，使得激进的量化机制比如W4A4（权重和激活值都用4bit）可以应用




