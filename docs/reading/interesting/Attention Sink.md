
Attention Sink现象是在量化，位置编码，以及对于Attention内容的改进中都很重要的现象

StreamLLM，如果不考虑KV的情况，Attention Sink指向的是一些特殊的token并且在位置编码为0的情况下，可以占据巨大的注意力内容

Attention Sink的起因的解读有很多，可以简单概括为这个描述，因为模型有时不需要所有的层来完成推理任务，因此Attention Block需要有输出不变的能力，在带有残差流的时候表现为输出为0。因此模型学到的范式是将前几个token在必要的时候，Attention Score计算得巨大，同时其Value又很小，导致整体的Attention输出趋近于0，不影响残差流。

这个现象可以通过Gated Attention这种方案来解决

## Outlier & Attention Sink & Rope

这是一篇来自微信公众号的[文章](https://mp.weixin.qq.com/s/RdL9HwbhSyGJ_BGaT42lQA?poc_token=HKrZZWmjC1cSfIKieqplBrd-rFEmnfcLQ0PlBolt)个人认为讲得不错，在做量化的时候，人们往往会关注到，LLM中存在一些离群值，它们往往出现在前几个token。并且这些token本身的数值不重要，重要的是它们的量级要高于其他数值，2000和500区别不大，但是2000到10这个量级就会有问题

![](asset/Pasted%20image%2020260116161224.png)

正如上文中所说的一样，这些离群值充当注意力的汇点，并且有很低的value，最终的结果就是形成了注意力的空操作

但是实际上，我们其实并不需要channel-wise outliers来形成注意力汇点，只需要让汇点token的key数值和普通token有足够大的差异即可，也就是说一种方案是靠channel维度里的几个很大的数值，另一种方案是依靠KV矩阵的学习到一些特征向量，可以把某些token和其他token正交开

![](asset/Pasted%20image%2020260116162107.png)

那么为什么两种路径中，LLM选择了channel-wise outlier这种做法呢，原因在于Rope这种位置编码方案

首先我们需要分析LLM对于长上下文的一些信息，原始的Attention操作，从理论上来说不相关文本越多，其困惑度自然会提升；但是在训练中，Attention架构自然演化出了方案来解决这种现象，Anthropic发表的论文中发现，多头注意力中有一些头（归纳头），专门用来关注局部信息（可以理解为关注topK的注意力权重的内容）；另一部分头（召回头），负责抓取长下文中于该token相关的信息，使其拥有更高的注意力权重

但是在Rope中，我们却要为远程的token分配更小的attention score，也就是说，模型不得不分配更大量级的Attention Score，来对抗Rope的衰减，因此大多数的离群值都出现在召回头上

![](asset/Pasted%20image%2020260116163017.png)

因此后续也衍生了一些操作，在部分Attention Head上禁用Rope，会拥有更好的效果