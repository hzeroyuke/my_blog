本笔记来自Mit 6.5940，一个关于TinyML的课程，主要讲述了关于剪枝，量化等一系列对于神经网络进行高效处理的机制

该课程一共分为几个部分

- 第一部分为Efficient Inference
	- Pruning & Sparsity
	- Quantization
	- Neural Architecture Search
	- Knowledge Distillation

## 1. Deep Neural Network

对于一个深度网络而言，同样的参数，宽且浅的网络对于硬件更加友好，因为其更容易提高并行度；相对的，窄且深的网络的性能更好。因此这里有个trade-off

卷积需要考虑的内容

- 卷积的dimension，1d卷积，2d卷积
- 卷积的Kernel size，kernel的大小
- 卷积的层数，决定了最后的输出的feature的感受野
- 卷积的步幅，扩大步幅是一个降采样的过程

在保持卷积核size一致的情况下，输出的feature的感受野receptive field随着模型的层数的增加而扩大

## 2. Pruning & Sparsity

剪枝Pruning技术是在参数层面对于模型进行修改，最终的结果是让模型在性能不变的情况下能够有更小的参数

![](asset/Pasted%20image%2020251213145116.png)

上图一定程度展示了我们为什么要做Pruning，对于现代计算硬件来说，Memory 移动远远比计算要昂贵

### 2.1. What is Pruning

![](asset/Pasted%20image%2020251213145440.png)

剪枝的过程是让神经网络变得稀疏的过程，减少模型的参数量，使得计算更加高效

### 2.2. Pruning Rules

剪枝的常规流程分为两步，一个是删减掉模型中的一些connections，随后进行第二次训练，使得其性能恢复到原有水平

**Pruning Granularity**

我们如何确定剪枝的粒度，我们有一些设计范式

![](asset/Pasted%20image%2020251213150621.png)

- Fine-granularity pruning 这个范式是最灵活的，我们可以裁剪掉任意位置的权重，因此其理论可以达到最高的剪枝率，但是这样子的无规则性会导致GPU的并行性能比较糟糕，难以去并行。
- Coarse-grained/Structured 粗粒度的剪枝，对于一整行/列去剪枝，然后压缩成密集矩阵，其优缺点于前者正好相反

以上是全连接层的剪枝，我们也可以对于卷积层进行剪枝，相比于全连接层，卷积层的剪枝方案就更多了

![](asset/Pasted%20image%2020251213150901.png)

比如第二种Pattern-based Pruning，就是在为剪枝的的时候引入一种规律，使其可以有效加速，比如2:4 Sparse Matrix（这是A100 GPU中特别优化过的算法）

![](asset/Pasted%20image%2020251213151247.png)

再比如Channel Pruning，这是相当粗粒度的剪枝，相当于剪掉了一整个通道，对于神经网络而言，会变成如下这样子。在这些例子中，如何确定稀疏度的大小，也是一个需要考量的范围。随机选择每层的稀疏度并不是一个好的选择

![](asset/Pasted%20image%2020251213151452.png)

**Pruning Criterion**

如何选择需要剪枝的内容，我们的核心准则自然是选择哪些对于模型输出影响最小的参数进行剪枝

神经网络流程本质还是矩阵乘法，因此参数的绝对值基本就能够代表其对于整个模型的影响，因此方案一就是比较绝对值（L1-norm）

![](asset/Pasted%20image%2020251213152548.png)

当我们要做更加规范的剪枝的时候，比如删除整行，可以考虑就是把整行参数的绝对值加起来，和其他行进行比较，进行剪枝

还有一种方案是引入可学习的参数，比如下图中的缩放因子，如果最终训练出来的结果，这些缩放因子比较小，可以推出这个部分的作用较小，我们考虑将其进行剪枝（这部分的缩放因子，甚至可以考虑复用batch norm中的缩放因子）

![](asset/Pasted%20image%2020251213152905.png)

参数的剪枝和连接的剪枝是不同的，参数的剪枝往往是很粗粒度的，其相当于矩阵乘法中一行的剪枝

![](asset/Pasted%20image%2020251213153612.png)

另外也存在一些动态的方案，比如Percentage-of-zero-based Pruning，之前讲述的方案都是静态分析参数，这种方案是在计算过程中动态分析激活值，将其中0的比例较高的一批channel给去掉，达到节省内存的目的

还有一种技巧是Regression-Based Pruning，这种方案旨在对于某一层进行剪枝，因为整个运行深度网络消耗巨大，因此在剪枝的时候可以仅仅将一层的输出和这一层的满状态进行对比，进行剪枝，基于这样子的回归监督，我们可以在不消耗大量资源的情况下，分析出哪个channel更适合剪枝

![](asset/Pasted%20image%2020251213154336.png)

**Pruning ratio**

分析一层该有的pruning ratio，我们采用一种sensitivity analysis的内容，有些层对于剪枝非常敏感，另一些层则不是

简单来说，我们就是对于某一层进行剪枝，随后分析accuracy的变化，来确定该层的敏感度

![](asset/Pasted%20image%2020251213162914.png)

> 这种方案同时也作为剪枝算法的正确性判断，一般来说在前60%大部分性能都不会严重下降，而在90%多的时候都会有下降，如果违反了类似的规律，可能是剪枝算法实现错误的表现

简单的方案是确定一个精度下降的阈值，基于这个阈值我们可以确定每层应该有的pruning ratio

![](asset/Pasted%20image%2020251213163226.png)

**Automatic Pruning**

我们想找到一种可扩展的pruning算法，而不是依赖于人类专家对于一个具体的模型去手动调试，该pruning算法应当在不同模型，不同的benchmark上都可以自动优化找到合适的方案

实际上代替人类专家的可验证任务，是一种合适的强化学习场景，因此我们会将其定义为一个RL任务，来我们找到合适的pruning ratio

![](asset/Pasted%20image%2020251213163710.png)

还有一种叫做NetAdapt的方案，也是以RL的方案优化模型pruning ratio

### 2.3. Finetuning & retraining

我们在Pruning之后往往会选择微调重训练来获得更好的性能，这里也有一些技巧

首先我们如何选择学习率，一般来说我们会选择更小一些的学习率，因为模型此时应该已经处于一个比较好的最小区间了

另外也可以选择迭代式的方案，比如先剪枝30%->微调->剪枝50%->微调，会比直接剪枝50%好

![](asset/Pasted%20image%2020251213165543.png)

### 2.4. System Support for Sparsity

**EIE: Efficient Inference Engine**






