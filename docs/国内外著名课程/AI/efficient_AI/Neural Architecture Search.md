NAS是指我们如何去设计一个好的神经网络架构，使其在准确度和高效方面上有一个比较好的权衡

## 1. Conv Block

![](asset/Pasted%20image%2020251215153756.png)

比如上图是ResNet-50的一个block的内容，其先用1x1卷积将输入进行降维到512，再进行昂贵的3x3卷积，最后再升维出去，这是一个比较好的设计

并且我们可以获得这个模型的计算量和参数量，比直接对于2048的原始输入进行3x3卷积都要优秀得多

对于卷积操作有众多优化的方案，比如Group Conv（ResNext）

![](asset/Pasted%20image%2020251215154019.png)

还有将depth-wise conv和channel-wise conv分离的方案，比如MobileNet

![](asset/Pasted%20image%2020251215155401.png)

## 2. Transformer Block

Transformer block 中的计算性能瓶颈主要是在Attention操作中，以下是对Attention操作的基本分析

![](asset/Pasted%20image%2020251215155747.png)

## 3. Automatic Neural Architecture Design

![](asset/Pasted%20image%2020251215160215.png)

这是一张图为我们展示了神经网络架构早期的设计历史，一部分是人们手工设计的结构，另一部分是利用机器学习，自动设计的网络结构，这种结构往往会更加高效

以下是神经网络搜索任务的抽象流程

![](asset/Pasted%20image%2020251215160628.png)