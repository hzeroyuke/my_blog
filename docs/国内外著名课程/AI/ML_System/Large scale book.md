
这是 huggingface 中找到的一个关于LLM训练的一本很好的系统指南，主要关于底层的存储和LLM system，并且有很好的可视化

https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview 这是链接

## 1. 单GPU训练

一次正常的模型训练，包括以下流程

1. 一次前向传播，计算Loss
2. 一次反向传播，计算梯度
3. 基于梯度进行参数更新

![](Pasted%20image%2020251014150435.png)

这时候我们涉及到了第一个超参数 Batch size，也是我们前向计算Loss的时候的数据批次大小，根据实验表明，小批次的学习效率更高，但是大批次的数据噪声更小

我们来定义第一个参数 $bst = bs * sequence\ size$ 这是指一个batch size中的token数量

接下来我们来分析在训练的过程中的内存，占据我们的内存的内容物主要是各种各样的张量，一个张量的存储由两个因素决定

- 形状：形状由各种超参决定，比如batch size, sequence length, attention head number
- 精度：FP8, FP32

从内存的含义上划分，可以分为下图中的各类内容

![](Pasted%20image%2020251014153331.png)


