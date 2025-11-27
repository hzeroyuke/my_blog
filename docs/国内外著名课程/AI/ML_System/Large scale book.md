
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

- 形状：形状由各种超参决定，比如 batch size, sequence length, attention head number
- 精度：FP8, FP32

从内存的含义上划分，可以分为下图中的各类内容

![](Pasted%20image%2020251014153331.png)

**Weights / Gradients / Optimizer states**

先从前三个内存占用开始

关于transformer LLM，其参数数量可以由以下公式给出

![](Pasted%20image%2020251020194241.png)

其中h是hidden states, v是词表大小，L是模型层数，而在全精度FP32的训练中，参数和梯度都是4字节，而Adam优化器要额外存储动量和方差

![](Pasted%20image%2020251020194423.png)

在如今更常见的混合精度训练中，用BF16来存储参数和梯度，也就是从4个字节降低为2个字节，但Adam的优化器状态为了稳定还是要保持全精度，除此之外还要保存一份全精度的参数副本

![](Pasted%20image%2020251020194918.png)

**Activation Memory**

接下来我们分析激活值所占的内存，[这篇文章](https://www.determined.ai/blog/act-mem-2) 很好地讲述了激活内存地含义，在混合精度地情形下，激活值所占地内存总量为

![](Pasted%20image%2020251020195732.png)

L表示层数，seq表示序列长度，bs表示batchsize，h表示hidden states，$n_{heads}$ 表示注意力头数量

![](Pasted%20image%2020251020195854.png)

由上图和上述公式可以看出，激活值所占地内存随着序列长度二次增长，面对这个难题，我们引入一个重要地Tricks

**Activation recomputation**

这项技术称为激活值重计算，就是在前向过程中丢弃一些激活值，在反向传播是花费一些时间将需要计算地激活值重新算一遍

![](Pasted%20image%2020251020200419.png)

当然还有一个trade-off是我们需要保存对应层地检查点，计算表明，如果我们丢弃所有地激活值，也就相当于在反向地时候重新做一遍前向，会增加30-40%的计算量

因此我们往往会有选择地进行激活值丢弃，研究表明，Attention操作的激活值增长很快，但是其前向的成本很低

现在最为常用的FlashAttention也很巧妙地将激活值重计算纳入框架中

**Gradient Accumulation**

对于激活值另一个大头的影响因素是Batch size，我们会采用Gradient Accumulation这个设计来尝试解决这个问题

梯度累积包括将一批次数据分成多个minibatch，然后根据每个minibatch进行前向和后方，计算梯度，在优化之前将所有minibatch的梯度相加

![](Pasted%20image%2020251020212805.png)

我们用mbs表示minibatch，用gbs表示全局的batch size，将gbs拆成minibatch，使得我们可以顺序处理minibatch来减少激活内存，但是也存在tradeoff也就是我们要多做几次前向反向

值得注意的是minibatch的梯度计算其实可以并行的，因此我们的多GPU也就有了用武之地

## 2. Data Parallelism

DP的理念就是在多个GPU上复制模型，对于这些模型的复制版本，可以进行MiniBatch的更新，其实和刚才的梯度累积的方案是一样的

因为我们对每个GPU进行梯度计算，得到的不同梯度我们需要进行同步，我们会使用all-reduce的分布式操作来对模型实例的梯度进行平均，随后再进行优化

![](Pasted%20image%2020251020214125.png)

一个简单的DP流程如上，但是上述流程仍然不是最终形态，因为此时GPU在通信过程中是空闲状态

方案一：重叠梯度计算和通信，因为我们没有必要等待整个模型的梯度计算完成再进行all-reduce，完全可以在一边计算梯度，一边更新

```python
def register_backward_hook(self, hook):
	for p in self.module.parameters():
		if p.requires_grad is True:
			p.register_post_accumulate_grad_hook(hook)
```

![](Pasted%20image%2020251020214647.png)

方案二：bucket梯度求和，因为GPU更适合对大张量做操作，而不是对多个小张量做操作，对于通信也是一样的，因此我们也可以让梯度按bucket进行all reduce

![](Pasted%20image%2020251020221202.png)


在Pytorch中，往往在不需要All-reduce的backward操作中加上 `model.no_sync()` 装饰器

DP效果很好，但是在GPU数量达到上百上千的时候，其协调开销会显著增加，并且网络开销会很大，导致其逐渐变得低效

并且DP能够成立的前提，也是至少一个前向过程能够塞入单个GPU内，这对于超大模型来说是无法做到的

![](Pasted%20image%2020251020222121.png)


## 3. Deepspeed Zero

DeepSpeed内部的zero机制，是一种旨在减少LLM训练中内存冗余的优化技术

DP中对于优化器状态，梯度以及参数的简单复制带来的冗余（激活值是每个DP副本不同的，因此不算在冗余之内，我们只能通过减小batch size来减小激活值），正是我们想要优化的目标

- Zero1 优化器状态分区
- Zero2 优化器状态+梯度分区
- Zero3 优化器状态+梯度+参数分区

![](Pasted%20image%2020251020222706.png)

Zero的设计理念在于把所有复制的数据都分片，只有在必要的时候才聚合起来使用

**Zero1**

在Zero1中，优化器状态被分为N份，每个GPU上都只维护1/N份优化器状态，更新的时候只优化对应的FP32权重，而在部分更新完成之后，通过一个all-gather操作，将各个GPU上的参数都更新到最新状态

**Zero2**

**Zero3**

## Appendix: code practice

### 1. Show GPT2 Architecture

```python
from transformers import GPT2Model
model = GPT2Model.from_pretrained('gpt2')
def count_params(model):
    params: int = sum(p.numel() for p in model.parameters())
    return f"{params / 1e6:.2f}M"

print(model)
print("Total # of params:", count_params(model))
```

得到的结果如下

```
GPT2Model(
  (wte): Embedding(50257, 768) # token embedding, from one-hot vector to 768-dimensional vector
  (wpe): Embedding(1024, 768)  # position embedding 支持最大1024序列长度
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-11): 12 x GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2Attention(
        (c_attn): Conv1D(nf=2304, nx=768)
        (c_proj): Conv1D(nf=768, nx=768)
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D(nf=3072, nx=768)
        (c_proj): Conv1D(nf=768, nx=3072)
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
Total # of params: 124.44M
```

### 2. Memory Compute

我们可以通过checkpoint这个机制，来实现激活值重计算

这个在小模型中很难体现这一点

### 3. Gradient Accumulation

```python
import torch
from torch import nn, optim

model = nn.Linear(10, 1).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

gradient_accumulation_steps = 4
microbatch_size = 2

# Fake data
inputs = torch.randn(gradient_accumulation_steps * microbatch_size, 10).cuda()
targets = torch.randn(gradient_accumulation_steps * microbatch_size, 1).cuda()

model.train()            # 切换到训练模型
optimizer.zero_grad()    # 清空之前的梯度，习惯性操作在训练之前
for i in range(gradient_accumulation_steps):
    start = i * microbatch_size
    end = (i + 1) * microbatch_size
    x = inputs[start:end]
    y = targets[start:end]

    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()  # 只算梯度，而不更新

    print(f"Step {i+1}: Loss = {loss.item():.4f}")

optimizer.step()   # 更新梯度，此时param.grad中已经保存了4个minibatch的梯度了
print("Optimizer step performed with accumulated gradients.")
```

上述是一个梯度累积的模拟，我们在一个循环中只计算梯度，而不更新，在循环结束之后累积梯度再更新

真正实践中，该方案内化成了如下的参数

```python
from transformers import Trainer

training_args = TrainingArguments(
    per_device_train_batch_size=2,      # micro-batch
    gradient_accumulation_steps=4,     # 累积 4 次
    # 实际等价于 global batch size = 2 × 4 × (GPU数量)
    # 比如 8 张卡：2 × 4 × 8 = 64
)
```

### 4. Data Paraellism

