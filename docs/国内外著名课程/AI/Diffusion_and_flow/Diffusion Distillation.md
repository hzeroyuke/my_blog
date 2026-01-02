这篇文章会用来分享一些有关于Diffusion&Flow加速相关的工作

这里的主要内容来自于这篇[文章](https://arxiv.org/abs/2512.13006v1) ，一篇Distillation Guide，主要是对于Text-to-Image，这篇文章将Diffusion的蒸馏划分为了两个范式，从分布和轨迹角度进行拟合教师模型

- Distribution Distillation: 包括DMD，LADD
- Trajectory-based Distillation: 包括sCM，IMM，MeanFlow

其中Distributon Distilation的范式已经有了很多实践验证，比如Qwen-Image-Lighting和Flux.1 Kontext dev都使用了上述技术，但是这三种Trajectory-based Distillation的方案，却只是在小规模（ImageNet）中做了验证，这篇论文为这些方案都加入更多的实验，提供了很多的Insights

## 1. Method Overview

### 1.1. sCM

用 TrigFlow 的范式来稳定 continuous consistecny model 的训练，包括架构上的改动，以及一些加权

### 1.2. MeanFlow

Flow Matching学习瞬时速度而Meanflow学习平均速度，输入两个时间点，计算平均速度

本文为MeanFlow提出了一个蒸馏方案，MeanFlow也是一个可以蒸馏可以从头训练的范式

![](asset/Pasted%20image%2020251230144649.png)

### 1.3. IMM

替代KL方案，IMM提出了一种MMD的方案，来衡量distirbutional difference

## 2. Comparative Insights

### 2.1. Flow matching & MeanFlow

Flow Matching和MeanFlow的关系可以理解为，FM是一种MeanFlow的特殊情况，也即r=t的时候，此时MeanFlow的目标就是在预测 $\boldsymbol{v}$ 

![](asset/Pasted%20image%2020251230150327.png)


### 2.2. FM & sCM

FM和TrigFlow在推理中是可以互相转换的，表现为，用一种方案训练的模型，可以用另一套范式的Sampler，而没有任何的能力退化，只需要通过一些线形组合

**用TrigFlow Sample运行FM**

![](asset/Pasted%20image%2020251230152503.png)

**用FM Sample运行TrigFlow**

![](asset/Pasted%20image%2020251230152529.png)

### 2.3. sCM & Meanflow

### 2.4. CM & IMM
















