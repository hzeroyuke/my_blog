- DAPO & GSPO 对于 GRPO 缺陷的优化
- [JustRL](https://relieved-cafe-fe1.notion.site/JustRL-Scaling-a-1-5B-LLM-with-a-Simple-RL-Recipe-24f6198b0b6b80e48e74f519bfdaf0a8) 用最简单的配置实现RL的性能持续提升
- https://arxiv.org/pdf/2511.19942 Differential Smoothing
- https://arxiv.org/abs/2505.22617 RL 中的熵机制
- https://arxiv.org/pdf/2512.01374 MiniRL：训推一致性的RL
- https://arxiv.org/abs/2512.07783 分析预训练，中训练，RL对于LLM推理能力的影响，构建了一个尽可能消除知识重叠的训练数据集，使得我们可以更好地分析每个训练阶段对模型能力带来地影响

对于LLM的RL，统一的视角应该是优化一个reward的期望值，并且附加一些截断，KL等技巧

整个2025年期间，RL有两个重要的命题

- **训练和推理的一致性**：尤其是对于Moe模型，因为现在更新模型的时候总是拆分minibatch，所以都有一定的off-policy。并且当推理框架和训练框架分离，使用不同的算子库的时候，存在精度等问题导致的不一致性
- **探索性和优化的平衡**：如果简单地对奖励进行优化，会导致其探索性快速下降，模型陷入局部最优，不再提升性能，表现为模型的熵下降，以及模型的Pass@1的性能提升的同时，Pass@k的性能不再提升

很多的论文都围绕这个部分来展开

而随着LLM RL从数学逻辑领域转向了复杂的长程Agent任务领域，比如Coding，办公任务等等，2026年初开始，RL开始转向了异步和Online的RL训练，其核心在于

- **asynchronous**：等待推理完成再训练实在是太慢了，如何缓解必要的off-policy
- **online**：如何从用户和agent的交互中获得有效的监督信号，如何构建有效的reward

就目前LLM的RL的来看，对于算法的改进是最不重要的，任务环境和训练环境的infra+data是更重要的

## 1. Basic RL algorithm

**From Policy Gradient to PPO**

这是早期RL算法的演变的过程，从最开始的Policy based Method + Value based Method 到后面的PPO的过程


**PPO**

![](asset/Pasted%20image%2020251224134542.png)

importance ratio + Advantage 优化，当Advantage  大于 0 的时候，该优化目标会迫使importance ratio变大

PPO的advantage计算是通过reward model和value model共同完成的，Reward model计算即时奖励，Value model预估外来的奖励，随后通过GAE来计算token level advantage

![](asset/Pasted%20image%2020251224140831.png)

随后依次更新Value model和policy model的梯度

**GRPO**

![](asset/Pasted%20image%2020251224141119.png)

GRPO通过一个group里的优势计算，来绕过value model的设计，使得训练成本大幅度下降，通过计算该条回答的reward相比于这个group的平均reward的优势，来衡量advantage

![](asset/Pasted%20image%2020251224141304.png)

**DAPO**

DAPO的方案对于GRPO的范式做了一系列的优化，增加了很多Tricks

- Clip Higher 原本的clip的上界过低，导致RL只会优化高概率高advantage的token，以至于其会迅速陷入局部最优，通过调高Clip的界限，可以有效帮助一些低概率高Advantage的token进行优化，有效提高模型的上限
- Dynamic Sampling 在训练中增加筛选，去掉全对和全错的样本
- Token-level Policy Gradient Loss 让一个mini-batch内部 repsonse token 的advantage权重相同
- Overlong Reward Shaping 增加response length的惩罚项数

**GSPO**

GRPO虽然本身也是Sequence Level的IS，但是它的IS对于每个token做clip来进行连乘，实际上还是对于token level进行操作，而GSPO转向完整的Sequence Level，不用token-level的clip，而是用整个Sequence的内容做几何平均归一化（就是在token的log上除以一个序列长度），来防止IS爆炸问题

![](asset/Pasted%20image%2020251224142207.png)

目标函数改为

![](asset/Pasted%20image%2020251224142238.png)


**VESPO**

来自小米团队的论文，开始转向off-policy的RL训练，能够做到在64x staleness ratios上做到稳定的训练

面对现有Policy Gradient的公式，想要提升算法，我们就要对Importance Weight做改动，之前最重要的做Importance Weight做改动的算法是GSPO，最核心的观点是将Token-Level的Importance Sampling修改到Sequence-Level，这篇论文指出其存在一定的问题，Sequence Level的IS将token level的进行叠加，导致单个token很大的IS会影响到整个序列，并且在长序列中表现得更为严重

面对这个问题，GRPO等该用Token Clip进行裁剪（实际上就是token level更新，放弃了序列级的统计），GSPO采用归一化的方案（存在长度依赖的偏差）

VESPO带我们重新审视对于Weight Reshaping这个操作，原生的IS是将这条老数据对应的分布和当前模型的分布进行对比，当我们做了Weight Reshaping之后，就是将老数据的分布和某个其他的分布进行对比，我们从优化一个IS的方案转向优化一个分布，这个分布应该具有一些特点

- 靠近老数据的分布
- 靠近新模型的分布
- 控制方差

比较直观地，我们可以通过一个权重系数来处理它

![](asset/Pasted%20image%2020260310163955.png)

另一方面，我们来看起如何做方差的控制，这是Sequence-Level的优化比较重要的方向。这就要套入之前的理论中，之前有很多理论来分析分布的方差

在重要性采样中，估计量的方差正比于二阶矩，二阶矩阵越大，相当于方差越高，有效的样本量就越少

![](asset/Pasted%20image%2020260310164700.png)

最终VESPO的结果是这样子的，其中W是原始的重要性采样

![](asset/Pasted%20image%2020260310165026.png)
## 2. Training-Rollout Consistency

在这个任务上，Slime应该是一个值得学习的项目

### 2.1. Multi-Turn Agent Training

token id 的解码编码问题

### 2.2. Precision Consistency

Nvidia在2月份出了一篇关于训推精度一致性的论文 [Jet RL](https://arxiv.org/pdf/2601.14243v1) ，对于RL而言，目前来看效率上的block就在rollout上面，因此最直接的想法我们会将直接推理加速的方案应用在RL的rollout上面，比如low-bit量化和稀疏计算。这篇论文就是研究了FP8 Rollout的技术，但是先前的低精度的rollout会和后续BF16的训练形成不一致，形成Off-Policy。

这篇论文在Verl上构建一个Rollout和Train均为FP8的RL过程

这个过程有如下的难点

- RL过程中，参数在频繁地更新，传统的FP8量化方案，需要每次参数更新之后再次计算缩放系数（重新校准），造成额外开销
- 如果不进行校准，直接将参数截断为FP8，会导致训练的不稳定，尤其是面对长文本和困难任务中


![](asset/Pasted%20image%2020260205144826.png)

反向传播的梯度保持BF16，其余内容全部转换成FP8。总结来看就是所有的算子都保持FP8，但是输出梯度的时候转换成BF16


## 3. Entropy

在LLM RL中，关于Entropy的思考是一个受人关注的话题，在早期对于math这个领域的RL，LLM总是展现出熵坍缩的现象，也即熵快速下降导致探索性能的下降。但是在后期的Agent RL的任务中，反而出现了一系列的熵增长的现象，也即LLM的探索性能反而在提升。

![](asset/Pasted%20image%2020260316192842.png)

上图来自于UI-Tars 2的论文，包括一些deep research的agent rl中也有类似的现象

个人的一些想法，模型的entropy的变化主要来自于是否有外界的信息输入，在math等任务中，rl只有问题和答案，模型在自己的能力中逐渐探索收敛；而在agent的任务中，模型有大量的工具调用，不论是GUI Agent还是Deep Research Agent，有大量的信息注入到训练过程中，一定程度上就扩展了模型的探索性

## Other Topics

这里介绍除了训推一致性和探索收敛以外的一些RL主题

### 1. Bottom-up Policy Optimization

这篇[论文](https://arxiv.org/pdf/2512.19673)通过分析LLM内部残差流的熵变化，来分析LLM的推理过程，对于一些主流开源模型进行了分析，发现了Llama模型在大量的层里面都保持较高的熵，而在最后几层快速收敛，而Qwen的模型则大多保持一个比较平稳的熵下降流程。在今年的RL实验中发现了Llama系列模型在做RL的时候相当不稳定，需要大量SFT数据预热才可以正常进行RL，这篇论文为我们提供了一个新的视角

![](asset/Pasted%20image%2020251230141134.png)

除了对于现在开源模型的研究以外，这篇论文还考虑能否对单个层的输出特性做优化，提出了一种新的优化方案，比如对于Qwen，我们可以固定后续收敛的层，仅对于前面几层进行优化，对于Llama，我们只固定最后几层，对其他部分进行优化，（这里只优化0-L层和最后的解嵌入矩阵）发现这种方案的RL可以作为预热，使得Qwen系列和Llama系列模型都有效涨点，并且维持Pass@k性能

本身在计算的过程中，使用第K层的输出logits来计算importance ratio，Reward还是来自于整个模型的输出，并且基于这个结果计算Loss，这个Loss自然依赖于前K层的梯度，因此在计算图中也就只会更新前K层，实现前K层单独的优化

关于RL对于模型内部层之间的影响，还有一篇Google的论文也分析了类似的现象，这篇[论文](https://arxiv.org/pdf/2512.20605) 提出了Hierarchical RL的新方法，同样是研究了内部的残差流，其中涌现出来线性可控的时序抽象，然后得到了一些观察

- 中间层有最丰富的子目标信息

### 2. KL in RL

这篇[论文](https://www.arxiv.org/pdf/2512.21852)提出了在RL训练中的KL应该采用K1估计器而不是现在常用的K3估计器，在RL中使用KL的原因是因为不希望模型偏离原本模型的分布太远，防止造成灾难性遗忘等问题。对于GRPO等算法，我们其实需要的是Sequence级别的KL计算，在实际操作中表现为将Token级别的KL进行累加

![](asset/Pasted%20image%2020260104201005.png)

之前的实践是采用将K3估计纳入到Loss计算中，值得注意的是，我们往往将RL中的Reward算作Loss的一部分，但是实际上是存在区别的，比如如果我们将KL项放在Reward中的话，其被视为标量奖励的一部分，梯度来源于对于policy采样的求导；而如果将KL项放在Loss中（现在常用的方案），现代AutoGrad框架会计算路径导数，因此两者其实存在显著的差异

![](asset/Pasted%20image%2020260104201452.png)

根据实验表现，将K1估计纳入到Reward中，比其他三种方式更接近无损的方案

![](asset/Pasted%20image%2020260104201607.png)








