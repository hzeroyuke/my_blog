- DAPO & GSPO 对于 GRPO 缺陷的优化
- [JustRL](https://relieved-cafe-fe1.notion.site/JustRL-Scaling-a-1-5B-LLM-with-a-Simple-RL-Recipe-24f6198b0b6b80e48e74f519bfdaf0a8) 用最简单的配置实现RL的性能持续提升
- https://arxiv.org/pdf/2511.19942 Differential Smoothing
- https://arxiv.org/abs/2505.22617 RL 中的熵机制
- https://arxiv.org/pdf/2512.01374 MiniRL：训推一致性的RL
- https://arxiv.org/abs/2512.07783 分析预训练，中训练，RL对于LLM推理能力的影响，构建了一个尽可能消除知识重叠的训练数据集，使得我们可以更好地分析每个训练阶段对模型能力带来地影响

对于LLM的RL，统一的视角应该是优化一个reward的期望值，并且附加一些截断，KL等技巧

RL现阶段有两个重要的命题

- 训练和推理的一致性：尤其是对于Moe模型，因为现在更新模型的时候总是拆分minibatch，所以都有一定的off-policy。并且当推理框架和训练框架分离，使用不同的算子库的时候，存在精度等问题导致的不一致性
- 探索性和优化的平衡：如果简单地对奖励进行优化，会导致其探索性快速下降，模型陷入局部最优，不再提升性能，表现为模型的熵下降，以及模型的Pass@1的性能提升的同时，Pass@k的性能不再提升

很多的论文都围绕这个部分来展开

## 1. Basic RL algorithm

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

将token level的advantage和importance ratio改为sequence level，importance ratio改为

![](asset/Pasted%20image%2020251224142207.png)

目标函数改为

![](asset/Pasted%20image%2020251224142238.png)



## Other Topics

这里介绍除了训推一致性和探索收敛以外的一些RL主题

### 1. Bottom-up Policy Optimization

这篇[论文](https://arxiv.org/pdf/2512.19673)通过分析LLM内部残差流的熵变化，来分析LLM的推理过程，对于一些主流开源模型进行了分析，发现了Llama模型在大量的层里面都保持较高的熵，而在最后几层快速收敛，而Qwen的模型则大多保持一个比较平稳的熵下降流程。在今年的RL实验中发现了Llama系列模型在做RL的时候相当不稳定，需要大量SFT数据预热才可以正常进行RL，这篇论文为我们提供了一个新的视角

![](asset/Pasted%20image%2020251230141134.png)

除了对于现在开源模型的研究以外，这篇论文还考虑能否对单个层的输出特性做优化，提出了一种新的优化方案，比如对于Qwen，我们可以固定后续收敛的层，仅对于前面几层进行优化，对于Llama，我们只固定最后几层，对其他部分进行优化，（这里只优化0-L层和最后的解嵌入矩阵）发现这种方案的RL可以作为预热，使得Qwen系列和Llama系列模型都有效涨点，并且维持Pass@k性能

关于RL对于模型内部层之间的影响，还有一篇Google的论文也分析了类似的现象，这篇[论文](https://arxiv.org/pdf/2512.20605) 提出了Hierarchical RL的新方法，同样是研究了内部的残差流，其中涌现出来线性可控的时序抽象，然后得到了一些观察

- 中间层有最丰富的子目标信息





