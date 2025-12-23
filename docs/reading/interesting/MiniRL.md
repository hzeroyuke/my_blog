
该论文是Qwen团队提出的，关于RL稳定性的一篇论文，相应的链接在[这里](https://arxiv.org/pdf/2512.01374)，主要是要解决RL中各个机制实际的作用，以及关于训推一致性的探讨，尤其是对于Moe模型

实验在30B的Moe模型上

RL最大的不匹配其实是reward是序列级别的，但是优化是token级别的，论文指出，只有在训推尽可能一致的时候，优化token级别的目标就可以提升序列级别的奖励

训推的不一致从两个方面产生

- Training-Inference Discrepancy: 推理引擎和训练引擎的区别：比如Kernel不同，推理端的优化包括低精度计算等等
- Policy Staleness: 策略滞后性：大batch拆成minibatch，导致后面的minibatch变成off-policy

基于这些思考之后，我们从最初的reinforce算法开始，做最小的改动

- Advantage Group Norm 降低方差
- Clipping 防止过度更新
- Token-Level IS correction 修正训练推理差异

![](asset/Pasted%20image%2020251204201933.png)

而针对于Moe，论文也提出了两种方案来解决Moe的训推不一致

- Vanilla Routing Replay(R2): 在梯度更新时，重放Rollout策略中选中的专家，也就是 $\pi_{old}$ 选择的专家
- Rollout Routing Replay(R3): 在梯度更新时，重放推理引擎中选择的专家

这两个方案都改动了当前token需要选择的专家，因此都有偏差










