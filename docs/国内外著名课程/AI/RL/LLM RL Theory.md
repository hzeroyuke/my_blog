- DAPO & GSPO 对于 GRPO 缺陷的优化
- [JustRL](https://relieved-cafe-fe1.notion.site/JustRL-Scaling-a-1-5B-LLM-with-a-Simple-RL-Recipe-24f6198b0b6b80e48e74f519bfdaf0a8) 用最简单的配置实现RL的性能持续提升
- https://arxiv.org/pdf/2511.19942 Differential Smoothing
- https://arxiv.org/abs/2505.22617 RL 中的熵机制
- https://arxiv.org/pdf/2512.01374 MiniRL：训推一致性的RL
- https://arxiv.org/abs/2512.07783 分析预训练，中训练，RL对于LLM推理能力的影响，构建了一个尽可能消除知识重叠的训练数据集，使得我们可以更好地分析每个训练阶段对模型能力带来地影响

对于LLM的RL，统一的视角应该是优化一个reward的期望值，并且附加一些截断，KL等技巧

RL现阶段有两个重要的命题

- 训练和推理的一致性：尤其是对于Moe模型，因为现在更新模型的时候总是拆分minibatch，所以都有一定的off-policy
- 探索性和优化的平衡：如果简单地对奖励进行优化，会导致其探索性快速下降，模型陷入局部最优，不再提升性能

很多的论文都围绕这个部分来展开


