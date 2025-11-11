
Prompt很重要，虽然在现阶段上下文工程火热的情况下，初始的System Prompt往往被人忽略，但是其在对话和特定领域上依然重要，我的理解中

- Context Engineer 主要在做Memory
- Prompt Engineer 主要在学习一种思考模式

因为大模型的能力边界一直没有被准确定义（人们称其为锯齿状的智能），如果要为一个模型定制一份Prompt，会花费很大的人力，因此由很多工作想要让LLM自动优化Prompt

相关工作有

- https://www.arxiv.org/abs/2510.04618 ACE
- https://arxiv.org/abs/2507.19457 GEPA
- https://arxiv.org/abs/2510.08191 Training Free GRPO
- https://arxiv.org/abs/2502.16923 Survey for APO

## 0. Survey of APO

![](Pasted%20image%2020251015162545.png)

一个APO流程一般包含以下元素

- Seed Prompt 初始的提示词
- Inference Evaluation and Feedback 执行一个任务并且获得反馈
	- Numeric Score Feedback
	- LLM Feedback
	- Human Feedback
- Candidate Prompt Generation 生成候选的提示词
	- 启发式Edit：直接让LLM生成新的提示词
	- 用专门训练的神经网络来提示词生成
	- 基于覆盖的方案
	- 程序合成
- Filter and Retain Promising Prompts 筛选和保留有希望的提示词
	- TopK
	- 上置信界
	- 区域联合所有
- Iteration Depth
	- 固定步数
	- 动态步数，根据性能进行动态调整

一个正常的流程大概是，首先是有一个初始的Prompt，随后跑一遍Benchmark，根据获得的Feedback来生成一批新的Candidate Prompt，基于某个规则进行筛选，维护一个Candidate Prompt Pool，如此循环

## 1. GEPA

![](Pasted%20image%2020251015182200.png)

这个算是我们现在的Baseline，其具体的流程图如上，其优化的结果示例如下

其优化具体的流程如下

- 初始化Prompt，直接进入Candidate Pool中
- 开始有一个优化循环
	- 进行Candidate Pool Filter，基于Pareto Frontier进行筛选（对于其中任何一个Prompt，都找不到另一个在所有任务上都比它好或跟它一样好的Prompt）
	- Propose New Candidate，每个Prompt其会和上一次执行的结果相连，根据这个信息直接优化提示词，或者其还有一种将两个提示词的优势结合的方案
	- 新生成的Prompt要经过一个小miniBatch的评测，如果没有提升直接丢弃，如果有则过一个Test集合，将这个结果纳入到得分矩阵中，并且将这个Prompt丢到Candidate Pool中
- 重复如上的优化循环，直到Budget用完

其比较核心的创新点在于引入了Pareto的概念，这个筛选机制可以保留那些在特定任务上表现突出、有特色的Prompt，而不仅仅是那些“平均分”高的Prompt，从而保证了候选Prompt的多样性，避免陷入“局部最优”。

![](Pasted%20image%2020251015182629.png)


## 2. ACE

- 没开源，也没有迭代完成的Prompt，名声很响但是莫得什么可用的东西，网上只能找到别人复现的结果

![](Pasted%20image%2020251015183614.png)

这篇论文核心要解决的问题是针对之前的APO工作存在

- 简洁偏差：之前的方案优化的过程都追求越短越好，导致细节的缺失
- 上下文坍缩：总是依赖LLM一次性重写整段Prompt，会导致最终提示词变为更短更泛的摘要，会有关键信息的确实

因此这个方案中其将提示词优化改为了维护一个PlayBook，设计了这么三个角色

- Generator：针对新查询产出完整推理轨迹。
- Reflector：仅负责“评判+提取”，把成败原因转化为可落地的战术句子；独立角色避免“评判”与“重写”耦合造成的信息丢失。
- Curator：将 Reflector 输出的战术封装成结构化条目（bullet），并以确定性逻辑合并到现有手册，杜绝 LLM 重写带来的方差。

并且在优化的过程中分两步走
- Grow：新的条目增加
- Refine：精炼，语义嵌入去重，合并或淘汰低价值条目

没有完整的示例和开源的代码

## 3. Training-Free GPRO

- https://github.com/TencentCloudADP/youtu-agent/tree/training_free_GRPO 有维护的比较好的Codebase

![](Pasted%20image%2020251015185748.png)

这篇文章follow GRPO的范式来做Prompt优化

将参数空间替换为上下文，将梯度更新替换为对于一个自然语言经验库进行增删查改，具体流程如下

- Rollout并且用奖励模型进行打分
- 组内奖励比较，基于这些内容进行Summary随后输出反思
- 基于反思来更新经验库

对于经验库可以进行增删查改

一个具体的优化流程如下，先是执行获得轨迹，分析轨迹和Feedback，发现出现的问题或者做的好的点，然后提炼出规则

![](Pasted%20image%2020251015191601.png)

其优化的结果节选如下

![](Pasted%20image%2020251015192317.png)

相比于一个文章形式的System Prompt，其更类似于维护一个知识库，对于多轮的回答或者多轮的Agent，比如Web Search等任务，这种形式或许比System prompt优化更好。对于Chat-Agent的优化任务，这种方案可以覆盖更多的边界情况的，也是类似于在做一个好的Memory机制


