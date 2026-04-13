

- real time vla
	- https://innovator-zero.github.io/FASTER/
	- https://arxiv.org/pdf/2512.01031v1 异步推理达到real time VLASH

## Diffusion Policy


- [扩散策略算法整理 知乎](https://zhuanlan.zhihu.com/p/6223910015?share_code=1qVRamye0oBu1&utm_psn=1953772172360278944)

所谓的Policy，表现在机器人之类的身上，就是输出一系列的参数

- Gassiuan Policy 一步神经网络，输出所有参数的均值和方差
- Diffusion Policy 多步神经网络，和diffusion一样，输出所有的参数






## 1. world action 

- https://mp.weixin.qq.com/s/QzUR6I6Gjx63vXbr0WLC1g
- 文章里为各个范式的论文都写了简单的介绍

这算是vla的一个分支，不是在MLLM的基础上构建action model，而是预测下一个物理状态，业界认为，一个完整的vla应该具有两方面的能力

- 根据observation & instruction来预测动作
- 根据obseravtion & action 来预测之后的世界状态

video pretrain > vision-language pretrain

基于上述共识，产生了一些技术路线

**1. Two-stage decoupled pipelines**

UniPi → VPP → Vidar → mimic-video → LAPA

使用视频模型预测未来的一段视频，然后用一个逆动力学视频，从video中反推出机器人执行的动作

**2. End2End joint video action generation**

PAD → VideoVLA → WorldVLA → Cosmos Policy → DreamZero

在Dit网络中同时预测video和动作

**3. Unified multi-function models**

UVA → UWM → LingBot-VA → Motus

一个模型同时做正向动力学、逆向动力学、策略推理、视频生成，其动机在于，训练数据适合scaling，这种范式能从纯视频中学习策略，之前的方案都需要大量的数据标注





