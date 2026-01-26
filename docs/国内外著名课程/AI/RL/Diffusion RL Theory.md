- [AWM 官方博客](https://zhuanlan.zhihu.com/p/1958843370878277526)
- [Diffusion RL summary](https://zhuanlan.zhihu.com/p/1995493539912623296?share_code=1pv5WzGwkLpnQ&utm_psn=1995940476071736428)
- [flow-grpo](https://github.com/yifan123/flow_grpo?tab=readme-ov-file)
- 

## 1. Flow-GRPO

这是从DS-R1之后，影响力较大的第一篇将GRPO迁移到flow matching上的工作

要将llm的rl范式迁移到flow matching，面对的最大问题就是flow matching的ode轨迹是确定的。我们知道GRPO实际是在优化重要性采样的那个比例，但是LLM的输出是整个token空间，而flow的ode输出的是具体的速度，也导致了其无法获得重要性采样的概率比，也就是无法优化，这个是首要需要解决的问题

重要性采样为

$$r^i(\theta)=\frac{p_\theta(x_{t-1}^{i}|x_{t-1}^i,c)}{{p_\theta}_{old}(x_{t-1}^{i}|x_{t-1}^i,c)}$$

问题就在于，flow模型ode是一个狄拉克分布: $p_\theta(x_{s}|x_{t},c),s<t$ ，因此要引入概率分布，就在特定的一步，将ode转换成sde，具体的方案如下


![](asset/Pasted%20image%2020260126200145.png)

Flow GRPO实际的policy gradient实际上是使用advantage加权的噪声，我们可以通过梯度的推导发现这一点，这个推导在Diffusion NFT的论文中也有过程

更通俗的来说，Flow GRPO的过程是找到了一个更好的噪声，然后让速度场往这个噪声的方向进行移动，噪声本身是没有意义的，只是这个噪声隐含了更优的速度方向，因此这个行为是有效的

在Flow GRPO后续出了一个系列工作，Flow GRPO Guard，里面也有这个梯度推导，结论就是，Flow GRPO在让速度场往一个噪声的方向移动，这个其实是有点奇怪的，因此LLM的预训练和后训练行为时一致的，但是Flow GRPO中，预训练是优化速度场，但是RL却变成了优化噪声，这个行为似乎是不够直接的

## 2. AWM

这篇论文正是发现了上述Flow GRPO的不一致问题

- RL优化的是Reverse一步转换的高斯分布的对数似然
- 预训练优化的是score/flow match loss

以Flow GRPO为代表的方案，将带噪数据推断训练目标，会将这个不确定性直接传到梯度目标里

而AWM提出了一种策略梯度一致的做法，损失函数依然是Flow matching那一套，只是做了一个加权

![](asset/Pasted%20image%2020260126212831.png)

## 3. Diffusion NFT

