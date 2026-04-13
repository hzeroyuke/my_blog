- [AWM 官方博客](https://zhuanlan.zhihu.com/p/1958843370878277526)
- [Diffusion RL summary](https://zhuanlan.zhihu.com/p/1995493539912623296?share_code=1pv5WzGwkLpnQ&utm_psn=1995940476071736428)
- [flow-grpo](https://github.com/yifan123/flow_grpo?tab=readme-ov-file)

## 1. Flow-GRPO

这是从DS-R1之后，影响力较大的第一篇将GRPO迁移到flow matching上的工作，同期有很多类似的工作，比如DanceGRPO

要将llm的rl范式迁移到flow matching，面对的最大问题就是flow matching的ode轨迹是确定的。我们知道GRPO实际是在优化重要性采样的那个比例，但是LLM的输出是整个token空间，而flow的ode输出的是具体的速度，也导致了其无法获得重要性采样的概率比，也就是无法优化，这个是首要需要解决的问题

重要性采样为

$$r^i(\theta)=\frac{p_\theta(x_{t-1}^{i}|x_{t-1}^i,c)}{{p_\theta}_{old}(x_{t-1}^{i}|x_{t-1}^i,c)}$$

问题就在于，flow模型ode是一个狄拉克分布: $p_\theta(x_{s}|x_{t},c),s<t$ ，因此要引入概率分布，就在特定的一步，将ode转换成sde，具体的方案如下


![](asset/Pasted%20image%2020260126200145.png)

Flow GRPO实际的policy gradient实际上是使用advantage加权的噪声，我们可以通过梯度的推导发现这一点，这个推导在Diffusion NFT的论文中也有过程

$$\begin{aligned}
\nabla_{v_{\theta}} \mathcal{L} & \approx A \nabla_{v_{\theta}} \log p_{\theta}\left(\boldsymbol{x}_{t-1}^i \mid \boldsymbol{x}_t^i\right) \\
& =-A \frac{2\left(C_1\left(\boldsymbol{v}_{o l d}-\boldsymbol{v}_{\theta}\right)+C_2 \boldsymbol{\epsilon}\right) C_1}{2 C_2^2} \\
& \approx-\frac{C_1}{C_2} A \boldsymbol{\epsilon}
\end{aligned}$$

更通俗的来说，Flow GRPO的过程是找到了一个更好的噪声，然后让速度场往这个噪声的方向进行移动，噪声本身是没有意义的，只是这个噪声隐含了更优的速度方向，因此这个行为是有效的，这里最后之所以是约等于，是因为假设了v old 和 v 是一样的，官方工作在flow grpo guard去掉了这一项，这样子就没有这个约等于，在数学上更严谨

在Flow GRPO后续出了一个系列工作，Flow GRPO Guard，里面也有这个梯度推导，结论就是，Flow GRPO在让速度场往一个噪声的方向移动，这个其实是有点奇怪的，因此LLM的预训练和后训练行为时一致的，但是Flow GRPO中，预训练是优化速度场，但是RL却变成了优化噪声，这个行为似乎是不够直接的


### 1.1 Flow GRPO CPS

- https://zhuanlan.zhihu.com/p/1948388095151026330
- https://arxiv.org/abs/2509.05952

这是在这篇文章里面提到的，关于Flow GRPO加噪的一个bug的问题，导致生成的图片带有明显的噪声。Flow GRPO的随机性注入并非是最优解，首先我们来看FlowGRPO里的噪声注入方案

![](asset/Pasted%20image%2020260205155827.png)

相比于ODE的Flow，其增加了一个扰动项和一个修正项，然后我们可以分解ODE的Flow和SDE的Flow

![](asset/Pasted%20image%2020260205160450.png)

但是分析上图中的Flow-SDE的过程，我们发现，Flow-SDE的每一步的噪声项的系数，比理想的DDIM过程要高得多，尤其是当t很小的时候

![](asset/Pasted%20image%2020260205160827.png)


针对这个问题，解决方案也很简单，只要让噪声项的系数能够保持一致就行了，也就是减去的幅度和新加入的幅度相符合，也就是这篇论文的主题 Coefficients-Preserving Sample 系数保持采样

我们进一步思考，为什么Flow SDE会存在这样的问题，实际是在Flow GRPO在推导的过程中，直接套用了Score Match那套的内容，但是实际上在Score Matching中，这个推导过程不是恒等的，而是基于 $\Delta t \rightarrow 0$ 这个假设，这个假设在当时DDPM的1000步左右的采样过程中，确实是可以容许的，但是现在的采样模型，往往都是几步完成采样，这个推导就不再能够成立  

![](asset/Pasted%20image%2020260205161738.png)

### 2.2. TempFlow GRPO

对于Flow GRPO里的优化进行时间加权

![](asset/Pasted%20image%2020260205165106.png)


## 2. AWM

这篇论文正是发现了上述Flow GRPO的不一致问题，本质上来说，模型应该学习的是好的速度而不是好的噪声，学习速度比学习噪声更高效

- RL优化的是Reverse一步转换的高斯分布的对数似然
- 预训练优化的是score/flow match loss

以Flow GRPO为代表的方案，将带噪数据推断训练目标，会将这个不确定性直接传到梯度目标里

$$Loss_{AWM}=A^i||v_\theta(x_t^i)-v_{gt}||$$

而AWM提出了一种策略梯度一致的做法，损失函数依然是Flow matching那一套，只是做了一个加权，用奖励做一个加权，其v gt是将其采样到x0的时候，将x0再加上一点噪声来的，但是如何判断这个速度值不值得学习，就是依赖reward，高advantage的时候，这个gt的速度值得学习，但是在低advantage的时候，这个gt的速度就作为负面样本

![](asset/Pasted%20image%2020260126212831.png)

实际优化的时候还要加上PPO Clip

## 3. Diffusion NFT

Diffusion NFT 在附录中解释了 Flow GRPO 实际的优化方向，就是advantage-weighted noise，相比于AWM，NFT的做法是off-policy的

这篇工作中重新思考了将RL应用到Flow中的范式，单纯讲GRPO这种范式应用到Flow Model中的学习在理论上并非最优，能够达到几十倍的收敛加速

![](asset/Pasted%20image%2020260327153820.png)

对于GRPO和PPO这种基于Policy Gradient的算法来说，有一个共性的要求就是其模型的likelihoods必须是可以计算的，这个过程在自回归模型中成立，但是在diffusion models中并不直接，因为现在的Diffusion过程本质是一种连续过程，你要计算似然就必须计算积分，但是这个现阶段是计算不出来的，只能考虑近似

如果想要计算的话，只有两种方案

- 求解概率 ODE（计算成本很高）
- 通过 SDE 的变分下界（本身就是近似值，不精确）

先有的工作使用离散化的逆向过程，离散化之后每一步可以用高斯分布去建模，增加了高斯分布这个约束之后就可以进行精确的似然计算了，例如FlowGRPO

![](asset/Pasted%20image%2020260327154448.png)

Diffusion NFT 定义了如下过程，一个模型对于一个prompt集合，每个prompt会采样出k个结果，然后用reward model/function给这k个结果打分为r，我们会将reward归一化到0-1之间，这样子可以靠reward之间的对比隐式地构建advantage

然后就是比较精妙的一部分，Diffusion NFT要对生成的结果进行好样本和坏样本二元分组，那么如何进行分组呢，我们设定一个结果会有r的概率进行好样本组，有1-r的概率进行坏样本组，这时候的两个样本组可以构建为这样子的分布

![](asset/Pasted%20image%2020260327182605.png)

上述分布隐含着一定的优劣关系，也即 $\pi^+>\pi^{old}>\pi^-$ 如果我们直接使用 $\pi^+$ 来微调，我们就实际上完成了之前的Diffusion领域的RFT（拒绝采样微调），但是作为一个RL范式，自然是也要用上负样本

![](asset/Pasted%20image%2020260327183333.png)

在实际的操作中，我们进行采样，采样完成之后打分，打分完成之后进行加噪，从加噪图像进行预测，加噪图像到之前的真实图像，即为真实速度；预测出来的即为old policy的速度

实际上的损失函数设计为，可以理论推导得到，最优的改进方向是正负样本的速度差异

$$\mathcal{L}(\theta) = \mathbb{E}_{\boldsymbol{c},\, \pi^{\text{old}}(\boldsymbol{x}_0|\boldsymbol{c}),\, t} \left[ r\|\boldsymbol{v}_\theta^+(\boldsymbol{x}_t, \boldsymbol{c}, t) - \boldsymbol{v}\|_2^2 + (1-r)\|\boldsymbol{v}_\theta^-(\boldsymbol{x}_t, \boldsymbol{c}, t) - \boldsymbol{v}\|_2^2 \right]$$

$$v_{\theta}^{+} = (1 - \beta)v_{old} + \beta v_{\theta}, \\ \ \\ \ \\ \\ 
v_{\theta}^{-} = (1 + \beta)v_{old} - \beta v_{\theta},$$



![](asset/Pasted%20image%2020260327184458.png)

其Loss地梯度推导如上


Diffusion NFT 可以用任意黑盒的Solver进行训练，且CFG无关

 