
- DDPM
- EDM
- Consistency Model
- sCM
- rCM

## 1. Continuous-Time Consistency Model

这篇[工作](https://arxiv.org/pdf/2410.11081)是来自OpenAI的，连续时间的一致性模型，最开始的consistency model是离散的，存在离散导致的误差累积问题，并且引入了额外的超参数

以前的连续时间的方案都存在训练不稳定的问题，这篇论文整体审视了这个问题，将Diffusion Model和Consistency Model纳入到同一个理论框架中

CM的核心目标还是加速Diffusion model，但是相比于各种蒸馏范式而言，它本身是一类独立的模型，有独特的采样优势，但是CM同样存在效果不如multi-step的diffusion，以及训练也更不稳定的缺陷

### 1.1. TrigFlow

这是一套理论框架，也是这篇论文的第一个贡献，统一了EDM和Flow Matching，并且显著简化了Diffusion相关的理论公式

$$x_t =  \alpha_tx_0\ +\ \sigma_tz$$

这是Diffusion过程最核心的式子，很多变式都是在调整$\alpha_t$和$\sigma_t$这两个系数

**EDM**

在EDM中我们设置 $\alpha_t=1, \sigma_t=t$ 在这个实现中，原始图像的权重保持不变，只是随着时间增加，噪声的比例一直增加，其核心的训练目标如下

$$\mathbb{E}[w(t) \| f_\theta^{DM}(x_t, t) - x_0\|^2]$$

其中$w(t)$是一个权重系数，因为不同时间步数的去噪难度不一样，其中$f_\theta$是我们的神经网络，给定$x_t$我们希望其预测干净的$x_0$，这个方案当然在不同时间步下难度不同，因此需要加权来避免训练崩溃

其中EDM的网络结果形式化如下

**Flow Matching**

在Flow matching我们设置 $\alpha_t=1-t, \sigma_t=t$ ，其训练目标如下，是我们很熟悉的预测速度场的结果，公式里面表现为系数的梯度的计算

$$\mathbb{E}[w(t) \| f_\theta^{DM}(x_t, t) - (\alpha_t^{\prime}x_0+\sigma_t^{\prime}z)\|^2]$$

PF-ODE的概念，EDM和Flow Matching本质上都是使用了PF-ODE，这也是Song Yang提出的一套理论对于任何一个随机的 SDE 扩散过程，都存在一个完全确定的 ODE 过程，它们的边缘概率分布是完全一模一样的。

**Discrete Consistency Model**

我们还是从训练目标开始，首先我们来看Discrete-time CM的训练目标，宏观理解上，这个CM的目标应该是拉近两个时间步之间的预测结果，并且增加一个硬性约束也就是当t=0的时候，输出等于真实图像，通过这两个约束就可以优化一个一致性模型

$$\mathbb{E}_{\boldsymbol{x}_t, t} \left[ w(t) d\big( \boldsymbol{f}_\theta(\boldsymbol{x}_t, t), \boldsymbol{f}_{\theta^-}(\boldsymbol{x}_{t-\Delta t}, t - \Delta t) \big) \right],$$

这里的d是指度量函数，一般和上述几个目标一样，也是采用L2 Loss等

实验中发现，Discrete Consistency Model的训练对于$\Delta t$的选取非常敏感，并且存在误差累积的问题，因为本质上我们获取$\boldsymbol{x}_{t-\Delta t}$是需要从$x_t$求解ODE来的，但是ODE的求解却一般得用Euler等数值法来近似，这里就存在误差，本身Loss的计算需要近似会导致训练产生偏差

**Contiuous Consistency Model**

连续的一致性模型，很显然的，就是将上述目标公式中的$\Delta t$趋近于0，避免了需要进行ODE求解导致的误差累积问题，但是实践发现，这种cCM的训练相当地不稳定

CM的训练有两种方式，一种是从头开始训练，一种是蒸馏现有的Diffusion Model


### 1.2 Simplifying Continuous-Time Consistency Model

早起的CM仿照了EDM中的模型参数结构，其参数的构成较为复杂

文章提出TrigFlow，提出了一种的新的参数化方案，利用三角函数，该形式是Flow Matching的一种特例

> 这里的证明还需要看一下


![](asset/Pasted%20image%2020251225142734.png)


### 1.3. Stabilizing Continuous-Time Consistency Model



## 2. rCM

尝试Scale up sCM，面临的挑战

- JVP Computation Infra
- Evaluation Benchmark

文章提出了一个基于FlashAttention 2的JVP Kernel，能够使得sCM模型在10B+的参数上训练，并且在此过程中发现了之前sCM的缺陷，并且提出了rCM的架构来解决这个问题

### 2.1. Infra

> 看Kernel的实现

- Flash Attention
- FSDP
- Context Parallelism

### 2.2. Algorithm

![](asset/Pasted%20image%2020251225150116.png)

sCM蒸馏总是在一些典型提示上做得很好，但是在一些复杂提示上，指令遵循下降，并且无法通过简单的scale up来解决，可以发现这里是存在一个误差累积的问题，构成误差累积的因素有很多，比如CM的目标是在one step去拟合教师模型的ODE流，但是此时如果产生一点偏差，在后续步骤中该偏差会被放大

针对于此种现象，本文提出了一种叫做rCM的方式，做Score-Regularized sCM

![](asset/Pasted%20image%2020251225150455.png)





## Code

一致性模型的Loss计算

```python
        # 两个相邻时间步
        t1 = torch.rand(128) * 0.9 + 0.1
        t2 = torch.clamp(t1 - 0.1, min=0.01)
        
        # 加噪声
        noise = torch.randn_like(batch)
        x1 = batch + t1.view(-1, 1) * noise
        x2 = batch + t2.view(-1, 1) * noise
        
        # 一致性损失：两个预测应该接近
        loss = F.mse_loss(model(x1, t1), model(x2, t2))
```

一致性模型的训练内容如下，取两个时间步，并且让两个时间步预测出来的内容尽可能相似

```python
class MinimalConsistencyModel(nn.Module):
    """最小一致性模型 - 只包含核心逻辑"""
    
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64), nn.ReLU(),
            nn.Linear(64, dim)
        )
    
    def forward(self, x, t):
        """x: 数据, t: 时间步"""
        # 拼接输入
        inp = torch.cat([x, t.view(-1, 1)], dim=1)
        
        # Skip connection: 当t→0时输出→x
        return (1 - t.view(-1, 1)) * x + t.view(-1, 1) * self.net(inp)
```

并且确保在t=0的时候，输出为真实图片即可

