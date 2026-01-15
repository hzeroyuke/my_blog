## 1. HPT

HPT（Hybrid Post-Training）的原论文

- https://arxiv.org/abs/2509.04419

![](Pasted%20image%2020251031140200.png)

通过计算大多常见的Post-Training的算法的梯度估计形式，可以统一成一个形式

![](Pasted%20image%2020251031140315.png)

这个Unified Policy Gradient Estimator可以拆分成以下四个组件，这个行为类似于RL中的GAE

**1. Stabilization Mask**

这部分指的是从PPO开始的一系列Clip方案，裁剪掉一些不稳定的梯度，后续的GSPO的sequence-level clip也被划分成这个部分，就是指训练中的梯度裁剪技巧

**2. Reference Policy Denominator**

这部分是指Reference Model，也即在RL中的采样Model，对于SFT中即为当前Model，对于一些离线强化学习算法中，由于不存在Reference Model，一般就直接将其设置为1

**3. Advantage Estimate**

在传统RL中，优势估计就是这个action带来的当前收益和预期的未来收益之和，但是对于LLM而言，优势估计往往是sequence-level的而不是token-level的，对于整个训练过程，核心过程就是最大化Advantage

**4. Likelihood Gradient**

梯度的计算，这部分是指如何讲上述内容合并起来进行梯度的计算，但是这块基本在所有范式中都是固定的


### 从Loss的视角到梯度的视角

## 2. Unified View of PostTraining

https://www.notion.so/From-Recipes-to-Regimes-A-Target-Operator-Diagnostics-Framework-for-LLM-Post-Training-2da6113bc4c58008b5a2c7af3e5815b2

跳脱出SFT和RL，进入完整的PostTraining的视角，人们发现SFT和RL之间的区别，相比于RL内部的Trick反而更小，显然整个PostTraining系统需要一套更宏观的解读

## 3. Entropy-Adaptive SFT

SFT的技巧对于RL有指导作用，反过来也是一样的，SFT总是迫使模型去服从输入数据的分布，这一定程度上和模型原始的能力会相互冲突，因此我们可以采用RL中的技巧，用Entropy based的方案，对于特定的一些token进行门控

![](asset/Pasted%20image%2020260112202845.png)












