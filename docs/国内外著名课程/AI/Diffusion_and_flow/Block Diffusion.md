
- Single Prompt Block Diffusion
	- Self Forcing, Self Forcing ++
	- Rolling Forcing
- Multiple Prompts
	- Longlive
	- Block Vid
- dLLM block diffusion
	- Fast-Dllm v2
- Efficient
	- Stream Diffusion v2
	- DeepForcing
- Context Forcing https://chenshuo20.github.io/Context_Forcing/
- Live
- Rolling Forcing https://arxiv.org/pdf/2509.25161
- casual forcing https://thu-ml.github.io/CausalForcing.github.io/
- [awesome video world models with ar diffusion](https://github.com/gracezhao1997/Awesome-Video-World-Models-with-AR-Diffusion)
- https://github.com/OpenDCAI/OpenWorldLib/blob/main/README_zh.md OpenWorldLib

## 0. reading

- https://arxiv.org/pdf/2602.24289
	- mode seek & mean seek，拆分Dit模块，一部分用真实的长视频做训练，一部分用类似Self-forcing的方案进行训练，长视频监督和短视频监督之间存在gap

## 1. Self-Forcing

前置工作有包括

- Teacher Forcing 最原始的ar diffusion，训练看gt frames，推理看自己生成的frames
- Diffusion Forcing 在Teacher Forcing的基础上，在训练中，在gt frames上增加一些噪声，进行训练，推理的时候也是看自己生成的frames，DF认为在训练时加入随机噪声的方案，可以提高模型应对错误累积的鲁棒性

[self-forcing](https://self-forcing.github.io/) 是 Block Diffusion 范式下有代表性的工作，主要是解决block diffusion之前在训练过程中的上下文为gt，而推理过程中的上下文为自己生成的内容，因此导致误差累积的问题

![](asset/Pasted%20image%2020260204195540.png)

非常简单的想法就是在训练的过程中，上下文就是自己生成的内容

相对应的，在Self-Forcing中，Loss的监督也从对于Next Frame的监督，改为了整段视频的监督。在这种自回归生成并且整体视频监督的范式下，训练的资源开销会变得非常大，因此Self Forcing中也提出了很多的Efficient的方案

- 阶段性的训练，用Self Forcing来微调之前的Forcing模型，而不是从头开始预训练
- Rolling KV-cache
- Gradient Truncation 梯度截断

![](asset/Pasted%20image%2020260310144138.png)

- Teacher Forcing
	- 上下文为干净的GT History
	- 每次训练的时候，所有帧预测的时间步是一致的
- Diffusion Forcing
	- 上下文为加噪扰动后的GT History
	- 每次训练的时候，所有帧预测的时间步是不同的

而在Self-Forcing中，所有History都用自己生成的Frames，这种训练显著地要比上述的方案要慢，因为其无法并行，上述方案对于一段分多个block的长视频，可以做到一次前向一次反向全部训完，而Self-Forcing的形式，理论上来说要经过block次前向和block反向才能训练完

为了应对这个问题，有一些比较常规的方案，比如使用few-step generation model，这个自然没有什么好说的，但是我们着重来看self-forcing中提出的比较特别的方案

**Gradient Truncation**

**Holistic Distribution Matching Loss**

Self-Forcing 的训练方案是依赖蒸馏的方案，因为其所有生成的内容都是自己产生的，所以可以对于整段long video施加DMD Loss

在DMD，SiD的形式下，这种方案是Data-Free的，但是需要一个teacher model，self-forcing里面使用的是14B的wan，实际上论文中也提供了一些其他的拟合分布的方案

- DMD
- SiD
- GAN

但是因为14B的wan本身是短视频生成，不可能对整个视频片段进行去噪，因此其会对整段long video做截断，截断为中间片段用teacher model和student model来做DMD

也就是说在整个训练阶段中，student model有两个角色，一个是casual gen model，生成长视频；一个是dmd student model，用于来继承teacher model的能力

**Rolling KV cache**

固定的KV cache长度，维护一个队列，先进先出

![](asset/Pasted%20image%2020260312162511.png)

> First frame 只编码了一张图片，并没有经过时序压缩，这个现象和3d VAE有关系
> 3d VAE基于因果卷机编码的话，对于首帧有另外的操作

对于KV cache而言，First Frames相比于其他的frames在统计意义上有很大的差异，因为在训练中First Frames总会出现在Context Length中，而在推理中一旦First Frame离开，就会导致很大的偏移。因此self-forcing的论文中，选择在训练中mask掉first frames，防止其依赖于first frames的情况


Self Forcing 代码解读

**推理流程**

对于Self-Attention的Transformer来说，只有Attention操作存在token间的交互，其他所有的操作都是token独立的

首先我们要知道现代video generation的dit的架构，wan的dit架构中兼具Self-Attention和Cross Attention，其中的Self Attention中应用3D的rope，因为有时间，长，宽三个维度，LLM是1维的Rope

Wan的Dit架构可以参考 Dit blog，Self Forcing对于其架构做了一定的修改，核心的内容在 `wan/modules/casual_model.py` 这个文件中，将wan model修改成了因果注意力的形式来适配 block diffusion，block内full attention，block之间casual attention。

随着视频的增长，其Rope编码会逐渐增长

```python
# casual rope apply
freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
```


**训练流程**

训练的流程有两部分：

- 第一部分是先采样一些ode轨迹，用这些轨迹，将Wan适配这种因果生成的形式
- 第二部分是DMD蒸馏，生成的部分加噪，并且用student model和teacher model对其进行去噪，从而进行学习

## 2. HiStream

使用Block Diffusion的范式，生成流式的，高效的，高分辨率1080P视频生成

其采用的加速方案是这样子的

- 空间压缩：前两步生成低分辨率的，后面几步生成高分辨率的，只做细节精修
- 时间压缩：生成的时候，KV-cache，只保留第一帧和最近的M-1帧，保留有上限的KV-cache
- 步数压缩：采用蒸馏的方案，将生成步数压缩到4步

相关工作整理有

- 高分辨率


高分辨率视觉生成

- 直接高分辨率训练 UltraPixel, Turbo2K
- Training-Free: I-MAX, CineScale, FreeScale
- 超分: Real-ESRGAN, FlashVSR

高效视觉生成

- 稀疏和滑动窗口，MOC, FasterCache, videoLCM
- 流式扩散: Self-Forcing, Magvi-1, Streaming Diffusion



## 3. Helios

来自 北大 + 字节 合作的项目，使用得是标准的teacher forcing，使用了diffusion forcing里的方案来抗扰动，其主要使用了对于history的压缩以及类似Var的不同尺寸的生成方案。这些方案虽然也能做到优秀的加速方案，但是相对于Sparse Attention和Quant Attention等可以即插即用（可能轻量微调）的方案来说，还是过于笨重，加速的方案和架构强绑定

**架构**

基于14B的wan，输入给Dit的结构包括一部分history，一部分噪声，根据history的不同，其在统一的架构中建模了t2v, i2v, v2v以及long video generation任务

![](asset/Pasted%20image%2020260307211156.png)

在计算 Self-attention 的时候，同时计算history和待生成部分，在计算 Cross-attention 也即文本注入的时候，mask掉history，只对待生成部分进行文本注入

**Generation Part**

- Relative Rope: 为了对抗long video生成的漂移现象，其使用Relative Rope，只对有限的History进行相对的时序建模，避免位置编码超出训练时候遇到的情况
- First Frame: 其在长视频生成的上下文中永远保留第一个frames
- Frame-Aware Corrupt: 类似于Diffusion Forcing，虽然其在训练的时候使用现实视频作为上下文来生成，但是仍然会对现实视频增加一些扰动，来提高模型的抗干扰能力

**Efficient Part**

- Multi-term Memory：金字塔形压缩模型
- Rescale Generation
- Step-Distillation

这三个套路叠加的结果，足够小的上下文，足够少的步数使其可以在单卡上运行14B的模型

## 4. Casual Forcing

这个主要是对Self-Forcing的改进，解决如何将双向注意力的video model蒸馏到ar的video model上，self-forcing主要分了两个部分

- ode轨迹蒸馏
- dmd精修

Casual Forcing，先用Teacher Forcing训练一个真的AR teacher model，再用这个model进行self-forcing的两阶段处理

## 5. Omni-Forcing

- https://omniforcing.com/

音视频同出的block diffusion，来自京东的工作，首个在audio-video diffusion上实现了因果蒸馏的工作


## 6. Linear Block Diffusion(SANA)

对于长视频生成来说，最大的问题就是Context难以提高，原因是KV cache太大了，而Linear Attention恰好能解决这种问题，在自回归模型下Linear attention和KV binding是等价的，可以构建出恒定内存开销的Block Diffusion工作

这个方向上最具代表性的工作就是SANA

- [blog](https://hanlab.mit.edu/blog/infinite-context-length-with-global-but-constant-attention-memory)
- [codebase](https://github.com/NVlabs/Sana)
- [paper](https://arxiv.org/pdf/2509.24695)

2B model，先训练双向模型，然后Self-Forcing蒸馏成线性的block diffusion model

![](asset/Pasted%20image%2020260330200621.png)



## New Distillation Method

在block diffusion范式下，除了一些infra的进展之外，算法上的主要想法在于如何有效蒸馏双向模型的能力上，因为长视频数据的稀缺性和block diffusion对于实时性的要求（few-step disitllation is necessary），这种范式在显得尤其得重要

- https://franklinz233.github.io/projects/astrolabe/
- https://arxiv.org/pdf/2603.09488


## Error Bank

是指将模型自身生成中出现的误差形式记录起来，在进行微调的过程注入这些特定的误差的形式，然后让模型能处理对应的误差情形，尽可能避免流式生成的误差累积

## Block Diffusion RL

- https://arxiv.org/pdf/2602.09022

针对Block Diffusion范式的长视频生成的RL存在一系列的困难，比如Rollout太慢，reward设计困难并且稀疏

这篇论文做了一系列的尝试包括

- Clip Level Rollout：对于一个需要优化的片段，共享前面所有的前缀，单独针对这个片段进行rollout，比如n个片段，n-1个片段都是一样的，就第n个片段反复rollout
- Reward Designs：一方面用3D model来估计相机轨迹，一方面用HPS v3来评判视觉质量
	- 3D model主要是用于相机轨迹的判断，在交互式生成的World Model中，比如实时交互的游戏生成中，主要的操作就是镜头移动和旋转，将输入的镜头移动和3D模型判断出来的镜头移动进行对比，就可以打reward
- Optimization algorithm：基于Diffusion NFT进行改进

![](asset/Pasted%20image%2020260303202203.png)

