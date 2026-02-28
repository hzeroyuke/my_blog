
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

## 1. Self-Forcing

前置工作有包括

- Teacher Forcing 最原始的ar diffusion，训练看gt frames，推理看自己生成的frames
- Diffusion Forcing 在Teacher Forcing的基础上，在训练中，在gt frames上增加一些噪声，进行训练，推理的时候也是看自己生成的frames，DF认为在训练时加入随机噪声的方案，可以提高模型应对错误累积的鲁棒性

[self-forcing](https://self-forcing.github.io/) 是 Block Diffusion 范式下比较新的工作，主要是解决block diffusion之前在训练过程中的上下文为gt，而推理过程中的上下文为自己生成的内容，因此导致误差累积的问题

![](asset/Pasted%20image%2020260204195540.png)

非常简单的想法就是在训练的过程中，上下文就是自己生成的内容

相对应的，在Self-Forcing中，Loss的监督也从对于Next Frame的监督，改为了整段视频的监督。在这种自回归生成并且整体视频监督的范式下，训练的资源开销会变得非常大，因此Self Forcing中也提出了很多的Efficient的方案

- 阶段性的训练，用Self Forcing来微调之前的Forcing模型，而不是从头开始预训练
- Rolling KV-cache
- Gradient Truncation 梯度截断

Self Forcing 代码解读

**推理流程**

对于Self-Attention的Transformer来说，只有Attention操作存在token间的交互，其他所有的操作都是token独立的

首先我们要知道现代video generation的dit的架构，wan的dit架构中兼具Self-Attention和Cross Attention，其中的Self Attention中应用3D的rope，因为有时间，长，宽三个维度，LLM是1维的Rope





**训练流程**

## HiStream

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



