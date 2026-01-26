
为了更好地提高生成模型的指令遵循能力，更进一步是为了构建生成理解统一的模型，其中一种解决方案是构建Semantic Rich的latent space，诞生了包括RAE等工作在内的进展

## 1. Vision Encoder

包括JEPA，Siglip，DINO v2等工作，代表的经典vision encoder工作，以及其性能分析

- https://arxiv.org/pdf/2512.19693

这篇论文分析了语义编码器（semantic encoder）和像素编码器（pixel encoder）的频谱差异，一个关注高频细节一个关注低频细节

- semantic encoder capture low-frequency components
- pixel encoder capture hight-frequency components


![](asset/Pasted%20image%2020260116153421.png)

现代的感知模型主要侧重于语义编码，而生成模型侧重于像素细节编码，并且通过一些实验来验证这个内容，比如对于text-image对齐任务，消除掉一些细节也没有影响，但是一旦过滤掉低频的色块，对齐任务的准确度马上就下降了

![](asset/Pasted%20image%2020260116154913.png)

并且本篇论文提出了一种相应的Unified AutoEncoder的训练方案，同时优化Semantic Loss（依靠一个教师模型）和Pixel Reconstruction Loss（直接重建），其是靠将DINO的初始化整个模型，然后将DINO的输出给按频率拆分，低频率部分采用一个重建的Loss

![](asset/Pasted%20image%2020260116155441.png)

## 2. Semantic Latent Space

### 2.1. RAE

论文指出，当前的 VAE 存在只注重低维度的重建表征，而忽视了其他信息，最终限制了现有的生成模型的质量，在RAE中其直接采用MLLM中的vision encoder作为编码器

这些模型不仅能够提供高纬表征和丰富的语义信息，并且基于 Transformer 架构

因为这是一个比较直接的想法，之前也有很多工作做过，但是很少有工作真的能在这个表征空间里把DIT做work，因为这些encoder并没有被训练掌握low-level的details，这篇论文也分析了这个地方会遇到的苦难和解决方案

![](asset/Pasted%20image%2020251228154706.png)

实验表明，在有效训练了对应的解码器的情况下，RAE的重建效果不仅更好而且效率更高，论文中的主体实验配置为

- encoder: DINO v2
- dit: LightingDit
- patch size: 1

研究者很快发现之前使用的Dit模型比如Dit-S和Dit-XL等模型在RAE的条件下表现不佳，并推测是以下原因

- Dit的架构问题
- 噪声调度的问题
- 小噪声的问题，之前的VAE对于噪声并不敏感

在实验中，研究者也发现在RAE架构中训练Dit的一些技巧

- Dit的架构：模型宽度问题，Dit的宽度必须要超过RAE输出token的纬度，Diffusion的数学流形有关
- 噪声调度的问题：迫使模型处理更强的噪声，RAE的输入是高纬度的，使用传统加噪方案使得模型的预测过于简单，因此将纬度也考虑到噪声调度中，使得模型在高纬中也能获得有效的去噪能力
- 小噪声的问题：RAE的空间对于噪声敏感，我们就在解码的时候使用noise-augmented decoding，在decoder的训练的时候就引入噪声，这也是为了应对RAE的离散空间限制，因为VAE的潜空间遵循连续分布

整个训练的流程是，先训练decoder，完成高质量的重建，并且使用到noist-augmented decoding，随后再训练Dit




### 2.2. scale rae

https://rae-dit.github.io/scale-rae/

scale rae 


### 2.3. VTP
