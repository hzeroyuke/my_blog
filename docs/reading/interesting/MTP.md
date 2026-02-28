
MTP 全称是 Multi-token prediction ，因为当前LLM的test-time scaling的推动，inference的压力变得越来越大，单靠硬件和系统加速存在上限，因此人们开始尝试在算法和架构层面提升推理的速度，最直观的想法就是MTP，让模型一次输出多个tokens

目前对于MTP的解决有一些方案比如Diffusion LLM和Speculative Decoding

## 1. Speculative Decoding

### 1.1. Eagle

[Eagle](https://github.com/SafeAILab/EAGLE)是一个加速大语言模型推理的开源的算法框架，属于Speculative Decoding的一种，相比于传统的LLM预测一个词，Eagle通过预测LLM倒数第二层的特征向量，通过一个回归头来提前预测后续多个token，然后用原始的LLM来并行验证这些token，留下正确的token，目前已经演化了三个版本

### 1.2 Dflash

[DFlash](https://z-lab.ai/projects/dflash/) 这是一个通过Diffusion+AR的范式，进行推测解码的方案，是用Diffusion Model进行高质量的草图生成，提高了了推测解码的上限，在指标上超越了Eagle

### Problem

SD 很重要的一个问题就是在大Batch Size的时候，效果不佳，这存在多种因素，本身SD的方案就是在计算打不满的情况下，让其通过多算一些东西来省一些forward。但是在大Batchsize下，计算逐渐也会被打满，因此SD的效果会逐渐变差。并且SD的做法也是在拉大一个batch size里的碎片化，因为有的request会猜对多个词语，有的request猜对的不多，从而有更大的对齐开销

现阶段SD的加速仅存在于小Batch Size下，可能真正的应用节点在于具身和边缘硬件场景









