
MTP 全称是 Multi-token prediction ，因为当前LLM的test-time scaling的推动，inference的压力变得越来越大，单靠硬件和系统加速存在上限，因此人们开始尝试在算法和架构层面提升推理的速度，最直观的想法就是MTP，让模型一次输出多个tokens

目前对于MTP的解决有一些方案比如Diffusion LLM和Speculative Decoding

## 1. Speculative Decoding

### 1.1. Eagle

[Eagle](https://github.com/SafeAILab/EAGLE)是一个加速大语言模型推理的开源的算法框架，属于Speculative Decoding的一种，相比于传统的LLM预测一个词，Eagle通过预测LLM倒数第二层的特征向量，通过一个回归头来提前预测后续多个token，然后用原始的LLM来并行验证这些token，留下正确的token，目前已经演化了三个版本



