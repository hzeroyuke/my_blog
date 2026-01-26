
Flash Attention并不是一个简单的优化技术，而已经事实上成为了现阶段大模型部署的基石，其核心是实现了softmax的tile-based处理，实现了等价且高效的attention计算

Flash Attention的版本更迭，是对新版本N卡的不断微调

- FA1 通过Tiling和重计算，将ON2的HBM访问转换成SRAM操作
- FA2 通过Ampere架构特性，引入sequence维度并行，将工作负载划分为Warp级别
- FA3 通过Hopper架构特性，利用TMA引擎实现数据的异步搬运，在Warp Group层面实现了GEMM计算和Softmax的深度流水线overlap、


## FlexAttention

虽然FlashAttention效果优秀，但是其代码实现较为复杂，而当大家想要实用不同的attention变体的时候，往往需要重写对应的cuda kernel

应对这个问题，Pytorch团队提出了Flex Attention，提供了一套灵活的API，让大家实现不同的attention变体的时候，能够自动实现算子融合

![](asset/Pasted%20image%2020260123154958.png)


