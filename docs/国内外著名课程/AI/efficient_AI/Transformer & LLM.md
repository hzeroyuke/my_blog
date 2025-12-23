课程的第二部分主要讲特定领域的优化技术，第一块是对LLM系列的优化技术

前面一大部分是LLM的基础知识，在很多地方已经记录过了，不再做赘述

## 1. KV-Cache

KV cache 是这类自回归模型中最重要的一类优化技术之一，通过对于context的KV进行存储，使得在进行Attention计算的时候，只需要计算最新进来的token和之前的context的attention即可

而在LLM追求Long Context的时候，KV cache的所占的空间也会变得非常的大

![](asset/Pasted%20image%2020251215162302.png)

一旦用户变多（Batch size变大）KV cache 占据的存储空间甚至很容易超过模型本身的大小

这里我们就引入了MHA MQA GQA等方案，GQA是这几个里面最优的解决方案，这个和早期对于Conv block的优化非常相似

![](asset/Pasted%20image%2020251215162915.png)

这些方案是在Attention Head上做处理，通过减少Attention Head的数量，来减少KV cache的存储，标准的MHA的我们有h个Attention Head，也就有h个QKV，MQA做的比较极端，我们只有一个KV权重，但是有h个Q权重，因为Q的计算是逃不掉的，但是这种做法过于极端导致性能较差，GQA就相对做得更好，我们让几个Q权重共享一个KV权重，这样子可以做到一个比较好的权衡

```python
class GroupedQueryAttention(nn.Module):
    """GQA：heads 分组，每组共享 1 组 KV"""
    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads // 8  # 默认 8 组
        self.d_k = d_model // num_heads
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_k)  # G 组 key
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_k)  # G 组 value
        self.W_o = nn.Linear(d_model, d_model)
```

显然这些优化都不是Training-Free的优化，需要在训练中就采用对应的变种

## 2. Linear Units

对于Linear Units的更新方案有一种是SwiGLU

![](asset/Pasted%20image%2020251215164346.png)


## 3. LLM Deployment

这是关于LLM部署相关技术的介绍，主要是三部分

- Quantization
- Sparsity
- Systems（like FlashAttention）

### 3.1. Quantization

前面的课程中已经介绍过了一些量化技术，但是它们大多在LLM scaling的时候会失效，并不适用于大型模型

![](asset/Pasted%20image%2020251215171730.png)

在大规模的训练中，会出现一些激活值的异常值，有些激活值会异常地大，在做常规地量化的时候，这个激活值会导致大量的小激活值被截断为0，训练就会崩溃

事实上在模型的训练中，参数的量化是比较容易，但是激活值的量化会相对困难

![](asset/Pasted%20image%2020251215172201.png)

针对此种现象我们可以采用一种对应的量化方案，就是将激活值中的较大的部分缩小，将对应的缩放因子转换到它们要计算的weight中，使得对应的权重扩大，就是将100x1变成10x10，在保证结果不变的情况下尽可能消除异常值

![](asset/Pasted%20image%2020251215190734.png)

将量化的压力在激活值和参数中取得一个平衡时一个比较好的方案，这种称为Smooth Quantization的方案，在具体的Transformer实现中如下

![](asset/Pasted%20image%2020251215192218.png)




