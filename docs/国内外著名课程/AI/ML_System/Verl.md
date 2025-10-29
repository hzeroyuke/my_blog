
RL的流程实际是一个数据流过程，相对于SFT来说更加复杂

- Multiple models: actor, critic, reference, reward
- Muliiple stages: generating, preparing experiences, training

![](Pasted%20image%2020251016212208.png)

在训练的过程中首先要确定各个Model放在哪些机器上，然后根据不同逻辑阶段来进行相应的执行

Verl这类框架的目标就是在用户定义了DataFlow之后，其可以帮助完成底层的资源分配以及高效的调度，提高并发

## 1. Verl Code Walkthrough

![](Pasted%20image%2020251016212732.png)

首先是进入main函数中的TaskRunner.run是整个verl的入口

第一步其实是资源的分配，会先初始化资源池子，然后规定好哪几块机器是做什么用途。Verl中默认的实现是每个workload都使用全局的资源池子，也即每个Workload占据全部GPU，导致了每个workload的过程其实都是串行的

第二步是Trainer的初始化，初始化Ray里面的worker

![](Pasted%20image%2020251016213105.png)

我们会把不同的workload纳入到同一个进程中（定义workerGroup），减少内存的碎片（GPU上的进程其实都在独自维护memory pool，因此每个进程都有没用到的但是占用的内存，进程越多这类碎片就越多）

第三步，训练的循环

![](Pasted%20image%2020251016213455.png)

采用Single Controller这个范式来最大化灵活性，在常规流程中，大家都使用全局资源池子的串行条件下，核心的PPO训练代码就上面这几行

- 过一个epoch
- 在epoch中迭代data batch
- Generating
- Preparing Experiences
- Training

内部其实涉及了很多通信的内容，虽然通信是很昂贵的，但是内部这边我们只用到了prompt和responses的通信，相对于参数的通信是开销较小的

第四步，Worker Procedure，Multi-Controller

![](Pasted%20image%2020251017120938.png)

现在我们开始涉及Worker内部，其内部我们用到SPMD等概念，其编程更复杂，但是其效率会更高。SPMD中进程代码是相同的，但是其处理的数据是不同的

## 2. SPMD in Verl

首先也是资源管理的部分

![](Pasted%20image%2020251017121316.png)

会在GPU上设置环境变量，并放置于ray的worker class中

其次是如何定义SPMD行为，也就是如何把对应的程序加载到worker上

![](Pasted%20image%2020251017121640.png)

register的定义，这会给修饰的函数添加一系列的属性

![](Pasted%20image%2020251017121913.png)



![](Pasted%20image%2020251017122303.png)

这个函数是核心模板，应对了每个worker里面实际执行的内容，之前的一些模块在这里组合成一个真正的function


## Rollout的一致性问题

![](Pasted%20image%2020251019234244.png)

关于Tokenizer的一些特性，在带有和环境交互的RL的过程中，是需要有一些注意事项的，那就是对于一个tokenizer，其encoder再decoder的结果是一致的，但是decoder再encoder的结果是不一定一致的

实际上由多个token_id可以被解码成同一个文本，但是一个文本只被编码成一个token_id

如上图，这个流程如果把轨迹直接encoder，其结果是可能不一致的，因为其先把response解码的结果用作toolcall，后续在编码回去，可能导致错误

