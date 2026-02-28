## Test Time Training

- https://arxiv.org/pdf/2512.23675

TTT 是一个Continual Learning的方案，早期的TTT还没有应用在 LLM 上，在没有scaling的情况下，域外数据对于模型是一个难题，因此人们当时采用了TTT，来处理图像分类等任务

**监督信号**

测试时的数据没有标签，因此人们要手动构造一些可供训练的数据出来，在最初的TTT论文上，人们将图片旋转特定角度，用模型来预测图片旋转的角度，来进行训练，后续发展出来MAE的方案，也即遮住图像的部分patch，然后预测遮住的部分，更加有效且通用

后续进入LLM领域之后，监督信号自然变成了Next token prediction

**参数更新**

对于TTT而言，更新全量参数往往不是一个现实的做法，尤其是进入LLM领域，人们往往考虑只更新

- Layer Norm
- Lora
- 内嵌一个几层的小模型

**LLM TTT**

LLM在早期也是引入了一个小型的神经网络，模拟RNN中的隐状态，每个时间步用自监督损失对W做梯度更新。但是后续人们为了支持更加实时地输出，努力降低计算开销，将神经网络替换成一个线性模型，也就是没有激活层的MLP，相当于乘了一个矩阵。

TTT本质是要做上下文学习，相比于静态的context输入，TTT可以改变部分参数来针对现在的上下文进行学习，人们开始考虑用TTT的线性模型来替代Self-Attention（么有替代qkv矩阵，只是替代了最后一步softmax操作）。本质上来说，Self-Attention机制不承担世界记忆，而是用作上下文内关联性的学习，和TTT要做的事情一致，但是TTT的优势在于其可以根据不同的上下文修改参数

这时候的TTT开始变为KV binding，仅增加一个线性模型的TTT就是KV binding

KV binding这个东西一开始和TTT并没有联系，是从KV cache那边发展出来的，KV cache 本身随着token数量的增加线性增长，人们开始考虑各种方案来降低这个开销，KV binding就是其中一个方案，维护一个固定大小的矩阵，将每个KV对以加法的形式存入这个矩阵

```python
for t in range(T):
	# 写入：将 (k_t, v_t) 绑定进记忆矩阵
	# k_t: (B, D) -> (B, D, 1)
	# v_t: (B, D) -> (B, 1, D)
	# 外积: (B, D, D)
	M = M + torch.bmm(k[:, t, :].unsqueeze(2),
					  v[:, t, :].unsqueeze(1))

	# 读出：用 q_t 从记忆矩阵中检索
	# q_t: (B, D) -> (B, 1, D)
	# o_t: (B, 1, D) -> (B, D)
	o_t = torch.bmm(q[:, t, :].unsqueeze(1), M).squeeze(1)
	outputs.append(o_t)

return torch.stack(outputs, dim=1)  # (B, T, D)
```

想要从这个矩阵中提取出KV对，就用对应的Q去乘以这个矩阵，从形式上来说就是对所有历史信息加权求和，对应的数学形式是

$$o_t​=q_t​M=q_t​\sum_{s≤t}​k_sT^​v_s​=\sum_{s≤t}​(q_t​k_s^T​)v_s​$$

我们来对比标准的softmax attention

$$o_t^{\text{softmax}} = \sum_{s \leq t} {softmax}(q_t k_s^T) v_s$$

区别恰好就在于Softmax操作带来的归一化，并且另一方面，KV binding计算形式和Linear Attention是一致的，Linear Attention的核心操作正是去掉softmax，交换qkv的计算顺序来获得线性的复杂度，但是由此我们也知道了KV binding自然也引入了Linear Attention低秩等缺陷

现在我们已经建立了KV binding和Linear Attention的关联，接下来我们来看KV binding和线性层TTT的关联，线性层TTT用一个矩阵替代Self-Attention层，然后根据其梯度下降，我们可以推导出和KV binding和Linear Attention一致的表达

人们用Key-Value对来训练这个线性层，本质上这个线性层用于提供查找的操作

这篇[论文](https://arxiv.org/pdf/2602.21204)揭示了Test Time Training with KV binding和Linear Attention的内在联系

对于Linear Attention，我们知道它是通过交换Attention操作的计算顺序来达到线性的计算复杂度，并且后续还衍生出了一系列的SSM模型，Mamba等SSM模型和Linear Attention的是同一类模型的不同视角

DeltaNet被证明为是单层线性层+零初始化的TTT特例


