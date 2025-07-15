# Diffusion and flow matching

## 2. Flow and Diffusion Models

Flow model 和 Diffusion model 分别在模拟ODE和SDE过程

### 2.1 Flow Models

我们从ODE过程开始，ODE的解是一个轨迹，其从时间 t 映射到空间$R^d$ ，任何一个样本都是这个空间中的一个点，不论是图像，视频或者是蛋白质结构

ODE的解是一个轨迹，下面这个式子中的X是一个function如下，t从[0,1]的范围中取，知道t你能够推出其在这个空间中的位置，这就是所谓的解是一个轨迹

$$
X:[0,1]\rightarrow R^d,\ \ \ t\rightarrow X_t
$$

对于每个时间和位置，ODE会定义一个vector field来告诉你在这个地方的速度向量u，u是一个function，如下

$$
u:R^d \times [0,1]\rightarrow R^d ,\ \ \ (x,t)\rightarrow u_t(x)
$$

在实践中，向量场u往往非常复杂，因此无法直接计算积分解决，一般是通过欧拉法，分步，假设单步之内的速度是不变的，进行计算解决