![](asset/Pasted%20image%2020251119153616.png)

例如 `__device__` 标记的函数，只能被其他在GPU上执行的代码所调用，也就是得在另外的 `__global__` 和 `__device__` 函数中调用

广播机制和strides机制

