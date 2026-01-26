Triton是OpenAI开发的基于Python来编写Cuda Kernel的库，本文主要包括了

## Triton Compile

```python
@triton.jit
def my_kernel(...):
    ...
my_kernel[grid](x_ptr, y_ptr, n)
```

一般来说我们使用triton编写内核的操作如上文

![](asset/Pasted%20image%2020260123160501.png)

Python DSL层级，只是做一系列的表述，并不实际执行kernel，在Compile层级将抽象出来的Python表示转换成可执行的代码形式，随后在Runtime层级跑起来