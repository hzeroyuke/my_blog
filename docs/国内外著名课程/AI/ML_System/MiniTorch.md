这是LLM System课程的大作业，这课程没有开源视频资源，但是开源它的大作业框架，写这个大作业，大概需要两周的时间

实现的代码后续会放到一个github链接中

## 1. 框架介绍

### 1.1. Tensor

**1.1. Tensor data**

这部分是tensor实际的存储结构，也是最底层的存储，在计算机中所有的底层存储都是一维数组，tensor data覆盖了一维数组到一个任意维度的矩阵的映射

```python
class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage.astype(datatype)
        else:
            self._storage = array(storage, dtype=datatype)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size
```

其中需要注意shape和strides这两个概念，shape是暴露在用户端的，也即这个tensor的形状，例如`[3, 4]`这样子的张量

而strides是和shape相辅相成的概念，其对应着shape，告诉底层访问一维数组的时候，你要跳过多少个元素，最经典的使用如下，当我们要访问某个具体的元素的index时候，要通过这个strides来具体访问到哪个元素，index是矩阵的视角，position是底层存储的视角

```python
def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """

    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride
    return position
```

上述都是概念上的python实现，实际上的pytorch中，这些都应该在cuda上实现

**1.2. Tensor ops**

```python
class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False
```

定义了基础的tensor操作

- map 一元操作，例如exp(x)，对于x中的所有元素进行一个exp
- zip 二元操作，例如x+y
- reduce 规约操作，例如sum(x)，将所有的元素以某种形式进行处理
- 矩阵乘法：特殊的优化操作

类似的实现了一个SimpleOps的类，是用python进行逻辑上的实现

```python
class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret
```

实际上我们应该进行cuda版本的实现，继承这个特殊类即可

**1.3. Tensor function**

这个类主要处理前向和反向的部分，在torch中，所有的操作都是针对tensor，这就衍生出了两个部分

1. 操作本身，例如加减乘除，exp，求和，比较
2. 操作到矩阵的扩张，例如map，zip，reduce，这些内容被称为并行操作原语

上述的Tensor ops是针对的第2部分，而这一部分是处理的第一部分，也即操作本身

```python
# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)
```

我们可以来看一个例子

```python
class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor: 
        out = a.f.tanh_map(a)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        out = ctx.saved_values[0]
        return grad_output * (-(out ** 2) + 1)
```

对于torch里的所有操作，我们都需要定义好他们的前向和反向传播，才能在后续做自动微分

这部分里同时还处理了自动微分的逻辑，以及tensor构造的逻辑

**1.4. tensor**

这是整个tensor系统中真正的接口，其中的backend是tensor ops的实现接口

```python
class Tensor:
    """
    Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend
```

- requires_grad_(): 启用/禁用梯度追踪
- backward(): 触发反向传播
- accumulate_derivative(): 累积梯度
- chain_rule(): 应用链式法则

### 1.2. cuda backend

在实验3开始我们正式引入cuda的后端实现，我们在`cuda_kernel_ops.py`中引入了cuda的接口

```python
def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # in_size
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorMap.restype = None
            
            # assert out.size == a.size, f"zip {out.size}, {a.size}"

            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret
```

这是在python中调用cuda后端的实现，这里给了两种实现

- cuda_kernel_ops.py 这个是上述我们自己写的kernel的调用方案
- cuda_ops.py 这个是用numba这个python的包里面的一些cuda kernel简单的组合，实现相同功能的实现



### 1.3. Module

这个模块中实现了各种神经网络组件，比如Linear，Dropout，Embedding，Norm，Attention等等，以下是一个例子

```python
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        if not self.training or self.p_dropout == 0.0:
            return x

        rate = self.p_dropout
        r = tensor_from_numpy(np.random.rand(*x.shape), backend=x.backend)
        drop = rate < r
        return x * drop / (1 - rate)
```

### 1.4. Scalar

标量计算，这部分是一个标量系统的实现，主要是实现一些标量操作的前向和反向操作

## 2. 实验内容

### 2.1 Cuda Programming

实验的第一部分是cuda programming，在给出python接口的情况，实现几个cuda函数

```cpp
__device__ int index_to_position(const int* index, const int* strides, int num_dims) {
    int position = 0;
    for (int i = 0; i < num_dims; ++i) {
        position += index[i] * strides[i];
    }
    return position;
}

__device__ void to_index(int ordinal, const int* shape, int* out_index, int num_dims) {
    int cur_ord = ordinal;
    for (int i = num_dims - 1; i >= 0; --i) {
        int sh = shape[i];
        out_index[i] = cur_ord % sh;
        cur_ord /= sh;
    }
}

__device__ void broadcast_index(const int* big_index, const int* big_shape, const int* shape, int* out_index, int num_dims_big, int num_dims) {
    for (int i = 0; i < num_dims; ++i) {
        if (shape[i] > 1) {
            out_index[i] = big_index[i + (num_dims_big - num_dims)];
        } else {
            out_index[i] = 0;
        }
    }
}
```

这三个函数，是框架中给出的内容，也是张量操作中很重要的三个操作

- index_to_position 将逻辑上的张量位置转换成一维存储中的position
- to_index 将ordinal代表的一维存储的position转换成具体的逻辑上的index
- broadcast_index 是张量操作中的广播机制的具体实现

这个广播机制比较复杂，我们来解释一下，这个函数是输入一个大数组的big_index，某个元素的位置，我们要知道它和哪个小数组的位置上的元素进行计算，也即out_index

![](asset/Pasted%20image%2020251126162514.png)

广播的机制是这样的，从维度的末端开始遍历，当前维度大于1，则该维度不能被广播，该维度中需要找到和大张量完全一致的坐标，如果没有找到，那么广播会失败；如果当前维度为1，则该维度可以被广播，广播的方式就是不论什么时候都访问第0个元素，视为第0个元素扩张到了所有地方

借助这三个函数，我们可以来实现reduce，zip，map等内容

比较简单的比如 map 操作，是对于张量中的每个元素都进行一样的操作，比如张量加上一个标量

```cpp
__global__ void mapKernel(
    float* out, 
    int* out_shape, 
    int* out_strides, 
    int out_size, 
    float* in_storage, 
    int* in_shape, 
    int* in_strides,
    int shape_size,
    int fn_id
) {
    int out_index[MAX_DIMS];
    int in_index[MAX_DIMS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < out_size){
      to_index(idx, out_shape, out_index, shape_size);  // 将数值上的索引，转移到具体的形状上的位置

      broadcast_index(out_index, out_shape, in_shape, in_index, shape_size, shape_size); // 将大的index广播到小的index上

      int in_pos = index_to_position(in_index, in_strides, shape_size);
      int out_pos = index_to_position(out_index, out_strides, shape_size);

      out[out_pos] = fn(fn_id, in_storage[in_pos]);
    }
}
```

其中比较复杂的是矩阵乘法这个操作

### 2.2. MiniTorch

第二部分的实验是让我熟悉MiniTorch这个框架，在这个实验中的MiniTorch是完全以Python为后端实现的简易版本

第一块的任务是让我们实现AutoDiff

```python
def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    from typing import Set
    order: List[Variable] = []
    visited: Set[int] = set()

    def dfs(var: Variable):
        if var.unique_id in visited or var.is_constant():
            return
        visited.add(var.unique_id)
        for parent in var.parents:
            dfs(parent)
        order.append(var)

    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph.
    """
    topo_order = list(topological_sort(variable))
    derivatives = {variable.unique_id: deriv}

    for var in topo_order:
        if var.unique_id not in derivatives:
            continue
            
        d_var = derivatives[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(d_var)
        elif not var.is_constant():
            for parent, d_parent in var.chain_rule(d_var):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = d_parent
                else:
                    derivatives[parent.unique_id] += d_parent
```

这部分和前面的框架介绍一致，这里只是在做梯度的传递，而没有做实际的梯度计算操作

这里的backpropagate函数在后续发现了一些问题

```python
    # derivatives = {variable.unique_id: deriv}

    derivatives = {v.unique_id:0 for v in topo_order}
    derivatives[variable.unique_id] = deriv
```

梯度的初始化需要这样做，才能通过测试，否则梯度会和测试结果很相似，但是在最后几位不一致

这里有个很细节的bug，我们在以下代码中，我们直接让`derivatives[parent.unique_id] = d_parent` ，这会导致这个字典上的数据直接引用了这个d_parent，一旦计算图中有多个地方引用了这个d_parent，后续有其中一个地方修改一个d_parent，其他所有地方都会相应的修改，但是其实这时候其他地方不应该进行变化

```python
            for parent, d_parent in var.chain_rule(d_var):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = d_parent
                else:
                    derivatives[parent.unique_id] += d_parent
```

了解了这个原因的时候，出了一开始就初始好所有东西的方案以外，还可以用以下这种方式进行拷贝，而不是引用

```python
            for parent, d_parent in var.chain_rule(d_var):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = d_parent + 0.0
                else:
                    derivatives[parent.unique_id] += d_parent
```

### 2.3. Transformer

第三部分的实验需要我们用MiniTorch框架实现Transformer，此时的MiniTorch补齐了基于cuda的后端，将前面两个实验组合起来

这一部分其实很像CS336的lab1，实现Transformer的各个组件，但是最大的区别在于我们是在自己的MiniTorch上实现，从Cuda后端到Tensor实现，以及自动微分，都在这个框架中，没有黑盒，这是一个很棒的事情

我们依次来看几个实现

```python
class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        self.weights = Parameter(ones_tensor_from_numpy((dim, ), backend=backend))
        self.bias = Parameter(zeros_tensor_from_numpy((dim, ), backend=backend))

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        mean = x.mean(dim=1).view(batch, 1)
        variance = x.var(dim=1).view(batch, 1)
        x_normalized = (x - mean) / ((variance + self.eps) ** 0.5) # eps 用于防止除0溢出的问题
        out = x_normalized * self.weights.value.view(1, dim) + self.bias.value.view(1, dim)
        return out
```

Layer Norm的实现，先对第1个维度计算均值和方差，随后进行norm，最后乘以一个可学习的参数

```python
class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        self.weights = Parameter(rand((num_embeddings, embedding_dim), backend=backend))
    
    def forward(self, x: Tensor):
        bs, seq_len = x.shape
        V = self.num_embeddings
        
        # 1. 将输入的词汇索引张量 x 转换成 One-Hot 张量 H (B, L, V)
        H = one_hot(x, V) 
        
        # 2. 扁平化 One-Hot 张量，使其变为 (B*L, V) 形状
        #    这样在 MatMul 内部，它和 (V, D) 的权重都会被视为 3D 张量 (1, M, K) 进行批处理
        H_flat = H.view(bs * seq_len, V) 
        
        # 3. 执行矩阵乘法 H_flat @ W。W 形状为 (V, D)。
        #    结果 output_flat 形状为 (B*L, D)
        output_flat = H_flat.__matmul__(self.weights.value)
        
        # 4. 重塑回期望的 (B, L, D) 形状
        return output_flat.view(bs, seq_len, self.embedding_dim)
```

Embedding 层，将一个词表索引（one-hot 向量）转换成一个低维度的向量表示


### 2.4. Transformer Accelerate by Cuda



