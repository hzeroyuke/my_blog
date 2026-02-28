这里主要讲述torch, cuda, cuda toolkit之间的依赖关系，以及如何用conda来管理这些东西

conda 能够管理的是用户级别的cuda包，不包括系统级别的，系统级别的cuda是nvidia-smi展现出来的cuda版本，conda能够管理的包括

- cuda-toolkit cuda的编译工具
- cuDNN 为代表的一系列cuda库

当我们平时下载torch的时候，其会自动为我们下载与其相匹配的cuda依赖和编译器，torch和我们当前的conda环境里的cuDNN等库相互匹配

conda和pip和互相使用，在conda启动环境之后，建议先用conda把核心环境torch等相关内容装好，再考虑用pip安装纯python的包

比如最典型的flash-attn的场景，这个包在默认安装条件下经常出问题

## 1. conda cuda

- `conda install cudatoolkit == <version> -c nvidia`
- `conda install pytorch ...`
- `conda install cuda-nvcc`

以前的管理方案基本如上，现在用conda管理不再主流，并且pytorch本身也不再提供给conda的包，现在的版本就是用uv来管理torch的环境

现在的torch包中往往自带CUDA运行时相关库，因此现在安装torch容易了许多，但是与此同时，安装其他需要nvcc等编译工具的python包的时候就会出现问题，比如flash attention