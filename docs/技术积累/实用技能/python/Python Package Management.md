
## 1. Conda

conda 是一个非常常用的Python包管理工具，常用于科学计算场景，因此其不止做了Python的包管理，同时还处理cuda等环境的管理。

常用命令包含

- `conda create -n <env_name> python=3.10`
- `conda activate <env_name>` & `conda deactivate <env_name>`
- `conda remove -n <env_name> --all`


在国内的话，需要考虑将conda对应的源转换成清华源


conda 可以同时管理 cuda 和 python 的包，但是python的包在conda上面不全，尤其是torch的新版本，因此建议是先用conda install下载完成对应的cuda环境，随后的用pip install下载


## 2. uv

uv 是一个非常新的python包管理工具，工具链很现代化，并且速度快的惊人，相对于conda而言

常用命令包括

- `uv init`
- `uv venv`
- `uv add <package_name>`

这里有一个[视频链接](https://www.youtube.com/watch?v=AMdG7IjgSPM&t=6s) 来介绍uv

当我们使用uv的venv之后，在激活虚拟环境之后有一系列的选择

- pip install 用pip原生的解析器
- uv pip install 用uv的解析器，速度更快
- uv add 使用uv解析器材的同时，还可以维护uv的pyproject.tomlO

uv 常用的镜像源

```bash
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
```


