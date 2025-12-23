# Docker

docker是现代最流行的容器技术，是运维技术和分布式技术的基础之一

## 1. 基础概念

docker中有两种基本概念 镜像 image 和容器 container

容器是镜像的一个实例，例如你有一个MySQL 8.0的镜像，你同时运行了4个，就有了4个MySQL容器

> - docker images 查看你系统上的所有镜像
- docker ps 查看你系统上的运行中容器
- docker ps -a 查看你系统上所有的容器
> 

Image可以从仓库Hub获取，例如最经典的Docker Hub上有很多的Image可以供我们使用，我们也可以自己编写的Dockerfile，把我们本地的程序打包成一个Docker

> - Docker pull <image_name>
- Docker push <image_name>
> 

除了image和container，docker中还有一个重要概念，就是卷 volume，container本身是无状态的，当容器被删除之后，其中数据会全部丢失，如果用户希望数据独立与单个container持久化的话，就需要用到volume

## 2. 基础命令

**关于镜像**

- Docker pull <image_name>
- Docker images
- Docker rmi <image_name> 删除镜像
- Docker build -t <image_name> <dockerfile所在目录>
    - 例如 `docker build -t my-custom-app:1.0 .`
- Docker push <username/repo_name>
- docker image prune -f 该命令可以清理为None的镜像
    - 什么时候镜像会变成None tag
    - 镜像的tag是什么意思

**关于容器**

- Docker run <image_name> 基于某个镜像启动容器，这个命令常常会带有大量参数
    - -name <container_name> 自定义容器名
    - d 后台运行不打印日志
    - v 定义卷路径
    - e 设置环境变量
    - it 交互式运行
- Docker ps
- Docker start <container_name / id> 启动已经停止的容器
- Docker stop <container_name / id> 停止已经启动的容器
- Docker rm <container_name / id> 删除存在的容器
- Docker logs <container_name / id>
- Docker exec -it <容器ID或容器名> <命令>
    - docker exec -it mysql-test bash 常用的启动bash命令

**关于卷**

## 3. 网络