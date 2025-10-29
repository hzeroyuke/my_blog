tmux是一个终端复用器，可以让user更加随心所欲地操作终端，以及托管任务，并且其保证了即便我们的连接断开，任务也不会中断，适合服务器形式开发

常见命令

Tmux的前缀键是Ctrl+b

-  tmux ls 列出所有的tmux对话
-  tmux new -s <session_name> 新建一个会话
-  tmux a -t <session_name> 连接一个已有的会话
-  左右拆分 Prefix + %
-  左右移动 Prefix + -> 
-  滚动模式 Prefix + \[ 退出 q
-  退出会话 Prefix + d


"""
huggingface-cli download Qwen/Qwen-Image \
  --local-dir /path/Qwen-Image \
  --local-dir-use-symlinks False

huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir /path/Flux_1_dev \
  --local-dir-use-symlinks False

/share/project/xiyan/huggingface/black-forest-labs/Flux.1-dev
"""