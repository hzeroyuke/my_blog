Huggingface类似于Github，是一个云托管的服务，但是其中大多是模型权重和数据集，相比于Github里的代码仓库，它们往往更加大型，也有各种优化网络传输和加载的方案

```
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir /mnt/nas_nfs/home/yuke/model/self_forcing/ --local-dir-use-symlinks False

huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 --local-dir /mnt/nas_nfs/home/yuke/model/LLaVA-Video-7B-Qwen2
```