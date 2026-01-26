
- Single Prompt Block Diffusion
	- Self Forcing, Self Forcing ++
	- Rolling Forcing
- Multiple Prompts
	- Longlive
	- Block Vid
- dLLM block diffusion
	- Fast-Dllm v2
- Efficient
	- Stream Diffusion v2
	- DeepForcing

## 1. Self-Forcing


## HiStream

使用Block Diffusion的范式，生成流式的，高效的，高分辨率1080P视频生成

其采用的加速方案是这样子的

- 空间压缩：前两步生成低分辨率的，后面几步生成高分辨率的，只做细节精修
- 时间压缩：生成的时候，KV-cache，只保留第一帧和最近的M-1帧，保留有上限的KV-cache
- 步数压缩：采用蒸馏的方案，将生成步数压缩到4步

相关工作整理有

- 高分辨率


高分辨率视觉生成

- 直接高分辨率训练 UltraPixel, Turbo2K
- Training-Free: I-MAX, CineScale, FreeScale
- 超分: Real-ESRGAN, FlashVSR

高效视觉生成

- 稀疏和滑动窗口，MOC, FasterCache, videoLCM
- 流式扩散: Self-Forcing, Magvi-1, Streaming Diffusion



