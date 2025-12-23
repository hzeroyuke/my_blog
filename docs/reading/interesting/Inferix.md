
论文链接在[这里](https://arxiv.org/abs/2511.20714) ，这是一篇旨在构建现代世界模型的推理引擎的文章，对标vllm，sglang等对于语言模型的优化。 inferix主要是对于长时间，高质量，可交互的video generation的推理优化

引入了序列并行，低比特量化，稀疏Attention，分布式并行等加速手段

我通过claude code辅助进行了这个codebase的理解

```bash
Inferix/
├── inferix/                    # 核心框架代码
│   ├── core/                   # 核心模块
│   ├── distributed/            # 分布式推理支持
│   ├── kvcache_manager/        # KV缓存管理
│   ├── models/                 # 模型实现
│   ├── pipeline/               # 推理管线
│   └── profiling/              # 性能分析
├── example/                    # 示例和配置
│   ├── self_forcing/           # Self-Forcing模型示例
│   ├── causvid/                # CausVid模型示例
│   ├── magi/                   # MAGI模型示例
│   ├── quantization/           # 量化示例
│   ├── profiling/              # 性能分析示例
│   └── rtmp_streaming/         # 视频流示例
├── LV-Bench/                   # 长视频基准测试
└── tests/                      # 测试代码
```

## 0. 理解推理框架

在阅读推理框架之前我们要了解一个问题，模型参数和推理框架的代码是如何互动的

我们有两个部分，一个是模型的权重，例如 `model.safetensors` 这类，当我们使用vllm这类的框架时，会进行如下的操作

```python
from vllm import LLM

model = LLM(model="meta-llama/Llama-2-7b-hf")
```

而在其内部，则会将这些权重进行封装，人们在发布模型的时候会标记好参数的权重的含义

我们简化vllm内部的实现，大概是这个样子的

```python
# 简化示例：vLLM 内部的权重加载逻辑
def load_weights(model, hf_model_path):
    # 1. 从 HuggingFace 加载权重
    state_dict = torch.load(f"{hf_model_path}/pytorch_model.bin")
    
    # 2. 权重名称映射
    # HF: model.layers.0.self_attn.q_proj.weight
    # vLLM: model.layers.0.attention.qkv_proj.weight (合并的QKV)
    
    for name, param in model.named_parameters():
        if "qkv_proj" in name:
            # vLLM 可能会合并 Q、K、V 权重以优化性能
            layer_idx = extract_layer_idx(name)
            q_weight = state_dict[f"layers.{layer_idx}.self_attn.q_proj.weight"]
            k_weight = state_dict[f"layers.{layer_idx}.self_attn.k_proj.weight"]
            v_weight = state_dict[f"layers.{layer_idx}.self_attn.v_proj.weight"]
            
            # 合并权重
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            param.data.copy_(qkv_weight)
```

我们会从权重中获取到 q k v 的权重，随后我们要进行flash attention还是分布式操作，就随着推理框架的代码走
## 1. pipeline

该项目的核心接口是一个pipeline，其支持以下三种实现

  **Self-Forcing** 其流程是单一提示词生成，支持T2V和I2V

  - 基础模型: Wan2.1-T2V-1.3B
  - 核心技术: DMD (Distribution Matching Distillation)
  - 架构特点:
    - 使用因果扩散模型 (CausalDiffusionInferencePipeline)
    - 支持 EMA (Exponential Moving Average) 权重
    - 基于 Wan Base 扩展

  **CausVid** 其流程是连续提示词生成，以及交互式的提示输入

  - 基础模型: Wan2.1-T2V-1.3B
  - 核心技术: 因果视频生成（Causal Video）
  - 架构特点:
    - 使用因果推理管线 (CausalInferencePipeline)
    - 支持 rollout 机制生成更长视频
    - 基于 Wan Base 扩展

  **MAGI** 支持V2V

  - 基础模型: 独立的多模态架构
  - 核心技术: 完全独立的 DiT (Diffusion Transformer) 架构
  - 架构特点:
    - 完全独立的模型实现（DiT + VAE + T5）
    - 不依赖 Wan Base
    - 支持多种规模（4.5B 和 24B）

我们来细看一下CausVid的代码，也就是基于wan模型的交互性生成的pipeline，这个部分有两个代码

- `CausalInferencePipeline.py` 这个是底层的推理，输入的噪声和提示，给出生成的video
- `pipeline.py` 这个是上层的接口，其编排整个生成流程，并保存结果

**CausalInferencePipeline**

这个代码的核心部分如下

```python
class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            wan_base_model_path,
            device,
            enable_kv_offload,
            parallel_config: Optional[ParallelConfig] = None
    ):
        super().__init__()

        self.parallel_config = parallel_config

        # Step 1: Initialize all models
        # Get model path from configuration (following README path convention)
        model_path = wan_base_model_path if wan_base_model_path is not None else 'weights/Wan2.1-T2V-1.3B'
        self.generator = WanDiffusionWrapper(
            model_path=model_path,
            **getattr(args, "model_kwargs", {}), enable_kv_offload=enable_kv_offload, parallel_config=parallel_config).to(device)
        self.text_encoder = WanTextEncoder(model_path=model_path)
        self.vae = WanVAEWrapper(model_path=model_path)

        if dist.is_initialized():
            dist.barrier()

        # Step 2: Initialize all causal hyperparmeters
        ...

	# Other Method
	...
	
	    def inference(self, noise: torch.Tensor, text_prompts: List[str], start_latents: Optional[torch.Tensor], return_latents: bool = True, kv_cache_manager: Optional[KVCacheManager] = None, kv_cache_requests: Optional[List] = None) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        
        ...
        
        if return_latents:
            return video, output
        else:
            return video
	
```

主体部分是初始化和推理，初始化部分主要是这三个模型，一个wan模型的基座，一个文本编码器，一个vae

```python
        self.generator = WanDiffusionWrapper(...)
        self.text_encoder = WanTextEncoder(model_path=model_path)
        self.vae = WanVAEWrapper(model_path=model_path)
```

这些部分的实现都放在 model/ 文件夹下

主要来看推理部分，推理的主干部分是两个循环，因为其设计是block diffusion，每个block块内根据时间步进行diffusion，然后外侧的循环是AR的过程

```python
	for block_index in range(num_blocks):
		noisy_input = noise[:, block_index *
							self.num_frame_per_block:(block_index + 1) * self.num_frame_per_block]
		... # 判定是否要填充start_latents
		for index, current_timestep in enumerate(self.denoising_step_list):
			# 迭代去噪
			...
		# 更新kv-cache
		...
```

这里还有一个`start_latents`的参数，这个参数是用于image2video和video2video的任务，这个参数表示的生成开头的前几帧的latents，比如我们在image2video的任务中，我们会先计算好这个image的latents，然后直接将其纳入到生成的pipeline中，也就是生成的前几帧并不需要去噪，而是直接将这些images的latents填充进入pipeline中，并且更新kv-cache

```python
if start_latents is not None and block_index < num_input_blocks:
	# 设置 timestep 为0，标记这为去噪完成的帧
	timestep = torch.ones(
		[batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * 0
	# 提取当前块的参考帧
	current_ref_latents = start_latents[:, block_index * self.num_frame_per_block:(
		block_index + 1) * self.num_frame_per_block]
	# 直接复制到输出
	output[:, block_index * self.num_frame_per_block:(
		block_index + 1) * self.num_frame_per_block] = current_ref_latents
	# 填充kv-cache
	self.generator(
		noisy_image_or_video=current_ref_latents,
		conditional_dict=conditional_dict,
		timestep=timestep * 0,
		current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
		current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length,
		kv_start=block_index * self.num_frame_per_block * self.per_rank_frame_seq_length,
		kv_end=(block_index + 1) * self.num_frame_per_block * self.per_rank_frame_seq_length,
		kv_cache_manager=kv_cache_manager,
		kv_cache_requests=kv_cache_requests,
	)
	continue # 填充完成之后进入下一步循环
```

过了这个start_latents的判断之后，就是进入了去噪的过程，去噪的过程非常直接，就是传统的ddpm的过程

```python
for index, current_timestep in enumerate(self.denoising_step_list):
	# set current timestep
	timestep = torch.ones(
		[batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * current_timestep
	
	if index < len(self.denoising_step_list) - 1:
		
		# 去噪
		denoised_pred = self.generator(
			noisy_image_or_video=noisy_input,
			conditional_dict=conditional_dict,
			timestep=timestep,
			...
		)
		... # 中间处理
		
		# 加噪
		noisy_input_flat = self.scheduler.add_noise(
			denoised_pred.flatten(0, 1),
			torch.randn_like(denoised_pred.flatten(0, 1)),
			timestep_tensor
		)
		# Ensure result is not None before view operation
		if noisy_input_flat is not None:
			noisy_input = noisy_input_flat.view(denoised_pred.shape)
		else:
			raise RuntimeError("scheduler.add_noise returned None")
	else:
		# 最后一步，获得最终输出
		denoised_pred = self.generator(
			noisy_image_or_video=noisy_input,
			conditional_dict=conditional_dict,
			timestep=timestep,
			...
		)
```

在完成一个block的diffusion之后，在最后一步再调用一遍generator，更新kv-cache

**pipeline**

我们看完了用于实际生成的pipeline，我们来看上层的业务接口

```bash
用户调用
   ↓
CausVidPipeline (业务层)
   ↓
CausalInferencePipeline (底层去噪)
   ↓
Generator (DiT/Transformer 模型)
```

pipeline中主要是处理如何分配每一次生成和上一次生成的start_latents的关系

```bash
假设 num_overlap_frames = 3

第一轮生成：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Frame:  0  1  2  3  4  5  ... 17 18 19 20
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                              └────┬────┘
                           准备作为下一轮的
                           start_latents
                           (Frame 18, 19, 20)

保存到 all_video: Frame 0-17 (去掉重叠的 18-20)

第二轮生成：
start_latents = [Frame 18, 19, 20] (来自上一轮)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Frame:  18 19 20 21 22 23 ... 35 36 37 38
        └──固定──┘ └────新生成────┘ └─重叠─┘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

保存到 all_video: Frame 21-35 (去掉 18-20 和 36-38)
```


## 2. models

该项目中的models被划分成了两层关系，一层是比较底层可复用的实现，比如attention；一层是高层的，和pipeline对接的部分，比如说causvid部分

## 3. kv-cache

KV cache的管控是项目中最重要的部分之一

## 4. profiler