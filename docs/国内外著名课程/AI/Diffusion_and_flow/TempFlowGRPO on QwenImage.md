
## 目录
- [概述](#概述)
- [第一次修改：Pipeline 适配](#第一次修改pipeline-适配)
- [第二次修改：训练脚本适配](#第二次修改训练脚本适配)
- [QwenImage 训练流程详解](#qwenimage-训练流程详解)
- [关键差异总结](#关键差异总结)


配置启动

```bash
conda create --prefix /share/project/wanli/env/verl-latest python=3.10

conda activate /share/project/wanli/env/flow_grpo
export PYTHONPATH=$PYTHONPATH:/share/project/wanli/wanli_yk/TempFlow-GRPO
export WANDB_API_KEY=API_KEY
```


- 改用了torch.run
- 新增了fsdp_utils.py
- dgx文件里新增了一个Qwen-image的配置
- sd3_sde_with_logprob.py 这个文件改为了最新版本Flow-GRPO的
---

## 概述

### 背景
项目中存在两种训练范式：
1. **常规训练**：在 SDE 窗口内采样，只保存窗口内的中间状态
2. **Per-step 训练**：对每个 denoising 步骤进行分支采样，生成完整图像并分别计算奖励

### 目标
将 Flux 模型的 per-step 训练方法适配到 QwenImage 模型，包括：
- Pipeline 层面的改造
- 训练脚本的完整适配

---

## 第一次修改：Pipeline 适配

### 源文件与目标文件
- **参考文件**：`flux_pipeline_with_logprob_perstep.py`
- **基础文件**：`qwenimage_pipeline_with_logprob.py`
- **目标文件**：`qwenimage_pipeline_with_logprob_perstep.py`

### 核心修改思路

#### 1. 移除窗口机制
**常规版本**有以下参数：
```python
noise_level: float = 0.7,
process_index: int = 0,
sde_window_size: int = 0,
sde_window_range: tuple[int, int] = (0, 5),
```

**Per-step 版本**替换为：
```python
determistic: bool = True,
```

**原因**：Per-step 版本需要遍历所有时间步，不需要窗口限制。

#### 2. 双轨处理机制
在每个时间步同时计算 ODE 和 SDE 两条路径：

```python
# ODE 分支 - 用于主流程推进
ode_latents, _, _, _ = sde_step_with_logprob(
    self.scheduler,
    noise_pred.float(),
    t.unsqueeze(0).repeat(latents.shape[0]),
    latents.float(),
    determistic=True,
)

# SDE 分支 - 用于生成训练样本
sde_latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
    self.scheduler,
    noise_pred.float(),
    t.unsqueeze(0),
    latents.float(),
    determistic=False,
)

# 主流程使用 ODE
latents = ode_latents
```

#### 3. Per-step 图像生成
从每个 SDE 采样点开始，使用 ODE 完成剩余步骤：

```python
# SDE sample branch, then ode process
inner_latents = sde_latents.clone()
for inner_i, inner_t in enumerate(timesteps[i+1:]):
    inner_timestep = inner_t.expand(inner_latents.shape[0]).to(inner_latents.dtype)

    # 执行完整的 transformer 前向传播（包含 CFG）
    inner_noise_pred = self.transformer(
        hidden_states=torch.cat([inner_latents, inner_latents], dim=0),
        timestep=torch.cat([inner_timestep, inner_timestep], dim=0) / 1000,
        guidance=guidance,
        encoder_hidden_states_mask=torch.cat([
            prompt_embeds_mask.repeat_interleave(inner_latents.shape[0]//prompt_embeds.shape[0], dim=0),
            negative_prompt_embeds_mask.repeat_interleave(inner_latents.shape[0]//prompt_embeds.shape[0], dim=0)
        ], dim=0),
        encoder_hidden_states=torch.cat([
            prompt_embeds.repeat_interleave(inner_latents.shape[0]//prompt_embeds.shape[0], dim=0),
            negative_prompt_embeds.repeat_interleave(inner_latents.shape[0]//prompt_embeds.shape[0], dim=0)
        ], dim=0),
        img_shapes=img_shapes*2,
        txt_seq_lens=txt_seq_lens+negative_txt_seq_lens,
    )[0]

    # 应用 CFG
    inner_noise_pred, inner_neg_noise_pred = inner_noise_pred.chunk(2, dim=0)
    inner_comb_pred = inner_neg_noise_pred + true_cfg_scale * (inner_noise_pred - inner_neg_noise_pred)
    inner_cond_norm = torch.norm(inner_noise_pred, dim=-1, keepdim=True)
    inner_noise_norm = torch.norm(inner_comb_pred, dim=-1, keepdim=True)
    inner_noise_pred = inner_comb_pred * (inner_cond_norm / inner_noise_norm)

    # ODE step
    inner_latents, _, _, _ = sde_step_with_logprob(
        self.scheduler,
        inner_noise_pred.float(),
        inner_t.unsqueeze(0),
        inner_latents.float(),
        determistic=True,
    )

# 解码为图像
inner_latents = self._unpack_latents(inner_latents, height, width, self.vae_scale_factor)
inner_latents = inner_latents.to(self.vae.dtype)
# QwenImage 特有的 latent 归一化
latents_mean = (
    torch.tensor(self.vae.config.latents_mean)
    .view(1, self.vae.config.z_dim, 1, 1, 1)
    .to(inner_latents.device, inner_latents.dtype)
)
latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
    inner_latents.device, inner_latents.dtype
)
inner_latents = inner_latents / latents_std + latents_mean
inner_image = self.vae.decode(inner_latents, return_dict=False)[0][:, :, 0]
inner_image = self.image_processor.postprocess(inner_image, output_type=output_type)
all_image.append(inner_image)
```

#### 4. QwenImage 特有适配

**a) Transformer 调用差异**

Flux 使用：
```python
noise_pred = self.transformer(
    hidden_states=latents,
    timestep=timestep / 1000,
    guidance=guidance,
    pooled_projections=pooled_prompt_embeds,
    encoder_hidden_states=prompt_embeds,
    txt_ids=text_ids,
    img_ids=latent_image_ids,
    joint_attention_kwargs=self.joint_attention_kwargs,
    return_dict=False,
)[0]
```

QwenImage 使用：
```python
noise_pred = self.transformer(
    hidden_states=torch.cat([latents, latents], dim=0),  # CFG 需要拼接
    timestep=torch.cat([timestep, timestep], dim=0) / 1000,
    guidance=guidance,
    encoder_hidden_states_mask=torch.cat([prompt_embeds_mask, negative_prompt_embeds_mask], dim=0),
    encoder_hidden_states=torch.cat([prompt_embeds], negative_prompt_embeds], dim=0),
    img_shapes=img_shapes*2,
    txt_seq_lens=txt_seq_lens+negative_txt_seq_lens,
)[0]
```

**关键差异**：
- QwenImage 需要手动实现 CFG（拼接正负样本）
- 需要提供 `encoder_hidden_states_mask`、`img_shapes`、`txt_seq_lens`
- 后续需要手动应用 CFG 缩放和归一化

**b) CFG 处理**

QwenImage 需要手动计算 CFG：
```python
noise_pred, neg_noise_pred = noise_pred.chunk(2, dim=0)
comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

# 归一化处理（保持条件预测的范数）
cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
noise_pred = comb_pred * (cond_norm / noise_norm)
```

**c) VAE 解码差异**

Flux：
```python
latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
image = self.vae.decode(latents, return_dict=False)[0]
```

QwenImage：
```python
latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
latents = latents.to(self.vae.dtype)
latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
latents = latents / latents_std + latents_mean
image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]  # 注意最后的 [:, :, 0]
```

#### 5. 返回值变化

**常规版本**：
```python
return {
    "images": image,
    "all_latents": all_latents,
    "all_log_probs": all_log_probs,
    "all_timesteps": all_timesteps,
    "prompt_embeds": prompt_embeds,
    "negative_prompt_embeds": negative_prompt_embeds,
    "prompt_embeds_mask": prompt_embeds_mask,
    "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
}
```

**Per-step 版本**：
```python
return all_image, all_latents, all_sde_latents, all_log_probs, prompt_embeds, negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask
```

返回 `all_image` 列表，包含每个时间步生成的完整图像。

---

## 第二次修改：训练脚本适配

### 源文件与目标文件
- **参考文件**：`train_flux_pr.py`
- **基础文件**：`train_qwenimage.py`
- **目标文件**：`train_qwenimage_pr.py`

### 核心修改思路

#### 1. 导入模块更新

添加 per-step 版本的函数：
```python
from flow_grpo.diffusers_patch.qwenimage_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.qwenimage_pipeline_with_logprob_perstep import pipeline_with_logprob as pipeline_with_logprob_perstep
from flow_grpo.diffusers_patch.sd3_sde_with_logprob_perstep import sde_step_with_logprob as sde_step_with_logprob_perstep
```

#### 2. compute_log_prob 函数修改

使用 per-step 版本的 sde_step：
```python
def compute_log_prob(transformer, pipeline, sample, j, config, rank):
    # ... transformer 前向传播和 CFG 处理 ...

    # 使用 perstep 版本，添加 determistic 参数
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob_perstep(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        determistic=False,  # 训练时使用随机采样
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t
```

#### 3. eval 函数修改

评估时使用确定性采样：
```python
def eval(pipeline, test_dataloader, config, rank, local_rank, world_size, device, global_step, reward_fn, executor, autocast, ema, transformer_trainable_parameters):
    # ...
    collected_data = pipeline_with_logprob(
        pipeline,
        prompts,
        negative_prompt=[" "]*len(prompts),
        num_inference_steps=config.sample.eval_num_steps,
        true_cfg_scale=config.sample.guidance_scale,
        output_type="pt",
        height=config.resolution,
        width=config.resolution,
        determistic=True,  # 评估时使用 ODE
    )
    # ...
```

#### 4. 采样阶段核心改动

**a) 使用 per-step pipeline**：
```python
with autocast():
    with torch.no_grad():
        images, latents, sde_latents, log_probs, prompt_embeds, negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask = pipeline_with_logprob_perstep(
            pipeline,
            prompts,
            negative_prompt=[" "]*len(prompts),
            num_inference_steps=config.sample.num_steps,
            true_cfg_scale=config.sample.guidance_scale,
            output_type="pil",
            height=config.resolution,
            width=config.resolution,
            generator=generator,
            determistic=False,  # 采样时使用 SDE
        )
```

**b) 分支探索机制**：
```python
exploration_k = 6  # 每个 prompt 生成 6 个分支

# 扩展 latents
latents = torch.stack(latents, dim=1).repeat_interleave(exploration_k, dim=0)
# (batch_size * exploration_k, num_steps + 1, seq_len, channels)

sde_latents = torch.stack(sde_latents, dim=1)
# (batch_size, num_steps, seq_len, channels)

log_probs = torch.stack(log_probs, dim=1)
# (batch_size, num_steps)

# 扩展 prompts
prompts = [prompts[i] for i in torch.arange(len(prompts)).repeat_interleave(exploration_k)]
prompt_metadata = [prompt_metadata[i] for i in torch.arange(len(prompt_metadata)).repeat_interleave(exploration_k)]
```

**c) 重建 timesteps**：
```python
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift

sigmas = np.linspace(1.0, 1 / config.sample.num_steps, config.sample.num_steps)
image_seq_len = latents.shape[2]
mu = calculate_shift(
    image_seq_len,
    pipeline.scheduler.config.get("base_image_seq_len", 256),
    pipeline.scheduler.config.get("max_image_seq_len", 4096),
    pipeline.scheduler.config.get("base_shift", 0.5),
    pipeline.scheduler.config.get("max_shift", 1.15),
)
timesteps, _ = retrieve_timesteps(
    pipeline.scheduler,
    config.sample.num_steps,
    device,
    sigmas=sigmas,
    mu=mu,
)
timesteps = timesteps.repeat(config.sample.train_batch_size*exploration_k, 1)
```

#### 5. Rewards 处理改动

**a) Per-step 奖励计算**：
```python
# images 是一个列表，每个元素对应一个时间步的所有图像
rewards = []
for im in images:  # 遍历每个时间步
    rewards.append(executor.submit(reward_fn, im, prompts, prompt_metadata, only_strict=True))
```

**b) 奖励聚合**：
```python
tmp_rewards = {}
for item in sample['rewards']:  # 遍历每个时间步的 reward future
    rewards, reward_metadata = item.result()
    for key, value in rewards.items():
        if key not in tmp_rewards.keys():
            tmp_rewards[key] = []

        if config.prompt_fn == "geneval" or list(config.reward_fn.keys()) == ["hpsv3"]:
            tmp_rewards[key].append(torch.as_tensor(value, device=device).float())
        else:
            if isinstance(value, list):
                value = torch.stack(value, dim=0)
            tmp_rewards[key].append(value)

# 按 timestep 维度堆叠
for key, value in tmp_rewards.items():
    tmp_rewards[key] = torch.stack(tmp_rewards[key], dim=1)
    # 结果形状：(batch_size, num_steps)
```

#### 6. 样本数据结构

```python
samples.append({
    "prompt_ids": prompt_ids.repeat_interleave(exploration_k, dim=0),
    "prompt_embeds": prompt_embeds.repeat_interleave(exploration_k, dim=0),
    "prompt_embeds_mask": prompt_embeds_mask.repeat_interleave(exploration_k, dim=0),
    "negative_prompt_embeds": negative_prompt_embeds.repeat_interleave(exploration_k, dim=0),
    "negative_prompt_embeds_mask": negative_prompt_embeds_mask.repeat_interleave(exploration_k, dim=0),
    "timesteps": timesteps,
    "latents": latents[:, :-1],  # 去掉最后一步（已经是完整图像）
    "next_latents": sde_latents,  # 使用 SDE 采样的下一步 latents
    "log_probs": log_probs,
    "rewards": rewards,  # 列表形式，每个时间步一个 future
})
```

#### 7. Rewards 维度处理

**常规版本**需要 repeat：
```python
samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
```

**Per-step 版本**已经是 per-timestep 的，不需要处理：
```python
samples["rewards"]["avg"] = samples["rewards"]["avg"]
# 已经是 (batch_size, num_steps) 形状
```

#### 8. Loss 计算改动

**常规版本**：
```python
unclipped_loss = -advantages * ratio
clipped_loss = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
```

**Per-step 版本**添加 `1.73 * std_dev_t` 系数：
```python
unclipped_loss = -1.73 * std_dev_t * advantages * ratio
clipped_loss = -1.73 * std_dev_t * advantages * torch.clamp(
    ratio,
    1.0 - config.train.clip_range,
    1.0 + config.train.clip_range,
)
policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
```

**系数含义**：
- `1.73 ≈ √3`，来自理论推导
- `std_dev_t`：当前时间步的标准差，用于加权不同时间步的损失

#### 9. 训练循环修改

**添加数据随机打乱**：
```python
for inner_epoch in range(config.train.num_inner_epochs):
    # 在每个 inner epoch 开始时打乱样本
    perm = torch.randperm(total_batch_size, device=device)
    samples_shuffled = {k: v[perm] for k, v in samples.items()}

    # 重新分批
    samples_batched = {
        k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
        for k, v in samples_shuffled.items()
    }
    # ...
```

---

## QwenImage 训练流程详解

### 常规版本流程 (train_qwenimage.py)

```
1. 初始化
   ├── 加载模型 (transformer, vae, text_encoder)
   ├── 应用 LoRA（可选）
   ├── FSDP 包装
   └── 设置优化器

2. 每个 Epoch
   ├── 评估（可选）
   │   └── 使用 ODE 采样生成图像
   │
   ├── 采样阶段
   │   ├── 从数据集获取 prompts
   │   ├── 使用 pipeline_with_logprob 生成图像
   │   │   ├── 随机选择 SDE 窗口 [start, end)
   │   │   ├── 在窗口外使用 ODE (noise_level=0)
   │   │   ├── 在窗口内使用 SDE (noise_level=0.7)
   │   │   └── 保存窗口内的 latents 和 log_probs
   │   ├── 异步计算 rewards
   │   └── 收集样本数据
   │
   ├── 等待 rewards 计算完成
   │   └── Padding prompt embeddings 到统一长度
   │
   ├── 处理 rewards
   │   ├── 跨进程 gather
   │   ├── 计算 advantages (标准化)
   │   └── 分配回各进程
   │
   └── 训练阶段
       ├── 对每个 batch
       │   └── 对窗口内每个 timestep
       │       ├── 重新计算 log_prob（当前策略）
       │       ├── 计算 ratio = exp(new_log_prob - old_log_prob)
       │       ├── PPO-style clipped loss
       │       ├── KL 散度（可选）
       │       ├── 反向传播
       │       └── 梯度累积 + 优化
       └── 更新 EMA（可选）

3. 保存检查点
```

**关键特点**：
- **窗口采样**：只在随机选择的连续窗口内使用 SDE
- **单一轨迹**：每个 prompt 只生成一个完整图像
- **窗口内训练**：只使用窗口内的中间步骤进行训练

### Per-step 版本流程 (train_qwenimage_pr.py)

```
1. 初始化
   └── （与常规版本相同）

2. 每个 Epoch
   ├── 评估（可选）
   │   └── 使用 determistic=True (ODE) 采样
   │
   ├── 采样阶段
   │   ├── 从数据集获取 prompts
   │   ├── 使用 pipeline_with_logprob_perstep 生成
   │   │   ├── 对每个时间步 t：
   │   │   │   ├── 计算 ODE 分支 → 主流程推进
   │   │   │   ├── 计算 SDE 分支 → 生成训练样本
   │   │   │   ├── 从 SDE 点开始用 ODE 完成剩余步骤
   │   │   │   └── 解码生成完整图像
   │   │   └── 返回：all_images, all_latents, all_sde_latents, all_log_probs
   │   │
   │   ├── 分支探索
   │   │   ├── 设置 exploration_k = 6
   │   │   ├── 将 latents 扩展 k 倍
   │   │   └── 扩展 prompts 和 metadata
   │   │
   │   ├── 为每个时间步的图像计算 rewards
   │   │   └── rewards 是列表：[reward_step0, reward_step1, ...]
   │   │
   │   └── 收集样本数据
   │       └── next_latents 使用 sde_latents（不是 latents[1:]）
   │
   ├── 等待 rewards 计算完成
   │   └── 将每个时间步的 rewards 沿时间维度堆叠
   │       └── 形状：(batch_size, num_steps)
   │
   ├── 处理 rewards
   │   ├── 跨进程 gather
   │   ├── 计算 advantages (已经是 per-timestep 的)
   │   └── 分配回各进程
   │
   └── 训练阶段
       ├── 随机打乱样本
       ├── 对每个 batch
       │   └── 对每个 timestep
       │       ├── 重新计算 log_prob
       │       ├── 计算 ratio
       │       ├── 使用加权的 PPO loss
       │       │   └── loss = 1.73 * std_dev_t * clipped_loss
       │       ├── KL 散度（可选）
       │       ├── 反向传播
       │       └── 梯度累积 + 优化
       └── 更新 EMA（可选）

3. 保存检查点
```

**关键特点**：
- **全程双轨**：每步都计算 ODE 和 SDE
- **分支探索**：每个 SDE 点生成完整图像，exploration_k=6
- **Per-step rewards**：每个时间步独立的奖励信号
- **加权损失**：考虑 std_dev_t 的时间步权重
- **全步训练**：在所有时间步上训练（不限于窗口）

---

## 关键差异总结

### Pipeline 层面

| 特性 | 常规版本 | Per-step 版本 |
|-----|---------|--------------|
| **窗口机制** | 随机选择 SDE 窗口 | 无窗口，遍历所有步 |
| **采样方式** | 窗口外 ODE + 窗口内 SDE | 每步 ODE+SDE 双轨 |
| **图像生成** | 只生成最终图像 | 每步生成完整图像 |
| **返回数据** | 字典（窗口内数据） | 元组（所有步数据） |
| **log_probs** | 窗口内的 log_probs | 所有步的 log_probs |

### 训练脚本层面

| 特性                | 常规版本                               | Per-step 版本                             |
| ----------------- | ---------------------------------- | --------------------------------------- |
| **exploration_k** | 通过 num_image_per_prompt 控制         | 固定为 6                                   |
| **采样函数**          | `pipeline_with_logprob`            | `pipeline_with_logprob_perstep`         |
| **sde_step**      | `sde_step_with_logprob`            | `sde_step_with_logprob_perstep`         |
| **输出类型**          | "pt" (tensor)                      | "pil" (PIL Image)                       |
| **next_latents**  | latents[:, 1:] (切片)                | sde_latents (独立采样)                      |
| **rewards 结构**    | 单个 future                          | 列表 (per-step futures)                   |
| **rewards 形状**    | (batch,) → repeat → (batch, steps) | (batch, steps) 原生                       |
| **loss 权重**       | advantages * ratio                 | 1.73 * std_dev_t * advantages * ratio   |
| **数据打乱**          | 无                                  | 每个 inner_epoch 打乱                       |
| **训练步数**          | num_train_timesteps (窗口大小)         | num_train_timesteps (timestep_fraction) |

### CFG 处理差异

| 模型            | CFG 实现方式                        |
| ------------- | ------------------------------- |
| **Flux**      | Transformer 内置支持，单次前向传播         |
| **QwenImage** | 手动实现：拼接 → 前向 → chunk → 混合 → 归一化 |

### VAE 解码差异

| 模型            | 解码流程                                             |
| ------------- | ------------------------------------------------ |
| **Flux**      | scaling + shift → decode                         |
| **QwenImage** | unpack → normalize (mean/std) → decode → [:,:,0] |

### 性能考虑

**常规版本优势**：
- 内存占用较小（只保存窗口内数据）
- 计算效率高（少量时间步）
- 适合快速迭代

**Per-step 版本优势**：
- 更精细的奖励信号（每步独立）
- 分支探索提高样本多样性
- 更好的梯度信号（全步训练）
- 适合高质量训练

**训练时间对比**：
- 常规版本：窗口大小 × 样本数
- Per-step 版本：总步数 × exploration_k × 样本数

---

## 实现细节注意事项

### 1. 内存管理
```python
# Per-step 版本内存占用更大
# 需要存储：
# - all_latents: (batch, steps+1, seq_len, channels)
# - all_sde_latents: (batch, steps, seq_len, channels)
# - all_images: list of (batch, 3, H, W) × steps
# - all_log_probs: (batch, steps)

# 解决方案：
# 1. 减小 batch size
# 2. 使用梯度检查点
# 3. 适当降低分辨率
```

### 2. 数据类型处理
```python
# QwenImage 需要注意 dtype 转换
latents_dtype = latents.dtype
latents = latents.float()  # sde_step 需要 float32
latents, log_prob, _, _ = sde_step_with_logprob(...)
if latents.dtype != latents_dtype:
    latents = latents.to(latents_dtype)  # 转回原始 dtype
```

### 3. 分布式训练
```python
# FSDP 模式下注意：
# 1. 所有 gather 操作需要指定 world_size
# 2. optimizer 同步通过 should_sync 控制
# 3. rewards 计算在 gather 之前完成
# 4. advantages 在 gather 后分配回各进程
```

### 4. 调试技巧
```python
# 检查形状
print(f"latents shape: {latents.shape}")
# 预期：(batch, seq_len, channels)

print(f"rewards shape: {rewards['avg'].shape}")
# 常规版本：(batch,)
# Per-step版本：(batch, steps)

# 检查 NaN
assert not torch.isnan(loss).any(), "Loss contains NaN"
assert not torch.isnan(log_prob).any(), "Log prob contains NaN"
```

---

## 配置文件建议

### 常规训练配置
```python
config.sample.sde_window_size = 5  # 窗口大小
config.sample.sde_window_range = (0, 10)  # 窗口范围
config.sample.noise_level = 0.7  # SDE 噪声水平
config.sample.num_image_per_prompt = 4  # 每个 prompt 的图像数
config.train.timestep_fraction = 1.0  # 不使用
```

### Per-step 训练配置
```python
# 移除窗口相关参数
config.train.timestep_fraction = 0.5  # 训练前 50% 的时间步
# exploration_k 在代码中硬编码为 6
config.sample.train_batch_size = 2  # 由于 exploration_k，实际为 12
```

---

## 总结

本次适配成功将 Flux 的 per-step 训练方法迁移到 QwenImage 模型，主要工作包括：

1. **Pipeline 适配**：处理 QwenImage 特有的 CFG、VAE、Transformer 接口
2. **训练脚本适配**：实现分支探索、per-step rewards、加权损失
3. **保持兼容性**：两种训练模式可以共存，根据需求选择

**使用建议**：
- 初期训练：使用常规版本快速收敛
- 精细调优：使用 per-step 版本提升质量
- 资源受限：优先常规版本
- 追求极致：选择 per-step 版本

**后续优化方向**：
- 动态调整 exploration_k
- 自适应时间步权重
- 混合训练策略
- 更高效的内存管理
