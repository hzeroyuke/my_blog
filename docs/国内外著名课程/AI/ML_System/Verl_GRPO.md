# GRPO_verl: 完整训练流程与模式切换机制

本文档详细说明 verl 框架中 GRPO (Group Relative Policy Optimization) 训练的完整流程，包括资源分配、数据流向、显存占用、FSDP 与 Rollout 的模式切换机制以及参数同步的具体实现。

---

## 目录

1. [整体架构：Hybrid Engine 模式](#1-整体架构hybrid-engine-模式)
2. [资源初始化阶段](#2-资源初始化阶段)
3. [模型加载与显存占用](#3-模型加载与显存占用)
4. [PPO 完整训练迭代流程](#4-ppo-完整训练迭代流程)
5. [模式切换的详细机制](#5-模式切换的详细机制)
6. [参数同步的底层实现](#6-参数同步的底层实现)
7. [显存管理时间线](#7-显存管理时间线)
8. [代码位置索引](#8-代码位置索引)
9. [性能优化建议](#9-性能优化建议)

---

## 1. 整体架构：Hybrid Engine 模式

### 1.1 为什么需要 Hybrid Engine？

PPO 训练需要两个模型：
- **Actor FSDP 模型**：用于训练，需要梯度计算和参数更新（分片存储）
- **Rollout 模型**：用于推理生成，需要高效的 KV Cache 管理（完整模型）

如果分别部署，需要 **2倍的 GPU 资源**。Hybrid Engine 将两者共置（colocation）在同一组 GPU 上，节省 50% 资源。

### 1.2 Hybrid Engine 架构

```
每个 GPU 上的模型布局（以 Qwen3-8B 为例）:

┌─────────────────────────────────────────────────────────────┐
│ GPU 0 (80GB)                                                │
├─────────────────────────────────────────────────────────────┤
│ Actor FSDP 模型（训练用）                                   │
│   ├── 参数分片 [0:1B]: 2GB (可 offload 到 CPU)             │
│   ├── 优化器状态 (AdamW): 4GB (可 offload 到 CPU)          │
│   └── 梯度缓冲: 2GB                                         │
│                                                             │
│ Rollout 模型（推理用，vLLM/SGLang）                         │
│   ├── 完整参数 [0:8B]: 16GB (固定在 GPU)                   │
│   └── KV Cache: 32GB (动态分配/释放)                        │
│                                                             │
│ 总显存占用: 18-62GB (取决于当前模式)                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 两种运行模式

| 模式 | Actor FSDP | Rollout 模型 | Rollout KV Cache | 典型显存 |
|------|-----------|-------------|-----------------|---------|
| **Trainer Mode** | 在 GPU (2-8GB) | 在 GPU (16GB) | 已释放 (0GB) | ~22GB |
| **Rollout Mode** | 可 offload (0-2GB) | 在 GPU (16GB) | 已分配 (32GB) | ~50GB |

**核心思想**：通过动态切换模式，复用显存空间。

---

## 2. 资源初始化阶段

### 2.1 创建资源池和 Ray Actors

**代码位置**：
- `verl/trainer/main_ppo.py:166-189` - 创建资源池
- `verl/trainer/ppo/ray_trainer.py:661-796` - 初始化 workers

```python
# ========== 步骤 1: 创建资源池 ==========
# 文件: verl/trainer/main_ppo.py:166-189
resource_pool = RayResourcePool(
    process_on_nodes=[8],  # 8 个进程（每个 GPU 一个）
    use_gpu=True,
)

# 创建 Placement Groups（GPU 资源分配）
pgs = resource_pool.get_placement_groups()
# 结果: 8 个 Placement Groups，每个包含 1 个 GPU bundle


# ========== 步骤 2: 定义共置 Worker 配置 ==========
# 文件: verl/trainer/ppo/ray_trainer.py:745-766
class_dict = {
    "actor_rollout": RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        args=(...),
        kwargs={
            "config": actor_config,
            "role": "actor_rollout",  # Actor + Rollout 功能
        }
    ),
}


# ========== 步骤 3: 创建融合 Worker 类（WorkerDict）==========
# 文件: verl/single_controller/ray/base.py:749-790
worker_dict_cls = create_colocated_worker_cls(class_dict)

# WorkerDict 内部结构：
class WorkerDict(Worker):
    def __init__(self):
        self.worker_dict = {
            "actor_rollout": ActorRolloutRefWorker(role="actor_rollout"),
        }

    # 自动绑定方法（带前缀）
    def actor_rollout_generate_sequences(self, data):
        return self.worker_dict["actor_rollout"].generate_sequences(data)

    def actor_rollout_compute_log_prob(self, data):
        return self.worker_dict["actor_rollout"].compute_log_prob(data)

    def actor_rollout_update_actor(self, data):
        return self.worker_dict["actor_rollout"].update_actor(data)


# ========== 步骤 4: 创建 RayWorkerGroup（启动 8 个 Ray Actors）==========
# 文件: verl/single_controller/ray/base.py:361-444
wg_dict = RayWorkerGroup(
    resource_pool=resource_pool,
    ray_cls_with_init=worker_dict_cls,
)

# 此时创建了 8 个 Ray Actors：
# GPU 0: Ray Actor 0 (WorkerDict 实例)
# GPU 1: Ray Actor 1 (WorkerDict 实例)
# ...
# GPU 7: Ray Actor 7 (WorkerDict 实例)


# ========== 步骤 5: Spawn 创建 WorkerGroup 视图 ==========
# 文件: verl/single_controller/ray/base.py:478-512
spawn_wg = wg_dict.spawn(prefix_set={"actor_rollout"})

# spawn_wg = {
#     "actor_rollout": RayWorkerGroup(
#         _workers=[Actor 0, ..., Actor 7],  # 共享相同的 8 个 Ray Actors
#         方法: generate_sequences, compute_log_prob, update_actor  # 去掉前缀
#     )
# }


# ========== 步骤 6: 提取最终的 WorkerGroup ==========
# 文件: verl/trainer/ppo/ray_trainer.py:763
self.actor_rollout_wg = spawn_wg["actor_rollout"]
```

### 2.2 Spawn 机制详解

**Spawn 的核心作用**：实现共置（Colocation）

```python
# 物理结构（8 个 Ray Actors）：
┌─────────────────────────────────────────────────────────────┐
│ GPU 0: Ray Actor 0 (WorkerDict)                             │
│   ├── worker_dict["actor_rollout"] = ActorRolloutRefWorker │
│   └── worker_dict["ref"] = RefWorker (如果配置了)           │
├─────────────────────────────────────────────────────────────┤
│ GPU 1: Ray Actor 1 (WorkerDict)                             │
│   ├── worker_dict["actor_rollout"] = ActorRolloutRefWorker │
│   └── worker_dict["ref"] = RefWorker                        │
├─────────────────────────────────────────────────────────────┤
│ ... (GPU 2-6)                                               │
├─────────────────────────────────────────────────────────────┤
│ GPU 7: Ray Actor 7 (WorkerDict)                             │
│   ├── worker_dict["actor_rollout"] = ActorRolloutRefWorker │
│   └── worker_dict["ref"] = RefWorker                        │
└─────────────────────────────────────────────────────────────┘

# 逻辑结构（3 个 RayWorkerGroup 对象）：

wg_dict (原始)
├── _workers = [Actor 0, Actor 1, ..., Actor 7]
├── 方法: actor_rollout_generate_sequences, ref_compute_ref_log_prob
└── 用途: 初始化，通常不直接使用

spawn_wg["actor_rollout"] (视图1)
├── _workers = [Actor 0, Actor 1, ..., Actor 7]  # 共享相同的 Actors！
├── 方法: generate_sequences, update_actor  # 只暴露 actor_rollout 的方法
└── 调用时路由到 worker_dict["actor_rollout"]

spawn_wg["ref"] (视图2, 如果有)
├── _workers = [Actor 0, Actor 1, ..., Actor 7]  # 共享相同的 Actors！
├── 方法: compute_ref_log_prob  # 只暴露 ref 的方法
└── 调用时路由到 worker_dict["ref"]
```

**关键点**：
- 物理上只有 **8 个 Ray Actors**（8 个进程）
- 逻辑上有多个 RayWorkerGroup 对象（视图）
- 所有视图都引用相同的 8 个 Ray Actors
- Spawn 只是创建了不同的"视图"，不创建新的 Ray Actors
- **目的**：避免为每个功能创建独立的 GPU 进程，节省资源

---

## 3. 模型加载与显存占用

### 3.1 `__init__` - Worker 初始化

**代码位置**：`verl/workers/fsdp_workers.py:139-263`

在 Ray Actor 创建时，首先调用 `ActorRolloutRefWorker.__init__()`：

```python
# 文件: verl/workers/fsdp_workers.py:139-263

class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config

        # ========== 步骤 1: 初始化分布式环境 ==========
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
            )

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # ========== 步骤 2: 创建 Device Mesh（分布式拓扑）==========
        from torch.distributed.device_mesh import init_device_mesh

        # Device Mesh 用于 FSDP/TP 分片
        # 例如：8 个 GPU → DeviceMesh([0, 1, 2, 3, 4, 5, 6, 7])
        self.device_mesh = init_device_mesh(
            get_device_name(),
            mesh_shape=(self.world_size,),
            mesh_dim_names=("fsdp",)
        )

        # ========== 步骤 3: 解析 role（决定创建哪些模型）==========
        # role 可能的值：
        # - "actor_rollout": 创建 Actor FSDP + Rollout 模型（Hybrid Engine）
        # - "actor": 仅创建 Actor FSDP
        # - "rollout": 仅创建 Rollout 模型
        # - "ref": 创建 Reference 模型

        self._is_actor = "actor" in role  # 是否包含 Actor 功能
        self._is_rollout = "rollout" in role  # 是否包含 Rollout 功能
        self._is_ref = "ref" in role  # 是否包含 Reference 功能

        # ========== 步骤 4: 配置 Offload 策略 ==========
        self._is_offload_param = config.actor.model.get("enable_parameter_offload", False)
        self._is_offload_optimizer = config.actor.optimizer.get("enable_optimizer_offload", False)

        # 注意：__init__ 阶段只是配置，不加载模型！
        # 实际的模型加载在 init_model() 中进行（由 Ray Trainer 调用）
```

**关键点**：
- `__init__` **不加载模型**，只初始化分布式环境和配置
- 模型加载在 `init_model()` 中（稍后调用）
- 此时显存占用：~0GB（仅分布式初始化开销 ~100MB）

---

### 3.2 `init_model()` - 模型初始化入口

**代码位置**：`verl/workers/fsdp_workers.py:760-820`

Ray Trainer 调用 `init_model()` 后，才真正加载模型：

```python
# 文件: verl/workers/fsdp_workers.py:760-820

@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    """初始化 Actor/Rollout/Ref 三个模型（FSDP 训练核心）"""

    # ========== 步骤 1: 加载 Actor FSDP 模型（用于训练）==========
    if self._is_actor:
        self.actor_module_fsdp, self.actor_optimizer, ... = self._build_model_optimizer(
            model_path=self.config.actor.model.path,  # 如 "Qwen/Qwen3-8B"
            fsdp_config=self.config.actor,
            role="actor",
        )
        # ↓ 详细流程见 3.3 节

    # ========== 步骤 2: 加载 Rollout 模型（用于推理）==========
    if self._is_rollout:
        self.rollout = self._build_rollout()
        # ↓ 详细流程见 3.4 节

    # ========== 步骤 3: 加载 Reference 模型（如果需要）==========
    if self._is_ref:
        self.ref_module_fsdp = self._build_model_optimizer(
            model_path=self.config.ref.model.path,
            fsdp_config=self.config.ref,
            role="ref",
        )

    # ========== 步骤 4: 配置 FSDP state_dict 类型 ==========
    # 文件: verl/workers/fsdp_workers.py:634-644
    if torch.distributed.get_world_size() == 1:
        FSDP.set_state_dict_type(
            self.actor_module_fsdp,
            state_dict_type=StateDictType.FULL_STATE_DICT,
        )
    else:
        FSDP.set_state_dict_type(
            self.actor_module_fsdp,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,  # ← 多 GPU 使用分片
        )

    # ========== 步骤 5: 切换到 Trainer 模式（初始状态）==========
    # 文件: verl/workers/fsdp_workers.py:650-656
    if rollout_config.mode == "sync" and self._is_actor:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.trainer_mode())
        # ↓ 切换到 Trainer 模式（释放 KV Cache）
```

---

### 3.3 `_build_model_optimizer()` - Actor FSDP 模型加载

**代码位置**：`verl/workers/fsdp_workers.py:268-589`

**核心问题**：显存中是否同时存在两份模型权重？

**答案**：**是的**！在 FSDP 初始化过程中，显存中短暂存在两份完整权重：

```python
# 文件: verl/workers/fsdp_workers.py:268-589

def _build_model_optimizer(self, model_path, fsdp_config, role="actor"):
    """构建 FSDP 模型和优化器"""

    # ========== 步骤 1: 加载 Tokenizer ==========
    # 文件: verl/workers/fsdp_workers.py:304-311
    self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
    self.processor = hf_processor(local_path, trust_remote_code=True)

    # ========== 步骤 2: 确定模型 dtype ==========
    # 文件: verl/workers/fsdp_workers.py:314-320
    torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
    # Actor: fp32（避免优化器在 bf16，影响训练稳定性）
    # Ref: bf16（节省显存，推理不需要高精度）

    # ========== 步骤 3: 加载模型配置 ==========
    # 文件: verl/workers/fsdp_workers.py:323-346
    actor_model_config = AutoConfig.from_pretrained(
        local_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2"
    )

    # ========== 步骤 4: 初始化模型（使用 meta tensor 或直接初始化）==========
    # 文件: verl/workers/fsdp_workers.py:348-389
    init_context = get_init_weight_context_manager(
        use_meta_tensor=not actor_model_config.tie_word_embeddings,
        mesh=self.device_mesh
    )
    # Meta tensor: 延迟初始化（不占用显存）
    # 如果 tie_word_embeddings=True，则直接初始化（占用显存）

    with init_context():
        # 加载预训练模型（从磁盘读取权重）
        actor_module = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,  # fp32 (Actor) 或 bf16 (Ref)
            config=actor_model_config,
            trust_remote_code=trust_remote_code,
        )

    # ========== 关键时刻：显存中有完整的模型权重 ==========
    # 此时每个 GPU 都加载了完整的 8B 参数（32GB，fp32 格式）
    # 显存占用：32GB / GPU

    # ========== 步骤 5: 应用优化（Liger/Monkey Patch/Gradient Checkpointing/LoRA）==========
    # 文件: verl/workers/fsdp_workers.py:392-432
    if use_liger:
        _apply_liger_kernel_to_instance(model=actor_module)  # 优化内核

    apply_monkey_patch(  # 注入自定义 forward
        model=actor_module,
        use_remove_padding=use_remove_padding,
        ulysses_sp_size=self.ulysses_sequence_parallel_size,
    )

    if enable_gradient_checkpointing:
        actor_module.gradient_checkpointing_enable()  # 省显存（recompute）

    if self._is_lora:
        actor_module = get_peft_model(actor_module, LoraConfig(...))  # LoRA 适配

    # ========== 步骤 6: FSDP 初始化（关键！）==========
    # 文件: verl/workers/fsdp_workers.py:447-540
    torch.distributed.barrier()  # 同步所有 GPU

    # 获取 FSDP 配置
    auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config)
    sharding_strategy = ShardingStrategy.FULL_SHARD  # ZeRO-3
    mixed_precision = MixedPrecision(...)  # bf16 计算, fp32 reduce

    fsdp_strategy = self.config.actor.strategy
    if fsdp_strategy == "fsdp":
        # ========== FSDP1（PyTorch < 2.4）==========
        # 文件: verl/workers/fsdp_workers.py:497-507
        actor_module_fsdp = FSDP(
            actor_module,  # 输入：完整模型（32GB）
            cpu_offload=None if role == "actor" else CPUOffload(offload_params=True),
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,  # 按层包装
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,  # ZeRO-3
            mixed_precision=mixed_precision,     # bf16 计算
            sync_module_states=True,             # ← 关键：同步初始化状态
            device_mesh=self.device_mesh,
        )
        # ↓ FSDP 内部执行：
        # 1. 将完整模型（32GB）分片为 8 份（每份 4GB）
        # 2. All-Reduce 同步参数（确保所有 GPU 的初始参数一致）
        # 3. 释放本地完整参数，只保留分片（4GB）
        # 4. 转换为 bf16 → 分片从 4GB 降到 2GB
        #
        # 显存变化：
        # - 初始化前：32GB（fp32 完整模型）
        # - FSDP 中：32GB (原模型) + 2GB (分片) = 34GB（峰值）
        # - 初始化后：2GB（bf16 分片，原模型被释放）

    elif fsdp_strategy == "fsdp2":
        # ========== FSDP2（PyTorch >= 2.4）==========
        # 文件: verl/workers/fsdp_workers.py:509-532
        fsdp_kwargs = {
            "mesh": fsdp_mesh,
            "mp_policy": mp_policy,
            "offload_policy": cpu_offload,
            "reshard_after_forward": fsdp_config.reshard_after_forward,
        }
        # 先获取完整参数
        full_state = actor_module.state_dict()  # 32GB（fp32）
        # 应用 FSDP2（in-place 转换）
        apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
        # 加载参数到分片模型
        fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
        # 释放 full_state
        del full_state

        # 显存变化：
        # - 初始化前：32GB（fp32 完整模型）
        # - FSDP2 中：32GB (原模型) + 32GB (full_state) = 64GB（峰值）
        # - 初始化后：2GB（bf16 分片）

    log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)
    # 此时显存：~2GB (参数分片)

    # ========== 步骤 7: 创建优化器（仅 Actor）==========
    # 文件: verl/workers/fsdp_workers.py:542-589
    if role == "actor":
        actor_optimizer = optim.AdamW(
            actor_module_fsdp.parameters(),
            lr=optim_config.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        # 优化器状态（momentum + variance）：2GB + 2GB = 4GB

        actor_lr_scheduler = get_lr_scheduler(...)

    return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

# ========== 显存占用总结 ==========
# 初始化峰值（FSDP1）: 34GB（fp32 完整模型 + bf16 分片）
# 初始化峰值（FSDP2）: 64GB（fp32 完整模型 + fp32 full_state）
# 初始化后稳定: 2GB（bf16 分片） + 4GB（优化器） = 6GB
```

**关键发现**：

1. **是的，显存中短暂存在两份模型权重**：
   - FSDP 初始化前：完整模型（32GB fp32）
   - FSDP 初始化中：完整模型 + 分片模型（34-64GB 峰值）
   - FSDP 初始化后：仅分片模型（2GB bf16）

2. **FSDP 初始化位置**：`verl/workers/fsdp_workers.py:497-507` (FSDP1) 或 `530-532` (FSDP2)

3. **sync_module_states=True 的作用**：
   - 在 FSDP 初始化时，所有 GPU 的参数必须一致
   - `sync_module_states=True` 确保 rank 0 的参数广播到所有 GPU
   - 这是分布式训练的必要步骤

---

### 3.4 `_build_rollout()` - Rollout 模型加载

**代码位置**：`verl/workers/fsdp_workers.py:591-630`

```python
# 文件: verl/workers/fsdp_workers.py:591-630

def _build_rollout(self):
    """构建 Rollout 模型（vLLM/SGLang）"""

    # ========== 步骤 1: 创建 Rollout Device Mesh ==========
    rollout_device_mesh = init_device_mesh(
        get_device_name(),
        mesh_shape=(self.world_size,),
        mesh_dim_names=("rollout",)
    )

    # ========== 步骤 2: 创建 Rollout Engine ==========
    rollout_config = self.config.rollout
    self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
        config=rollout_config,
        model_config=model_config,
        device_mesh=rollout_device_mesh
    )
    # → 内部调用 vLLM 或 SGLang 初始化
    # → 从磁盘加载完整模型（16GB bf16）
    # → 分配 KV Cache（32GB）

    log_gpu_memory_usage(f"After building {rollout_config.name} rollout", logger=logger)
    # 此时显存：16GB (参数) + 32GB (KV Cache) = 48GB

    return self.rollout

# ========== Rollout 模型特点 ==========
# - 完整模型（不分片）：16GB（bf16）
# - KV Cache：32GB（动态分配）
# - 与 Actor FSDP 共存：同一 GPU 上
```

**关键点**：
- Rollout 模型**不使用 FSDP**，是完整模型
- 直接从磁盘加载到 GPU（16GB）
- 与 Actor FSDP 模型**共享同一 GPU**

---

### 3.5 初始化后的模式状态

**代码位置**：`verl/workers/fsdp_workers.py:650-656`

```python
# 文件: verl/workers/fsdp_workers.py:650-656

# 5. switch to trainer mode
# NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
if rollout_config.mode == "sync" and self._is_actor:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(self.trainer_mode())
    # ↓ 切换到 Trainer 模式
```

**初始化完整流程**：

```
1. __init__() → 初始化分布式环境（~100MB）
   ↓
2. init_model() → 调用 _build_model_optimizer()
   ↓
3. _build_model_optimizer():
   ├── 加载完整 Actor 模型（32GB fp32）
   ├── FSDP 初始化（峰值 34-64GB）
   ├── 释放完整模型，保留分片（2GB bf16）
   └── 创建优化器（4GB）
   ↓ 此时显存：6GB

4. _build_rollout():
   ├── 加载 Rollout 模型（16GB bf16）
   └── 分配 KV Cache（32GB）
   ↓ 此时显存：6GB + 48GB = 54GB

5. trainer_mode():
   ├── 释放 KV Cache（-32GB）
   └── 保留 Rollout 参数（16GB）
   ↓ 最终显存：6GB + 16GB = 22GB
```

**初始化后的显存占用（每个 GPU）**：

```
GPU 0 (Trainer Mode):
├── Actor FSDP 参数分片: 2GB (bf16)
├── Actor 优化器状态: 4GB (fp32)
├── Rollout 参数（完整）: 16GB (bf16)
├── Rollout KV Cache: 0GB (已释放)
└── 总计: ~22GB / 80GB
```

### 3.2 初始化后的模式状态

**关键点**：初始化后，系统处于 **Trainer Mode**，而不是 Rollout Mode！

**代码位置**：`verl/workers/fsdp_workers.py:650-656`

```python
# 文件: verl/workers/fsdp_workers.py:650-656

# 5. switch to trainer mode
# NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
# For sync mode, we directly switch to trainer mode here.
# For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
if rollout_config.mode == "sync" and self._is_actor:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(self.trainer_mode())  # ← 这里切换到 Trainer Mode
```

**初始化流程**：

```
1. _build_model_optimizer() → Actor FSDP 加载（2GB 参数分片 + 4GB 优化器）
2. _build_rollout() → Rollout 模型加载（16GB 完整参数 + 32GB KV Cache）
   ↓ 此时显存: ~54GB
3. trainer_mode() 被调用:
   ↓ rollout.release() 释放 KV Cache（-32GB）
   ↓ 此时显存: ~22GB
4. 初始化完成，系统处于 Trainer Mode
```

**为什么初始化后要切换到 Trainer Mode？**

根据代码注释：
> It's critical that hybrid engine in trainer mode initially to load checkpoint.

原因：
1. **加载 checkpoint** 时需要 Actor FSDP 在 GPU（用于加载参数）
2. Rollout KV Cache 在初始化时会占用大量显存，需要释放
3. 训练开始前可能需要执行验证（validation），需要 Trainer Mode

---

## 4. PPO 完整训练迭代流程

### 4.1 训练循环入口

**代码位置**：`verl/trainer/ppo/ray_trainer.py:962-1259`

```python
# 文件: verl/trainer/ppo/ray_trainer.py:962-1259

def fit(self):
    """
    The training loop of PPO.
    The driver process only need to call the compute functions of the worker group through RPC
    to construct the PPO dataflow.
    The light-weight advantage computation is done on the driver process.
    """

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            # ========== 阶段 1: 生成序列（Rollout Phase）==========
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            # ========== 阶段 2: 计算 Reward（TaskRunner CPU）==========
            reward_tensor, reward_extra_infos = compute_reward(batch, self.reward_fn)

            # ========== 阶段 3: 计算 Old Log Prob（Actor FSDP）==========
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

            # ========== 阶段 4: 计算 Advantage（TaskRunner CPU）==========
            batch = compute_advantage(batch, adv_estimator="grpo", ...)

            # ========== 阶段 5: 更新 Actor 模型（FSDP 训练）==========
            actor_output = self.actor_rollout_wg.update_actor(batch)

            # 下一轮迭代...
```

### 4.2 阶段 1：生成序列（Rollout Phase）

**代码位置**：
- 调用入口：`verl/trainer/ppo/ray_trainer.py:1042-1050`
- Worker 实现：`verl/workers/fsdp_workers.py:927-984`

```python
# ========== TaskRunner 调用 ==========
# 文件: verl/trainer/ppo/ray_trainer.py:1042-1050

gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)


# ========== 在每个 GPU Worker 中执行 ==========
# 文件: verl/workers/fsdp_workers.py:927-984

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
@DistProfiler.annotate(color="red", role="rollout_generate")
def generate_sequences(self, prompts: DataProto):
    """生成序列（Rollout 阶段）"""

    # ========== 步骤 1.1: 切换到 Rollout 模式 ==========
    # 文件: verl/workers/fsdp_workers.py:945-950
    if self._is_actor:  # For rollout only, we do not switch context.
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.rollout_mode())  # ← 切换到 Rollout Mode
        # → Actor 权重同步到 Rollout Engine
        # → Actor 模型 offload 到 CPU（如果配置了）
        log_gpu_memory_usage("After switch to rollout mode", logger=logger)

    # ========== 步骤 1.2: 使用 Rollout 推理 ==========
    # 文件: verl/workers/fsdp_workers.py:952-957
    with simple_timer("generate_sequences", timing_generate):
        output = self.rollout.generate_sequences(prompts=prompts)
        # → vLLM/SGLang 执行推理
        # → 返回：sequences, token_level_scores, ...

    # ========== 步骤 1.3: 切换回 Trainer 模式 ==========
    # 文件: verl/workers/fsdp_workers.py:960-964
    if self._is_actor:
        loop.run_until_complete(self.trainer_mode())  # ← 切换回 Trainer Mode
        # → Rollout Engine offload
        # → Actor 模型加载回 GPU
        log_gpu_memory_usage("After switch to trainer mode", logger=logger)

    return output
```

### 4.3 阶段 2：计算 Reward（TaskRunner CPU）

**代码位置**：`verl/trainer/ppo/ray_trainer.py:1088-1098`

```python
# 文件: verl/trainer/ppo/ray_trainer.py:1088-1098

# ========== 在 TaskRunner（CPU）中计算 Reward ==========
# 对于 GRPO，使用规则 Reward（如 GSM8K 答案匹配）

with marked_timer("reward", timing_raw, color="yellow"):
    # compute reward model score
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)

    if self.config.reward_model.launch_reward_fn_async:
        future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
    else:
        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

# 不需要模式切换
# 不占用 GPU 显存
```

### 4.4 阶段 3：计算 Old Log Prob（使用 Actor FSDP）

**代码位置**：
- 调用入口：`verl/trainer/ppo/ray_trainer.py:1100-1110`
- Worker 实现：`verl/workers/fsdp_workers.py:986-1026`

```python
# 文件: verl/trainer/ppo/ray_trainer.py:1100-1110

# recompute old_log_probs
with marked_timer("old_log_prob", timing_raw, color="blue"):
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    entropys = old_log_prob.batch["entropys"]
    response_masks = batch.batch["response_mask"]
    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
    metrics.update(old_log_prob_metrics)
    old_log_prob.batch.pop("entropys")
    batch = batch.union(old_log_prob)


# 文件: verl/workers/fsdp_workers.py:986-1026

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
@DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
def compute_log_prob(self, data: DataProto):
    """计算旧策略的 Log Probs（用于 PPO importance sampling）"""

    # ========== 不需要模式切换！==========
    # 因为当前已经在 Trainer Mode（上一步 trainer_mode() 完成）

    # ========== 步骤 3.1: 加载 Actor FSDP 到 GPU（如果 offload 了）==========
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        # → 从 CPU 加载参数分片（2GB）

    # ========== 步骤 3.2: 使用 Actor FSDP 计算 log prob ==========
    output, entropys = self.actor.compute_log_prob(
        data=data,
        calculate_entropy=True
    )
    # → Forward pass（Teacher Forcing）
    # → 计算每个 token 的 log probability
    # → 使用的是 **当前训练中的 Actor 模型**（不是 Rollout）

    # ========== 步骤 3.3: Offload Actor FSDP 回 CPU（如果配置了）==========
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        # → 释放 2GB 显存

    return output
```

### 4.5 阶段 4：计算 Advantage（TaskRunner CPU）

**代码位置**：`verl/trainer/ppo/ray_trainer.py:1132-1164`

```python
# 文件: verl/trainer/ppo/ray_trainer.py:1132-1164

with marked_timer("adv", timing_raw, color="brown"):
    # we combine with rule-based rm
    reward_extra_infos_dict: dict[str, list]
    if self.config.reward_model.launch_reward_fn_async:
        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
    batch.batch["token_level_scores"] = reward_tensor

    if reward_extra_infos_dict:
        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

    # compute rewards. apply_kl_penalty if available
    if self.config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(
            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
        )
        metrics.update(kl_metrics)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

    # compute advantages, executed on the driver process
    norm_adv_by_std_in_grpo = self.config.algorithm.get(
        "norm_adv_by_std_in_grpo", True
    )  # GRPO adv normalization factor

    batch = compute_advantage(
        batch,
        adv_estimator=self.config.algorithm.adv_estimator,
        gamma=self.config.algorithm.gamma,
        lam=self.config.algorithm.lam,
        num_repeat=self.config.actor_rollout_ref.rollout.n,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=self.config.algorithm,
    )

# 在 TaskRunner（CPU）中执行
# 不需要模式切换
# 不占用 GPU 显存
```

### 4.6 阶段 5：更新 Actor 模型（FSDP 训练）

**代码位置**：
- 调用入口：`verl/trainer/ppo/ray_trainer.py:1174-1180`
- Worker 实现：`verl/workers/fsdp_workers.py:877-923`

```python
# 文件: verl/trainer/ppo/ray_trainer.py:1174-1180

# implement critic warmup
if self.config.trainer.critic_warmup <= self.global_steps:
    # update actor
    with marked_timer("update_actor", timing_raw, color="red"):
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        actor_output = self.actor_rollout_wg.update_actor(batch)
    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    metrics.update(actor_output_metrics)


# 文件: verl/workers/fsdp_workers.py:877-923

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
@DistProfiler.annotate(color="red", role="actor_update")
def update_actor(self, data: DataProto):
    """执行 PPO 策略更新"""

    # ========== 不需要模式切换！==========
    # 因为当前已经在 Trainer Mode

    # ========== 步骤 5.1: 加载模型和优化器到 GPU ==========
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        # → 加载 Actor 参数分片（2GB）
    if self._is_offload_optimizer:
        load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
        # → 加载优化器状态（4GB）

    # ========== 步骤 5.2: 调用 PPO Actor 训练 ==========
    with self.ulysses_sharding_manager:
        data = data.to("cpu")  # 先移到 CPU，后续按 micro-batch 移到 GPU
        output = self.actor.update_policy(data)
        # ↓ 内部执行 mini-batch 训练循环
        # ↓ Forward, Backward, Optimizer Step
        # ↓ FSDP 自动同步梯度和参数（NCCL All-Reduce）

    # ========== 步骤 5.3: Offload 模型和优化器回 CPU ==========
    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        # → 释放 2GB
    if self._is_offload_optimizer:
        offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        # → 释放 4GB

    return output
```

---

## 5. 模式切换的详细机制

### 5.1 `rollout_mode()` - 从 Trainer 切换到 Rollout

**代码位置**：`verl/workers/fsdp_workers.py:658-739`

**调用时机**：每次 `generate_sequences()` 开始时

```python
# 文件: verl/workers/fsdp_workers.py:658-739

async def rollout_mode(self):
    """Context switch hybridengine to rollout mode."""

    # ========== 子步骤 1: 清理缓存 ==========
    aggressive_empty_cache(force_sync=True)
    # → torch.cuda.empty_cache()
    # → 清理碎片化的显存

    # ========== 子步骤 2: 加载 Actor FSDP 到 GPU（如果之前 offload 了）==========
    log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        # → 从 CPU 加载参数分片到 GPU
        # → 显存增加 2GB
    log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

    # ========== 子步骤 3: 收集 FSDP 参数（分片 → 完整）==========
    peft_config = None
    peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)

    if hasattr(peft_model, "peft_config"):  # LoRA 模式
        # 仅收集 LoRA 参数（轻量级）
        params = collect_lora_params(
            module=self.actor_module_fsdp,
            layered_summon=self.config.rollout.get("layered_summon", False),
            base_sync_done=self.base_sync_done,
        )
    else:  # 完整模型
        # 收集完整参数（重量级）
        params = self.actor_module_fsdp.state_dict()
        # → 内部调用 FSDP 的 state_dict()
        # → 根据配置决定是 SHARDED_STATE_DICT 还是 FULL_STATE_DICT

    # ========== 子步骤 4: 转换参数格式 ==========
    params = convert_weight_keys(
        params,
        getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
    )
    # → 转换 FSDP 的 key 格式为 Rollout 期望的格式
    # → 例如：移除 "_fsdp_wrapped_module." 前缀

    # ========== 子步骤 5: 转换为 full_tensor（关键！）==========
    # 对于 FSDP2 使用 DTensor，需要显式转换
    if fsdp_version(self.actor_module_fsdp) == 2:  # FSDP2
        device = get_device_id()
        per_tensor_param = (
            (name, param.to(device, non_blocking=True).full_tensor()
             if isinstance(param, DTensor) else param)
            for name, param in params.items()
        )
    else:  # FSDP1
        per_tensor_param = params.items()

    # DTensor.full_tensor() 的作用：
    # 1. 触发 all-gather 操作（NCCL 通信）
    # 2. 将 8 个 GPU 的分片聚合到每个 GPU
    # 3. 返回完整的 tensor（16GB）
    #
    # 通信量：
    # - GPU 0 发送 2GB，接收 14GB → 总计 16GB
    # - GPU 1 发送 2GB，接收 14GB → 总计 16GB
    # - ...
    # - 总通信量：8 * 14GB = 112GB
    # - 通信时间：~200ms（NVLink 带宽 600GB/s）

    # ========== 子步骤 6: 恢复 Rollout Engine（重新加载 KV Cache）==========
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["weights"])
        # → 重新分配 KV Cache 空间（32GB）
        # → 显存增加 32GB
    log_gpu_memory_usage("After resume weights", logger=logger)

    # ========== 子步骤 7: 更新 Rollout 参数（核心！）==========
    # 文件: verl/workers/fsdp_workers.py:725-728
    await self.rollout.update_weights(per_tensor_param, peft_config=peft_config)
    # → 将 Actor FSDP 的参数复制到 Rollout 模型
    # → in-place 更新，不需要额外显存
    # → 耗时：~100ms（复制 16GB 数据）

    log_gpu_memory_usage("After update_weights", logger=logger)
    del params, per_tensor_param
    aggressive_empty_cache(force_sync=True)

    # ========== 子步骤 8: 恢复 KV Cache ==========
    if self.config.rollout.free_cache_engine:
        await self.rollout.resume(tags=["kv_cache"])
        # → 重新分配 KV Cache（32GB）
    log_gpu_memory_usage("After resume kv_cache", logger=logger)

    # ========== 子步骤 9: 保存/恢复随机状态 ==========
    self.base_sync_done = True
    # 保存 Trainer 的随机状态
    self.torch_random_states = get_torch_device().get_rng_state()
    # 恢复 Rollout 的随机状态
    get_torch_device().set_rng_state(self.gen_random_states)
    # → 确保生成的随机性可复现
```

**显存变化时间线**：

```
初始状态（Trainer Mode）:
├── Actor FSDP 分片: 2GB (GPU)
├── Actor 优化器: 4GB (GPU 或 CPU offload)
├── Rollout 参数: 16GB (GPU)
└── 总计: 18-22GB

执行 rollout_mode():
├── load_fsdp_model_to_gpu: +2GB (如果之前 offload)
├── params.full_tensor(): +16GB (临时，all-gather)
├── update_weights: 0GB (in-place 替换)
├── 释放临时参数: -16GB
├── resume KV Cache: +32GB
└── 总计: 50GB

最终状态（Rollout Mode）:
├── Actor FSDP 分片: 2GB (GPU 或 CPU offload)
├── Rollout 参数: 16GB (GPU, 已更新)
├── Rollout KV Cache: 32GB (GPU)
└── 总计: 50GB (如果 Actor offload 则 48GB)
```

### 5.2 `trainer_mode()` - 从 Rollout 切换回 Trainer

**代码位置**：`verl/workers/fsdp_workers.py:741-757`

**调用时机**：每次 `generate_sequences()` 结束时

```python
# 文件: verl/workers/fsdp_workers.py:741-757

async def trainer_mode(self):
    """Context switch hybridengine to trainer mode."""

    # ========== 子步骤 1: 释放 Rollout KV Cache（关键！）==========
    if self.config.rollout.free_cache_engine:
        log_gpu_memory_usage("Before rollout offload", logger=logger)
        await self.rollout.release()
        # → 释放 KV Cache（32GB）
        # → 保留 Rollout 参数（16GB）
        log_gpu_memory_usage("After rollout offload", logger=logger)

    # rollout.release() 内部逻辑：
    # async def release(self):
    #     """释放 KV Cache 和其他缓存"""
    #     # 1. 释放所有 KV Cache 块
    #     self.kv_cache_manager.clear()  # → 释放 32GB
    #
    #     # 2. 清理其他缓存（如 PageTable）
    #     self.cache_engine.clear()
    #
    #     # 3. 保留模型参数（不释放）
    #     # self.model.parameters() 仍然在 GPU（16GB）
    #
    #     torch.cuda.empty_cache()

    # ========== 子步骤 2: 切换 Actor 模型为训练模式 ==========
    self.actor_module_fsdp.train()
    # → 启用 Dropout、BatchNorm 等训练特性

    # ========== 子步骤 3: 清理缓存 ==========
    aggressive_empty_cache(force_sync=True)
    # → torch.cuda.empty_cache()
    # → 整理碎片化的显存

    # ========== 子步骤 4: 设置可扩展段（PyTorch 显存管理优化）==========
    set_expandable_segments(True)
    # → 允许 PyTorch 动态扩展显存段
    # → 减少显存碎片化

    # ========== 子步骤 5: 恢复随机状态 ==========
    # 保存 Rollout 的随机状态
    self.gen_random_states = get_torch_device().get_rng_state()
    # 恢复 Trainer 的随机状态
    get_torch_device().set_rng_state(self.torch_random_states)
    # → 确保训练的随机性可复现
```

**显存变化时间线**：

```
初始状态（Rollout Mode）:
├── Actor FSDP 分片: 2GB (可能在 CPU)
├── Rollout 参数: 16GB (GPU)
├── Rollout KV Cache: 32GB (GPU)
└── 总计: 50GB

执行 trainer_mode():
├── rollout.release(): -32GB (释放 KV Cache)
├── empty_cache(): 0GB (整理碎片)
└── 总计: 18GB

最终状态（Trainer Mode）:
├── Actor FSDP 分片: 2GB (GPU)
├── Actor 优化器: 4GB (GPU 或 CPU offload)
├── Rollout 参数: 16GB (GPU, 保留但不使用)
└── 总计: 18-22GB
```

---

## 6. 参数同步的底层实现

### 6.1 为什么每次 Rollout 都需要同步参数？

因为 Actor 模型在每次训练后都会更新！

```python
# ========== 时间线 ==========
时刻 t0: 初始化
  Actor FSDP 参数 = W0
  Rollout 参数 = W0
  （两者一致）

时刻 t1: 第一次训练
  generate_sequences():
    rollout_mode() → Actor (W0) → Rollout (W0)  # 同步（实际上参数已一致，但仍执行）
    Rollout 推理（使用 W0）
  update_actor():
    训练更新 → Actor (W1)  # 参数已改变！

时刻 t2: 第二次训练
  generate_sequences():
    rollout_mode() → Actor (W1) → Rollout (W1)  # 同步（必须！Rollout 还是 W0）
    Rollout 推理（使用 W1）  # 使用最新的策略
  update_actor():
    训练更新 → Actor (W2)

时刻 t3: 第三次训练
  generate_sequences():
    rollout_mode() → Actor (W2) → Rollout (W2)  # 同步
    Rollout 推理（使用 W2）
  ...
```

**如果不同步会怎样？**

```python
# 错误示例：不同步
时刻 t1: Actor = W0, Rollout = W0 → 生成使用 W0 ✓
时刻 t2: Actor = W1, Rollout = W0 → 生成使用 W0 ✗ (应该用 W1!)
时刻 t3: Actor = W2, Rollout = W0 → 生成使用 W0 ✗ (应该用 W2!)

结果：
- Rollout 使用的是初始策略，不是当前策略
- PPO 的 importance sampling 失效
- 训练发散！
```

### 6.2 FSDP state_dict() 的机制

**代码位置**：
- 配置 state_dict 类型：`verl/workers/fsdp_workers.py:634-644`
- 调用 state_dict()：`verl/workers/fsdp_workers.py:679`

```python
# ========== 步骤 1: 配置 state_dict 类型 ==========
# 文件: verl/workers/fsdp_workers.py:634-644

if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
    FSDP.set_state_dict_type(
        self.actor_module_fsdp,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(),
    )
elif fsdp_version(self.actor_module_fsdp) == 1:
    FSDP.set_state_dict_type(
        self.actor_module_fsdp,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,  # ← 多 GPU 使用分片模式
        state_dict_config=ShardedStateDictConfig(),
    )


# ========== 步骤 2: 调用 state_dict() ==========
# 文件: verl/workers/fsdp_workers.py:679

params = self.actor_module_fsdp.state_dict()

# FSDP state_dict() 的内部行为（根据配置）：

# 1. 如果配置为 SHARDED_STATE_DICT：
#    - 每个 GPU 只返回本地分片（2GB）
#    - 不需要 all-gather
#    - 显存开销：0GB（仅返回已有分片的引用）
#    - 返回的 params 是分片的（DTensor）

# 2. 如果配置为 FULL_STATE_DICT：
#    - 内部调用 FSDP.summon_full_params()
#    - All-Gather 聚合所有分片到每个 GPU
#    - 临时显存开销：16GB（完整参数）
#    - 返回后立即释放临时参数


# ========== verl 使用 SHARDED_STATE_DICT 的原因 ==========
# 1. 节省显存：不在 state_dict() 阶段触发 all-gather
# 2. 延迟聚合：在后续的 full_tensor() 调用时才触发 all-gather
# 3. 灵活性：可以选择性地聚合某些层（layered_summon）
```

### 6.3 DTensor.full_tensor() - All-Gather 的触发点

**代码位置**：`verl/workers/fsdp_workers.py:708-714`

```python
# 文件: verl/workers/fsdp_workers.py:708-714

if fsdp_version(self.actor_module_fsdp) == 2:  # FSDP2
    device = get_device_id()
    per_tensor_param = (
        (name, param.to(device, non_blocking=True).full_tensor()
         if isinstance(param, DTensor) else param)  # ← 这里触发 all-gather
        for name, param in params.items()
    )
else:  # FSDP1
    per_tensor_param = params.items()


# ========== DTensor.full_tensor() 的作用 ==========
# DTensor 是分片的 tensor，每个 GPU 只有一部分

# 例如：一个 [8192, 8192] 的权重矩阵
# - GPU 0: DTensor，本地分片 [1024, 8192] (2GB / 8)
# - GPU 1: DTensor，本地分片 [1024, 8192]
# - ...
# - GPU 7: DTensor，本地分片 [1024, 8192]

# 调用 full_tensor() 后：
# 1. 触发 NCCL all-gather 操作
# 2. 每个 GPU 收集所有分片
# 3. 返回完整的 tensor [8192, 8192] (16GB / 模型层数)

# 通信模式（All-Gather）：
#     GPU 0       GPU 1       GPU 2   ...   GPU 7
#       ↓           ↓           ↓             ↓
#    [shard0]    [shard1]    [shard2]      [shard7]
#       ↓           ↓           ↓             ↓
#    ┌───────────────────────────────────────────┐
#    │         NCCL All-Gather Ring              │
#    │   每个 GPU 广播自己的分片到所有 GPU        │
#    └───────────────────────────────────────────┘
#       ↓           ↓           ↓             ↓
#    [full]      [full]      [full]        [full]
#    16GB        16GB        16GB          16GB

# 通信量计算：
# - 每个 GPU 发送：2GB (自己的分片)
# - 每个 GPU 接收：14GB (其他 7 个 GPU 的分片)
# - 总通信量（per GPU）：2GB + 14GB = 16GB
# - 全局总通信量：8 * 14GB = 112GB
# - 通信时间：~200ms（NVLink 带宽 600GB/s）
```

### 6.4 rollout.update_weights() - In-Place 参数更新

**代码位置**：`verl/workers/fsdp_workers.py:728`

```python
# 文件: verl/workers/fsdp_workers.py:728

await self.rollout.update_weights(per_tensor_param, peft_config=peft_config)

# ========== rollout.update_weights() 内部（vLLM/SGLang）==========
# 伪代码（基于 vLLM 实现）：

async def update_weights(self, per_tensor_param, peft_config=None):
    """更新 Rollout 模型的参数"""

    # 1. 遍历所有参数
    for name, new_param in per_tensor_param:
        # new_param: [完整的 8B 参数中的某一层]
        # 例如: 'model.layers.0.self_attn.q_proj.weight' -> [4096, 4096] (32MB)

        # 2. 获取 Rollout 模型中对应的参数
        old_param = self.model.get_parameter(name)  # 也是 [4096, 4096]

        # 3. 直接替换（in-place 更新）
        old_param.data.copy_(new_param.data)
        # → GPU 上的内存复制（cudaMemcpy）
        # → 不需要额外显存（in-place）
        # → 耗时：~100ms（复制 16GB 数据）

    # 4. 更新完成后，释放临时参数
    del per_tensor_param
    torch.cuda.empty_cache()


# ========== 为什么是 in-place 更新？ ==========
# 1. 节省显存：不需要分配新的 16GB 空间
# 2. 高效：直接在 GPU 上复制，无需 CPU 中转
# 3. 保持 Rollout Engine 状态：KV Cache、PageTable 等不受影响
```

### 6.5 优化：分层同步（Layered Summon）

**代码位置**：`verl/utils/fsdp_utils.py:569-608`

对于大模型，可以逐层同步参数，减少显存峰值：

```python
# 文件: verl/utils/fsdp_utils.py:569-608

def layered_summon_lora_params(fsdp_module) -> OrderedDict:
    """逐层聚合参数，减少显存峰值"""
    lora_params = OrderedDict()

    # 遍历每一层（如 32 层 Transformer）
    prefix_list = [
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        # ...
    ]

    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            # 只聚合当前层的参数
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=submodule.state_dict())
                    sub_lora_params = {
                        f"{prefix}.{name}": param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in sub_lora_params.items()
                    }
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                # → 显存峰值：16GB / 32 = 0.5GB（单层）

            # 当前层处理完后立即释放
            get_torch_device().empty_cache()
            # → 显存立即恢复

    return lora_params

# 优势：
# - 不分层：峰值 16GB（所有层一次性聚合）
# - 分层：峰值 0.5GB（每次只聚合一层）
# - 代价：通信次数增加（32 次 vs 1 次）
```

---

## 7. 显存管理时间线

```
┌────────────────────────────────────────────────────────────────────┐
│               PPO 单次迭代的显存变化（GPU 0）                      │
└────────────────────────────────────────────────────────────────────┘

时刻 0s: Trainer Mode (初始)
├── Actor FSDP 分片: 2GB
├── Actor 优化器: 4GB (或 CPU offload: 0GB)
├── Rollout 参数: 16GB
└── 总计: 22GB (或 18GB)

时刻 0-1s: generate_sequences() → rollout_mode()
├── load_fsdp_model_to_gpu: 22GB → 24GB (+2GB, 如果之前 offload)
├── params.full_tensor(): 24GB → 40GB (+16GB, 临时)
├── update_weights: 40GB → 40GB (in-place)
├── 释放临时参数: 40GB → 24GB (-16GB)
├── resume KV Cache: 24GB → 56GB (+32GB)
└── 总计: 56GB ← 全局峰值

时刻 1-2s: Rollout 推理
├── KV Cache 动态使用: 56GB → 50GB (部分使用)
└── 总计: ~50GB

时刻 2s: trainer_mode()
├── release KV Cache: 50GB → 18GB (-32GB)
└── 总计: 18GB

时刻 2-3s: compute_log_prob()
├── load_fsdp_model_to_gpu: 18GB → 20GB (+2GB, 如果之前 offload)
├── Forward Pass 激活: 20GB → 24GB (+4GB)
├── offload_fsdp_model_to_cpu: 24GB → 22GB (-2GB, 如果配置了)
└── 总计: 22GB

时刻 3-6s: update_actor()
├── load_fsdp_model_to_gpu: 22GB → 24GB (+2GB)
├── load_fsdp_optimizer: 24GB → 28GB (+4GB)
├── Forward + Backward: 28GB → 38GB (+10GB, 激活 + 梯度)
├── Optimizer Step: 38GB → 38GB (in-place)
├── offload: 38GB → 22GB (-16GB)
└── 总计: 22GB

时刻 6s: 迭代结束，回到初始状态
└── 总计: 22GB

┌────────────────────────────────────────────────────────────────────┐
│                        峰值显存分析                                │
└────────────────────────────────────────────────────────────────────┘

全局峰值: 56GB (Rollout Mode + KV Cache)
  ├── rollout_mode() 中的 params.full_tensor(): 40GB
  ├── rollout_mode() 中的 resume KV Cache: 56GB ← 全局最大
  └── update_actor() 中的 Forward + Backward: 38GB

优化方向:
1. 启用 Parameter Offload: 减少 Actor 参数显存（-2GB）
2. 启用 Optimizer Offload: 减少优化器显存（-4GB）
3. 减少 KV Cache: 调整 gpu_memory_utilization（-10GB）
4. 使用 Layered Summon: 减少参数同步峰值（-10GB）
5. 减少 batch size: 减少激活值显存（-4GB）
```

---

## 8. 代码位置索引

### 8.1 资源初始化

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| 创建资源池 | `verl/trainer/main_ppo.py` | 166-189 |
| 初始化 workers | `verl/trainer/ppo/ray_trainer.py` | 661-796 |
| 创建融合 Worker 类 | `verl/single_controller/ray/base.py` | 749-790 |
| 创建 RayWorkerGroup | `verl/single_controller/ray/base.py` | 361-444 |
| Spawn 机制 | `verl/single_controller/ray/base.py` | 478-512 |

### 8.2 模型初始化

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| **`__init__()` Worker 初始化** | `verl/workers/fsdp_workers.py` | **139-263** |
| `init_model()` 入口 | `verl/workers/fsdp_workers.py` | 760-820 |
| **`_build_model_optimizer()` FSDP 构建** | `verl/workers/fsdp_workers.py` | **268-589** |
| - 加载 Tokenizer | `verl/workers/fsdp_workers.py` | 304-311 |
| - 确定模型 dtype | `verl/workers/fsdp_workers.py` | 314-320 |
| - 加载模型配置 | `verl/workers/fsdp_workers.py` | 323-346 |
| - 初始化模型（from_pretrained） | `verl/workers/fsdp_workers.py` | 348-389 |
| - 应用优化（Liger/Gradient Checkpointing/LoRA） | `verl/workers/fsdp_workers.py` | 392-432 |
| **- FSDP1 初始化（关键）** | `verl/workers/fsdp_workers.py` | **497-507** |
| **- FSDP2 初始化（关键）** | `verl/workers/fsdp_workers.py` | **530-532** |
| - 创建优化器 | `verl/workers/fsdp_workers.py` | 542-589 |
| **`_build_rollout()` Rollout 构建** | `verl/workers/fsdp_workers.py` | **591-630** |
| 配置 state_dict 类型 | `verl/workers/fsdp_workers.py` | 634-644 |
| **初始化后切换到 Trainer Mode** | `verl/workers/fsdp_workers.py` | **650-656** |

### 8.3 训练循环

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| PPO 训练循环 | `verl/trainer/ppo/ray_trainer.py` | 962-1259 |
| 生成序列（调用） | `verl/trainer/ppo/ray_trainer.py` | 1042-1050 |
| 计算 Reward | `verl/trainer/ppo/ray_trainer.py` | 1088-1098 |
| 计算 Old Log Prob（调用） | `verl/trainer/ppo/ray_trainer.py` | 1100-1110 |
| 计算 Advantage | `verl/trainer/ppo/ray_trainer.py` | 1132-1164 |
| 更新 Actor（调用） | `verl/trainer/ppo/ray_trainer.py` | 1174-1180 |

### 8.4 模式切换（核心）

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| **`rollout_mode()` 定义** | `verl/workers/fsdp_workers.py` | **658-739** |
| **`trainer_mode()` 定义** | `verl/workers/fsdp_workers.py` | **741-757** |
| `generate_sequences()` - 切换到 Rollout | `verl/workers/fsdp_workers.py` | **945-950** |
| `generate_sequences()` - 切换回 Trainer | `verl/workers/fsdp_workers.py` | **960-964** |

### 8.5 参数同步

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| 收集 FSDP 参数 | `verl/workers/fsdp_workers.py` | 679 |
| 转换为 full_tensor | `verl/workers/fsdp_workers.py` | 708-714 |
| 更新 Rollout 参数 | `verl/workers/fsdp_workers.py` | 728 |
| 分层同步（Layered Summon） | `verl/utils/fsdp_utils.py` | 569-608 |
| 收集 LoRA 参数 | `verl/utils/fsdp_utils.py` | 611-650 |

### 8.6 Worker 方法实现

| 功能 | 文件路径 | 行号 |
|------|---------|------|
| `generate_sequences()` | `verl/workers/fsdp_workers.py` | 927-984 |
| `compute_log_prob()` | `verl/workers/fsdp_workers.py` | 986-1026 |
| `update_actor()` | `verl/workers/fsdp_workers.py` | 877-923 |

---

## 9. 性能优化建议

### 9.1 显存优化

| 优化方法 | 节省显存 | 代价 | 配置参数 |
|---------|---------|------|---------|
| Parameter Offload | 2GB | CPU↔GPU 传输延迟 ~50ms | `actor.model.enable_parameter_offload=True` |
| Optimizer Offload | 4GB | CPU↔GPU 传输延迟 ~100ms | `actor.optimizer.enable_optimizer_offload=True` |
| 减少 gpu_memory_utilization | 10-20GB | KV Cache 容量减小，batch size 受限 | `rollout.gpu_memory_utilization=0.4` |
| Layered Summon (LoRA) | 10-15GB | 增加通信次数（32x） | `rollout.layered_summon=True` |
| free_cache_engine=True | 32GB | 每次重新分配 KV Cache 开销 | `rollout.free_cache_engine=True` |
| 减少 rollout batch size | 5-10GB | 推理吞吐量降低 | `rollout.rollout_batch_size=128` |

### 9.2 通信优化

| 优化方法 | 节省时间 | 适用场景 |
|---------|---------|---------|
| 使用 NVLink/NVSwitch | ~50% | 多 GPU 同机训练 |
| 启用 NCCL 压缩 | ~20% | 低带宽网络 |
| 减少参数同步频率 | ~30% | 小步长更新 |
| Gradient Checkpointing | ~10% | 减少激活值显存，间接减少通信 |

### 9.3 推理优化

| 优化方法 | 提升吞吐 | 配置参数 |
|---------|---------|---------|
| 增加 KV Cache | ~50% | `rollout.gpu_memory_utilization=0.8` |
| 使用 PagedAttention | ~2x | vLLM/SGLang 默认启用 |
| 启用 Continuous Batching | ~3x | vLLM/SGLang 默认启用 |
| 使用 FP8 量化 | ~2x | `rollout.quantization=fp8` |

### 9.4 训练优化

| 优化方法 | 提升速度 | 配置参数 |
|---------|---------|---------|
| Gradient Accumulation | ~30% | `actor.gradient_accumulation_steps=4` |
| Mixed Precision (BF16) | ~50% | `actor.mixed_precision=bf16` |
| Flash Attention | ~2x | `actor.use_flash_attention=True` |
| Activation Checkpointing | ~20% | `actor.enable_gradient_checkpointing=True` |

---

## 10. 常见问题 (FAQ)

### Q1: 为什么初始化后要切换到 Trainer Mode？

**A**: 根据代码注释（`verl/workers/fsdp_workers.py:651`）：
> It's critical that hybrid engine in trainer mode initially to load checkpoint.

原因：
1. **加载 checkpoint** 时需要 Actor FSDP 在 GPU（用于加载参数）
2. Rollout KV Cache 在初始化时会占用大量显存（32GB），需要释放
3. 训练开始前可能需要执行验证（validation），需要 Trainer Mode

### Q2: 为什么每次 Rollout 都要同步参数，不能只同步一次？

**A**: 因为 Actor 模型在每次训练迭代后都会更新参数（`update_actor()`）。如果不同步，Rollout 会一直使用初始参数 W0，而不是最新参数 W_t，导致 PPO 的 importance sampling 失效，训练发散。

### Q3: FSDP 的 state_dict() 为什么配置为 SHARDED_STATE_DICT？

**A**:
1. **节省显存**：不在 `state_dict()` 阶段触发 all-gather
2. **延迟聚合**：在后续的 `full_tensor()` 调用时才触发 all-gather
3. **灵活性**：可以选择性地聚合某些层（layered_summon）

如果配置为 `FULL_STATE_DICT`，会在 `state_dict()` 时就触发 all-gather，峰值显存更高。

### Q4: rollout.update_weights() 为什么是 in-place 更新？

**A**:
1. **节省显存**：不需要分配新的 16GB 空间
2. **高效**：直接在 GPU 上复制（cudaMemcpy），无需 CPU 中转
3. **保持 Rollout Engine 状态**：KV Cache、PageTable 等不受影响

### Q5: 如何减少 rollout_mode() 的峰值显存？

**A**:
1. 启用 Parameter Offload：Actor FSDP offload 到 CPU（-2GB）
2. 使用 Layered Summon：逐层聚合参数（峰值从 40GB 降到 ~2GB）
3. 减少 KV Cache：调整 `gpu_memory_utilization`（-10-20GB）

### Q6: Spawn 机制的本质是什么？

**A**: Spawn 创建了多个"视图"（RayWorkerGroup 对象），它们共享相同的底层 Ray Actors（物理进程），但只暴露特定前缀的方法，实现了：
1. **资源复用**：8 个 GPU 而不是 16 个 GPU
2. **接口隔离**：逻辑上分离 Actor 和 Rollout 功能
3. **零额外开销**：只是 Python 对象层面的引用

### Q7: init 时显存中是否同时存在两份模型权重？

**A**: **是的**！在 FSDP 初始化过程中，显存中短暂存在两份权重：

**FSDP1 初始化**（`verl/workers/fsdp_workers.py:497-507`）：
```
1. from_pretrained() → 加载完整模型（32GB fp32）
2. FSDP() 调用 → 创建分片模型（2GB bf16）
   ├── 此时峰值：32GB + 2GB = 34GB
3. FSDP 内部释放完整模型 → 仅保留分片（2GB）
```

**FSDP2 初始化**（`verl/workers/fsdp_workers.py:530-532`）：
```
1. from_pretrained() → 加载完整模型（32GB fp32）
2. state_dict() → 复制完整参数（32GB）
   ├── 此时峰值：32GB + 32GB = 64GB
3. apply_fsdp2() → 应用分片
4. 释放原模型和 state_dict → 仅保留分片（2GB）
```

**总结**：
- 峰值显存：34GB (FSDP1) 或 64GB (FSDP2)
- 稳定显存：2GB (分片) + 4GB (优化器) = 6GB
- 这是 FSDP 初始化的必要开销，**只在初始化时发生一次**

### Q8: FSDP 初始化在哪里？后续每次 rollout 完都会重新初始化 FSDP 吗？

**A**: **不会**！FSDP 只在 `init_model()` 时初始化一次。

**FSDP 初始化位置**：
- `verl/workers/fsdp_workers.py:497-507` (FSDP1)
- `verl/workers/fsdp_workers.py:530-532` (FSDP2)
- 调用时机：Ray Trainer 启动时调用 `init_model()`（**只执行一次**）

**后续训练流程**：
```
初始化阶段（只执行一次）：
├── __init__() → 初始化分布式环境
├── init_model() → 加载模型
│   ├── _build_model_optimizer() → FSDP 初始化 ← 只执行一次
│   └── _build_rollout() → Rollout 初始化 ← 只执行一次
└── trainer_mode() → 切换到 Trainer Mode

训练迭代（重复执行）：
├── generate_sequences()
│   ├── rollout_mode() → 同步参数到 Rollout（不重新初始化 FSDP）
│   ├── Rollout 推理
│   └── trainer_mode() → 切换回 Trainer Mode
├── compute_log_prob() → 使用已有的 FSDP 模型
├── update_actor() → 更新已有的 FSDP 参数
└── 下一轮迭代...
```

**关键点**：
1. **FSDP 只初始化一次**（在 `init_model()` 中）
2. **参数同步不是重新初始化**：
   - `rollout_mode()` 只是将 FSDP 的**当前参数**复制到 Rollout
   - 不会重新调用 `FSDP()` 或 `apply_fsdp2()`
3. **模式切换不改变 FSDP 结构**：
   - `rollout_mode()` 和 `trainer_mode()` 只控制显存分配（KV Cache）
   - FSDP 模型结构保持不变

### Q9: Actor FSDP 模型和 Rollout 模型的参数是独立的吗？

**A**: **是的**，它们是两份独立的权重：

**物理存储**：
```
GPU 0:
├── Actor FSDP 参数分片: 2GB (bf16)
│   └── 存储：GPU 显存
│   └── 用途：训练（梯度计算、参数更新）
│
├── Rollout 参数（完整）: 16GB (bf16)
    └── 存储：GPU 显存
    └── 用途：推理（生成序列）
```

**为什么需要两份权重**：
1. **FSDP 模型是分片的**：
   - 每个 GPU 只有 1/8 的参数（2GB）
   - 无法直接用于推理（缺少其他 GPU 的参数）
   - 训练时通过 All-Gather 临时聚合完整参数

2. **Rollout 模型是完整的**：
   - 每个 GPU 有完整的 8B 参数（16GB）
   - 推理时可以独立工作，无需跨 GPU 通信
   - vLLM/SGLang 需要完整模型才能高效推理

**参数同步流程**：
```
训练更新 → Actor FSDP 参数改变（2GB 分片）
   ↓
rollout_mode() 调用
   ↓
1. full_tensor() → All-Gather 聚合所有 GPU 的分片（临时 16GB）
2. update_weights() → 复制到 Rollout 模型（16GB）
3. 释放临时聚合的参数
   ↓
Rollout 模型参数已更新（16GB）
```

**总结**：
- 两份权重独立存储（2GB FSDP 分片 + 16GB Rollout 完整）
- 通过参数同步保持一致（每次 rollout 前同步）
- 总显存：2GB + 16GB = 18GB（不包括优化器和 KV Cache）

---

## 11. 总结

### 核心要点

1. **Hybrid Engine 架构**：
   - 在同一组 GPU 上共置 Actor FSDP 和 Rollout 模型
   - 通过动态切换 Trainer Mode 和 Rollout Mode 复用显存
   - 节省 50% GPU 资源

2. **模式切换机制**：
   - **初始化后**：系统处于 **Trainer Mode**（`verl/workers/fsdp_workers.py:650-656`）
   - **每次 generate_sequences()**：
     - 开始时调用 `rollout_mode()`（`verl/workers/fsdp_workers.py:945-950`）
     - 结束时调用 `trainer_mode()`（`verl/workers/fsdp_workers.py:960-964`）
   - **compute_log_prob() 和 update_actor()**：无需切换（已在 Trainer Mode）

3. **参数同步流程**：
   - Actor FSDP `state_dict()` → 返回分片参数（DTensor）
   - `full_tensor()` → All-Gather 聚合为完整参数（NCCL 通信 ~200ms）
   - `rollout.update_weights()` → In-Place 复制到 Rollout 模型（~100ms）
   - **必须每次同步**：因为 Actor 参数在每次训练后都更新

4. **显存峰值管理**：
   - 全局峰值：56GB（Rollout Mode + KV Cache）
   - 优化方向：Parameter/Optimizer Offload、Layered Summon、减少 KV Cache

5. **Spawn 共置机制**：
   - 物理层面：8 个 Ray Actors（共享）
   - 逻辑层面：多个 RayWorkerGroup 视图（隔离）
   - 目的：避免为每个功能创建独立 GPU 进程

---

**文档版本**: v1.0
**最后更新**: 2025-01-XX
**适用 verl 版本**: v0.1.0+