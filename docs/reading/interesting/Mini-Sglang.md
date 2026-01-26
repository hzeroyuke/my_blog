
## 1. Overview

以下是官方对于Mini-SGlang的架构介绍

The source code is located in `python/minisgl`. Here is a breakdown of the modules for developers:

- `minisgl.core`: Provides core dataclasses `Req` and `Batch` representing the state of requests, class `Context` which holds the global state of the inference context, and class `SamplingParams` holds the sampling parameters provided by users.
- `minisgl.distributed`: Provides the interface to all-reduce and all-gather in tensor parallelism, and dataclass `DistributedInfo` which holds the TP information for a TP worker.
- `minisgl.layers`: Implements basic building blocks for building LLMs with TP support, including linear, layernorm, embedding, RoPE, etc. They share common base classes defined in `minisgl.layers.base`.
- `minisgl.models`: Implements LLM models, including Llama and Qwen3. Also defines utilities for loading weights from huggingface and sharding weights.
- `minisgl.attention`: Provides interface of attention Backends and implements backends of `flashattention` and `flashinfer`. They are called by `AttentionLayer` and uses metadata stored in `Context`.
- `minisgl.kvcache`: Provides interface of KVCache pool and KVCache manager, and implements `MHAKVCache`, `NaiveCacheManager` and `RadixCacheManager`.
- `minisgl.utils`: Provides a collection of utilities, including logger setup and wrappers around zmq.
- `minisgl.engine`: Implements `Engine` class, which is a TP worker on a single process. It manages the model, context, KVCache, attention backend and cuda graph replaying.
- `minisgl.message`: Defines messages exchanged (in zmq) between api_server, tokenizer, detokenizer and scheduler. All message types support automatic serialization and deserialization.
- `minisgl.scheduler`: Implements `Scheduler` class, which runs on each TP worker process and manages the corresponding `Engine`. The rank 0 scheduler receives msgs from tokenizer, communicates with scheduler on other TP workers, and sends msgs to detokenizer.
- `minisgl.server`: Defines cli arguments and `launch_server` which starts all the subprocesses of Mini-SGLang. Also implements a FastAPI server in `minisgl.server.api_server` acting as a frontend, providing endpoints such as `/v1/chat/completions`.
- `minisgl.tokenizer`: Implements `tokenize_worker` function which handles tokenization and detokenization requests.
- `minisgl.llm`: Provides class `LLM` as a python interface to interact with the Mini-SGLang system easily.
- `minisgl.kernel`: Implements custom CUDA kernels, supported by `tvm-ffi` for python binding and jit interface.
- `minisgl.benchmark`: Benchmark utilities.

## 2. RadixAttention

相对于vllm中的PagedAttention机制，sglang中的显存管理方案为RadixAttention，是基于前缀技术的显存管理方案。对于LLM而言的大量请求都共享相同的Prefix，例如system prompt和多轮的对话

sglang将kv cache将kv cache组织成一颗radix tree。当新的请求到达的时候，系统在radix tree匹配最长前缀，匹配成功之后复用一用KV cache

![](asset/Pasted%20image%2020260123154425.png)