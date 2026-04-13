## OpenClaw

![](asset/Pasted%20image%2020260305203929.png)

上图是Openclaw项目的架构图，比较清晰易懂，通过一个统一网关暴露给用户，用户可以用各种形式访问这个网关，常见的方案是通过各种聊天App

- Pi Agent
	- 统一管理一批Code Agent，可以初始化它们，并且划分它们的工作范围
	- 同样支持skills
- Channel 各种通信渠道
	- Chat App
	- CLI

**Memory**

各种memory都用markdown存储，并且支持关键词查询和向量查询

**Long time task**

Pi Agent和Claude code一样，具备启动sub agent以及并发执行sub agent的能力，以及压缩上下文，用心跳机制检测后台的Agent任务等等

**Workflow**

这就是

![](asset/Pasted%20image%2020260305204956.png)


**Summary**

本质上来说，相对于我们之前看到的大量的Agent框架

- 构建大量Tools
	- Composio，MCP等等
	- Browser Use等重量级工具
- 构建复杂的Memory机制
	- MemOS，Mem0

OpenClaw等到了最重要的一个节点就是Code Agent能力的增强，不必堆叠的大量的工具，可以让模型自己写code构建工具。不必构建复杂的Memory机制，用简单的检索工具，就可以让模型自己去管理自己的记忆

但是面对要复杂验证的任务，OpenClaw仍然是无能为力，存在很多反代码反代理的任务，比如用户注册等等

## RL for OpenClaw

- RL Anything
- OpenClaw RL

现在对于OpenClaw的RLHF的工作有两个方向的发展

- 异步：将Agent的使用和训练解耦开，不能互相block
	- Agent Lightning：微软发布的工作（占坑的）解耦Agent的使用和训练
	- Areal
- RLHF for history：从用户的对话信息中获得监督信号，进行RLHF
	- Online RLHF