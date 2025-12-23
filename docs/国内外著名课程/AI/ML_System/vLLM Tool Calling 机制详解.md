
## 📋 目录

1. [核心概念](#核心概念)
2. [工作流程](#工作流程)
3. [DeepSeek V3.1 完整示例](#deepseek-v31-完整示例)
4. [Llama 3.1 完整示例](#llama-31-完整示例)
5. [两者对比](#两者对比)
6. [源码解析](#源码解析)
7. [实战指南](#实战指南)

---

## 核心概念

### ❓ 问题：为什么需要 vLLM 的 Tool Calling 机制？

很多模型（如 DeepSeek V3.1、Llama 3.1）的**原生 `chat_template` 并不支持处理 `tools` 参数**。即使模型在预训练时学会了 function calling 的格式（special tokens），tokenizer 的模板也无法自动将工具描述注入到 prompt 中。

### ✅ vLLM 的解决方案

vLLM 通过以下三个核心组件实现完整的 tool calling 支持：

1. **自定义 Chat Template** - 负责将工具描述注入到 prompt
2. **ToolParser** - 负责解析模型输出中的工具调用
3. **服务层集成** - 将上述组件无缝整合到 OpenAI 兼容的 API 中

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 用户请求（含 messages + tools）                                │
└───────────────┬─────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. vLLM 服务层                                                    │
│    - 选择对应的 chat_template（deepseekv31.jinja/llama3.1.jinja）│
│    - 调用 apply_chat_template(messages, tools=tools)            │
└───────────────┬─────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Chat Template 处理                                             │
│    - 检测到 tools 参数不为空                                      │
│    - 将工具描述格式化并注入到 system_prompt                        │
│    - 拼接完整的 prompt（含工具定义 + 用户消息）                    │
└───────────────┬─────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. 模型推理                                                       │
│    - 接收完整 prompt（已包含工具描述）                             │
│    - 生成输出（使用预训练时学习的 special tokens）                 │
└───────────────┬─────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. ToolParser 解析                                                │
│    - DeepSeekV31ToolParser: 解析 <｜tool▁call▁begin｜>... 格式   │
│    - Llama3JsonToolParser: 解析 {"name": "...", "parameters"...} │
│    - 提取结构化的工具调用信息                                      │
└───────────────┬─────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. 返回给用户                                                     │
│    - 标准 OpenAI 格式的 tool_calls                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## DeepSeek V3.1 完整示例

### 1. 用户请求

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
payload = {
    "model": "deepseek-ai/DeepSeek-V3.1",
    "messages": [
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ],
    "tool_choice": "auto"
}

response = requests.post(url, json=payload)
print(response.json())
```

### 2. Chat Template 处理

**文件**: `examples/tool_chat_template_deepseekv31.jinja`

**关键代码**（第 19-26 行）：

```jinja2
{% if tools is defined and tools is not none %}
  {% set tool_ns = namespace(text='## Tools\nYou have access to the following tools:\n') %}
  {% for tool in tools %}
    {% set tool_ns.text = tool_ns.text + '\n### ' + tool.function.name +
         '\nDescription: ' + tool.function.description +
         '\n\nParameters: ' + (tool.function.parameters | tojson) + '\n' %}
  {% endfor %}
  {% set tool_ns.text = tool_ns.text +
       "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n" +
       "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>" +
       "tool_call_arguments<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n..." %}

  {% set ns.system_prompt = ns.system_prompt + '\n\n' + tool_ns.text %}
{% endif %}
```

**生成的工具描述**（注入到 system_prompt）：

```
## Tools
You have access to the following tools:

### get_weather
Description: 获取指定城市的天气信息

Parameters: {"type":"object","properties":{"city":{"type":"string","description":"城市名称"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"温度单位"}},"required":["city"]}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜><｜tool▁calls▁end｜>

Where:

- `tool_call_name` must be an exact match to one of the available tools
- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
- For multiple tool calls, chain them directly without separators or spaces
```

### 3. 完整 Prompt（发送给模型）

```
<｜begin▁of▁sentence｜>## Tools
You have access to the following tools:

### get_weather
Description: 获取指定城市的天气信息

Parameters: {"type":"object","properties":{"city":{"type":"string","description":"城市名称"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"温度单位"}},"required":["city"]}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜><｜tool▁calls▁end｜>

<｜User｜>北京今天天气怎么样？<｜Assistant｜></think>
```

### 4. 模型输出

```
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city":"北京","unit":"celsius"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>
```

### 5. ToolParser 解析

**文件**: `vllm/entrypoints/openai/tool_parsers/deepseekv31_tool_parser.py`

**关键代码**（第 78-120 行）：

```python
def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
    # 检查是否包含工具调用标记
    if self.tool_calls_start_token not in model_output:
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    # 使用正则表达式提取工具调用
    # 正则: <｜tool▁call▁begin｜>(?P<function_name>.*?)<｜tool▁sep｜>(?P<function_arguments>.*?)<｜tool▁call▁end｜>
    function_call_tuples = self.tool_call_regex.findall(model_output)

    tool_calls = []
    for match in function_call_tuples:
        function_name, function_args = match
        tool_calls.append(
            ToolCall(
                type="function",
                function=FunctionCall(
                    name=function_name,
                    arguments=function_args  # 保持为 JSON 字符串
                ),
            )
        )

    # 提取工具调用前的文本内容
    content = model_output[:model_output.find(self.tool_calls_start_token)]

    return ExtractedToolCallInformation(
        tools_called=True,
        tool_calls=tool_calls,
        content=content if content else None,
    )
```

**解析结果**：

```python
ExtractedToolCallInformation(
    tools_called=True,
    tool_calls=[
        ToolCall(
            type="function",
            function=FunctionCall(
                name="get_weather",
                arguments='{"city":"北京","unit":"celsius"}'
            )
        )
    ],
    content=None
)
```

### 6. 返回给用户（OpenAI 格式）

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "deepseek-ai/DeepSeek-V3.1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"北京\",\"unit\":\"celsius\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

---

## Llama 3.1 完整示例

### 1. 用户请求

```python
payload = {
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
}
```

### 2. Chat Template 处理

**文件**: `examples/tool_chat_template_llama3.1_json.jinja`

**关键代码**（第 56-77 行）：

```jinja2
{%- if tools_in_user_message and not tools is none %}
    {#- 提取第一条用户消息 #}
    {%- if messages[0]['content'] is string %}
        {%- set first_user_message = messages[0]['content']|trim %}
    {%- else %}
        {%- set first_user_message = messages[0]['content'] | selectattr('type', 'equalto', 'text') | map(attribute='text') | map('trim') | join('\n') %}
    {%- endif %}
    {%- set messages = messages[1:] %}

    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. ' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}
```

**生成的工具描述**（注入到第一条用户消息）：

```
<|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string"
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "celsius",
                        "fahrenheit"
                    ]
                }
            },
            "required": [
                "city"
            ]
        }
    }
}

What's the weather like in San Francisco?<|eot_id|>
```

### 3. 完整 Prompt（发送给模型）

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question.<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}

What's the weather like in San Francisco?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### 4. 模型输出

```
{"name": "get_weather", "parameters": {"city": "San Francisco", "unit": "celsius"}}
```

### 5. ToolParser 解析

**文件**: `vllm/entrypoints/openai/tool_parsers/llama_tool_parser.py`

**关键代码**（第 69-125 行）：

```python
def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
    # 快速检查
    if not (self.bot_token in model_output or "{" in model_output):
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    # 使用正则查找 JSON 对象
    match = self.tool_call_regex.search(model_output)
    if not match:
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )

    try:
        json_str = match.group(0)
        # 支持分号分隔的多个 JSON
        json_objects = [obj.strip() for obj in json_str.split(";")]

        tool_calls = []
        for json_obj in json_objects:
            if not json_obj:
                continue
            obj = json.loads(json_obj)
            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=obj["name"],
                        arguments=json.dumps(
                            obj["arguments"] if "arguments" in obj else obj["parameters"],
                            ensure_ascii=False
                        )
                    )
                )
            )

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=None
        )

    except Exception:
        logger.exception("Error in extracting tool call from response.")
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )
```

**解析结果**：

```python
ExtractedToolCallInformation(
    tools_called=True,
    tool_calls=[
        ToolCall(
            type="function",
            function=FunctionCall(
                name="get_weather",
                arguments='{"city":"San Francisco","unit":"celsius"}'
            )
        )
    ],
    content=None
)
```

---

## 两者对比

| 特性 | DeepSeek V3.1 | Llama 3.1 |
|------|--------------|-----------|
| **工具描述注入位置** | System Prompt（第 25 行） | 第一条 User Message（第 77 行） |
| **工具定义格式** | Markdown 风格（## Tools, ### function_name） | JSON 格式（完整 tool schema） |
| **输出格式** | Special Tokens：`<｜tool▁call▁begin｜>name<｜tool▁sep｜>args<｜tool▁call▁end｜>` | Pure JSON：`{"name": "...", "parameters": {...}}` |
| **解析方式** | 正则提取 special token 包裹的内容 | 正则查找 JSON 对象 + JSON 解析 |
| **多工具调用** | 链式：`<｜tool▁call▁begin｜>...｜><｜tool▁call▁begin｜>...` | 分号分隔：`{...}; {...}` |
| **Template 文件** | `tool_chat_template_deepseekv31.jinja` | `tool_chat_template_llama3.1_json.jinja` |
| **Parser 类** | `DeepSeekV31ToolParser` | `Llama3JsonToolParser` |
| **启动参数** | `--tool-call-parser deepseek_v31` | `--tool-call-parser llama3_json` |

---

## 源码解析

### 1. 服务层集成

**文件**: `vllm/entrypoints/openai/serving_engine.py:1094-1108`

```python
# 检查是否需要解析工具调用
should_parse_tools = tool_parser is not None and (
    hasattr(request, "tool_choice") and request.tool_choice != "none"
)

if should_parse_tools:
    if not isinstance(request, ChatCompletionRequest):
        msg = "Tool usage is only supported for Chat Completions API"
        raise NotImplementedError(msg)

    # 调用 ToolParser 的 adjust_request() 方法
    # DeepSeek V3.1 没有重写此方法，直接返回原请求
    # 某些 parser 可能会修改请求（如添加额外的 system message）
    request = tool_parser(tokenizer).adjust_request(request=request)
```

### 2. Chat Template 应用

**文件**: `vllm/entrypoints/chat_utils.py:1525-1561`

```python
def apply_hf_chat_template(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    conversation: list[ConversationMessage],
    chat_template: str | None,
    tools: list[dict[str, Any]] | None,
    *,
    model_config: ModelConfig,
    **kwargs: Any,
) -> str:
    # 解析 chat template（可能是自定义的，也可能是 tokenizer 自带的）
    hf_chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        model_config=model_config,
    )

    if hf_chat_template is None:
        raise ValueError("Chat template is required")

    # 解析模板中使用的变量
    resolved_kwargs = resolve_chat_template_kwargs(
        tokenizer=tokenizer,
        chat_template=hf_chat_template,
        chat_template_kwargs=kwargs,
    )

    # 调用 tokenizer 的 apply_chat_template
    # tools 参数会被传入 Jinja2 模板中
    return tokenizer.apply_chat_template(
        conversation=conversation,
        tools=tools,  # ← 关键：传入 tools
        chat_template=hf_chat_template,
        tokenize=False,
        **resolved_kwargs,
    )
```

### 3. 输出解析（流式）

**文件**: `vllm/entrypoints/openai/serving_chat.py:974-988`

```python
# 在流式输出的每个 chunk 中调用
if tool_parser:
    delta_message = tool_parser.extract_tool_calls_streaming(
        previous_text=previous_text,
        current_text=current_text,
        delta_text=delta_text,
        previous_token_ids=previous_token_ids,
        current_token_ids=current_token_ids,
        delta_token_ids=delta_token_ids,
        request=request,
    )

    # delta_message 可能包含：
    # - content: 普通文本
    # - tool_calls: 工具调用的增量更新
    #   - index: 工具调用的索引（支持多个工具）
    #   - function.name: 首次出现时发送
    #   - function.arguments: 增量发送（如 "{\"city\"" -> ":\"Beijing\"}")
```

### 4. ToolParser 注册机制

**文件**: `vllm/entrypoints/openai/tool_parsers/abstract_tool_parser.py:86-155`

```python
class ToolParserManager:
    tool_parsers: dict[str, type] = {}

    @classmethod
    def register_module(cls, name: str | list[str] | None = None, ...):
        """装饰器，用于注册 ToolParser"""
        def _register(module):
            cls._register_module(module=module, module_name=name, ...)
            return module
        return _register

    @classmethod
    def get_tool_parser(cls, name) -> type:
        """根据名称获取 ToolParser 类"""
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]
        raise KeyError(f"tool helper: '{name}' not found")
```

**使用示例**：

```python
# 在 deepseekv31_tool_parser.py 中
@ToolParserManager.register_module("deepseek_v31")
class DeepSeekV31ToolParser(ToolParser):
    ...

# 在 llama_tool_parser.py 中
@ToolParserManager.register_module("llama3_json")
@ToolParserManager.register_module("llama4_json")
class Llama3JsonToolParser(ToolParser):
    ...
```

---

## 实战指南

### 启动 vLLM 服务器

#### DeepSeek V3.1

```bash
vllm serve deepseek-ai/DeepSeek-V3.1 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v31 \
  --chat-template examples/tool_chat_template_deepseekv31.jinja
```

#### Llama 3.1

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

### 发送请求

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.1",
    messages=[
        {"role": "user", "content": "What's the weather in Beijing?"}
    ],
    tools=tools,
    tool_choice="auto"
)

# 处理工具调用
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Tool: {tool_call.function.name}")
        print(f"Args: {tool_call.function.arguments}")
```

### 多轮对话（工具调用 + 工具输出 + 最终回答）

```python
# 第一轮：用户请求
messages = [
    {"role": "user", "content": "What's the weather in Tokyo?"}
]

response1 = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.1",
    messages=messages,
    tools=tools
)

# 提取工具调用
tool_call = response1.choices[0].message.tool_calls[0]
messages.append(response1.choices[0].message)

# 第二轮：返回工具执行结果
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": '{"temperature": 18, "condition": "sunny"}'
})

# 第三轮：获取最终回答
response2 = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.1",
    messages=messages,
    tools=tools
)

print(response2.choices[0].message.content)
# 输出: "The weather in Tokyo is currently sunny with a temperature of 18°C."
```

### 自定义 ToolParser

如果你想为自己的模型实现 tool calling，需要：

1. **创建自定义 chat template**（`.jinja` 文件）
2. **实现 ToolParser 子类**
3. **注册 ToolParser**

**示例**：

```python
# my_tool_parser.py
import regex as re
from vllm.entrypoints.openai.tool_parsers import (
    ToolParser, ToolParserManager
)

@ToolParserManager.register_module("my_custom_parser")
class MyCustomToolParser(ToolParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # 定义你的 special tokens
        self.tool_start = "<|tool|>"
        self.tool_end = "<|/tool|>"
        # 定义正则表达式
        self.regex = re.compile(r"<\|tool\|>(.*?)<\|/tool\|>", re.DOTALL)

    def extract_tool_calls(self, model_output, request):
        # 实现解析逻辑
        matches = self.regex.findall(model_output)
        # ... 处理并返回 ExtractedToolCallInformation

    def extract_tool_calls_streaming(self, ...):
        # 实现流式解析逻辑
        # ... 返回 DeltaMessage
```

**启动时指定**：

```bash
vllm serve my-model \
  --enable-auto-tool-choice \
  --tool-call-parser my_custom_parser \
  --tool-parser-plugin /path/to/my_tool_parser.py \
  --chat-template /path/to/my_template.jinja
```

---

## 总结

### 关键要点

1. ✅ **vLLM 通过自定义 chat template 注入工具描述**，而不是依赖模型原生 tokenizer 的模板
2. ✅ **不同模型使用不同的输出格式**：
   - DeepSeek：Special Tokens（`<｜tool▁call▁begin｜>...`）
   - Llama：Pure JSON（`{"name": "...", "parameters": {...}}`）
3. ✅ **ToolParser 负责解析模型输出**，提取结构化的工具调用信息
4. ✅ **完全兼容 OpenAI API**，客户端无需感知底层实现差异

### 核心流程回顾

```
用户 tools → Chat Template 注入 → 模型输出 → ToolParser 解析 → OpenAI 格式返回
```

### 参考资料

- vLLM 官方文档：https://docs.vllm.ai/en/stable/features/tool_calling.html
- 源码仓库：https://github.com/vllm-project/vllm
- DeepSeek V3.1 Template：`examples/tool_chat_template_deepseekv31.jinja`
- Llama 3.1 Template：`examples/tool_chat_template_llama3.1_json.jinja`
- ToolParser 实现：`vllm/entrypoints/openai/tool_parsers/`

---

**文档生成时间**: 2025-01-21
**vLLM 版本**: Based on latest main branch
**作者**: Claude Code Analysis
