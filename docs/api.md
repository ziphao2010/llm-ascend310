# llm-ascend310 — API Documentation

Base URL: `http://<your-server>:8000`
Auth: `Authorization: Bearer <LLM_API_KEY>`

## 端点

### `GET /health`

服务健康检查。

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model": "MiniCPM5-1B",
  "hardware": "Ascend310",
  "version": "1.0.0"
}
```

---

### `GET /v1/models`

列出可用模型。

```bash
curl -H "Authorization: Bearer $LLM_API_KEY" http://localhost:8000/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {"id": "minicpm1", "object": "model", "created": 1784297693, "owned_by": "empero-ai"},
    {"id": "minicpm2", "object": "model", ...},
    {"id": "minicpm3", "object": "model", ...},
    {"id": "minicpm4", "object": "model", ...}
  ]
}
```

---

### `POST /v1/chat/completions`

聊天补全（支持流式）。

#### 请求

```json
{
  "model": "minicpm1",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.1,
  "max_tokens": 256,
  "stream": false
}
```

| 参数 | 类型 | 默认 | 说明 |
|:----|:----|:----|:-----|
| model | str | minicpm1 | 模型实例 (minicpm1-4) |
| messages | array | — | 对话消息 |
| temperature | float | 0.1 | 采样温度 (0.01-1.0) |
| max_tokens | int | 256 | 最大生成长度 |
| stream | bool | false | SSE 流式输出 |
| seed | int | null | 随机种子 |

#### 非流式响应

```json
{
  "id": "chatcmpl-1784297693",
  "object": "chat.completion",
  "created": 1784297693,
  "model": "minicpm1",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "你好！"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 11,
    "completion_tokens": 4,
    "total_tokens": 15
  }
}
```

#### 流式响应 (SSE)

```
data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"choices":[{"delta":{"content":"你好"},"index":0}]}

data: {"choices":[{"delta":{"content":"！"},"index":0}]}

data: [DONE]
```

---

### `POST /v1/completions`

文本补全（非对话）。

```json
{
  "model": "minicpm1",
  "prompt": "The capital of France is",
  "temperature": 0.1,
  "max_tokens": 64
}
```

---

### `POST /v1/chat/completions` (流式)

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm1",
    "messages": [{"role": "user", "content": "讲个故事"}],
    "max_tokens": 256,
    "stream": true
  }'
```

## OpenAI 客户端兼容

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="http://your-server:8000/v1",
    api_key=os.environ["LLM_API_KEY"],
)

# Chat
resp = client.chat.completions.create(
    model="minicpm1",
    messages=[{"role": "user", "content": "你好！"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="minicpm2",  # 使用第二个实例
    messages=[{"role": "user", "content": "讲个笑话"}],
    max_tokens=128,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 模型路由

| model 参数 | 芯片 | 备注 |
|:----------|:----|:-----|
| minicpm1 | Chip 0 | 默认，省略 model 时使用 |
| minicpm2 | Chip 1 | — |
| minicpm3 | Chip 2 | — |
| minicpm4 | Chip 3 | 请求满时自动轮询 |

未指定 `model` 或指定错误名称时，自动 Round-robin 负载均衡。

## 环境变量

| 变量 | 必需 | 默认 | 说明 |
|:----|:----:|:----|:-----|
| LLM_API_KEY | ✅ | — | API 认证密钥 |
| LLM_MODEL_PATH | — | /root/models/MiniCPM5-1B | 模型路径 |
| LLM_MAX_CONTEXT | — | 32768 | 最大上下文长度 |
| LLM_INSTANCES | — | 4 | 模型实例数 |
| PORT | — | 8000 | 服务端口 |
