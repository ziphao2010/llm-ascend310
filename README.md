# llm-ascend310

在 **Huawei Ascend 310** 上运行 LLaMA 架构大语言模型的高性能推理引擎。

## 特点

- 🚀 **纯 NPU 推理** — 所有算子跑在 Ascend 310 Cube Unit，CPU 仅做数据搬运
- 🔌 **通用架构** — 自动适配任意 LLaMA 模型（从 config.json 读取参数）
- ⚡ **高性能** — MiniCPM5-1B 达 **3.3 tok/s**（单芯片），4 芯片并发 **13 tok/s**
- 📏 **长上下文** — 原生支持 128K 上下文（MiniCPM5-1B），KV Cache 高效管理
- 🔗 **OpenAI 兼容** — 完整 `/v1/chat/completions` + `/v1/completions` + SSE 流式
- 🔀 **多实例** — 4 芯片各自独立运行模型实例，Round-robin 负载均衡

## 快速开始

### 环境要求

```bash
# 硬件: Atlas 300I 3010 (4×Ascend 310)
# 软件: CANN 7.0.0 + 驱动 24.1.1.3
# 系统: Ubuntu 22.04
```

### 安装

```bash
# 1. 设置 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=Ascend310

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载模型（以 MiniCPM5-1B 为例）
python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/MiniCPM5-1B', local_dir='/root/models/MiniCPM5-1B')"
```

### 编译算子

```bash
python -m engine.compile --model /root/models/MiniCPM5-1B --output ./om_models --soc Ascend310
```

### 启动服务器

```bash
# 方式一：一键脚本
bash scripts/deploy.sh start

# 方式二：手动启动
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/7.0.0/lib64:/usr/local/Ascend/driver/lib64
export PYTHONPATH=/root/llm-ascend310:$PYTHONPATH
python server/api.py
```

### 调用

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer wsh101007" \
  -H "Content-Type: application/json" \
  -d '{"model":"minicpm1","messages":[{"role":"user","content":"你好！"}],"max_tokens":64}'
```

Python OpenAI 客户端:

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.1.199:8000/v1", api_key="wsh101007")
resp = client.chat.completions.create(
    model="minicpm1",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(resp.choices[0].message.content)
```

## 支持的模型

| 模型 | 参数量 | 架构 | 状态 |
|:----|:-----:|:----|:----:|
| MiniCPM5-1B | 1.08B | LLaMA (24层, 1536H, 16Q/2KV) | ✅ 已优化 |
| Qwen2.5-1.5B | 1.54B | LLaMA | 🟡 算子就绪 |
| SmolLM2-1.7B | 1.7B | LLaMA | 🟡 计划中 |

## 性能

### MiniCPM5-1B @ Ascend 310 × 1

| 指标 | 值 |
|:----|:---:|
| 单token推理 | **0.3s** (3.3 tok/s) |
| Prefill (100 token) | ~2.8s |
| 权重加载 | ~2s (234张量) |
| 峰值显存 | ~2.8GB / 8GB |
| 上下文 | 131072 token |

### 4芯片并发

| 指标 | 值 |
|:----|:---:|
| 总吞吐 | **~13 tok/s** |
| 负载均衡 | Round-robin |
| 实例隔离 | 完全独立（无D2D同步） |

## 项目结构

```
llm-ascend310/
├── engine/                  ← 核心引擎
│   ├── base.py              ← ACL封装 (Device, malloc/free/exec)
│   ├── model_loader.py      ← 通用模型加载器
│   ├── llama_engine.py      ← LLaMA前向引擎
│   └── compile.py           ← ATC算子编译器
├── models/                  ← 模型配置
│   └── minicpm5-1b/         ← MiniCPM5-1B 特化
├── server/
│   └── api.py               ← OpenAI API (4实例)
├── scripts/
│   ├── deploy.sh            ← 一键部署
│   └── compile_ops.sh       ← 算子编译
├── tests/
│   ├── test_forward.py      ← 前向正确性测试
│   ├── test_quality.py      ← 质量/采样测试
│   └── benchmark.py         ← 性能压测
├── om_models/               ← ATC编译的 .om 算子
└── requirements.txt
```

## 算子列表

| 算子 | 形状 | 用途 |
|:----|:----|:-----|
| `mm_1_1536_2048` | [1,1536]×[1536,2048] | Q投影 |
| `mm_1_1536_256` | [1,1536]×[1536,256] | K/V投影 |
| `mm_1_1536_1536` | [1,1536]×[1536,1536] | O投影、z门控 |
| `mm_1_1536_4608` | [1,1536]×[1536,4608] | MLP gate/up |
| `mm_1_2304_1536` | [1,2304]×[2304,1536] | MLP down (拆分) |
| `ops_silu_{n}` | [n]→[n] | SiLU激活 |
| `ops_add` | [1,n]→[1,n] | 残差连接 |
| `ops_mul` | [1,n]→[1,n] | 逐元素乘 |

## 故障排查

**libascendcl.so not found**: 设置 LD_LIBRARY_PATH 指向 CANN 安装目录。

**ATC 编译失败**: 检查 SOC_VERSION=Ascend310，确认 ONNX opset=13。

**算子返回 507011**: 跨芯片显存访问错误，调用 `aclrtSetDevice` 后再 malloc。

**模型输出多语言混杂**: MiniCPM5-1B 以中文训练为主，英文能力有限。尝试中文 prompt。

## License

Apache 2.0
