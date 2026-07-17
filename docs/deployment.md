# llm-ascend310 — Deployment Guide

## 环境

```bash
# 硬件
4× Ascend 310 (Atlas 300I 3010)
每芯片: 8GB HBM, 44 TFLOPS FP16
互联: PM8532 PCIe 3.0 交换机

# 软件
OS: Ubuntu 22.04
CANN: 7.0.0
驱动: 24.1.1.3
Python: 3.10+
```

## 部署

### 1. 下载模型

```bash
# HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openbmb/MiniCPM5-1B', local_dir='/root/models/MiniCPM5-1B')
"
```

### 2. 配置环境

```bash
# CANN 环境变量（必须）
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/7.0.0/lib64:\
/usr/local/Ascend/ascend-toolkit/7.0.0/lib64/plugin/opskernel:\
/usr/local/Ascend/ascend-toolkit/7.0.0/lib64/plugin/nnengine:\
/usr/local/Ascend/driver/lib64:\
/usr/local/Ascend/driver/lib64/common:\
/usr/local/Ascend/driver/lib64/driver

# 项目路径
export PYTHONPATH=/root/llm-ascend310:$PYTHONPATH

# API 密钥（必设，无默认值）
export LLM_API_KEY=your_secret_key_here

# 可选配置
export LLM_MAX_CONTEXT=32768   # 最大上下文 (默认 32768)
export LLM_INSTANCES=4         # 芯片实例数 (默认 4)
```

### 3. 编译算子

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/llm-ascend310
python -m engine.compile \
  --model /root/models/MiniCPM5-1B \
  --output ./om_models \
  --soc Ascend310
```

### 4. 启动

```bash
cd /root/llm-ascend310
bash scripts/deploy.sh start

# 查看状态
bash scripts/deploy.sh status

# 查看日志
tail -f /root/llm_server.log
```

## 4芯片部署架构

```
                     ┌─ HTTP ─┐
                     │ 8000   │
                     └───┬────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
     ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
     │ Chip 0    │ │ Chip 1    │ │ Chip 2    │ │ Chip 3    │
     │ MiniCPM1  │ │ MiniCPM2  │ │ MiniCPM3  │ │ MiniCPM4  │
     │ 128K ctx  │ │ 128K ctx  │ │ 128K ctx  │ │ 128K ctx  │
     │ 0.23s/tok │ │ 0.23s/tok │ │ 0.23s/tok │ │ 0.23s/tok │
     └───────────┘ └───────────┘ └───────────┘ └───────────┘
```

模型选择：`model` 参数取 `"minicpm1"` 到 `"minicpm4"`，或省略自动轮询。
