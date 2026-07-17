"""
llm-ascend310 — Generic LLaMA NPU Inference Engine for Ascend 310.

Architecture:
  models/<model_name>/          ← model config + weights
    ├── config.json             ← HuggingFace/ModelScope format
    ├── model.safetensors       ← weights
    └── tokenizer.json          ← tokenizer
  
  engine/
    ├── base.py                 ← ACL wrappers (Chip, Device, Tensor)
    ├── model_loader.py         ← Loads ANY LLaMA-model from config.json
    ├── llama_forward.py        ← Generic 32/24/16-layer LLaMA forward
    ├── sampler.py              ← Top-K/Top-P sampler
    ├── cache.py                ← KV Cache (128K+ support)
    └── compile.py              ← ATC operator compiler for any dims
  
  models/minicpm5-1b/
    ├── compile_ops.sh          ← Pre-compiled operators script
    └── __init__.py             ← Override defaults (special tuning)

  server/
    ├── api.py                  ← OpenAI-compatible API
    └── worker.py               ← Multi-instance worker pool
  
The engine auto-detects model architecture from config.json:
  hidden_size → operator dimensions
  num_hidden_layers → forward loop count
  num_attention_heads / num_key_value_heads → attention shapes
  max_position_embeddings → KV cache sizing
"""
