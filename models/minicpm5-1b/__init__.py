"""
MiniCPM5-1B — OpenBMB's 1B LLaMA model for Ascend 310.
Special optimization: tuned operator ordering, FP32 matmul fallbacks, 128K RoPE.
"""
from engine.base import Device
from engine.model_loader import ModelConfig

# MiniCPM5-1B specific overrides
# These are already in config.json, but we provide them here for reference
# and potential override
CONFIG_OVERRIDES = {
    "hidden_size": 1536,
    "num_layers": 24,
    "num_heads": 16,
    "num_kv_heads": 2,
    "head_dim": 128,
    "intermediate_size": 4608,
    "vocab_size": 130560,
    "max_position": 131072,
    "rope_theta": 5000000.0,
    "rms_norm_eps": 1e-6,
}

# Operator name mapping for MiniCPM5-1B
# Maps (op_type, hidden_dim, output_dim) → .om filename
OPERATOR_MAP = {
    # QKV projections
    (1536, 2048): "mm_1_1536_2048",     # Q proj
    (1536, 256):  "mm_1_1536_256",      # K/V proj
    (1536, 1536): "mm_1_1536_1536",     # O proj, z gate
    (1536, 4608): "mm_1_1536_4608",     # gate/up proj
    (2304, 1536): "mm_1_2304_1536",     # down proj split
    # Normalization
    (1536,):     "ops_rmsnorm_1536",
    # Activation
    (4608,):     "ops_silu_4608",
    (1536,):     "ops_silu_1536",
    # Element-wise (model-name independent)
    "add":       "ops_add",
    "mul":       "ops_mul",
}

# RoPE optimization: use CPU precomputed table for 128K
# No NPU RoPE operator needed — CPU is fast enough (<0.1ms per token)
USE_CPU_ROPE = True

# Attention: use CPU for all positions (NPU fused_attn not needed for 2 KV heads)
ATTENTION_MODE = "cpu"  # 'cpu' | 'npu_fused'

# KV Cache: use ring buffer to avoid OOM at 128K
KV_CACHE_MODE = "ring_buffer"  # 'list' | 'ring_buffer'
KV_RING_SIZE = 131072  # match max_position
