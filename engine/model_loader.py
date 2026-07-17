"""
Generic model loader for LLaMA-architecture models.
Reads config.json and loads model parameters dynamically.

Supports any model with:
  - LlamaForCausalLM architecture
  - config.json with hidden_size, num_attention_heads, num_hidden_layers, etc.
"""
import json, os, sys, numpy as np
from typing import Dict, List, Optional, Any


class ModelConfig:
    """Model configuration loaded from HuggingFace config.json."""

    def __init__(self, model_path: str):
        with open(f"{model_path}/config.json") as f:
            cfg = json.load(f)

        self.model_path = model_path
        self.hidden_size = cfg.get("hidden_size", 1536)
        self.num_layers = cfg.get("num_hidden_layers", 24)
        self.num_heads = cfg.get("num_attention_heads", 16)
        self.num_kv_heads = cfg.get("num_key_value_heads",
                                   cfg.get("num_kv_heads", 2))
        self.head_dim = cfg.get("head_dim", 128)
        self.intermediate_size = cfg.get("intermediate_size", 4608)
        self.vocab_size = cfg.get("vocab_size", 130560)
        self.max_position = cfg.get("max_position_embeddings", 131072)
        self.rms_norm_eps = cfg.get("rms_norm_eps", 1e-6)
        self.rope_theta = cfg.get("rope_theta", 5000000.0)
        self.hidden_act = cfg.get("hidden_act", "silu")
        self.torch_dtype = cfg.get("torch_dtype", "bfloat16")
        self.tie_word = cfg.get("tie_word_embeddings", False)
        self.model_type = cfg.get("model_type", "llama")

        # Derived dimensions
        self.q_dim = self.num_heads * self.head_dim       # 16*128=2048
        self.k_dim = self.num_kv_heads * self.head_dim     # 2*128=256
        self.v_dim = self.k_dim  # same as k for LLaMA GQA

        self.hidden_bytes = self.hidden_size * 2  # FP16 bytes
        self.rope_dim = self.head_dim  # standard LLaMA applies RoPE to full head

    def weight_shape(self, weight_key: str) -> Optional[tuple]:
        """Get expected shape for a weight tensor."""
        shapes = {
            "model.embed_tokens.weight": (self.vocab_size, self.hidden_size),
            "model.norm.weight": (self.hidden_size,),
            "lm_head.weight": (self.vocab_size, self.hidden_size),
            # Per-layer shapes
            "input_layernorm.weight": (self.hidden_size,),
            "post_attention_layernorm.weight": (self.hidden_size,),
            "self_attn.q_proj.weight": (self.q_dim, self.hidden_size),
            "self_attn.k_proj.weight": (self.k_dim, self.hidden_size),
            "self_attn.v_proj.weight": (self.v_dim, self.hidden_size),
            "self_attn.o_proj.weight": (self.hidden_size, self.q_dim),
            "mlp.gate_proj.weight": (self.intermediate_size, self.hidden_size),
            "mlp.up_proj.weight": (self.intermediate_size, self.hidden_size),
            "mlp.down_proj.weight": (self.hidden_size, self.intermediate_size),
        }
        return shapes.get(weight_key)

    def operator_name(self, op_type: str, m: int, k: int, n: int = 0) -> str:
        """Generate canonical operator name for given dimensions."""
        if op_type == "mm":
            # matmul: [1, M] @ [M, N] → [1, N]
            return f"mm_1_{m}_{n}"
        elif op_type == "rmsnorm":
            return f"ops_rmsnorm_{m}"
        elif op_type == "silu":
            return f"ops_silu_{m}"
        elif op_type == "softmax":
            return f"ops_softmax"
        elif op_type == "add":
            return f"ops_add"
        elif op_type == "mul":
            return f"ops_mul"
        return f"ops_{op_type}"

    def __repr__(self):
        return (f"ModelConfig({self.model_path.split('/')[-1]}: "
                f"{self.num_layers}L, {self.hidden_size}H, "
                f"{self.num_heads}Q/{self.num_kv_heads}KV, "
                f"{self.max_position}ctx)")


class WeightLoader:
    """Load weights from safetensors files into CPU memory."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.cfg = ModelConfig(model_path)
        self.weights: Dict[str, np.ndarray] = {}
        self._loaded = False

    def load(self) -> Dict[str, np.ndarray]:
        """Load all weights from safetensors."""
        if self._loaded:
            return self.weights

        # Find safetensors files
        import glob
        files = sorted(glob.glob(f"{self.model_path}/*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors found in {self.model_path}")

        from safetensors import safe_open
        for fpath in files:
            print(f"  Loading {os.path.basename(fpath)}")
            with safe_open(fpath, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Convert to float16 (weights are usually bfloat16)
                    tensor = tensor.float().numpy().astype(np.float16)
                    self.weights[key] = tensor

        self._loaded = True
        return self.weights

    def get_layer_weights(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Get weights for a specific layer index."""
        prefix = f"model.layers.{layer_idx}"
        return {k[len(prefix)+1:]: v
                for k, v in self.weights.items()
                if k.startswith(prefix)}

    def upload_layer(self, device, layer_idx: int) -> Optional[Dict[str, int]]:
        """Upload one layer's weights to NPU device. Returns {name: device_ptr}."""
        lw = self.get_layer_weights(layer_idx)
        w = {}
        for k, v in lw.items():
            d = v
            # Transpose down_proj for matmul compatibility
            if "down_proj" in k:
                d = np.ascontiguousarray(v.T.astype(np.float16))
            try:
                ptr = device.malloc(d.nbytes)
                device.h2d(ptr, d)
                w[k] = ptr
            except RuntimeError:
                # OOM — skip this weight
                print(f"    OOM on layer {layer_idx} weight {k}")
                return None
        return w if len(w) >= 8 else None  # need at least 8 weight tensors per layer

    def layer_weight_keys(self, layer_idx: int) -> List[str]:
        """List all weight keys for a given layer."""
        prefix = f"model.layers.{layer_idx}"
        return [k for k in self.weights if k.startswith(prefix)]

    def __repr__(self):
        return (f"WeightLoader({self.cfg})"
                + (f" {len(self.weights)} tensors" if self._loaded else " (not loaded)"))
