"""
Generic LLaMA forward pass for Ascend 310 NPU.

Loads model config and operators dynamically — works with ANY LLaMA variant.
The engine auto-detects hidden_size, num_layers, num_heads etc. from config.json.
"""
import json, numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from ctypes import c_void_p

from .base import Device
from .model_loader import ModelConfig, WeightLoader


class LLaMAEngine:
    """
    Generic LLaMA NPU inference engine.

    Auto-adapts to any LLaMA-architecture model via ModelConfig.
    Manages 1-4 NPU devices (chips).

    Usage:
        cfg = ModelConfig("/path/to/model")
        engine = LLaMAEngine(cfg, device_ids=[0,1,2,3])
        h_out = engine.forward(h_in, kv_cache)
    """

    def __init__(self, config: ModelConfig, device_ids: List[int] = None):
        self.cfg = config
        self.device_ids = device_ids or [0]
        self.devices = [Device(d) for d in self.device_ids]
        self.H = config.hidden_size
        self.HB = config.hidden_bytes

        # Derived constants for attention
        self.NH = config.num_heads          # 16
        self.NKV = config.num_kv_heads      # 2
        self.HD = config.head_dim           # 128
        self.QD = config.q_dim              # 2048
        self.KD = config.k_dim              # 256
        self.IM = config.intermediate_size  # 4608

        # Pre-compute attention scale
        self.attn_scale = self.HD ** -0.5

        # Weight pointers (loaded per device)
        self.wc: List[Optional[Dict[int, Dict]]] = [None] * len(self.device_ids)

        # RoPE precomputation (128K context)
        self._precompute_rope()

    def _precompute_rope(self):
        """Precompute RoPE cos/sin tables for up to max_position."""
        # For standard LLaMA RoPE on half of head_dim
        T = self.cfg.max_position
        dim = self.HD
        half = dim // 2

        inv_freq = 1.0 / (self.cfg.rope_theta **
                          (np.arange(0, dim, 2, dtype=np.float32) / dim))
        pos = np.arange(T, dtype=np.float32)
        freqs = np.outer(pos, inv_freq)  # [T, half]
        self.rope_cos = np.cos(freqs).astype(np.float16)  # [T, half]
        self.rope_sin = np.sin(freqs).astype(np.float16)

    def apply_rope(self, x: np.ndarray, pos: int) -> np.ndarray:
        """
        Apply RoPE to a query or key tensor.
        x: [num_heads, head_dim] in contiguous layout
        pos: token position
        """
        half = self.HD // 2
        cos = self.rope_cos[pos]   # [half]
        sin = self.rope_sin[pos]
        x_rot = x.reshape(-1, self.HD)  # [num_heads, HD]
        x_half1 = x_rot[:, :half]
        x_half2 = x_rot[:, half:]
        rotated = np.concatenate([
            x_half1 * cos - x_half2 * sin,
            x_half1 * sin + x_half2 * cos
        ], axis=-1)
        return rotated.reshape(x.shape)

    def load_weights(self, wl: WeightLoader, device_idx: int = 0):
        """
        Load all weights for this engine onto the specified device.
        Distributes layers across devices if multiple chips.
        """
        dev = self.devices[device_idx]
        layers_per_device = (self.cfg.num_layers + len(self.devices) - 1) // len(self.devices)
        start_layer = device_idx * layers_per_device
        end_layer = min(start_layer + layers_per_device, self.cfg.num_layers)

        print(f"  Device {device_idx}: loading layers {start_layer}-{end_layer-1}")
        dev.set_device()

        w = {}
        for i in range(start_layer, end_layer):
            lw = wl.upload_layer(dev, i)
            if lw:
                w[i] = lw

        # Load non-layer weights (embed_tokens, norm, lm_head)
        # These are loaded on CPU since they're too large for NPU
        self.embed = wl.weights.get("model.embed_tokens.weight")
        self.norm_weight = wl.weights.get("model.norm.weight")
        self.lm_head = wl.weights.get("lm_head.weight")

        self.wc[device_idx] = w
        return w

    def forward(self, h_cpu: np.ndarray, kv_cache: List, token_pos: int,
                device_idx: int = 0) -> np.ndarray:
        """
        Forward pass through all layers.

        Args:
            h_cpu: (hidden_size,) float16 CPU array — embedding for this token
            kv_cache: list per-layer KV cache
            token_pos: absolute token position (for RoPE)
            device_idx: which device to use

        Returns:
            (hidden_size,) float16 — output hidden state
        """
        dev = self.devices[device_idx]
        w = self.wc[device_idx]
        if w is None:
            raise RuntimeError(f"Device {device_idx} weights not loaded")

        cfg = self.cfg
        HB = self.HB
        dev.set_device()

        # Upload initial hidden state
        h_ptr = dev.malloc(HB)
        dev.h2d(h_ptr, h_cpu)

        # Determine layer range
        lpd = (cfg.num_layers + len(self.devices) - 1) // len(self.devices)
        start = device_idx * lpd
        end = min(start + lpd, cfg.num_layers)

        for i in range(start, end):
            lw = w.get(i)
            if lw is None:
                continue

            def g(k):
                return lw.get(k, lw.get(f".{k}"))

            # ── RMSNorm ──
            hn = dev.exec(f"ops_rmsnorm_{cfg.hidden_size}",
                         [(h_ptr, HB), (g("input_layernorm.weight"), HB)])[0]

            # ── QKV Projection ──
            q = dev.exec(f"mm_1_{cfg.hidden_size}_{cfg.q_dim}",
                        [(hn, HB), (g("self_attn.q_proj.weight"), cfg.q_dim * cfg.hidden_size * 2)])[0]
            k = dev.exec(f"mm_1_{cfg.hidden_size}_{cfg.k_dim}",
                        [(hn, HB), (g("self_attn.k_proj.weight"), cfg.k_dim * cfg.hidden_size * 2)])
            v = dev.exec(f"mm_1_{cfg.hidden_size}_{cfg.v_dim}",
                        [(hn, HB), (g("self_attn.v_proj.weight"), cfg.v_dim * cfg.hidden_size * 2)])

            # Download Q, K, V for CPU attention + RoPE
            q_cpu = np.empty(self.QD, dtype=np.float16); dev.d2h(q_cpu, q)
            k_cpu = np.empty(self.KD, dtype=np.float16); dev.d2h(k_cpu, k[0])
            v_cpu = np.empty(self.KD, dtype=np.float16); dev.d2h(v_cpu, v[0])
            dev.free(q); dev.free(k[0]); dev.free(v[0])

            # Reshape for multi-head
            q_reshaped = q_cpu.reshape(self.NH, self.HD).astype(np.float32)
            k_reshaped = k_cpu.reshape(self.NKV, self.HD).astype(np.float32)
            v_reshaped = v_cpu.reshape(self.NKV, self.HD).astype(np.float32)

            # Apply RoPE to Q and K
            q_reshaped = self.apply_rope(q_reshaped, token_pos)
            k_reshaped = self.apply_rope(k_reshaped, token_pos)

            # Append to KV cache (as float32 for precision)
            kv_cache[i].append((k_reshaped.copy(), v_reshaped.copy()))

            # ── CPU Attention ──
            T = len(kv_cache[i])
            # GQA: expand KV heads NH/NKV times
            ka = np.array([kv_cache[i][t][0] for t in range(T)]
                         ).reshape(-1, self.NKV, self.HD).astype(np.float32)
            va = np.array([kv_cache[i][t][1] for t in range(T)]
                         ).reshape(-1, self.NKV, self.HD).astype(np.float32)
            # Expand KV: each Q head has its own KV head
            ka = ka.repeat(self.NH // self.NKV, axis=1)  # [T, NH, HD]
            va = va.repeat(self.NH // self.NKV, axis=1)
            k2d = ka.reshape(-1, self.HD)  # [T*NH, HD]
            v2d = va.reshape(-1, self.HD)

            # Scaled dot-product attention
            scores = q_reshaped @ k2d.T  # [NH, T*NH]
            scores = scores * self.attn_scale
            # Softmax (stable)
            scores -= np.max(scores, axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn = attn / np.sum(attn, axis=-1, keepdims=True)
            # Weighted sum
            out = (attn @ v2d).astype(np.float16)  # [NH, HD]

            # Upload attention output for O projection
            out_ptr = dev.malloc(HB)
            dev.h2d(out_ptr, out.ravel())
            op = dev.exec(f"mm_1_{cfg.hidden_size}_{cfg.hidden_size}",
                         [(out_ptr, HB), (g("self_attn.o_proj.weight"), cfg.hidden_size * cfg.q_dim * 2)])[0]
            dev.free(out_ptr)

            # ── Residual + MLP ──
            dev.free(hn)
            add_r = dev.exec("ops_add", [(h_ptr, HB), (op, HB)])[0]
            dev.d2d(h_ptr, add_r, HB)
            dev.free(add_r)
            dev.free(op)

            # MLP
            pn = g("post_attention_layernorm.weight")
            gp = g("mlp.gate_proj.weight")
            up = g("mlp.up_proj.weight")
            dp = g("mlp.down_proj.weight")
            if all([pn, gp, up, dp]):
                hn2 = dev.exec(f"ops_rmsnorm_{cfg.hidden_size}",
                              [(h_ptr, HB), (pn, HB)])[0]
                gg = dev.exec(f"mm_1_{cfg.hidden_size}_{cfg.intermediate_size}",
                             [(hn2, HB), (gp, cfg.intermediate_size * cfg.hidden_size * 2)])
                uu = dev.exec(f"mm_1_{cfg.hidden_size}_{cfg.intermediate_size}",
                             [(hn2, HB), (up, cfg.intermediate_size * cfg.hidden_size * 2)])
                dev.free(hn2)
                sg = dev.exec(f"ops_silu_{cfg.intermediate_size}",
                             [(gg[0], cfg.intermediate_size * 2)])
                gu = dev.exec("ops_mul", [(sg[0], cfg.intermediate_size * 2),
                                         (uu[0], cfg.intermediate_size * 2)])
                dev.free(gg[0]); dev.free(uu[0]); dev.free(sg[0])
                # down_proj uses split halves (works around Cube limit)
                half_im = cfg.intermediate_size // 2
                dd = dev.exec(f"mm_1_{half_im}_{cfg.hidden_size}",
                             [(gu[0], half_im * 2), (dp, half_im * cfg.hidden_size * 2)])
                dd2 = dev.exec(f"mm_1_{half_im}_{cfg.hidden_size}",
                              [(gu[0] + half_im * 2, half_im * 2),
                               (dp + half_im * cfg.hidden_size * 2, half_im * cfg.hidden_size * 2)])
                dev.free(gu[0])
                ds = dev.exec("ops_add", [(dd[0], HB), (dd2[0], HB)])[0]
                dev.free(dd[0]); dev.free(dd2[0])
                r2 = dev.exec("ops_add", [(h_ptr, HB), (ds, HB)])[0]
                dev.d2d(h_ptr, r2, HB)
                dev.free(ds); dev.free(r2)

        # Download result
        h_out = np.empty(self.H, dtype=np.float16)
        dev.d2h(h_out, h_ptr)
        dev.free(h_ptr)
        return h_out
