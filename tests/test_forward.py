#!/usr/bin/env python3
"""
MiniCPM5-1B engine test on Ascend 310.
Uses CPU RMSNorm (via numpy) instead of NPU ops_rmsnorm for now.
"""
import sys, time, json, numpy as np
from ctypes import c_void_p

sys.path.insert(0, "/root/llm-ascend310")
from engine.base import Device
from engine.model_loader import ModelConfig, WeightLoader

# ── Config ──
MP = "/root/models/MiniCPM5-1B"
cfg = ModelConfig(MP)
H, HB = cfg.hidden_size, cfg.hidden_bytes
QD = cfg.q_dim    # 2048
KD = cfg.k_dim    # 256
VD = cfg.v_dim    # 256
IM = cfg.intermediate_size  # 4608
NH, NKV, HD = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim  # 16, 2, 128

print(f"MiniCPM5-1B: {cfg.num_layers}L, {H}H, {NH}Q/{NKV}KV, {cfg.max_position}ctx")

# ── Load weights ──
wl = WeightLoader(MP); wl.load()
cw = wl.weights
embed = cw["model.embed_tokens.weight"]  # [130560, 1536]
norm_w = cw["model.norm.weight"]          # [1536]
lm_head = cw["lm_head.weight"]            # [130560, 1536]
VS = lm_head.shape[0]

# Pre-convert LM head for speed
lm_t = np.ascontiguousarray(lm_head.T.astype(np.float32))

# ── Init Device ──
dev = Device(0)
dev.set_device()
print(f"Device 0 ready, mem: {dev.mem_info()[0]/1024/1024:.0f}MB free")

# ── Upload weights ──
def load_weights():
    w = {}
    for i in range(cfg.num_layers):
        lw = wl.get_layer_weights(i)
        for k, v in lw.items():
            if "down_proj" in k:
                d = np.ascontiguousarray(v.T.astype(np.float16))
            else:
                d = v.astype(np.float16)
            try:
                p = dev.malloc(d.nbytes)
                dev.h2d(p, d)
                w[f"L{i}.{k}"] = p
            except RuntimeError:
                print(f"  OOM at layer {i} {k}")
                return None
        print(f"  Layer {i}: {len(lw)} weights", end="\r")
    print(f"\nWeights loaded: {len(w)} tensors")
    return w

t0 = time.time()
wc = load_weights()
print(f"Load time: {time.time()-t0:.0f}s")

def g(layer_idx, key):
    return wc.get(f"L{layer_idx}.{key}")

# ── Helper functions ──
def rmsnorm_cpu(h_cpu, weight_cpu):
    """CPU RMSNorm for when NPU ops_rmsnorm not available."""
    h32 = h_cpu.astype(np.float32)
    rms = np.sqrt(np.mean(h32**2) + 1e-6)
    return ((h32 / rms) * weight_cpu).astype(np.float16)

def apply_rope(x, pos, theta=5000000.0):
    """Apply RoPE. x: [num_heads, head_dim]."""
    half = HD // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
    cos = np.cos(pos * inv_freq).astype(np.float16)
    sin = np.sin(pos * inv_freq).astype(np.float16)
    x_half1 = x[:, :half]
    x_half2 = x[:, half:]
    rotated = np.concatenate([
        x_half1 * cos - x_half2 * sin,
        x_half1 * sin + x_half2 * cos
    ], axis=-1)
    return rotated

def forward_layer(h_ptr, layer_idx, kv_cache, token_pos):
    """Run one transformer layer."""
    i = layer_idx

    # ── CPU RMSNorm (until ops_rmsnorm_1536.om is compiled) ──
    h_cpu = np.empty(H, dtype=np.float16)
    dev.d2h(h_cpu, h_ptr)
    # Get layernorm weight from NPU
    ln_w = np.empty(H, dtype=np.float16)
    dev.d2h(ln_w, g(i, "input_layernorm.weight"))
    hn_cpu = rmsnorm_cpu(h_cpu, ln_w)
    hn_ptr = dev.malloc(HB)
    dev.h2d(hn_ptr, hn_cpu)

    # ── Q Projection ──
    q = dev.exec("mm_1_1536_2048",
                [(hn_ptr, HB), (g(i, "self_attn.q_proj.weight"), QD * H * 2)])[0]
    k = dev.exec("mm_1_1536_256",
                [(hn_ptr, HB), (g(i, "self_attn.k_proj.weight"), KD * H * 2)])
    v = dev.exec("mm_1_1536_256",
                [(hn_ptr, HB), (g(i, "self_attn.v_proj.weight"), VD * H * 2)])

    # Download QKV
    q_cpu = np.empty(QD, dtype=np.float16); dev.d2h(q_cpu, q)
    k_cpu = np.empty(KD, dtype=np.float16); dev.d2h(k_cpu, k[0])
    v_cpu = np.empty(KD, dtype=np.float16); dev.d2h(v_cpu, v[0])
    dev.free(q); dev.free(k[0]); dev.free(v[0])

    # Reshape and RoPE
    q_v = q_cpu.reshape(NH, HD).astype(np.float32)
    k_v = k_cpu.reshape(NKV, HD).astype(np.float32)
    q_rot = apply_rope(q_v, token_pos)
    k_rot = apply_rope(k_v, token_pos)

    # KV Cache
    kv_cache[i].append((k_rot.copy(), v_cpu.reshape(NKV, HD).astype(np.float32).copy()))

    # CPU Attention
    T = len(kv_cache[i])
    ka = np.array([kv_cache[i][t][0] for t in range(T)]).reshape(T, NKV, HD)
    va = np.array([kv_cache[i][t][1] for t in range(T)]).reshape(T, NKV, HD)
    ka = ka.repeat(NH // NKV, axis=1).reshape(-1, HD)
    va = va.repeat(NH // NKV, axis=1).reshape(-1, HD)
    scores = (q_rot @ ka.T) * (HD ** -0.5)
    scores -= np.max(scores, -1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / np.sum(attn, -1, keepdims=True)
    out = (attn @ va).astype(np.float16).ravel()

    # Upload + O projection
    out_ptr = dev.malloc(HB); dev.h2d(out_ptr, out)
    op = dev.exec("mm_1_1536_1536",
                 [(out_ptr, HB), (g(i, "self_attn.o_proj.weight"), H * QD * 2)])[0]
    dev.free(out_ptr)
    dev.free(hn_ptr)

    # Residual
    add_r = dev.exec("ops_add", [(h_ptr, HB), (op, HB)])[0]
    dev.d2d(h_ptr, add_r, HB)
    dev.free(add_r); dev.free(op)

    # ── MLP ──
    pn = g(i, "post_attention_layernorm.weight")
    gp = g(i, "mlp.gate_proj.weight")
    up = g(i, "mlp.up_proj.weight")
    dp = g(i, "mlp.down_proj.weight")
    if all([pn, gp, up, dp]):
        # CPU RMSNorm
        pa_w = np.empty(H, dtype=np.float16)
        dev.d2h(pa_w, pn)
        h2_cpu = np.empty(H, dtype=np.float16)
        dev.d2h(h2_cpu, h_ptr)
        hn2_cpu = rmsnorm_cpu(h2_cpu, pa_w)
        hn2_ptr = dev.malloc(HB); dev.h2d(hn2_ptr, hn2_cpu)

        gg = dev.exec("mm_1_1536_4608",
                     [(hn2_ptr, HB), (gp, IM * H * 2)])
        uu = dev.exec("mm_1_1536_4608",
                     [(hn2_ptr, HB), (up, IM * H * 2)])
        dev.free(hn2_ptr)

        # SiLU on CPU (no ops_silu_4608.om needed for test)
        gg_cpu = np.empty(IM, dtype=np.float16); dev.d2h(gg_cpu, gg[0])
        uu_cpu = np.empty(IM, dtype=np.float16); dev.d2h(uu_cpu, uu[0])
        dev.free(gg[0]); dev.free(uu[0])

        gg_f32 = gg_cpu.astype(np.float32)
        sig = 1.0 / (1.0 + np.exp(-gg_f32))
        gu_cpu = (gg_f32 * sig * uu_cpu.astype(np.float32)).astype(np.float16)

        # Down projection (split)
        half_im = IM // 2
        gu_ptr = dev.malloc(IM * 2); dev.h2d(gu_ptr, gu_cpu)
        dd = dev.exec("mm_1_2304_1536",
                     [(gu_ptr, half_im * 2), (dp, half_im * H * 2)])
        dd2 = dev.exec("mm_1_2304_1536",
                      [(gu_ptr + half_im * 2, half_im * 2),
                       (dp + half_im * H * 2, half_im * H * 2)])
        dev.free(gu_ptr)

        ds = dev.exec("ops_add", [(dd[0], HB), (dd2[0], HB)])[0]
        dev.free(dd[0]); dev.free(dd2[0])
        r2 = dev.exec("ops_add", [(h_ptr, HB), (ds, HB)])[0]
        dev.d2d(h_ptr, r2, HB)
        dev.free(ds); dev.free(r2)

def forward(h_cpu, kv_cache, token_pos=0):
    """Forward one token through all layers."""
    h_ptr = dev.malloc(HB)
    dev.h2d(h_ptr, h_cpu)
    for i in range(cfg.num_layers):
        forward_layer(h_ptr, i, kv_cache, token_pos)
    h_out = np.empty(H, dtype=np.float16)
    dev.d2h(h_out, h_ptr)
    dev.free(h_ptr)
    return h_out

# ═══ TEST ═══
print("\n=== Test: Single token forward ===")
kv = [[] for _ in range(cfg.num_layers)]

# Use first embedding
h0 = embed[0].astype(np.float16)

t0 = time.time()
h_out = forward(h0, kv, token_pos=0)
t1 = time.time()
print(f"Forward: {t1-t0:.3f}s")
print(f"Output: [{h_out.min():.4f}, {h_out.max():.4f}] mean={h_out.mean():.4f}")

# LM head test
ll = h_out.astype(np.float32) @ lm_t
print(f"Logits: [{ll.min():.1f}, {ll.max():.1f}] NaN={np.any(np.isnan(ll))}")

# ── Multi-token: prefill + decode ──
print("\n=== Test: Multi-token ===")
from tokenizers import Tokenizer
tk = Tokenizer.from_file(f"{MP}/tokenizer.json")

# Simple prompt encoding
prompt = "Hello! What is 2+2?"
try:
    ids = tk.encode(prompt).ids[:64]
except:
    ids = [1, 15043, 680, 364, 220, 382, 15, 2]  # fallback
print(f"Prompt: {len(ids)} tokens: {ids[:8]}...")

kv2 = [[] for _ in range(cfg.num_layers)]

# Prefill
t0 = time.time()
for pos, tid in enumerate(ids):
    h = embed[tid].astype(np.float16)
    forward(h, kv2, token_pos=pos)
t_pre = time.time()
print(f"Prefill {len(ids)} tokens: {t_pre-t0:.1f}s")

# Decode
print("Decoding:")
last_tid = ids[-1]  # start from last prompt token
for step in range(8):
    t1 = time.time()
    h = embed[last_tid].astype(np.float16)
    h = forward(h, kv2, token_pos=len(ids) + step)

    # Final norm (CPU)
    h32 = h.astype(np.float32)
    rms = np.sqrt(np.mean(h32**2) + 1e-6)
    h_norm = ((h32 / rms) * norm_w).astype(np.float16)

    # LM head
    ll = h_norm.astype(np.float32) @ lm_t
    if np.any(np.isnan(ll)):
        print(f"  NaN at step {step}"); break

    # Sample with temperature sweep
    temps = [0.01, 0.1, 0.5, 0.8]
    temp = temps[min(step, len(temps)-1)]
    ll = (ll / temp).clip(-100, 100)
    # Top-40 filtering
    kth = np.partition(ll, -40)[-40]
    ll[ll < kth] = -np.inf
    ll -= np.max(ll)
    p = np.exp(ll) / np.sum(np.exp(ll))
    tid = int(np.random.choice(VS, p=p))
    last_tid = tid  # ← feed back as next input
    print(f"  [{time.time()-t1:.1f}s] {tid}: {repr(tk.decode([tid]))}")
