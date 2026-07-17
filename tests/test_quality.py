#!/usr/bin/env python3
"""
MiniCPM5-1B NPU engine: chat template + temperature tuning + quality test.
"""
import sys, time, json, numpy as np, copy

sys.path.insert(0, "/root/llm-ascend310")
from engine.base import Device
from engine.model_loader import ModelConfig, WeightLoader

MP = "/root/models/MiniCPM5-1B"
cfg = ModelConfig(MP)
H, HB, QD, KD, VD, IM = cfg.hidden_size, cfg.hidden_bytes, cfg.q_dim, cfg.k_dim, cfg.v_dim, cfg.intermediate_size
NH, NKV, HD, VS = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.vocab_size

wl = WeightLoader(MP); wl.load()
embed, norm_w, lm_h = wl.weights["model.embed_tokens.weight"], wl.weights["model.norm.weight"], wl.weights["lm_head.weight"]
lm_t = np.ascontiguousarray(lm_h.T.astype(np.float32))

dev = Device(0); dev.set_device()

def load_all_weights():
    w = {}
    for i in range(cfg.num_layers):
        lw = wl.get_layer_weights(i)
        for k, v in lw.items():
            # ALL matmul weights need .T for ONNX's [in_dim, out_dim] layout
            # PyTorch weights: [out_dim, in_dim]
            # ONNX matmul B:   [in_dim, out_dim]
            # So we upload weight.T
            v_f16 = v.astype(np.float16)
            if any(x in k for x in ["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj"]):
                d = np.ascontiguousarray(v_f16.T)
            elif "down_proj" in k:
                d = np.ascontiguousarray(v_f16.T)  # (4608, 1536) for split
            else:
                d = v_f16  # norms, biases
            p = dev.malloc(d.nbytes)
            dev.h2d(p, d)
            w[f"L{i}.{k}"] = p
        print(f"  Layer {i}", end="\r")
    print(f"\nLoaded {len(w)} tensors")
    return w

wc = load_all_weights()
def g(i, k): return wc.get(f"L{i}.{k}")

def rmsnorm_cpu(h, w):
    h32 = h.astype(np.float32)
    rms = np.sqrt(np.mean(h32**2) + 1e-6)
    return ((h32 / rms) * w).astype(np.float16)

def apply_rope(x, pos, theta=5000000.0):
    half = HD // 2
    inv_f = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
    c, s = np.cos(pos * inv_f).astype(np.float16), np.sin(pos * inv_f).astype(np.float16)
    x1, x2 = x[:, :half], x[:, half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)

def forward_layer(h_ptr, idx, kv_cache, tpos):
    i = idx
    h_cpu = np.empty(H, dtype=np.float16); dev.d2h(h_cpu, h_ptr)
    lnw = np.empty(H, dtype=np.float16); dev.d2h(lnw, g(i, "input_layernorm.weight"))
    hn_cpu = rmsnorm_cpu(h_cpu, lnw)
    hn_ptr = dev.malloc(HB); dev.h2d(hn_ptr, hn_cpu)

    q = dev.exec("mm_1_1536_2048", [(hn_ptr, HB), (g(i, "self_attn.q_proj.weight"), QD*H*2)])[0]
    k = dev.exec("mm_1_1536_256", [(hn_ptr, HB), (g(i, "self_attn.k_proj.weight"), KD*H*2)])
    v = dev.exec("mm_1_1536_256", [(hn_ptr, HB), (g(i, "self_attn.v_proj.weight"), VD*H*2)])
    qc = np.empty(QD, dtype=np.float16); dev.d2h(qc, q)
    kc = np.empty(KD, dtype=np.float16); dev.d2h(kc, k[0])
    vc = np.empty(KD, dtype=np.float16); dev.d2h(vc, v[0])
    dev.free(q); dev.free(k[0]); dev.free(v[0])

    qv = qc.reshape(NH, HD).astype(np.float32)
    kv_r = kc.reshape(NKV, HD).astype(np.float32)
    q_rot = apply_rope(qv, tpos)
    k_rot = apply_rope(kv_r, tpos)
    kv_cache[i].append((k_rot.copy(), vc.reshape(NKV, HD).astype(np.float32).copy()))

    T = len(kv_cache[i])
    ka = np.array([kv_cache[i][t][0] for t in range(T)]).reshape(T, NKV, HD)
    va = np.array([kv_cache[i][t][1] for t in range(T)]).reshape(T, NKV, HD)
    ka = ka.repeat(NH//NKV, axis=1).reshape(-1, HD)
    va = va.repeat(NH//NKV, axis=1).reshape(-1, HD)
    scores = (q_rot @ ka.T) * (HD**-0.5)
    scores -= np.max(scores, -1, keepdims=True)
    attn = np.exp(scores); attn /= np.sum(attn, -1, keepdims=True)
    out = (attn @ va).astype(np.float16).ravel()

    out_ptr = dev.malloc(HB); dev.h2d(out_ptr, out)
    op = dev.exec("mm_1_1536_1536", [(out_ptr, HB), (g(i, "self_attn.o_proj.weight"), H*QD*2)])[0]
    dev.free(out_ptr); dev.free(hn_ptr)
    add_r = dev.exec("ops_add", [(h_ptr, HB), (op, HB)])[0]
    dev.d2d(h_ptr, add_r, HB); dev.free(add_r); dev.free(op)

    pn, gp, up, dp = g(i, "post_attention_layernorm.weight"), g(i, "mlp.gate_proj.weight"), g(i, "mlp.up_proj.weight"), g(i, "mlp.down_proj.weight")
    if all([pn, gp, up, dp]):
        paw = np.empty(H, dtype=np.float16); dev.d2h(paw, pn)
        h2c = np.empty(H, dtype=np.float16); dev.d2h(h2c, h_ptr)
        hn2c = rmsnorm_cpu(h2c, paw)
        hn2p = dev.malloc(HB); dev.h2d(hn2p, hn2c)
        gg = dev.exec("mm_1_1536_4608", [(hn2p, HB), (gp, IM*H*2)])
        uu = dev.exec("mm_1_1536_4608", [(hn2p, HB), (up, IM*H*2)])
        dev.free(hn2p)
        ggc = np.empty(IM, dtype=np.float16); dev.d2h(ggc, gg[0])
        uuc = np.empty(IM, dtype=np.float16); dev.d2h(uuc, uu[0])
        dev.free(gg[0]); dev.free(uu[0])
        g32 = ggc.astype(np.float32)
        gu_cpu = (g32 * (1.0/(1.0+np.exp(-g32))) * uuc.astype(np.float32)).astype(np.float16)
        half_im = IM//2
        gup = dev.malloc(IM*2); dev.h2d(gup, gu_cpu)
        dd = dev.exec("mm_1_2304_1536", [(gup, half_im*2), (dp, half_im*H*2)])
        d2 = dev.exec("mm_1_2304_1536", [(gup+half_im*2, half_im*2), (dp+half_im*H*2, half_im*H*2)])
        dev.free(gup)
        ds = dev.exec("ops_add", [(dd[0], HB), (d2[0], HB)])[0]
        dev.free(dd[0]); dev.free(d2[0])
        r2 = dev.exec("ops_add", [(h_ptr, HB), (ds, HB)])[0]
        dev.d2d(h_ptr, r2, HB); dev.free(ds); dev.free(r2)

def forward(h_cpu, kv_cache, tpos=0):
    hp = dev.malloc(HB); dev.h2d(hp, h_cpu)
    for i in range(cfg.num_layers): forward_layer(hp, i, kv_cache, tpos)
    ho = np.empty(H, dtype=np.float16); dev.d2h(ho, hp); dev.free(hp)
    return ho

# ═══════════════════════════════════════════════
# CHAT TEMPLATE + TEMPERATURE TUNING
# ═══════════════════════════════════════════════
from tokenizers import Tokenizer
tk = Tokenizer.from_file(f"{MP}/tokenizer.json")

# ═══ Test 1: Check if logits are reasonable ═══
print("\n=== Quality: Logits distribution ===")
kv = [[] for _ in range(cfg.num_layers)]
chat_prompt = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
ids = tk.encode(chat_prompt).ids

t0 = time.time()
for p, tid in enumerate(ids):
    forward(embed[tid].astype(np.float16), kv, tpos=p)
print(f"Prefill {len(ids)} tok in {time.time()-t0:.1f}s")

# Check distribution of NEXT token (first decode)
h_out = forward(embed[ids[-1]].astype(np.float16), kv, tpos=len(ids))
h32 = h_out.astype(np.float32)
h_norm = ((h32 / np.sqrt(np.mean(h32**2)+1e-6)) * norm_w).astype(np.float16)
ll = h_norm.astype(np.float32) @ lm_t

top20 = np.argsort(ll)[-20:][::-1]
print(f"Logits range: [{ll.min():.1f}, {ll.max():.1f}]")
print(f"Top 20 tokens:")
for t in top20:
    probs = np.exp(ll-ll.max()) / np.sum(np.exp(ll-ll.max()))
    print(f"  [{ll[t]:6.1f}] p={probs[t]:.6f} {t}: {repr(tk.decode([t]))[:40]}")

# ═══ Test 2: Decode with best temperature ═══
print("\n=== Decode: temperature scan ===")
for temp in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]:
    kv_t = [[] for _ in range(cfg.num_layers)]
    for p, tid in enumerate(ids):
        forward(embed[tid].astype(np.float16), kv_t, tpos=p)

    last_tid = ids[-1]
    tokens = []
    for step in range(6):
        h_o = forward(embed[last_tid].astype(np.float16), kv_t, tpos=len(ids)+step)
        h32 = h_o.astype(np.float32)
        h_norm = ((h32 / np.sqrt(np.mean(h32**2)+1e-6)) * norm_w).astype(np.float16)
        ll = h_norm.astype(np.float32) @ lm_t

        ll_s = (ll / temp).clip(-100, 100)
        kth = np.partition(ll_s, -40)[-40]
        ll_s[ll_s < kth] = -np.inf
        ll_s -= np.max(ll_s[np.isfinite(ll_s)])
        p = np.exp(ll_s) / np.sum(np.exp(ll_s))
        tid = int(np.random.choice(VS, p=p))
        tokens.append(tid)
        last_tid = tid
        if tid in {1, 130073}:  # EOS
            break
    text = tk.decode(tokens)
    print(f"  T={temp:.3f}: {repr(text[:80])}")

# ═══ Test 3: Full chat with Chinese prompt ═══
print("\n=== Chat test (Chinese, temp=0.01) ===")
conversation = "<|im_start|>user\n你好！请介绍一下你自己。<|im_end|>\n<|im_start|>assistant\n"
ids2 = tk.encode(conversation).ids
kv5 = [[] for _ in range(cfg.num_layers)]

t0 = time.time()
for p, tid in enumerate(ids2):
    forward(embed[tid].astype(np.float16), kv5, tpos=p)
print(f"Prefill {len(ids2)} tok in {time.time()-t0:.1f}s")

ltid = ids2[-1]
resp = []
for step in range(32):
    t1 = time.time()
    h_o = forward(embed[ltid].astype(np.float16), kv5, tpos=len(ids2)+step)
    h32 = h_o.astype(np.float32)
    h_norm = ((h32 / np.sqrt(np.mean(h32**2)+1e-6)) * norm_w).astype(np.float16)
    ll = h_norm.astype(np.float32) @ lm_t
    ll_s = (ll / 0.01).clip(-100, 100)
    # No top-k — let argmax decide
    ll_s -= np.max(ll_s[np.isfinite(ll_s)])
    p = np.exp(ll_s) / np.sum(np.exp(ll_s))
    ltid = int(np.random.choice(VS, p=p))
    resp.append(ltid)
    if ltid in {1, 130073}: break

# Show top tokens at first decode step
print(f"First token stats:")
first_ll = ll_s  # from the last forward before decoding
# re-get the actual first logits
kv_t = [[] for _ in range(cfg.num_layers)]
for p, tid in enumerate(ids2):
    forward(embed[tid].astype(np.float16), kv_t, tpos=p)
h_o = forward(embed[ids2[-1]].astype(np.float16), kv_t, tpos=len(ids2))
top10 = np.argsort(h_o.astype(np.float32) @ lm_t)[-10:][::-1]
for t in top10:
    print(f"  {t}: {repr(tk.decode([t]))[:40]}")

print(f"Response: {repr(tk.decode(resp))}")
