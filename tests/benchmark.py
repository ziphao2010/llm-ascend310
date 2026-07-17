#!/usr/bin/env python3
"""
Benchmark: performance measurements for llm-ascend310.
TTFT, TPOT, throughput, memory.
"""
import sys, time, json, numpy as np, argparse

sys.path.insert(0, "/root/llm-ascend310")
from engine.base import Device
from engine.model_loader import WeightLoader, ModelConfig
from tokenizers import Tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/MiniCPM5-1B")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompt", default="Hello! What is AI?")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    MP = args.model
    cfg = ModelConfig(MP)
    H, QD, KD, IM, NH, NKV, HD, VS = cfg.hidden_size, cfg.q_dim, cfg.k_dim, cfg.intermediate_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.vocab_size

    print(f"=== llm-ascend310 Benchmark ===")
    print(f"Model: {cfg}")
    print(f"Device: {args.device}")

    # Load
    wl = WeightLoader(MP); wl.load()
    embed = wl.weights["model.embed_tokens.weight"].astype(np.float16)
    norm_w = wl.weights["model.norm.weight"]
    lm_t = np.ascontiguousarray(wl.weights["lm_head.weight"].T.astype(np.float32))

    dev = Device(args.device); dev.set_device()
    tk = Tokenizer.from_file(f"{MP}/tokenizer.json")

    # Upload
    t0 = time.time()
    w = {}
    for i in range(cfg.num_layers):
        for k, v in wl.get_layer_weights(i).items():
            vf=v.astype(np.float16)
            d=np.ascontiguousarray(vf.T) if any(x in k for x in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]) else vf
            p=dev.malloc(d.nbytes);dev.h2d(p,d);w[f"{i}.{k}"]=p
    upload_t = time.time()-t0

    def g(i,k): return w.get(f"{i}.{k}")
    def rms(h,wgt):h32=h.astype(np.float32);return((h32/np.sqrt(np.mean(h32**2)+1e-6))*wgt).astype(np.float16)
    def rope(x,p):
        hd=HD//2;inv=1.0/(5000000.0**(np.arange(0,HD,2,dtype=np.float32)/HD))
        c,s=np.cos(p*inv).astype(np.float16),np.sin(p*inv).astype(np.float16)
        x1,x2=x[:,:hd],x[:,hd:];return np.concatenate([x1*c-x2*s,x1*s+x2*c],axis=-1)

    def forward(h_cpu,kv,tpos):
        hp=dev.malloc(H*2);dev.h2d(hp,h_cpu)
        for i in range(cfg.num_layers):
            ln=np.empty(H,dtype=np.float16);dev.d2h(ln,g(i,"input_layernorm.weight"))
            hc=np.empty(H,dtype=np.float16);dev.d2h(hc,hp)
            hn=rms(hc,ln)
            hnp=dev.malloc(H*2);dev.h2d(hnp,hn)
            q=dev.exec(f"mm_1_{H}_{QD}",[(hnp,H*2),(g(i,"self_attn.q_proj.weight"),QD*H*2)])[0]
            k=dev.exec(f"mm_1_{H}_{KD}",[(hnp,H*2),(g(i,"self_attn.k_proj.weight"),KD*H*2)])
            v=dev.exec(f"mm_1_{H}_{KD}",[(hnp,H*2),(g(i,"self_attn.v_proj.weight"),KD*H*2)])
            dev.free(hnp)
            qc=np.empty(QD,dtype=np.float16);dev.d2h(qc,q);dev.free(q)
            kc=np.empty(KD,dtype=np.float16);dev.d2h(kc,k[0]);dev.free(k[0])
            vc=np.empty(KD,dtype=np.float16);dev.d2h(vc,v[0]);dev.free(v[0])
            qv=qc.reshape(NH,HD).astype(np.float32);kr=kc.reshape(NKV,HD).astype(np.float32)
            qr=rope(qv,tpos);krot=rope(kr,tpos)
            kv[i].append((krot.copy(),vc.reshape(NKV,HD).astype(np.float32).copy()))
            T=len(kv[i])
            ka=np.array([kv[i][t][0] for t in range(T)]).reshape(T,NKV,HD)
            va=np.array([kv[i][t][1] for t in range(T)]).reshape(T,NKV,HD)
            ka=ka.repeat(NH//NKV,1).reshape(-1,HD);va=va.repeat(NH//NKV,1).reshape(-1,HD)
            sc=(qr@ka.T)*(HD**-0.5);sc-=np.max(sc,-1,keepdims=True)
            an=np.exp(sc)/np.sum(np.exp(sc),-1,keepdims=True)
            ao=(an@va).astype(np.float16).ravel()
            ap=dev.malloc(H*2);dev.h2d(ap,ao)
            op=dev.exec(f"mm_1_{H}_{H}",[(ap,H*2),(g(i,"self_attn.o_proj.weight"),H*QD*2)])[0]
            dev.free(ap)
            ar=dev.exec("ops_add",[(hp,H*2),(op,H*2)])[0]
            dev.d2d(hp,ar,H*2);dev.free(ar);dev.free(op)
            pw=np.empty(H,dtype=np.float16);dev.d2h(pw,g(i,"post_attention_layernorm.weight"))
            h2=np.empty(H,dtype=np.float16);dev.d2h(h2,hp)
            h2n=rms(h2,pw)
            h2p=dev.malloc(H*2);dev.h2d(h2p,h2n)
            gp=dev.exec(f"mm_1_{H}_{IM}",[(h2p,H*2),(g(i,"mlp.gate_proj.weight"),IM*H*2)])
            up=dev.exec(f"mm_1_{H}_{IM}",[(h2p,H*2),(g(i,"mlp.up_proj.weight"),IM*H*2)])
            dev.free(h2p)
            gg=np.empty(IM,dtype=np.float16);dev.d2h(gg,gp[0]);dev.free(gp[0])
            uu=np.empty(IM,dtype=np.float16);dev.d2h(uu,up[0]);dev.free(up[0])
            g32=gg.astype(np.float32)
            gu=(g32*(1.0/(1.0+np.exp(-g32)))*uu.astype(np.float32)).astype(np.float16)
            hi=IM//2
            gup=dev.malloc(IM*2);dev.h2d(gup,gu)
            dp=g(i,"mlp.down_proj.weight")
            dd=dev.exec(f"mm_1_{hi}_{H}",[(gup,hi*2),(dp,hi*H*2)])
            d2=dev.exec(f"mm_1_{hi}_{H}",[(gup+hi*2,hi*2),(dp+hi*H*2,hi*H*2)])
            dev.free(gup)
            ds=dev.exec("ops_add",[(dd[0],H*2),(d2[0],H*2)])[0]
            dev.free(dd[0]);dev.free(d2[0])
            r2=dev.exec("ops_add",[(hp,H*2),(ds,H*2)])[0]
            dev.d2d(hp,r2,H*2);dev.free(ds);dev.free(r2)
        ho=np.empty(H,dtype=np.float16);dev.d2h(ho,hp);dev.free(hp)
        return ho

    # Warmup
    print(f"\nWarmup ({args.warmup}x)...")
    for _ in range(args.warmup):
        kv=[[] for _ in range(cfg.num_layers)]
        forward(embed[0].astype(np.float16),kv,0)

    # Benchmark 1: Single forward
    print("\n--- Benchmark 1: Single forward ---")
    times=[]
    kv=[[] for _ in range(cfg.num_layers)]
    for _ in range(5):
        t0=time.time()
        forward(embed[0].astype(np.float16),kv,0)
        times.append(time.time()-t0)
    avg_fwd=np.mean(times)
    print(f"  5 runs: min={min(times):.3f}s max={max(times):.3f}s avg={avg_fwd:.3f}s")
    print(f"  = {1/avg_fwd:.1f} tok/s")

    # Benchmark 2: Prefill
    ids=tk.encode(args.prompt).ids[:64]
    print(f"\n--- Benchmark 2: Prefill ({len(ids)} tokens) ---")
    kv=[[] for _ in range(cfg.num_layers)]
    t0=time.time()
    for p,tid in enumerate(ids):
        forward(embed[tid].astype(np.float16),kv,tpos=p)
    prefill_t=time.time()-t0
    print(f"  Prefill: {prefill_t:.1f}s ({len(ids)/prefill_t:.1f} tok/s)")

    # Benchmark 3: Decode
    print(f"\n--- Benchmark 3: Decode ({args.max_tokens} tokens) ---")
    last=ids[-1];decode_times=[]
    for step in range(args.max_tokens):
        t1=time.time()
        ho=forward(embed[last].astype(np.float16),kv,tpos=len(ids)+step)
        h32=ho.astype(np.float32)
        hn=((h32/np.sqrt(np.mean(h32**2)+1e-6))*norm_w).astype(np.float16)
        ll=hn.astype(np.float32)@lm_t
        ll=(ll/0.1).clip(-100,100)
        kth=np.partition(ll,-40)[-40];ll[ll<kth]=-np.inf
        ll-=np.max(ll[np.isfinite(ll)])
        p=np.exp(ll)/np.sum(np.exp(ll))
        last=int(np.random.choice(VS,p=p))
        decode_times.append(time.time()-t1)

    avg_dec=np.mean(decode_times)
    print(f"  Avg: {avg_dec:.3f}s/token = {1/avg_dec:.1f} tok/s")
    print(f"  Min: {min(decode_times):.3f}s  Max: {max(decode_times):.3f}s")

    # Memory
    free,total=dev.mem_info()
    print(f"\n--- Memory ---")
    print(f"  Used: {(total-free)/1024/1024:.0f}MB / {total/1024/1024:.0f}MB")

    # Summary
    print(f"\n=== Summary ===")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Weight upload | {upload_t:.0f}s |")
    print(f"| Single forward | {avg_fwd:.3f}s |")
    print(f"| Throughput | {1/avg_fwd:.1f} tok/s |")
    print(f"| Prefill ({len(ids)} tok) | {prefill_t:.1f}s |")
    print(f"| Decode avg | {avg_dec:.3f}s/token ({1/avg_dec:.1f} tok/s) |")
    print(f"| Peak memory | {(total-free)/1024/1024:.0f}MB |")

if __name__=="__main__":
    main()
