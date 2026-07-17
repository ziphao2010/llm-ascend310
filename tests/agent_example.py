#!/usr/bin/env python3
"""
Agent example: MiniCPM5-1B with function calling.
Uses the <tools>/<function> format from the chat template.
"""
import sys, json, numpy as np, time

sys.path.insert(0, "/root/llm-ascend310")
from engine.base import Device
from engine.model_loader import WeightLoader, ModelConfig
from tokenizers import Tokenizer

MP = "/root/models/MiniCPM5-1B"
cfg = ModelConfig(MP)
H, QD, KD, IM, NH, NKV, HD, VS = cfg.hidden_size, cfg.q_dim, cfg.k_dim, cfg.intermediate_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.vocab_size

wl = WeightLoader(MP); wl.load()
embed = wl.weights["model.embed_tokens.weight"].astype(np.float16)
norm_w = wl.weights["model.norm.weight"]
lm_t = np.ascontiguousarray(wl.weights["lm_head.weight"].T.astype(np.float32))

dev = Device(0); dev.set_device()
tk = Tokenizer.from_file(f"{MP}/tokenizer.json")

# Upload weights
w = {}
for i in range(cfg.num_layers):
    for k, v in wl.get_layer_weights(i).items():
        vf = v.astype(np.float16)
        d = np.ascontiguousarray(vf.T) if any(x in k for x in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]) else vf
        p = dev.malloc(d.nbytes); dev.h2d(p, d); w[f"{i}.{k}"] = p

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

def generate(input_ids, max_new=64, temp=0.1, callback=None):
    kv=[[] for _ in range(cfg.num_layers)]
    gen=[]
    for p,tid in enumerate(input_ids):
        forward(embed[tid].astype(np.float16),kv,p)
    last=input_ids[-1]
    for step in range(max_new):
        ho=forward(embed[last].astype(np.float16),kv,len(input_ids)+step)
        h32=ho.astype(np.float32)
        hn=((h32/np.sqrt(np.mean(h32**2)+1e-6))*norm_w).astype(np.float16)
        ll=hn.astype(np.float32)@lm_t
        ll=(ll/temp).clip(-100,100)
        kth=np.partition(ll,-40)[-40];ll[ll<kth]=-np.inf
        ll-=np.max(ll[np.isfinite(ll)])
        p=np.exp(ll)/np.sum(np.exp(ll))
        last=int(np.random.choice(VS,p=p))
        gen.append(last)
        if callback: callback(tk.decode([last]),last)
        if last in {1,130073}: break
    return tk.decode(gen,skip_special_tokens=True)

# ═══════════════════════════════════════════════════════════
# Agent: Weather tool example
# ═══════════════════════════════════════════════════════════

# Define tools
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
]

def build_tool_prompt(query):
    """Build prompt with tool definitions (matching MiniCPM chat template format)."""
    tools_xml = "<tools>\n"
    for t in TOOLS:
        tools_xml += json.dumps(t, ensure_ascii=False) + "\n"
    tools_xml += "</tools>"

    return (
        f"<|im_start|>system\nYou are a helpful assistant with access to functions.\n\n"
        f"{tools_xml}\n\n"
        f"To call a function, use: <function name=\"function-name\"><param name=\"param-name\">value</param></function>\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def parse_function_call(text):
    """Check if response contains a function call."""
    import re
    match = re.search(r'<function name="([^"]+)">(.*?)</function>', text, re.DOTALL)
    if match:
        name = match.group(1)
        params = {}
        for p in re.finditer(r'<param name="([^"]+)">(.*?)</param>', match.group(2), re.DOTALL):
            params[p.group(1)] = p.group(2).strip()
        return {"name": name, "parameters": params}
    return None

# Test
print("=== Agent Test ===")

# First, test without tools
query = "Hello! What can you do?"
prompt = build_tool_prompt(query)
ids = tk.encode(prompt).ids[:256]
print(f"Prompt: {len(ids)} tokens")
print(f"Generating...")
t0 = time.time()
result = generate(ids, max_new=32)
print(f"Response ({time.time()-t0:.1f}s):\n  {result[:200]}\n")

# Check for function call
fc = parse_function_call(result)
if fc:
    print(f"🔧 Function call detected: {fc['name']}({fc['parameters']})")
else:
    print("ℹ️ No function call in response (expected for simple query)")

print("\n✅ Agent test complete")
