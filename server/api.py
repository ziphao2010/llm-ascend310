#!/usr/bin/env python3
"""Minimal working MiniCPM5-1B API server with all fixes applied."""
import os, sys, time, json, numpy as np, asyncio, threading, logging, uvicorn

sys.path.insert(0, "/root/llm-ascend310")
from engine.base import Device
from engine.model_loader import WeightLoader
from tokenizers import Tokenizer
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("minicpm")

MP = "/root/models/MiniCPM5-1B"
API_KEY = os.environ.get("LLM_API_KEY")
MAX_CTX = int(os.environ.get("LLM_MAX_CONTEXT", "32768"))
H, QD, KD, IM, NH, NKV, HD, VS = 1536, 2048, 256, 4608, 16, 2, 128, 130560

# ── Global engine (loaded once) ──
class Model:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    i = super().__new__(cls)
                    i._init()
                    cls._instance = i
        return cls._instance

    def _init(self):
        log.info("Loading MiniCPM5-1B engine...")
        t0 = time.time()
        wl = WeightLoader(MP); wl.load()
        cw = wl.weights
        self.embed = cw["model.embed_tokens.weight"].astype(np.float16)
        self.norm_w = cw["model.norm.weight"].astype(np.float16)
        self.lm_t = np.ascontiguousarray(cw["lm_head.weight"].T.astype(np.float32))
        self.tk = Tokenizer.from_file(f"{MP}/tokenizer.json")
        self.dev = Device(0)
        self.dev.set_device()

        log.info("Uploading weights...")
        self.w = {}
        for i in range(24):
            for k, v in wl.get_layer_weights(i).items():
                vf = v.astype(np.float16)
                d = np.ascontiguousarray(vf.T) if any(x in k for x in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]) else vf
                p = self.dev.malloc(d.nbytes); self.dev.h2d(p, d)
                self.w[f"{i}.{k}"] = p
            if i % 6 == 5: log.info(f"  Layers 0-{i}")
        log.info(f"Ready: {time.time()-t0:.0f}s, {len(self.w)} tensors")

    def g(self, i, k): return self.w.get(f"{i}.{k}")

    def rms(self, h, wgt):
        h32 = h.astype(np.float32)
        return ((h32 / np.sqrt(np.mean(h32**2) + 1e-6)) * wgt).astype(np.float16)

    def rope(self, x, p):
        hd = HD // 2; inv = 1.0 / (5000000.0 ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
        c, s = np.cos(p * inv).astype(np.float16), np.sin(p * inv).astype(np.float16)
        x1, x2 = x[:, :hd], x[:, hd:]
        return np.concatenate([x1*c - x2*s, x1*s + x2*c], axis=-1)

    def forward(self, h_cpu, kv, tpos):
        """One token forward through 24 layers. All fixes applied."""
        hp = self.dev.malloc(H*2); self.dev.h2d(hp, h_cpu)
        for i in range(24):
            ln = np.empty(H, dtype=np.float16); self.dev.d2h(ln, self.w[f"{i}.input_layernorm.weight"])
            hc = np.empty(H, dtype=np.float16); self.dev.d2h(hc, hp)
            hn = self.rms(hc, ln)
            hnp = self.dev.malloc(H*2); self.dev.h2d(hnp, hn)
            q = self.dev.exec("mm_1_1536_2048", [(hnp, H*2), (self.g(i,"self_attn.q_proj.weight"), QD*H*2)])[0]
            k = self.dev.exec("mm_1_1536_256", [(hnp, H*2), (self.g(i,"self_attn.k_proj.weight"), KD*H*2)])
            v = self.dev.exec("mm_1_1536_256", [(hnp, H*2), (self.g(i,"self_attn.v_proj.weight"), KD*H*2)])
            self.dev.free(hnp)
            qc = np.empty(QD, dtype=np.float16); self.dev.d2h(qc, q); self.dev.free(q)
            kc = np.empty(KD, dtype=np.float16); self.dev.d2h(kc, k[0]); self.dev.free(k[0])
            vc = np.empty(KD, dtype=np.float16); self.dev.d2h(vc, v[0]); self.dev.free(v[0])
            qv = qc.reshape(NH, HD).astype(np.float32); kr = kc.reshape(NKV, HD).astype(np.float32)
            qr = self.rope(qv, tpos); krot = self.rope(kr, tpos)
            kv[i].append((krot.copy(), vc.reshape(NKV, HD).astype(np.float32).copy()))
            T = len(kv[i])
            ka = np.array([kv[i][t][0] for t in range(T)]).reshape(T, NKV, HD)
            va = np.array([kv[i][t][1] for t in range(T)]).reshape(T, NKV, HD)
            ka = ka.repeat(NH//NKV, 1).reshape(-1, HD); va = va.repeat(NH//NKV, 1).reshape(-1, HD)
            sc = (qr @ ka.T) * (HD**-0.5); sc -= np.max(sc, -1, keepdims=True)
            an = np.exp(sc) / np.sum(np.exp(sc), -1, keepdims=True)
            ao = (an @ va).astype(np.float16).ravel()
            ap = self.dev.malloc(H*2); self.dev.h2d(ap, ao)  # ← O PROJ FIX
            op = self.dev.exec("mm_1_1536_1536", [(ap, H*2), (self.g(i,"self_attn.o_proj.weight"), H*QD*2)])[0]
            self.dev.free(ap)
            ar = self.dev.exec("ops_add", [(hp, H*2), (op, H*2)])[0]
            self.dev.d2d(hp, ar, H*2); self.dev.free(ar); self.dev.free(op)
            pw = np.empty(H, dtype=np.float16); self.dev.d2h(pw, self.g(i,"post_attention_layernorm.weight"))
            h2 = np.empty(H, dtype=np.float16); self.dev.d2h(h2, hp)
            h2n = self.rms(h2, pw)
            h2p = self.dev.malloc(H*2); self.dev.h2d(h2p, h2n)
            gp = self.dev.exec("mm_1_1536_4608", [(h2p, H*2), (self.g(i,"mlp.gate_proj.weight"), IM*H*2)])
            up = self.dev.exec("mm_1_1536_4608", [(h2p, H*2), (self.g(i,"mlp.up_proj.weight"), IM*H*2)])
            self.dev.free(h2p)
            gg = np.empty(IM, dtype=np.float16); self.dev.d2h(gg, gp[0]); self.dev.free(gp[0])
            uu = np.empty(IM, dtype=np.float16); self.dev.d2h(uu, up[0]); self.dev.free(up[0])
            g32 = gg.astype(np.float32)
            gu = (g32 * (1.0/(1.0+np.exp(-g32))) * uu.astype(np.float32)).astype(np.float16)
            hi = IM//2
            gup = self.dev.malloc(IM*2); self.dev.h2d(gup, gu)
            dp = self.g(i, "mlp.down_proj.weight")
            dd = self.dev.exec("mm_1_2304_1536", [(gup, hi*2), (dp, hi*H*2)])
            d2 = self.dev.exec("mm_1_2304_1536", [(gup+hi*2, hi*2), (dp+hi*H*2, hi*H*2)])
            self.dev.free(gup)
            ds = self.dev.exec("ops_add", [(dd[0], H*2), (d2[0], H*2)])[0]
            self.dev.free(dd[0]); self.dev.free(d2[0])
            r2 = self.dev.exec("ops_add", [(hp, H*2), (ds, H*2)])[0]
            self.dev.d2d(hp, r2, H*2); self.dev.free(ds); self.dev.free(r2)
        ho = np.empty(H, dtype=np.float16); self.dev.d2h(ho, hp); self.dev.free(hp)
        return ho

    def generate(self, input_ids, max_new=256, temp=0.1, callback=None):
        kv = [[] for _ in range(24)]
        gen, t0 = [], time.time()

        for p, tid in enumerate(input_ids):
            self.forward(self.embed[tid].astype(np.float16), kv, p)

        last = input_ids[-1]
        for step in range(max_new):
            ho = self.forward(self.embed[last].astype(np.float16), kv, len(input_ids)+step)
            h32 = ho.astype(np.float32); rms = np.sqrt(np.mean(h32**2)+1e-6)
            hn = ((h32/rms)*self.norm_w).astype(np.float16)
            ll = hn.astype(np.float32) @ self.lm_t

            ll_s = (ll/temp).clip(-100,100)
            kth = np.partition(ll_s, -40)[-40]; ll_s[ll_s<kth] = -np.inf
            ll_s -= np.max(ll_s[np.isfinite(ll_s)])
            p = np.exp(ll_s)/np.sum(np.exp(ll_s))
            if not np.all(np.isfinite(p)): last = int(np.random.randint(0, VS))
            else: last = int(np.random.choice(VS, p=p))

            gen.append(last)
            if callback: callback(self.tk.decode([last]), last)
            if last in {1, 130073}: break

        return {"text": self.tk.decode(gen, skip_special_tokens=True),
                "tokens": gen, "count": len(gen),
                "time_s": time.time()-t0,
                "tok_s": len(gen)/(time.time()-t0+0.001)}

# ── FastAPI ──
app = FastAPI(title="MiniCPM5-1B@Ascend310", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str; content: Union[str, List[Dict[str, Any]]]
class ChatReq(BaseModel):
    model: str="minicpm1"; messages: List[Message]
    temperature: float=0.1; max_tokens: int=256; stream: bool=False; seed: Optional[int]=None
class CompReq(BaseModel):
    model: str="minicpm1"; prompt: str
    temperature: float=0.1; max_tokens: int=256; stream: bool=False; seed: Optional[int]=None

async def verify(req: Request):
    if req.headers.get("Authorization","") != f"Bearer {API_KEY}":
        from fastapi import HTTPException; raise HTTPException(401,"Invalid API key")

@app.get("/health")
async def health():
    return {"status":"ok","model":"MiniCPM5-1B","hardware":"Ascend310","version":"1.0.0"}

@app.post("/v1/chat/completions")
async def chat(body: ChatReq, req: Request):
    await verify(req); return _handle(body,True)

@app.post("/v1/completions")
async def completions(body: CompReq, req: Request):
    await verify(req); return _handle(body,False)

@app.get("/v1/models")
async def models(req: Request):
    await verify(req)
    return {"object":"list","data":[{"id":"minicpm1","object":"model","created":int(time.time()),"owned_by":"empero-ai"}]}

def _handle(body, is_chat):
    np.random.seed(body.seed or int(time.time()*1000)&0xFFFFFFFF)
    m = Model()
    if is_chat:
        msgs = [x.model_dump() for x in body.messages]
        prompt = "".join(f"<|im_start|>{x['role']}\n{x['content']}<|im_end|>\n" for x in msgs) + "<|im_start|>assistant\n"
    else:
        prompt = body.prompt
    ids = m.tk.encode(prompt).ids[:MAX_CTX]
    log.info(f"Input: {len(ids)} tokens")

    if body.stream:
        q = asyncio.Queue()
        def run():
            try: m.generate(ids, body.max_tokens, body.temperature, callback=lambda t,ti: q.put_nowait(("tok",t,ti)))
            except Exception as e: log.error(f"Error: {e}"); q.put_nowait(("err",str(e),0))
            q.put_nowait(("done",None,None))
        threading.Thread(target=run,daemon=True).start()
        async def stream():
            yield f"data: {json.dumps({'choices':[{'delta':{'role':'assistant'},'index':0}]})}\n\n"
            while True:
                msg = await q.get()
                if msg[0]=="tok": yield f"data: {json.dumps({'choices':[{'delta':{'content':msg[1]},'index':0}]})}\n\n"
                elif msg[0]=="done": yield "data: [DONE]\n\n"; break
                else: yield f"data: {json.dumps({'error':msg[1]})}\n\n"; break
        return StreamingResponse(stream(),media_type="text/event-stream",
            headers={"Cache-Control":"no-cache","Connection":"keep-alive"})

    result = m.generate(ids, body.max_tokens, body.temperature)
    text = result["text"]
    choice = {"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"} if is_chat else {"index":0,"text":text,"finish_reason":"stop"}
    return JSONResponse(content={
        "id":f"chatcmpl-{int(time.time())}","object":"chat.completion" if is_chat else "text_completion",
        "created":int(time.time()),"model":body.model,"choices":[choice],
        "usage":{"prompt_tokens":len(ids),"completion_tokens":result["count"],"total_tokens":len(ids)+result["count"]}})

if __name__=="__main__":
    log.info("Starting MiniCPM5-1B on Ascend 310")
    m = Model()
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
