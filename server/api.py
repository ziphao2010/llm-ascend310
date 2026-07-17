"""
OpenAI-compatible API server for llm-ascend310.
Manages up to 4 independent model instances across 4 NPU chips.
"""
import os, sys, time, json, threading, logging, asyncio, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.base import Device
from engine.model_loader import ModelConfig, WeightLoader
from engine.llama_engine import LLaMAEngine

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("llm-server")

API_KEY = os.environ.get("LLM_API_KEY", "llm101007")
DEFAULT_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", "/root/models/MiniCPM5-1B")
MAX_CONTEXT = int(os.environ.get("LLM_MAX_CONTEXT", "131072"))
INSTANCES = int(os.environ.get("LLM_INSTANCES", "4"))
EOS_TOKENS = {1, 130073}  # MiniCPM5 EOS


# ═══════════════════════════════════════════════════════════════════
# MODEL INSTANCE (1 per chip)
# ═══════════════════════════════════════════════════════════════════
class ModelInstance:
    """One model instance running on one NPU chip."""

    def __init__(self, instance_id: int, device_id: int, model_path: str):
        self.id = instance_id
        self.device_id = device_id
        self.model_path = model_path
        self.name = f"MiniCPM{instance_id + 1}"
        self.lock = threading.Lock()  # serializes requests to this instance

        log.info(f"Instance {self.name}: loading on device {device_id}...")
        t0 = time.time()

        self.cfg = ModelConfig(model_path)
        self.dev = Device(device_id)
        self.engine = LLaMAEngine(self.cfg, device_ids=[device_id])

        # Load weights
        wl = WeightLoader(model_path)
        wl.load()
        self.engine.load_weights(wl, device_idx=0)

        # Load tokenizer
        import transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

        # Pre-convert LM head for speed (CPU matmul)
        self.lm_head_f32_t = np.ascontiguousarray(
            self.engine.lm_head.T.astype(np.float32))
        self.norm_weight = self.engine.norm_weight

        log.info(f"Instance {self.name}: ready ({time.time()-t0:.0f}s)")

    def final_norm(self, h: np.ndarray) -> np.ndarray:
        """Final RMSNorm before LM head."""
        h32 = h.astype(np.float32)
        rms = np.sqrt(np.mean(h32 ** 2) + 1e-6)
        return ((h32 / rms) * self.norm_weight).astype(np.float16)

    def logits(self, h: np.ndarray) -> np.ndarray:
        """LM head: hidden → vocab logits."""
        return h.astype(np.float32) @ self.lm_head_f32_t

    def sample(self, logits: np.ndarray, temperature=0.6, top_p=0.9, top_k=50) -> int:
        """Sample next token with temperature/top-p/top-k."""
        if temperature > 0:
            logits = logits / temperature
        if top_k > 0 and top_k < logits.shape[0]:
            kth = np.partition(logits, -top_k)[-top_k]
            logits[logits < kth] = -np.inf
        if top_p < 1.0 and top_p > 0:
            si = np.argsort(logits)[::-1]
            sl = logits[si]
            mx = np.max(sl[np.isfinite(sl)])
            if np.isfinite(mx):
                cs = np.cumsum(np.exp(sl - mx))
                sl[cs / cs[-1] > top_p] = -np.inf
                logits[si] = sl
        finite = logits[np.isfinite(logits)]
        if len(finite) == 0:
            return int(np.random.randint(0, logits.shape[0]))
        mx = np.max(finite)
        probs = np.exp((logits - mx).clip(-100, 100))
        probs = probs / np.sum(probs)
        if not np.all(np.isfinite(probs)) or np.sum(probs) <= 0:
            return int(np.random.randint(0, logits.shape[0]))
        return int(np.random.choice(logits.shape[0], p=probs))

    def generate(self, input_ids, max_new=256, temperature=0.6,
                 top_p=0.9, top_k=50, callback=None):
        """Generate tokens with KV cache management."""
        with self.lock:
            kv = [[] for _ in range(self.cfg.num_layers)]
            generated = []
            t0 = time.time()

            # Prefill
            for pos, tid in enumerate(input_ids):
                h = self.engine.embed[tid].astype(np.float16)
                self.engine.forward(h, kv, pos, device_idx=0)

            # Decode
            last_id = input_ids[-1]
            for step in range(max_new):
                h = self.engine.embed[last_id].astype(np.float16)
                h = self.engine.forward(h, kv, len(input_ids) + step,
                                        device_idx=0)

                h = self.final_norm(h)
                ll = self.logits(h)
                tid = self.sample(ll, temperature, top_p, top_k)

                generated.append(tid)
                if callback:
                    callback(self.tokenizer.decode([tid]), tid)

                last_id = tid
                if tid in EOS_TOKENS:
                    break

            return {
                "text": self.tokenizer.decode(generated, skip_special_tokens=True),
                "tokens": generated,
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(generated),
                "total_tokens": len(input_ids) + len(generated),
                "time_s": time.time() - t0,
            }

    def format_chat(self, messages):
        """Apply chat template."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
        except Exception as e:
            log.warning(f"Chat template fallback: {e}")
            out = ""
            for m in messages:
                out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            return out + "<|im_start|>assistant\n"


# ═══════════════════════════════════════════════════════════════════
# INSTANCE POOL (load balancing across instances)
# ═══════════════════════════════════════════════════════════════════
class InstancePool:
    """Manages multiple model instances across chips."""

    def __init__(self, model_path: str, num_instances: int = 4):
        self.instances = []
        for i in range(num_instances):
            inst = ModelInstance(i, i, model_path)
            self.instances.append(inst)
        self._rr = 0  # round-robin counter
        log.info(f"Instance pool ready: {num_instances} instances on {num_instances} chips")

    def get_next(self) -> ModelInstance:
        """Round-robin load balancing."""
        inst = self.instances[self._rr % len(self.instances)]
        self._rr += 1
        return inst

    def get_by_name(self, name: str) -> ModelInstance:
        for inst in self.instances:
            if inst.name == name:
                return inst
        return None

    def list_models(self):
        return [{"id": inst.name, "object": "model",
                 "created": int(time.time()), "owned_by": "llm-ascend310"}
                for inst in self.instances]


# ═══════════════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════════════
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any

app = FastAPI(title="llm-ascend310 API", version="1.0.0")
pool: InstancePool = None


class Message(BaseModel):
    role: str; content: Union[str, List[Dict[str, Any]]]
class ChatReq(BaseModel):
    model: str = "MiniCPM1"
    messages: List[Message]
    temperature: float = 0.6; top_p: float = 0.9; top_k: int = 50
    max_tokens: int = 256; stream: bool = False; seed: Optional[int] = None
class CompReq(BaseModel):
    model: str = "MiniCPM1"
    prompt: str
    temperature: float = 0.6; top_p: float = 0.9; top_k: int = 50
    max_tokens: int = 256; stream: bool = False; seed: Optional[int] = None


async def verify(req: Request):
    auth = req.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(401, "Invalid API key")


@app.get("/health")
async def health():
    return {"status": "ok", "instances": len(pool.instances) if pool else 0,
            "max_context": MAX_CONTEXT, "version": "1.0.0"}

@app.get("/v1/models")
async def list_models(req: Request):
    await verify(req)
    return {"object": "list", "data": pool.list_models()}

@app.post("/v1/chat/completions")
async def chat_completions(body: ChatReq, req: Request):
    await verify(req)
    return _handle(body, is_chat=True)

@app.post("/v1/completions")
async def completions(body: CompReq, req: Request):
    await verify(req)
    return _handle(body, is_chat=False)


def _handle(body, is_chat):
    np.random.seed(body.seed if body.seed else int(time.time() * 1000) & 0xFFFFFFFF)

    # Route to instance
    inst = pool.get_by_name(body.model) or pool.get_next()
    log.info(f"Route to {inst.name} (device {inst.device_id})")

    if is_chat:
        msgs = [m.model_dump() for m in body.messages]
        prompt = inst.format_chat(msgs)
    else:
        prompt = body.prompt

    input_ids = inst.tokenizer.encode(prompt, truncation=True, max_length=MAX_CONTEXT)
    log.info(f"  Input: {len(input_ids)} tokens, max_new={body.max_tokens}")

    if body.stream:
        return _stream(inst, input_ids, body, is_chat)

    result = inst.generate(input_ids, body.max_tokens,
                           body.temperature, body.top_p, body.top_k)
    log.info(f"  Done: {result['completion_tokens']} tok in {result['time_s']:.1f}s")

    if is_chat:
        choice = {"index": 0,
                  "message": {"role": "assistant", "content": result["text"]},
                  "finish_reason": "stop"}
    else:
        choice = {"index": 0, "text": result["text"], "finish_reason": "stop"}

    return JSONResponse(content={
        "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion" if is_chat else "text_completion",
        "created": int(time.time()), "model": body.model, "choices": [choice],
        "usage": {"prompt_tokens": result["prompt_tokens"],
                  "completion_tokens": result["completion_tokens"],
                  "total_tokens": result["total_tokens"]}})


def _stream(inst, input_ids, body, is_chat):
    tq = asyncio.Queue()
    def run():
        try:
            inst.generate(input_ids, body.max_tokens, body.temperature,
                          body.top_p, body.top_k,
                          callback=lambda text, tid: tq.put_nowait(("token", text, tid)))
            tq.put_nowait(("done", None, None))
        except Exception as ex:
            log.error(f"Stream error: {ex}")
            tq.put_nowait(("error", str(ex), None))
    threading.Thread(target=run, daemon=True).start()

    async def stream():
        if is_chat:
            yield f"data: {json.dumps({'choices':[{'delta':{'role':'assistant'},'index':0}]})}\n\n"
        while True:
            msg = await tq.get()
            if msg[0] == "token":
                yield f"data: {json.dumps({'choices':[{'delta':{'content':msg[1]},'index':0}]})}\n\n"
            elif msg[0] == "done":
                yield "data: [DONE]\n\n"; break
            elif msg[0] == "error":
                yield f"data: {json.dumps({'error':msg[1]})}\n\n"; break

    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("=" * 50)
    log.info("llm-ascend310 — Multi-Instance LLM Server")
    log.info("=" * 50)

    model_path = os.environ.get("LLM_MODEL_PATH", DEFAULT_MODEL_PATH)
    num_instances = INSTANCES

    log.info(f"Model: {model_path}")
    log.info(f"Instances: {num_instances}")
    log.info(f"Max context: {MAX_CONTEXT}")

    pool = InstancePool(model_path, num_instances)

    port = int(os.environ.get("PORT", 8000))
    log.info(f"Listening on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)
