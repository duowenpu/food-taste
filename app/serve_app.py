"""Recipe tastiness demo server: stdlib HTTP + the trained scorer. GET / -> UI, POST /score {"text":...} -> scores."""
import json, threading, torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
torch.cuda.set_per_process_memory_fraction(0.75)
AUX = ["bland", "dry", "soggy", "tough", "watery", "make_again", "hard"]
MU = 4.661
print("loading model...", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.8-27B", dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
trunk = base.model if hasattr(base, "model") else base.language_model
del base.lm_head
trunk = PeftModel.from_pretrained(trunk, "/work/out/scorer_ckpt").eval()
head = torch.nn.Sequential(torch.nn.LayerNorm(5120), torch.nn.Linear(5120, 512), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(512, 8))
head.load_state_dict(torch.load("/work/out/scorer_ckpt/head.pt", map_location="cuda")); head = head.to("cuda", torch.float32).eval()
lock = threading.Lock()
print("model ready", flush=True)

def score(text):
    with lock, torch.no_grad():
        e = tok(text, return_tensors="pt", truncation=True, max_length=768).to("cuda")
        h = trunk(**e).last_hidden_state
        out = head(h[0, -1].float().unsqueeze(0))[0].cpu().tolist()
    return {"rating": round(out[0] + MU, 2), "risks": {k: round(max(0.0, min(1.0, v)), 3) for k, v in zip(AUX, out[1:])}}

PAGE = open("/work/app/index.html", "rb").read()
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(PAGE)
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            res = score(body["text"][:8000])
            out = json.dumps(res).encode()
            self.send_response(200)
        except Exception as ex:
            out = json.dumps({"error": str(ex)[:200]}).encode(); self.send_response(500)
        self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)
HTTPServer(("0.0.0.0", 8095), H).serve_forever()
