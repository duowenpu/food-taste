"""Cross-lingual test: score matched EN/ZH versions of 1000 test recipes."""
import json, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from scipy.stats import spearmanr
torch.cuda.set_per_process_memory_fraction(0.75)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.8-27B", dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
trunk = base.model if hasattr(base, "model") else base.language_model
del base.lm_head
trunk = PeftModel.from_pretrained(trunk, "out/scorer_ckpt").eval()
head = torch.nn.Sequential(torch.nn.LayerNorm(5120), torch.nn.Linear(5120, 512), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(512, 8))
head.load_state_dict(torch.load("out/scorer_ckpt/head.pt", map_location="cuda")); head = head.to("cuda", torch.float32).eval()

zh = [json.loads(l) for l in open("data/test_zh.jsonl")]
en_all = {r["id"]: r for r in (json.loads(l) for l in open("data/scorer_test.jsonl"))}
pairs = [(r["id"], r["y"], en_all[r["id"]]["text"], r["text"]) for r in zh if r["id"] in en_all]
print(f"matched pairs: {len(pairs)}", flush=True)

def batch_score(texts):
    enc = sorted(((tok(t, truncation=True, max_length=768)["input_ids"], i) for i, t in enumerate(texts)), key=lambda x: len(x[0]))
    preds = np.zeros(len(texts)); cur = []
    def flush(cur):
        L = max(len(x[0]) for x in cur); pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(cur), L), pad, dtype=torch.long); att = torch.zeros((len(cur), L), dtype=torch.long)
        for k, (x, _) in enumerate(cur): ids[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
        with torch.no_grad():
            h = trunk(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
            out = head(h[torch.arange(len(cur), device="cuda"), att.sum(1).cuda() - 1].float())
        for k, (_, i) in enumerate(cur): preds[i] = out[k, 0].item()
    for item in enc:
        if cur and (len(cur) >= 48 or max(len(cur[0][0]), len(item[0])) * (len(cur) + 1) > 8192): flush(cur); cur = []
        cur.append(item)
    if cur: flush(cur)
    return preds

y = np.array([p[1] for p in pairs])
t0=time.time(); pe = batch_score([p[2] for p in pairs]); print(f"EN scored {time.time()-t0:.0f}s", flush=True)
t0=time.time(); pz = batch_score([p[3] for p in pairs]); print(f"ZH scored {time.time()-t0:.0f}s", flush=True)
print(f"\n=== CROSS-LINGUAL (n={len(y)}) ===")
print(f"English input : Spearman vs truth {spearmanr(pe, y).correlation:+.4f}")
print(f"Chinese input : Spearman vs truth {spearmanr(pz, y).correlation:+.4f}")
print(f"EN-ZH score agreement: Spearman {spearmanr(pe, pz).correlation:+.4f}")
json.dump({"en": pe.tolist(), "zh": pz.tolist(), "y": y.tolist()}, open("out/crosslingual.json", "w"))
