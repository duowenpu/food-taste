"""Score a recipe with the trained scorer (base Qwen3.8-27B + LoRA + head).

Usage (inside pytorch image, PYTHONPATH=/work/.deps):
  python3 predict.py --ckpt out/scorer_ckpt --text "Garlic butter shrimp\n20 minutes, 5 steps\nIngredients: shrimp; butter...\nSteps: 1. ..."
  python3 predict.py --ckpt out/scorer_ckpt --file recipes.jsonl   # rows: {"id":..., "text":...}
Outputs tastiness (predicted bayes rating) + aspect risk estimates.
"""
import argparse, json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

AUX = ["bland", "dry", "soggy", "tough", "watery", "make_again", "hard"]
MU = 4.661
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/scorer_ckpt"); ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
ap.add_argument("--text"); ap.add_argument("--file"); ap.add_argument("--max-len", type=int, default=768)
a = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.75)

tok = AutoTokenizer.from_pretrained(a.model)
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
trunk = base.model if hasattr(base, "model") else base.language_model
del base.lm_head
trunk = PeftModel.from_pretrained(trunk, a.ckpt).eval()
head = torch.nn.Sequential(torch.nn.LayerNorm(5120), torch.nn.Linear(5120, 512), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(512, 1 + len(AUX)))
head.load_state_dict(torch.load(f"{a.ckpt}/head.pt", map_location="cuda")); head = head.to("cuda", torch.float32).eval()

def score(texts):
    enc = tok(texts, truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to("cuda")
    last = enc["attention_mask"].sum(1) - 1
    with torch.no_grad():
        h = trunk(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        out = head(h[torch.arange(h.size(0), device="cuda"), last].float())
    res = []
    for row in out.cpu().tolist():
        res.append({"tastiness": round(row[0] + MU, 3), **{f"risk_{k}": round(max(0.0, min(1.0, v)), 3) for k, v in zip(AUX, row[1:])}})
    return res

if a.text:
    print(json.dumps(score([a.text])[0], ensure_ascii=False, indent=1))
elif a.file:
    rows = [json.loads(l) for l in open(a.file)]
    for i in range(0, len(rows), 16):
        chunk = rows[i:i+16]
        for r, sc in zip(chunk, score([x["text"] for x in chunk])):
            print(json.dumps({"id": r.get("id"), **sc}, ensure_ascii=False))
else:
    sys.exit("need --text or --file")
