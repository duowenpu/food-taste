"""Final test-set evaluation of the trained scorer + demo predictions."""
import json, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
torch.cuda.set_per_process_memory_fraction(0.75)
AUX = ["richness_bland_rate","texture_dry_rate","texture_soggy_rate","texture_tough_rate","texture_watery_rate","would_make_again_rate","difficulty_hard_rate"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.8-27B", dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
trunk = base.model if hasattr(base, "model") else base.language_model
del base.lm_head
trunk = PeftModel.from_pretrained(trunk, "out/scorer_ckpt").eval()
head = torch.nn.Sequential(torch.nn.LayerNorm(5120), torch.nn.Linear(5120, 512), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(512, 8))
head.load_state_dict(torch.load("out/scorer_ckpt/head.pt", map_location="cuda")); head = head.to("cuda", torch.float32).eval()

rows = [json.loads(l) for l in open("data/scorer_test.jsonl")]
print(f"test rows: {len(rows)}", flush=True)
enc = sorted(((tok(r["text"], truncation=True, max_length=768)["input_ids"], i) for i, r in enumerate(rows)), key=lambda x: len(x[0]))
preds = np.zeros((len(rows), 8), dtype=np.float64); t0 = time.time()
B = 8192; MAXSEQ = 48
cur = []
def flush(cur):
    L = max(len(x[0]) for x in cur); pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.full((len(cur), L), pad, dtype=torch.long); att = torch.zeros((len(cur), L), dtype=torch.long)
    for k, (x, _) in enumerate(cur): ids[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
    with torch.no_grad():
        h = trunk(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
        out = head(h[torch.arange(len(cur), device="cuda"), att.sum(1).cuda() - 1].float())
    for k, (_, i) in enumerate(cur): preds[i] = out[k].cpu().numpy()
done = 0
for item in enc:
    if cur and (len(cur) >= MAXSEQ or max(len(cur[0][0]), len(item[0])) * (len(cur) + 1) > B): flush(cur); done += len(cur); cur = []
    cur.append(item)
    if done and done % 2000 < 20: pass
if cur: flush(cur)
print(f"scored in {time.time()-t0:.0f}s", flush=True)

y = np.array([r["y"] for r in rows]); p = preds[:, 0]
from scipy.stats import spearmanr
rng = np.random.RandomState(0); ids_ = np.arange(len(y))
a, b = rng.choice(ids_, 200000), rng.choice(ids_, 200000)
mask = np.abs(y[a] - y[b]) >= 0.3
pa = np.mean((p[a[mask]] > p[b[mask]]) == (y[a[mask]] > y[b[mask]]))
print(f"\n=== TEST RESULTS (n={len(y)}) ===")
print(f"Qwen3.8-27B scorer: Spearman={spearmanr(p, y).correlation:+.4f}  pairwise-acc(gap>=0.3)={pa:.1%}  (n_pairs={mask.sum()})")
print(f"Ridge baseline    : Spearman=+0.275   pairwise-acc=71.0%")
for j, k in enumerate(AUX, start=1):
    t = np.array([r["aux"][k] if r["aux"][k] is not None else np.nan for r in rows]); m = ~np.isnan(t)
    if m.sum() > 300:
        print(f"aux {k:26s} Spearman={spearmanr(preds[m, j], t[m]).correlation:+.3f} (n={m.sum()})")

print("\n=== DEMO ===")
demos = [
 ("Garlic Butter Herb Roast Chicken", "Garlic butter herb roast chicken\n90 minutes, 8 steps\nIngredients: whole chicken; butter; garlic; rosemary; thyme; lemon; salt; black pepper\nSteps: 1. pat chicken dry and season generously inside and out 2. stuff cavity with lemon and herbs 3. rub softened garlic butter under and over the skin 4. roast at 425f for 20 minutes 5. reduce to 350f and roast until juices run clear 6. baste twice with pan drippings 7. rest 15 minutes before carving 8. serve with pan sauce"),
 ("Fat-Free Microwave 'Alfredo'", "Fat-free microwave alfredo pasta\n12 minutes, 4 steps\nIngredients: pasta; fat-free evaporated milk; cornstarch; garlic powder; artificial butter sprinkles; fat-free processed cheese slices\nSteps: 1. boil pasta 2. microwave evaporated milk with cornstarch until thick 3. stir in cheese slices and butter sprinkles 4. pour over pasta"),
]
for name, text in demos:
    e = tok(text, return_tensors="pt", truncation=True, max_length=768).to("cuda")
    with torch.no_grad():
        h = trunk(**e).last_hidden_state
        out = head(h[0, -1].float().unsqueeze(0))[0].cpu().tolist()
    aux_str = ", ".join(f"{k.split('_')[1] if 'texture' in k else k.split('_')[0]}={max(0,min(1,v)):.2f}" for k, v in zip(AUX, out[1:]))
    print(f"{name}: predicted rating {out[0]+4.661:.2f}  | {aux_str}")
