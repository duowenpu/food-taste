"""Qwen3.8-27B + LoRA + multi-task regression head -> recipe tastiness scorer.

Data: data/scorer_{train,val,test}.jsonl  {text, y (centered bayes_rating), w, aux{7 rates or null}}
Pooling: last-token hidden state. Loss: w-weighted MSE(y) + 0.5 * masked MSE(aux).
Batching: length-sorted token-budget micro-batches. Eval: Spearman on val; saves best adapter+head.
Usage: python3 train_scorer.py [--pilot N] [--epochs 2] [--resume out/scorer_ckpt]
"""
import argparse, glob, json, math, os, random, time
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

AUX = ["richness_bland_rate", "texture_dry_rate", "texture_soggy_rate", "texture_tough_rate",
       "texture_watery_rate", "would_make_again_rate", "difficulty_hard_rate"]
P = argparse.ArgumentParser()
P.add_argument("--pilot", type=int, default=0); P.add_argument("--epochs", type=int, default=3)
P.add_argument("--budget", type=int, default=16384)     # tokens per micro-batch
P.add_argument("--accum", type=int, default=1); P.add_argument("--max-len", type=int, default=768)
P.add_argument("--lr", type=float, default=5e-5); P.add_argument("--head-lr", type=float, default=5e-4)
P.add_argument("--lora-r", type=int, default=32); P.add_argument("--eval-every", type=int, default=100)
P.add_argument("--out", default="out/scorer_ckpt"); P.add_argument("--model", default="Qwen/Qwen3.8-27B")
P.add_argument("--data-prefix", default="data/scorer_"); P.add_argument("--init-adapter", default="")
a = P.parse_args()
torch.manual_seed(0); random.seed(0)
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(0.75)   # hard cap ~89GB: fail fast, never freeze the unified-memory box

def mem_available_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) / 1024**2
    return 999.0

def load(split):
    rows = [json.loads(l) for l in open(f"{a.data_prefix}{split}.jsonl")]
    if a.pilot and split == "train": rows = rows[: a.pilot]
    return rows
train, val = load("train"), load("val")
print(f"train {len(train)} val {len(val)}", flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
def encode(rows):
    out = []
    for r in rows:
        ids = tok(r["text"], truncation=True, max_length=a.max_len)["input_ids"]
        aux = [(r["aux"][k] if r["aux"][k] is not None else 0.0) for k in AUX]
        msk = [float(r["aux"][k] is not None) for k in AUX]
        out.append((ids, r["y"], r["w"], aux, msk))
    return out
tr_enc, va_enc = encode(train), encode(val)
print("tokens/recipe p50/p90:", sorted(len(x[0]) for x in tr_enc)[len(tr_enc)//2], sorted(len(x[0]) for x in tr_enc)[9*len(tr_enc)//10], flush=True)

def batches(enc, budget, shuffle=True):
    enc = sorted(enc, key=lambda x: len(x[0]))
    bs, cur, cur_max = [], [], 0
    for item in enc:
        L = max(cur_max, len(item[0]))
        if cur and L * (len(cur) + 1) > budget:
            bs.append(cur); cur, cur_max = [], 0; L = len(item[0])
        cur.append(item); cur_max = L
    if cur: bs.append(cur)
    if shuffle: random.shuffle(bs)
    return bs

def collate(batch):
    L = max(len(x[0]) for x in batch); pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.full((len(batch), L), pad, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long); last = torch.zeros(len(batch), dtype=torch.long)
    for i, (x, *_ ) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1; last[i] = len(x) - 1
    y  = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    w  = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    ax = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    mx = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    return ids.to(dev), att.to(dev), last.to(dev), y.to(dev), w.to(dev), ax.to(dev), mx.to(dev)

print("loading model (bf16)...", flush=True); t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
trunk = model.model if hasattr(model, "model") else model.language_model     # drop lm_head from the graph
del model.lm_head
hidden = trunk.config.hidden_size if hasattr(trunk, "config") else 5120
print(f"loaded in {time.time()-t0:.0f}s, hidden={hidden}", flush=True)
lcfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])  # GDN projections excluded: perturbing the recurrent state destabilizes training
trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
if hasattr(trunk, "enable_input_require_grads"):
    trunk.enable_input_require_grads()
if a.init_adapter:
    from peft import PeftModel
    trunk = PeftModel.from_pretrained(trunk, a.init_adapter, is_trainable=True)
    print(f"continued from adapter {a.init_adapter}", flush=True)
else:
    trunk = get_peft_model(trunk, lcfg)
trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})   # re-assert after peft wrap
gc_flags = {getattr(m, "gradient_checkpointing", None) for m in trunk.modules() if hasattr(m, "gradient_checkpointing")}
print("gradient_checkpointing flags on modules:", gc_flags, flush=True)
trunk.print_trainable_parameters()
print(f"host MemAvailable after load: {mem_available_gb():.0f}G", flush=True)
head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 1 + len(AUX))).to(dev, torch.float32)
if a.init_adapter and os.path.exists(f"{a.init_adapter}/head.pt"):
    head.load_state_dict(torch.load(f"{a.init_adapter}/head.pt", map_location=dev)); print("head loaded from init adapter", flush=True)

opt = torch.optim.AdamW([{"params": [p for p in trunk.parameters() if p.requires_grad], "lr": a.lr},
                         {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.01)
steps_per_epoch = len(batches(tr_enc, a.budget, False)) // a.accum + 1   # +1: leftover-gradient step at epoch end
total = steps_per_epoch * a.epochs
sched = get_cosine_schedule_with_warmup(opt, max(30, int(0.04 * total)), total)
print(f"optimizer steps/epoch ~{steps_per_epoch}, total {total}", flush=True)

def forward(ids, att, last):
    h = trunk(input_ids=ids, attention_mask=att).last_hidden_state
    pooled = h[torch.arange(h.size(0), device=dev), last]
    return head(pooled.float())

def evaluate(enc):
    trunk.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in batches(enc, a.budget * 2, shuffle=False):
            ids, att, last, y, w, ax, mx = collate(b)
            out = forward(ids, att, last)
            ys += y.tolist(); ps += out[:, 0].tolist()
    trunk.train()
    ys, ps = np.array(ys), np.array(ps)
    ra, rb = np.argsort(np.argsort(ys)), np.argsort(np.argsort(ps))
    return float(np.corrcoef(ra, rb)[0, 1])

os.makedirs(a.out, exist_ok=True)
trunk.train(); head.train()   # from_pretrained returns eval mode; HF layers skip checkpointing unless self.training
best = -1; gstep = 0; t0 = time.time(); lose = []
def do_eval_and_save(tag):
    global best
    sp = evaluate(va_enc)
    print(f"== eval {tag}: val Spearman {sp:+.4f} (best {best:+.4f})", flush=True)
    if sp > best:
        best = sp
        trunk.save_pretrained(a.out); torch.save(head.state_dict(), f"{a.out}/head.pt")
        json.dump({"tag": str(tag), "val_spearman": sp}, open(f"{a.out}/best.json", "w"))

for ep in range(a.epochs):
    pending = False
    for i, b in enumerate(batches(tr_enc, a.budget)):
        ids, att, last, y, w, ax, mx = collate(b)
        out = forward(ids, att, last)
        loss_y = (w * (out[:, 0] - y) ** 2).sum() / w.sum()
        la = ((out[:, 1:] - ax) ** 2 * mx).sum() / mx.sum().clamp(min=1)
        loss = loss_y + 0.5 * la
        (loss / a.accum).backward(); lose.append(loss.item()); pending = True
        if (i + 1) % a.accum == 0:
            pending = False
            torch.nn.utils.clip_grad_norm_([p for p in trunk.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); gstep += 1
            if mem_available_gb() < 12:
                trunk.save_pretrained(a.out + "_emergency"); torch.save(head.state_dict(), a.out + "_emergency/head.pt")
                raise SystemExit(f"WATCHDOG: MemAvailable < 12G at step {gstep}; checkpoint saved, aborting before the box freezes")
            if gstep == 1 or gstep % 5 == 0:
                if gstep == 1: print(f"cuda mem peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB reserved {torch.cuda.memory_reserved()/1e9:.1f}GB", flush=True)
                print(f"ep{ep} step {gstep}/{total} loss {np.mean(lose[-100:]):.4f} "
                      f"{(time.time()-t0)/gstep:.1f}s/step eta {(total-gstep)*(time.time()-t0)/gstep/3600:.1f}h", flush=True)
            if gstep % a.eval_every == 0:
                do_eval_and_save(f"step{gstep}")
    if pending:      # flush leftover accumulated gradients
        torch.nn.utils.clip_grad_norm_([p for p in trunk.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
        opt.step(); opt.zero_grad(); gstep += 1
    do_eval_and_save(f"epoch{ep}-end")
print(f"DONE best val Spearman {best:+.4f}", flush=True)
