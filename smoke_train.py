import time, torch, torch.nn as nn
def mem(tag):
    ram = int(open('/proc/meminfo').readline().split()[1]) // 1024**2
    avail = [l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1]
    print(f"[{tag}] cuda alloc {torch.cuda.memory_allocated()/1e9:.1f}G reserved {torch.cuda.memory_reserved()/1e9:.1f}G | host avail {int(avail)//1024**2}G", flush=True)
t0=time.time()
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
mem("start")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.8-27B", dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
print(f"load {time.time()-t0:.0f}s", flush=True); mem("loaded")
trunk = model.model if hasattr(model, "model") else model.language_model
del model.lm_head
trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
trunk.enable_input_require_grads() if hasattr(trunk,'enable_input_require_grads') else None
lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","in_proj_qkvz","in_proj_ba","out_proj"])
trunk = get_peft_model(trunk, lcfg); mem("lora")
head = nn.Sequential(nn.Linear(5120, 512), nn.GELU(), nn.Linear(512, 8)).to("cuda", torch.float32)
ids = tok(["Chicken soup with garlic and thyme. Steps: 1. simmer."]*4, return_tensors="pt", padding=True).to("cuda")
for it in range(3):
    t1=time.time()
    h = trunk(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"]).last_hidden_state
    out = head(h[:, -1].float()); loss = (out ** 2).mean()
    print(f"fwd{it} {time.time()-t1:.1f}s", flush=True); mem(f"fwd{it}")
    t1=time.time(); loss.backward()
    print(f"bwd{it} {time.time()-t1:.1f}s", flush=True); mem(f"bwd{it}")
    for p in trunk.parameters():
        if p.grad is not None: p.grad = None
print("SMOKE OK", flush=True)
