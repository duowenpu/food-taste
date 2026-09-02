"""Warm the prefix cache so the shared prompt prefix is cached at mamba block boundaries.
Sends a completion whose prompt is exactly the first k*BLOCK tokens of the shared prefix."""
import asyncio, sys, time, httpx
from transformers import AutoTokenizer
from prompt import build_messages
BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 1072
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
BASE = "http://127.0.0.1:8092"

def prefix_ids():
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok.apply_chat_template(build_messages("X", ["a"], 5, ""), tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids, list) else ids["input_ids"])

async def hits(c):
    m = (await c.get(f"{BASE}/metrics")).text
    h = q = 0.0
    for line in m.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total"): h = float(line.split()[-1])
        if line.startswith("vllm:prefix_cache_queries_total"): q = float(line.split()[-1])
    return h, q

async def chat_probe(c, tag):
    h0, q0 = await hits(c)
    for i in range(3):
        r = await c.post(f"{BASE}/v1/chat/completions", json={"model": MODEL, "max_tokens": 1, "temperature": 0,
              "messages": build_messages("Probe Dish", ["a", "b"], 4, f"probe {i} {time.time()} tasty but salty"),
              "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
    h1, q1 = await hits(c)
    print(f"{tag}: 3 chat requests -> hits/request={(h1-h0)/3:.0f} tokens, queries/request={(q1-q0)/3:.0f}", flush=True)

async def main():
    ids = prefix_ids(); k = len(ids) // BLOCK; n = k * BLOCK
    print(f"prefix tokens={len(ids)} block={BLOCK} -> priming with first {n} tokens ({k} blocks)", flush=True)
    async with httpx.AsyncClient(timeout=120) as c:
        await chat_probe(c, "BEFORE prime")
        t0 = time.time()
        r = await c.post(f"{BASE}/v1/completions", json={"model": MODEL, "prompt": ids[:n], "max_tokens": 1, "temperature": 0})
        r.raise_for_status(); print(f"prime request ok ({time.time()-t0:.1f}s), usage={r.json().get('usage')}", flush=True)
        await chat_probe(c, "AFTER prime ")
        # second prime at k*BLOCK - nothing else; also try priming each boundary separately
        for j in range(1, k + 1):
            r = await c.post(f"{BASE}/v1/completions", json={"model": MODEL, "prompt": ids[:j*BLOCK], "max_tokens": 1, "temperature": 0}); r.raise_for_status()
        await chat_probe(c, "AFTER all-boundary prime")
asyncio.run(main())
