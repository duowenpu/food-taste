import asyncio, random, time
from openai import AsyncOpenAI
from prompt import build_messages
random.seed(1)
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
WORDS = "butter garlic onion salt pepper chicken flour sugar lemon cream tomato basil rice pasta oven bake simmer stir minutes tender crispy sauce".split()
async def go(client, kind, n, conc):
    sem = asyncio.Semaphore(conc); pt = 0; errs = {}
    async def one(i):
        nonlocal pt
        async with sem:
            try:
                msgs = build_messages("Test Dish", ["a","b","c"], 5, f"review {i}: " + " ".join(random.choice(WORDS) for _ in range(40))) if kind=="shared" \
                       else [{"role":"user","content":"Summarize in one word: " + " ".join(random.choice(WORDS) for _ in range(700))}]
                r = await client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=1, temperature=0,
                                                         extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                pt += r.usage.prompt_tokens
            except Exception as e:
                errs[type(e).__name__] = errs.get(type(e).__name__, 0) + 1
    t0 = time.time(); await asyncio.gather(*[one(i) for i in range(n)]); dt = time.time() - t0
    print(f"{kind:7s} n={n} conc={conc}: {dt:6.1f}s prompt_tok/s={pt/dt:7.0f} avg_prompt={pt/max(n-sum(errs.values()),1):.0f} errs={errs}", flush=True)
async def main():
    client = AsyncOpenAI(base_url="http://127.0.0.1:8092/v1", api_key="x", timeout=60, max_retries=0)
    r = await client.chat.completions.create(model=MODEL, messages=build_messages("X", ["a"], 5, ""), max_tokens=1, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    print("PREFIX TOKENS (empty review):", r.usage.prompt_tokens, flush=True)
    await go(client, "shared", 8, 8)
    await go(client, "shared", 128, 64)
    await go(client, "unique", 64, 64)
asyncio.run(main())
