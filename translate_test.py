"""Translate 1,000 Food.com test recipes to Chinese via local vLLM (for cross-lingual scoring test)."""
import asyncio, json, random
from openai import AsyncOpenAI
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
rows = [json.loads(l) for l in open("data/scorer_test.jsonl")]
random.seed(5); random.shuffle(rows); rows = rows[:1000]
SYS = "把下面的英文食谱完整翻译成自然的中文菜谱，保持同样的结构（名称、时间步数、Ingredients: 换成 食材：、Steps: 换成 步骤：），只输出翻译。"
async def one(client, sem, r):
    async with sem:
        try:
            resp = await client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=900,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": r["text"][:3000]}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return {"id": r["id"], "y": r["y"], "text": resp.choices[0].message.content.strip()}
        except Exception as e:
            return {"id": r["id"], "y": r["y"], "error": str(e)[:100]}
async def main():
    client = AsyncOpenAI(base_url="http://127.0.0.1:8092/v1", api_key="x", timeout=300, max_retries=2)
    sem = asyncio.Semaphore(256)
    res = await asyncio.gather(*[one(client, sem, r) for r in rows])
    ok = [r for r in res if "text" in r]
    with open("data/test_zh.jsonl", "w") as f:
        for r in ok: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"translated {len(ok)}/{len(rows)}")
    print("sample:", ok[0]["text"][:200].replace(chr(10), " | "))
asyncio.run(main())
