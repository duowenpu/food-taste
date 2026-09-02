"""Zero-shot LLM judge: predict tastiness 0-100 from recipe-only info; compare to bayes_rating."""
import asyncio, json, random, sys, time
from openai import AsyncOpenAI
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
recipes = {}
for l in open("data/recipes.jsonl"):
    r = json.loads(l); recipes[r["id"]] = r
tests = [json.loads(l) for l in open("out/test_split.jsonl")]
random.seed(11); random.shuffle(tests); tests = tests[:N]
SYS = ("You are a culinary expert. Given a recipe's name, ingredients and cook time, predict how tasty the home-cooking "
       "community would find it: an integer 0-100 (50 = mediocre, 80 = very good, 95 = outstanding). Judge flavor balance, "
       "ingredient quality and appeal, not healthiness. Output JSON only.")
SCHEMA = {"type": "object", "properties": {"tastiness": {"type": "integer", "minimum": 0, "maximum": 100}}, "required": ["tastiness"]}
async def one(client, sem, t):
    rec = recipes[str(t["id"])]
    async with sem:
        try:
            r = await client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=20,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": f"Recipe: {rec['name']}\nIngredients: {', '.join(rec['ingredients'])}\nTotal time: {rec['minutes']} min, {rec['n_steps']} steps"}],
                response_format={"type": "json_schema", "json_schema": {"name": "T", "schema": SCHEMA, "strict": True}},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return t["id"], t["y"], json.loads(r.choices[0].message.content)["tastiness"]
        except Exception:
            return t["id"], t["y"], None
async def main():
    client = AsyncOpenAI(base_url="http://127.0.0.1:8092/v1", api_key="x", timeout=600, max_retries=3)
    sem = asyncio.Semaphore(512); t0 = time.time()
    res = await asyncio.gather(*[one(client, sem, t) for t in tests])
    ok = [(i, y, s) for i, y, s in res if s is not None]
    print(f"judged {len(ok)}/{len(tests)} in {time.time()-t0:.0f}s")
    with open("out/judge_scores.jsonl", "w") as f:
        for i, y, s in ok: f.write(json.dumps({"id": i, "y": y, "judge": s}) + "\n")
asyncio.run(main())
