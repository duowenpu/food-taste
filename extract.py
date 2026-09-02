"""High-concurrency review labeler against a local vLLM OpenAI endpoint.

Input JSONL rows: {"id":..., "recipe_name":..., "ingredients":[...], "rating":..., "text":...}
Output JSONL rows: {"id":..., "label":{...}} or {"id":..., "error":"..."}; resumable.

Usage (inside the vLLM image):
  python3 extract.py --in data/x.jsonl --out out/x.labels.jsonl --concurrency 256 [--limit N]
"""
import argparse, asyncio, json, os, sys, time
from openai import AsyncOpenAI
import importlib
_S = importlib.import_module(os.environ.get("SCHEMA_MOD", "schema"))
JSON_SCHEMA, ReviewLabel = _S.JSON_SCHEMA, _S.ReviewLabel
import prompt as _P
if hasattr(_S, "to_v2"):
    _P.FEWSHOT = [(u, _S.to_v2(a)) for u, a in _P.FEWSHOT]
build_messages = _P.build_messages


async def label_one(client, sem, model, row, max_tokens, use_schema=True):
    async with sem:
        res, usage = await _label(client, model, row, max_tokens, use_schema, 0.0)
        if "error" in res and "ValidationError" in res["error"]:      # grammar glitch -> one retry with a little temperature
            res2, usage2 = await _label(client, model, row, max_tokens, use_schema, 0.3)
            usage = (usage[0] + usage2[0], usage[1] + usage2[1])
            if "error" not in res2:
                res = res2
        return res, usage


async def _label(client, model, row, max_tokens, use_schema, temperature):
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=build_messages(row["recipe_name"], row["ingredients"], row.get("rating"), row["text"]),
                temperature=temperature, max_tokens=max_tokens,
                **({"frequency_penalty": 0.4} if temperature > 0 else {}),          # retry only: break byte-loop degenerations
                **({"response_format": {"type": "json_schema",
                                 "json_schema": {"name": "ReviewLabel", "schema": JSON_SCHEMA, "strict": True}}} if use_schema else {}),
                extra_body={"chat_template_kwargs": {"enable_thinking": False},
                            **({"repetition_penalty": 1.15} if temperature > 0 else {})},
            )
            content = r.choices[0].message.content
            label = ReviewLabel.model_validate_json(content).model_dump()
            u = r.usage
            return {"id": row["id"], "label": label}, (u.prompt_tokens, u.completion_tokens)
        except Exception as e:  # noqa
            return {"id": row["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"}, (0, 0)


async def main(a):
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "label" in row:            # only successes count as done; errors get retried on resume
                        done.add(row["id"])
                except Exception:
                    pass
    rows = []
    with open(a.inp) as f:
        for line in f:
            row = json.loads(line)
            if row["id"] not in done:
                rows.append(row)
            if a.limit and len(rows) >= a.limit:
                break
    print(f"todo={len(rows)} already_done={len(done)} concurrency={a.concurrency}", flush=True)
    if not rows:
        return

    client = AsyncOpenAI(base_url=a.base_url, api_key="x", timeout=600, max_retries=3)
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time(); n = 0; pt = ct = 0; errs = 0
    out = open(a.out, "a")
    def handle(res, p, c):
        nonlocal n, pt, ct, errs
        out.write(json.dumps(res, ensure_ascii=False) + "\n")
        n += 1; pt += p; ct += c; errs += "error" in res
        if n % a.log_every == 0 or n == len(rows):
            out.flush(); dt = time.time() - t0
            print(f"{n}/{len(rows)} errs={errs} {dt:.0f}s | {n/dt:.1f} req/s | "
                  f"gen {ct/dt:.0f} tok/s | prompt {pt/dt:.0f} tok/s | avg out {ct/max(n-errs,1):.0f} tok", flush=True)
    if a.wave:
        # overlapping waves: submit a burst up to `concurrency`, then top up with a new burst whenever
        # in-flight drops below refill_at * concurrency (keeps prefill batched, avoids straggler tails)
        low = max(1, int(a.concurrency * a.refill_at)); pending = set(); it = iter(rows)
        def submit(k):
            for r in it:
                pending.add(asyncio.create_task(label_one(client, sem, a.model, r, a.max_tokens, not a.no_schema)))
                k -= 1
                if k <= 0: break
        submit(a.concurrency)
        while pending:
            done_set, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for fut in done_set:
                pending.discard(fut); res, (p, c) = fut.result(); handle(res, p, c)
            if len(pending) <= low:
                submit(a.concurrency - len(pending))
    else:
        tasks = [asyncio.create_task(label_one(client, sem, a.model, r, a.max_tokens, not a.no_schema)) for r in rows]
        for fut in asyncio.as_completed(tasks):
            res, (p, c) = await fut
            handle(res, p, c)
    out.close()
    dt = time.time() - t0
    print(f"DONE n={n} errs={errs} wall={dt:.1f}s req/s={n/dt:.2f} gen_tok/s={ct/dt:.0f} "
          f"prompt_tok/s={pt/dt:.0f} avg_prompt={pt/max(n,1):.0f} avg_out={ct/max(n-errs,1):.0f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--no-schema", action="store_true")
    ap.add_argument("--wave", action="store_true", help="burst submission: fill to --concurrency, refill when in-flight <= refill-at fraction")
    ap.add_argument("--refill-at", type=float, default=0.4)
    asyncio.run(main(ap.parse_args()))
