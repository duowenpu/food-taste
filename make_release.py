"""Build copyright-safe release files: my results keyed by Kaggle IDs, no source text.

- release/review_labels-XX.jsonl.gz : per-review structured labels, id = "{user_id}_{recipe_id}_{date}"
- release/recipe_aspects.jsonl.gz   : per-recipe aggregates (rates, bayes rating, top mods)
- release/splits.csv.gz             : recipe_id,split used for all reported numbers
- release/judge_and_crosslingual.json.gz : baseline judge scores + EN/ZH scorer predictions by recipe_id
Rebuild: join on RAW_interactions.csv (user_id, recipe_id, date) and RAW_recipes.csv (id).
"""
import gzip, json, os, hashlib
os.makedirs("release", exist_ok=True)
FORBID = ("text", "review", "steps", "name")

def clean_label_row(r):
    assert not any(k in r for k in FORBID), r.keys()
    uid, rid, date = r["id"].split("_", 2)
    return {"user_id": uid, "recipe_id": rid, "date": date, "label": r["label"]}

n = 0; shard = 0; out = None; SHARD_ROWS = 300000
for line in open("out/reviews.labels.jsonl"):
    r = json.loads(line)
    if "label" not in r: continue
    if n % SHARD_ROWS == 0:
        if out: out.close()
        out = gzip.open(f"release/review_labels-{shard:02d}.jsonl.gz", "wt", compresslevel=9); shard += 1
    out.write(json.dumps(clean_label_row(r), ensure_ascii=False, separators=(",", ":")) + "\n"); n += 1
out.close()
print(f"review labels: {n} rows in {shard} shards")

with gzip.open("release/recipe_aspects.jsonl.gz", "wt", compresslevel=9) as f:
    m = 0
    for line in open("out/recipe_aspects.jsonl"):
        r = json.loads(line); assert not any(k in r for k in FORBID)
        f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"); m += 1
print(f"recipe aspects: {m} rows")

with gzip.open("release/splits.csv.gz", "wt") as f:
    f.write("recipe_id,split\n")
    for split in ("train", "val", "test"):
        for line in open(f"data/scorer_{split}.jsonl"):
            f.write(f"{json.loads(line)['id']},{split}\n")
print("splits written")

extra = {"zeroshot_judge": [], "crosslingual": []}
for r in json.load(open("out/tada_zeroshot.json")) if os.path.exists("out/tada_zeroshot.json") else []:
    pass  # tada demo data excluded entirely (its ratings are placeholders)
for line in open("out/judge_scores.jsonl"):
    r = json.loads(line); extra["zeroshot_judge"].append({"recipe_id": r["id"], "judge_0_100": r["judge"]})
cl = json.load(open("out/crosslingual.json"))
zh_ids = [json.loads(l)["id"] for l in open("data/test_zh.jsonl")]
extra["crosslingual"] = [{"recipe_id": i, "pred_en": round(e, 4), "pred_zh": round(z, 4)} for i, e, z in zip(zh_ids, cl["en"], cl["zh"])]
with gzip.open("release/judge_and_crosslingual.json.gz", "wt") as f:
    json.dump(extra, f)
print(f"extras: judge {len(extra['zeroshot_judge'])}, crosslingual {len(extra['crosslingual'])}")

with open("release/SHA256SUMS", "w") as f:
    for fn in sorted(os.listdir("release")):
        if fn == "SHA256SUMS": continue
        h = hashlib.sha256(open(f"release/{fn}", "rb").read()).hexdigest()
        f.write(f"{h}  {fn}\n")
print("checksums done")
