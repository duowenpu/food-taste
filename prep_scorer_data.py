"""Build scorer training data: recipe full text + multi-task targets.
Out: data/scorer_{train,val,test}.jsonl with {id, text, y (centered bayes_rating), w (weight), aux{...} with nulls}."""
import ast, csv, json
import random
csv.field_size_limit(1 << 30)
targets = {}
for l in open("out/recipe_aspects.jsonl"):
    r = json.loads(l)
    if r["n_rated"] >= 5:
        targets[r["recipe_id"]] = r
test_ids = {str(json.loads(l)["id"]) for l in open("out/test_split.jsonl")}
MU = 4.661
AUX = ["richness_bland_rate", "texture_dry_rate", "texture_soggy_rate", "texture_tough_rate", "texture_watery_rate", "would_make_again_rate", "difficulty_hard_rate"]
random.seed(7)
counts = {"train": 0, "val": 0, "test": 0}
outs = {k: open(f"data/scorer_{k}.jsonl", "w") for k in counts}
lens = []
for r in csv.DictReader(open("data/RAW_recipes.csv")):
    t = targets.get(r["id"])
    if not t: continue
    try:
        ings = ast.literal_eval(r["ingredients"]); steps = ast.literal_eval(r["steps"]); nut = ast.literal_eval(r["nutrition"])
    except Exception:
        continue
    text = (f"{r['name'].strip()}\n{r['minutes']} minutes, {r['n_steps']} steps\n"
            f"Ingredients: {'; '.join(ings)}\n"
            f"Steps: " + " ".join(f"{i+1}. {s.strip()}" for i, s in enumerate(steps)))
    lens.append(len(text))
    aux = {k: t.get(k) for k in AUX}
    row = {"id": r["id"], "text": text[:6000], "y": round(t["bayes_rating"] - MU, 4),
           "w": round(min(t["n_rated"], 50) ** 0.5, 3), "aux": aux, "n_rated": t["n_rated"], "bayes": t["bayes_rating"]}
    split = "test" if r["id"] in test_ids else ("val" if random.random() < 0.06 else "train")
    outs[split].write(json.dumps(row, ensure_ascii=False) + "\n"); counts[split] += 1
for f in outs.values(): f.close()
lens.sort()
print(counts, "| text chars p50/p90/p99:", lens[len(lens)//2], lens[9*len(lens)//10], lens[99*len(lens)//100])
import collections
nn = collections.Counter()
for l in open("data/scorer_train.jsonl"):
    r = json.loads(l)
    for k, v in r["aux"].items(): nn[k] += v is not None
print("aux coverage in train:", dict(nn))
