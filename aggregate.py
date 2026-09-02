"""Aggregate per-review labels to recipe level.
Inputs: data/reviews.jsonl (+ data/reviews.skipped.jsonl), out/reviews.labels.jsonl
Output: out/recipe_aspects.jsonl  one row per recipe with rates, top modifications, bayes rating.
Pure stdlib (no pandas needed). Usage: python3 aggregate.py [--min-support 5] [--prior-m 10]
"""
import json, argparse, re
from collections import defaultdict, Counter
ap = argparse.ArgumentParser(); ap.add_argument("--min-support", type=int, default=5); ap.add_argument("--prior-m", type=float, default=10.0)
a = ap.parse_args()

labels = {}
for line in open("out/reviews.labels.jsonl"):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:      # file may still be appended to by a live run
        continue
    if "label" in r:
        labels[r["id"]] = r["label"]

JUNK = re.compile(r"(.)\1{5,}|[�]")   # runaway strings like "0000000" or garbage bytes
def clean_mod(m):
    ing = (m.get("ingredient") or "").strip().lower()
    if not ing or JUNK.search(ing) or "time" in ing or ing in ("recipe","it","this","that","the recipe","everything","nothing","dish","amounts","ingredients"):
        return None
    return (m["action"], ing, m.get("kind", "did"), (m.get("amount") or "").lower())

rec = defaultdict(lambda: {"n_reviews": 0, "n_rated": 0, "sum_rating": 0.0, "n_informative": 0,
                           "taste": defaultdict(Counter), "texture": Counter(), "timing": Counter(), "difficulty": Counter(),
                           "as_written": Counter(), "again": Counter(), "audience": Counter(), "mods": Counter(), "mod_amounts": defaultdict(list)})
all_ratings = []
for path in ("data/reviews.jsonl", "data/reviews.skipped.jsonl"):
    try:
        f = open(path)
    except FileNotFoundError:
        continue
    for line in f:
        r = json.loads(line); R = rec[r["recipe_id"]]
        R["n_reviews"] += 1
        if r["rating"] > 0:
            R["n_rated"] += 1; R["sum_rating"] += r["rating"]; all_ratings.append(r["rating"])
        lab = labels.get(r["id"])
        if not lab or not lab.get("informative"):
            continue
        R["n_informative"] += 1
        for k, v in (lab.get("taste") or {}).items():
            if v is not None:
                R["taste"][k][str(v)] += 1
        for t in lab.get("texture_issues") or []: R["texture"][t] += 1
        if lab.get("timing"): R["timing"][lab["timing"]] += 1
        if lab.get("difficulty"): R["difficulty"][lab["difficulty"]] += 1
        if lab.get("made_as_written") is not None: R["as_written"][str(lab["made_as_written"])] += 1
        if lab.get("would_make_again") is not None: R["again"][str(lab["would_make_again"])] += 1
        for au in lab.get("audience") or []: R["audience"][au] += 1
        for m in lab.get("modifications") or []:
            key = clean_mod(m)
            if key:
                R["mods"][key[:3]] += 1
                if key[3]: R["mod_amounts"][key[:3]].append(key[3])

mu = sum(all_ratings) / max(len(all_ratings), 1)
def rate(counter, key, denom):
    return round(counter[key] / denom, 3) if denom >= a.min_support else None

with open("out/recipe_aspects.jsonl", "w") as out:
    for rid, R in rec.items():
        n_inf = R["n_informative"]
        row = {"recipe_id": rid, "n_reviews": R["n_reviews"], "n_rated": R["n_rated"], "n_informative": n_inf,
               "mean_rating": round(R["sum_rating"] / R["n_rated"], 3) if R["n_rated"] else None,
               "bayes_rating": round((R["sum_rating"] + a.prior_m * mu) / (R["n_rated"] + a.prior_m), 3)}
        for k, c in R["taste"].items():
            mentioned = sum(c.values())
            row[f"{k}_mention_rate"] = rate(Counter({"m": mentioned}), "m", n_inf)
            for v in c:
                row[f"{k}_{v}_rate"] = rate(c, v, mentioned)          # denominator = reviews that mention this axis
        for t, cnt in R["texture"].items(): row[f"texture_{t}_rate"] = rate(R["texture"], t, n_inf)
        for t in R["timing"]: row[f"timing_{t}_rate"] = rate(R["timing"], t, sum(R["timing"].values()))
        for d in R["difficulty"]: row[f"difficulty_{d}_rate"] = rate(R["difficulty"], d, sum(R["difficulty"].values()))
        row["as_written_rate"] = rate(R["as_written"], "True", sum(R["as_written"].values()))
        row["would_make_again_rate"] = rate(R["again"], "True", sum(R["again"].values()))
        for au, cnt in R["audience"].items(): row[f"audience_{au}_rate"] = rate(R["audience"], au, n_inf)
        row["top_mods"] = [{"action": k[0], "ingredient": k[1], "kind": k[2], "count": c, "share": round(c / n_inf, 3),
                            "amounts": Counter(R["mod_amounts"][k]).most_common(2)} for k, c in R["mods"].most_common(5)] if n_inf >= a.min_support else []
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"recipes: {len(rec)}  global mean rating: {mu:.3f}  -> out/recipe_aspects.jsonl")
