"""Ta-da demo (3,000 Chinese recipes, GBK) -> scorer_tada_{train,test}.jsonl"""
import csv, json, random, re, statistics
rows = []
with open("data/tada/demo-recipe.csv", encoding="gbk", errors="replace", newline="") as f:
    for r in csv.DictReader(f):
        try:
            avg = float(r["平均分"])
            fav = int(re.search(r"\d+", r["收藏"]).group()) if re.search(r"\d+", r["收藏"]) else 0
            view = int(re.search(r"\d+", r["浏览"]).group()) if re.search(r"\d+", r["浏览"]) else 0
            main = "、".join(re.findall(r"'([^']+)'", r["主料"])) or r["主料"]
            aux  = "、".join(re.findall(r"'([^']+)'", r["辅料"])) or r["辅料"]
        except Exception:
            continue
        if not (0 < avg <= 5): continue
        text = (f"{r['名字']}\n工艺：{r['工艺']}｜口味：{r['口味']}｜时间：{r['时间']}｜难度：{r['难度']}\n"
                f"主料：{main}\n辅料：{aux}")
        rows.append({"id": r["menus_id"], "text": text, "avg": avg, "fav": fav, "view": view})
mu = statistics.mean(x["avg"] for x in rows)
print(f"recipes: {len(rows)}  mean 平均分: {mu:.3f}  min/max: {min(x['avg'] for x in rows)}/{max(x['avg'] for x in rows)}")
random.seed(42); random.shuffle(rows)
n_test = len(rows) // 5
for split, part in [("test", rows[:n_test]), ("train", rows[n_test:])]:
    with open(f"data/scorer_tada_{split}.jsonl", "w") as f:
        for x in part:
            f.write(json.dumps({"id": x["id"], "text": x["text"], "y": round(x["avg"] - mu, 4),
                                "w": round(min(max(x["fav"], 1), 50) ** 0.5, 3), "bayes": x["avg"], "n_rated": x["fav"],
                                "aux": {k: None for k in ["richness_bland_rate","texture_dry_rate","texture_soggy_rate","texture_tough_rate","texture_watery_rate","would_make_again_rate","difficulty_hard_rate"]}},
                               ensure_ascii=False) + "\n")
    print(split, len(part))
print("sample:", json.dumps(json.loads(open('data/scorer_tada_train.jsonl').readline()), ensure_ascii=False)[:300])
