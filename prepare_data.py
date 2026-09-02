"""Kaggle Food.com (shuyangli94/food-com-recipes-and-user-interactions) -> JSONL for extract.py

Expects in data/:  RAW_recipes.csv  RAW_interactions.csv
Writes: data/reviews.jsonl      (all reviews with a recipe, in the extract.py input format)
        data/reviews.skipped.jsonl (reviews skipped by the cheap pre-filter, kept for the record)
        data/recipes.jsonl      (recipe id -> name, ingredients, minutes, n_steps, tags, nutrition)
Usage: python3 prepare_data.py [--min-words 6]
"""
import ast, csv, json, re, sys, argparse
csv.field_size_limit(1 << 30)
ap = argparse.ArgumentParser(); ap.add_argument("--min-words", type=int, default=6); a = ap.parse_args()

recipes = {}
with open("data/RAW_recipes.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            ings = ast.literal_eval(r["ingredients"])
        except Exception:
            ings = []
        recipes[r["id"]] = {"id": r["id"], "name": r["name"].strip(), "ingredients": ings, "minutes": r["minutes"],
                            "n_steps": r["n_steps"], "n_ingredients": r["n_ingredients"], "tags": r["tags"], "nutrition": r["nutrition"]}
with open("data/recipes.jsonl", "w") as f:
    for v in recipes.values():
        f.write(json.dumps(v, ensure_ascii=False) + "\n")
print(f"recipes: {len(recipes)}")

# cheap pre-filter: reviews with no concrete content ("yummy!", "will try") are marked informative=false without the LLM
GENERIC = re.compile(r"^(yum+y?|delicious|great|good|excellent|awesome|perfect|thanks?( you)?( for (sharing|posting))?|love(d)? (it|this)|will try( soon)?|so good|wonderful|fantastic)[\s!.]*$", re.I)
n_all = n_keep = n_skip = 0
with open("data/RAW_interactions.csv", newline="", encoding="utf-8") as f, \
     open("data/reviews.jsonl", "w") as out, open("data/reviews.skipped.jsonl", "w") as skipped:
    for r in csv.DictReader(f):
        n_all += 1
        rec = recipes.get(r["recipe_id"])
        text = (r["review"] or "").strip()
        row = {"id": f'{r["user_id"]}_{r["recipe_id"]}_{r["date"]}', "recipe_id": r["recipe_id"], "user_id": r["user_id"],
               "date": r["date"], "rating": int(float(r["rating"] or 0)), "text": text,
               "recipe_name": rec["name"] if rec else "", "ingredients": rec["ingredients"] if rec else []}
        words = len(text.split())
        if rec is None or words == 0 or (words < a.min_words and GENERIC.match(text)) or words < 3:
            n_skip += 1; skipped.write(json.dumps(row, ensure_ascii=False) + "\n"); continue
        n_keep += 1; out.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"interactions: {n_all}  -> to LLM: {n_keep}  skipped (pre-filter): {n_skip}")
