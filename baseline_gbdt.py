"""Tabular baselines: predict bayes_rating (and bland-risk) from recipe-only features.
Models: Ridge on sparse bag-of-ingredients+tags; HistGradientBoosting on numeric + top ingredients.
Split by recipe. Metrics: Spearman, pairwise accuracy (gap >= 0.3). Writes test split for the LLM judge."""
import ast, csv, json, sys
import numpy as np, pandas as pd
from collections import Counter
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
csv.field_size_limit(1 << 30)

targets = {}
for l in open("out/recipe_aspects.jsonl"):
    r = json.loads(l)
    if r["n_rated"] >= 5:
        targets[r["recipe_id"]] = (r["bayes_rating"], r.get("richness_bland_rate"), r["n_rated"])
print(f"recipes with >=5 ratings: {len(targets)}")

rows = []
for r in csv.DictReader(open("data/RAW_recipes.csv")):
    if r["id"] not in targets: continue
    try:
        ings = [i.strip().lower() for i in ast.literal_eval(r["ingredients"])]
        tags = [t.strip().lower() for t in ast.literal_eval(r["tags"])]
        nut  = [float(x) for x in ast.literal_eval(r["nutrition"])]  # cal, fat, sugar, sodium, protein, satfat, carbs (PDV)
    except Exception:
        continue
    y, bland, n_rated = targets[r["id"]]
    rows.append(dict(id=r["id"], name=r["name"], y=y, bland=bland, n_rated=n_rated,
                     minutes=min(float(r["minutes"] or 0), 1440), n_steps=int(r["n_steps"] or 0),
                     n_ings=len(ings), ings=ings, tags=tags,
                     cal=min(nut[0],5000), fat=min(nut[1],500), sugar=min(nut[2],500), sodium=min(nut[3],500),
                     protein=min(nut[4],500), satfat=min(nut[5],500), carbs=min(nut[6],500)))
df = pd.DataFrame(rows)
rng = np.random.RandomState(42); df["split"] = np.where(rng.rand(len(df)) < 0.85, "train", "test")
tr, te = df[df.split=="train"], df[df.split=="test"]
print(f"train {len(tr)}  test {len(te)}")

ing_vocab = [w for w,c in Counter(w for l in tr.ings for w in l).items() if c >= 20]
tag_vocab = [w for w,c in Counter(w for l in tr.tags for w in l).items() if c >= 50]
ing_ix = {w:i for i,w in enumerate(ing_vocab)}; tag_ix = {w:i+len(ing_vocab) for i,w in enumerate(tag_vocab)}
D = len(ing_vocab)+len(tag_vocab)
def enc(part):
    m = sparse.lil_matrix((len(part), D+9), dtype=np.float32)
    for k,(_,r) in enumerate(part.iterrows()):
        for w in r.ings:
            if w in ing_ix: m[k, ing_ix[w]] = 1
        for w in r.tags:
            if w in tag_ix: m[k, tag_ix[w]] = 1
        m[k, D:] = [np.log1p(r.minutes), r.n_steps, r.n_ings, np.log1p(r.cal), r.fat/100, r.sugar/100, r.sodium/100, r.protein/100, r.satfat/100]
    return m.tocsr()
Xtr, Xte = enc(tr), enc(te)
print(f"features: {D} sparse + 9 numeric")

def report(name, yhat, ytrue):
    s = spearmanr(yhat, ytrue).correlation
    ids = np.arange(len(ytrue)); rng2 = np.random.RandomState(0)
    a, b = rng2.choice(ids, 200000), rng2.choice(ids, 200000)
    mask = np.abs(ytrue[a]-ytrue[b]) >= 0.3
    pa = np.mean((yhat[a[mask]] > yhat[b[mask]]) == (ytrue[a[mask]] > ytrue[b[mask]]))
    print(f"{name:34s} Spearman={s:+.3f}   pairwise-acc(gap>=0.3)={pa:.1%}  (n_pairs={mask.sum()})")
    return s

ytr, yte = tr.y.values, te.y.values
ridge = Ridge(alpha=3.0).fit(Xtr, ytr)
report("Ridge bag-of-ings+tags", ridge.predict(Xte), yte)
hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=None, min_samples_leaf=40, random_state=0)
hgb.fit(Xtr[:, list(range(D, D+9))].toarray(), ytr)
report("HistGB numeric-only", hgb.predict(Xte[:, list(range(D, D+9))].toarray()), yte)
w = np.asarray(np.abs(ridge.coef_[:D])).ravel(); top = np.argsort(-w)[:600]
Xtr2 = np.hstack([Xtr[:, top].toarray(), Xtr[:, list(range(D, D+9))].toarray()])
Xte2 = np.hstack([Xte[:, top].toarray(), Xte[:, list(range(D, D+9))].toarray()])
hgb2 = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, min_samples_leaf=30, random_state=0).fit(Xtr2, ytr)
report("HistGB numeric+top600-sparse", hgb2.predict(Xte2), yte)

mask_tr, mask_te = tr.bland.notna().values, te.bland.notna().values
if mask_tr.sum() > 2000:
    rb = Ridge(alpha=3.0).fit(Xtr[mask_tr], tr.bland.values[mask_tr])
    s = spearmanr(rb.predict(Xte[mask_te]), te.bland.values[mask_te]).correlation
    print(f"{'Ridge -> bland_rate (aux task)':34s} Spearman={s:+.3f}  (n_test={mask_te.sum()})")

vocab_names = np.array(ing_vocab + tag_vocab)
coef = np.asarray(ridge.coef_[:D]).ravel()
print("\nRidge: most positive features:", ', '.join(vocab_names[np.argsort(-coef)[:12]]))
print("Ridge: most negative features:", ', '.join(vocab_names[np.argsort(coef)[:12]]))
te[["id","name","y","n_rated"]].to_json("out/test_split.jsonl", orient="records", lines=True)
print(f"\ntest split saved: out/test_split.jsonl ({len(te)})")
