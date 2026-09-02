# food-taste — 1.12M recipe reviews turned into structured taste facts, and a scorer trained on them

Food.com's star ratings are 72% five-star — nearly useless as a label. The reviews
underneath them are not. This repo releases, to our knowledge, the first large-scale
**structured taste annotation of the Food.com corpus**: every one of 1,123,604 reviews
read by a local 35B model into a fixed schema of taste facts, plus the 27B LoRA scorer
trained on top of it. Everything ran on one machine; no text ever left it.

## The dataset (`release/`, 56 MB)

**1,123,604 reviews → 1,043,300 with at least one concrete fact → 1,607,451 individual
recipe modifications**, each split into what the reviewer *did* versus what they
*suggest*, with ingredient and amount. Field coverage across the corpus:

| field | reviews asserting it |
|---|---|
| modifications (did / suggests, ingredient, amount) | 727,680 reviews · 1.61M items |
| would_make_again | 788,444 |
| made_as_written | 609,437 |
| richness (bland / fine / rich) | 445,727 |
| audience (kids, guests, meal-prep, …) | 363,378 |
| difficulty | 165,679 |
| sweet / spicy / salt / sour (too little–too much) | 94,139 |
| texture issues (dry, soggy, tough, watery, …) | 46,511 |
| timing (stated time wrong, and which way) | 21,115 |

Plus `recipe_aspects.jsonl.gz`: all of it aggregated to **231,637 recipes** — per-recipe
rates for every aspect, top modifications with counts, and Bayesian-shrunk mean ratings.
Aggregate sanity holds: recipes called bland correlate −0.57 with rating,
would-make-again +0.63, every texture failure negative.

**No Food.com text is redistributed.** Every row carries only Kaggle's own keys — join
them back to the source you download yourself:

```python
import gzip, json
labels = [json.loads(l) for f in range(4)
          for l in gzip.open(f"release/review_labels-{f:02d}.jsonl.gz", "rt")]
# each row: {"user_id", "recipe_id", "date", "label": {...}}
# join key -> RAW_interactions.csv (user_id, recipe_id, date)
# recipe_aspects.jsonl.gz joins RAW_recipes.csv on recipe_id
```

Label quality is **hand-audited** (283 labels, per-field precision in
[`release/gold_verdicts.md`](release/gold_verdicts.md)): bland / modifications /
difficulty / made_as_written ≥ 90%; would_make_again ~70%; "rich" and texture flags are
aggregate-only. Read the audit before trusting any single label.

Things this enables that raw ratings cannot: aspect-aware recommenders, "how do people
actually fix this recipe" mining, explanation generation, and — as demonstrated below —
training a scorer that says *why*, not just *how much*.

## The scorer (7,616 held-out recipes)

| model | Spearman | pairwise acc (gap ≥ 0.3★) |
|---|---|---|
| **Qwen3.8-27B LoRA, trained on the labels above** | **+0.326** | **75.2%** |
| Ridge, bag-of-ingredients + tags | +0.275 | 71.0% |
| Zero-shot 35B judge | +0.125 | 61.5% |

Auxiliary heads predict per-recipe risks (bland +0.315 on test, soggy +0.227, dry +0.194),
which is what lets the demo app explain its score. Chinese input, zero-shot: Spearman
+0.271 — 80% of the ranking power retained, EN–ZH agreement 0.826.

Also in `release/`: the exact train/val/test split behind every number
(`splits.csv.gz`), baseline judge scores, and the EN/ZH cross-lingual predictions
(`judge_and_crosslingual.json.gz`), with `SHA256SUMS`.

## Rebuild from scratch

1. Download `RAW_recipes.csv` + `RAW_interactions.csv` from Kaggle
   (`shuyangli94/food-com-recipes-and-user-interactions`) into `data/`. Check the licence.
2. `./run_extract.sh server` — vLLM container serving Qwen3.6-35B-A3B-FP8 (port 8092).
3. `./run_extract.sh prepare && ./run_extract.sh extract` — ~60 h at ~5.7 reviews/s.
4. `./run_extract.sh aggregate` → `out/recipe_aspects.jsonl`.
5. `python3 prep_scorer_data.py && python3 baseline_gbdt.py` — splits + baselines.
6. `train_scorer.py` (LoRA, ~24 h), `eval_test.py`, `score_zh.py` — scorer, test eval, cross-lingual.
7. `app/serve_app.py` — local demo on :8095, warm inference 0.4 s.

Python deps go into `.deps/` inside the NGC pytorch container (`pip install --target .deps
transformers peft accelerate scikit-learn flash-linear-attention`), not onto the host.
Hard-won GB10/vLLM/hybrid-model notes: on hybrid GDN models the prefix cache only hits
in whole 1,072-token blocks (size the shared prompt prefix past it); EAGLE/MTP speculation
silently drops the last cached block per request; n-gram speculation corrupts output on
this architecture; unified memory needs `set_per_process_memory_fraction` + a MemAvailable
watchdog or a runaway run can freeze the whole box.

The trained LoRA adapter (638 MB) is not in this repo; open an issue if you want it.
