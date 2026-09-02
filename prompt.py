"""System prompt + few-shot turns. Everything here is a constant prefix -> vLLM prefix cache hits."""
import json

SYSTEM = """You label recipe reviews from a cooking website into a fixed JSON schema.

Rules:
- Use ONLY what the review text says. If a field is not mentioned, output null (or [] for lists). Never guess or infer.
- informative=true whenever ANY other field gets a value, or the review says something concrete about the result
  (how it tasted or felt, texture, how it was served, who ate it, how easy it was). informative=false ONLY for content-free
  praise/thanks ("Delicious!", "Great recipe, thanks for posting") or when the reviewer has not actually cooked it yet.
- taste.*: "too_much" only if the reviewer says it was too salty/sweet/sour/spicy, or reduced it and it was better.
  "too_little" if they wanted more. "ok" only if they explicitly say the level was right.
- richness: "bland" if flat/boring/needed more seasoning/"missing something". "rich" ONLY when flavor depth or intensity is
  described (flavorful, rich, deep, savory, decadent, "so much flavor"). Generic praise (delicious, tasty, awesome, great, loved it) -> null.
- modifications: every change the reviewer DID (kind="did") or RECOMMENDS (kind="suggests"), including ingredient swaps like
  "I used X instead" or "I used boneless breasts". Map ingredients to the recipe's ingredient list when possible. amount is a short literal.
- made_as_written: true ONLY if they explicitly say they followed it exactly ("as written", "exactly", "didn't change a thing").
  false if they mention any change. Nothing said about changes -> null.
- would_make_again: true/false only when stated or clearly implied ("keeper", "make it often", "going in the rotation", "won't bother again").
- timing: cook/bake time in the recipe being wrong (too_short = needed longer). Put time changes ONLY in timing, never in modifications.
- difficulty: only if effort is discussed (easy/quick/simple -> easy; fussy/lots of steps/too much work -> hard).
- texture_issues: NEGATIVE problems only. Praise like "crispy", "tender", "moist" is NOT an issue -> leave empty.
- audience: only if mentioned. kids/picky_eaters need actual children or picky eaters (a spouse is NOT a kid); guests = company,
  party, potluck, served to others; weeknight = quick weekday dinner; meal_prep = leftovers/reheats/freezes/make ahead;
  healthy = diet/light/low-fat; budget = cheap/pantry; holiday = named holiday or special occasion.
Output compact JSON only. Omit any field that would be null or an empty list (they default to null/[]); always include "informative"."""


def user_msg(recipe_name: str, ingredients: list[str], rating, text: str) -> str:
    return (
        f"Recipe: {recipe_name}\n"
        f"Ingredients: {', '.join(ingredients)}\n"
        f"Stars given: {rating if rating not in (None, 0, '0') else 'not given'}\n"
        f"Review: {text.strip()}"
    )


FEWSHOT = [
    (
        user_msg("Garlic Butter Shrimp", ["shrimp", "butter", "garlic", "lemon", "parsley", "salt", "red pepper flakes"], 4,
                 "Really good and fast. I cut the butter in half and added a splash of white wine. "
                 "A bit salty for us, next time I'll skip the added salt since the butter is salted. Kids ate it all."),
        {"informative": True, "made_as_written": False, "would_make_again": True,
         "taste": {"salt": "too_much", "sweet": None, "sour": None, "spicy": None, "richness": None, "greasy": None},
         "texture_issues": [], "timing": None, "difficulty": "easy",
         "modifications": [
             {"action": "reduce", "ingredient": "butter", "amount": "half", "reason": None, "kind": "did"},
             {"action": "add", "ingredient": "white wine", "amount": "splash", "reason": None, "kind": "did"},
             {"action": "remove", "ingredient": "salt", "amount": None, "reason": "butter is already salted", "kind": "suggests"}],
         "audience": ["kids", "weeknight"]},
    ),
    (
        user_msg("Classic Banana Bread", ["bananas", "flour", "sugar", "butter", "eggs", "baking soda", "salt", "vanilla"], 5,
                 "Delicious!! Made it exactly as written. Thank you for sharing."),
        {"informative": True, "made_as_written": True, "would_make_again": None,
         "taste": {"salt": None, "sweet": None, "sour": None, "spicy": None, "richness": None, "greasy": None},
         "texture_issues": [], "timing": None, "difficulty": None, "modifications": [], "audience": []},
    ),
    (
        user_msg("Oven Baked Chicken Thighs", ["chicken thighs", "olive oil", "paprika", "garlic powder", "salt", "pepper"], 3,
                 "Flavor was pretty bland and 25 minutes was nowhere near enough, mine needed 40 to cook through. "
                 "The skin never got crispy. I'd double the spices."),
        {"informative": True, "made_as_written": None, "would_make_again": None,
         "taste": {"salt": None, "sweet": None, "sour": None, "spicy": None, "richness": "bland", "greasy": None},
         "texture_issues": ["undercooked", "soggy"], "timing": "too_short", "difficulty": None,
         "modifications": [{"action": "increase", "ingredient": "paprika, garlic powder", "amount": "double", "reason": "bland", "kind": "suggests"}],
         "audience": []},
    ),
    (
        user_msg("Easy Chocolate Mousse", ["dark chocolate", "heavy cream", "eggs", "sugar"], 0,
                 "yum"),
        {"informative": False, "made_as_written": None, "would_make_again": None,
         "taste": {"salt": None, "sweet": None, "sour": None, "spicy": None, "richness": None, "greasy": None},
         "texture_issues": [], "timing": None, "difficulty": None, "modifications": [], "audience": []},
    ),
    (
        user_msg("Spicy Peanut Noodles", ["rice noodles", "peanut butter", "soy sauce", "sriracha", "lime", "garlic", "honey", "scallions"], 5,
                 "Huge hit with my picky 6 year old once I left out the sriracha. I also swapped honey for maple syrup "
                 "since that's what we had. Sauce was rich and perfectly balanced, not greasy at all. Going in the rotation."),
        {"informative": True, "made_as_written": False, "would_make_again": True,
         "taste": {"salt": None, "sweet": None, "sour": None, "spicy": "too_much", "richness": "rich", "greasy": False},
         "texture_issues": [], "timing": None, "difficulty": None,
         "modifications": [
             {"action": "remove", "ingredient": "sriracha", "amount": None, "reason": "picky kid", "kind": "did"},
             {"action": "substitute", "ingredient": "honey", "amount": None, "reason": "maple syrup, what we had", "kind": "did"}],
         "audience": ["kids", "picky_eaters"]},
    ),
    (
        user_msg("Slow Cooker Beef Stew", ["beef chuck", "potatoes", "carrots", "beef broth", "tomato paste", "flour", "thyme", "salt", "pepper"], 2,
                 "Disappointing. Followed it to the letter and the beef was still tough after 8 hours, and the whole thing "
                 "was really under-seasoned, I had to add a lot of salt at the table. Way too much work for the result. Won't bother again."),
        {"informative": True, "made_as_written": True, "would_make_again": False,
         "taste": {"salt": "too_little", "sweet": None, "sour": None, "spicy": None, "richness": "bland", "greasy": None},
         "texture_issues": ["tough"], "timing": "too_short", "difficulty": "hard",
         "modifications": [{"action": "increase", "ingredient": "salt", "amount": "a lot", "reason": "under-seasoned", "kind": "did"}],
         "audience": []},
    ),
    (
        user_msg("Lemon Garlic Roasted Chicken", ["whole chicken", "lemons", "garlic", "olive oil", "rosemary", "salt", "pepper"], 3,
                 "Made it exactly as written. Two whole lemons was too much, it came out really sour and the breast was overcooked "
                 "at the full 90 minutes, I'd pull it at 70. The skin was lovely though."),
        {"informative": True, "made_as_written": True, "would_make_again": None,
         "taste": {"salt": None, "sweet": None, "sour": "too_much", "spicy": None, "richness": None, "greasy": None},
         "texture_issues": ["overcooked"], "timing": "too_long", "difficulty": None,
         "modifications": [{"action": "reduce", "ingredient": "lemons", "amount": None, "reason": "too sour", "kind": "suggests"}],
         "audience": []},
    ),
    (
        user_msg("Pumpkin Cheesecake", ["cream cheese", "pumpkin puree", "sugar", "eggs", "graham crackers", "butter", "cinnamon", "nutmeg"], 5,
                 "Made this for Thanksgiving and it was the star of the table, rich and creamy with a perfect crust. "
                 "I doubled the cinnamon and nutmeg. Definitely making it again for Christmas."),
        {"informative": True, "made_as_written": False, "would_make_again": True,
         "taste": {"salt": None, "sweet": None, "sour": None, "spicy": None, "richness": "rich", "greasy": None},
         "texture_issues": [], "timing": None, "difficulty": None,
         "modifications": [{"action": "increase", "ingredient": "cinnamon, nutmeg", "amount": "double", "reason": None, "kind": "did"}],
         "audience": ["guests", "holiday"]},
    ),
    (
        user_msg("Turkey Chili", ["ground turkey", "kidney beans", "diced tomatoes", "onion", "chili powder", "cumin", "chicken broth", "salt"], 4,
                 "Cheap, healthy and makes a ton, I froze half. It was pretty watery and needed a good bit more salt, so I'd cut the broth "
                 "to 1 cup and maybe stir in a spoon of cornstarch. Great base recipe."),
        {"informative": True, "made_as_written": None, "would_make_again": None,
         "taste": {"salt": "too_little", "sweet": None, "sour": None, "spicy": None, "richness": None, "greasy": None},
         "texture_issues": ["watery"], "timing": None, "difficulty": None,
         "modifications": [
             {"action": "reduce", "ingredient": "chicken broth", "amount": "1 cup", "reason": "watery", "kind": "suggests"},
             {"action": "add", "ingredient": "cornstarch", "amount": "a spoon", "reason": "watery", "kind": "suggests"},
             {"action": "increase", "ingredient": "salt", "amount": None, "reason": "needed more salt", "kind": "suggests"}],
         "audience": ["meal_prep", "healthy", "budget"]},
    ),
    (
        user_msg("Buttermilk Pancakes", ["flour", "buttermilk", "eggs", "butter", "sugar", "baking powder", "baking soda", "salt"], 5,
                 "Saving this one, will make again next weekend."),
        {"informative": True, "made_as_written": None, "would_make_again": True,
         "taste": {"salt": None, "sweet": None, "sour": None, "spicy": None, "richness": None, "greasy": None},
         "texture_issues": [], "timing": None, "difficulty": None, "modifications": [], "audience": []},
    ),
]


import os as _os
_sel = _os.environ.get("FEWSHOT_IDS")           # e.g. "0,1,2,3,4,5" to pick a subset of examples
if _sel:
    FEWSHOT = [FEWSHOT[int(i)] for i in _sel.split(",")]
if _os.environ.get("NO_GLOSSARY"):
    SYSTEM = SYSTEM.split("\n\nLabel glossary")[0]


def compact(v):
    """drop null / empty values recursively (schema defaults restore them on validation)"""
    if isinstance(v, dict):
        out = {k: compact(x) for k, x in v.items()}
        return {k: x for k, x in out.items() if x not in (None, [], {})}
    if isinstance(v, list):
        return [compact(x) for x in v]
    return v


def build_messages(recipe_name, ingredients, rating, text):
    msgs = [{"role": "system", "content": SYSTEM}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": json.dumps(compact(a), separators=(",", ":"))})
    msgs.append({"role": "user", "content": user_msg(recipe_name, ingredients, rating, text)})
    return msgs
