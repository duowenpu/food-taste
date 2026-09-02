from typing import Literal
from pydantic import BaseModel
L3 = Literal["na", "too_little", "ok", "too_much"]
YN = Literal["na", "yes", "no"]

class Modification(BaseModel):
    action: Literal["reduce", "increase", "add", "remove", "substitute"]
    ingredient: str
    amount: str          # "" if none
    reason: str          # "" if none
    kind: Literal["did", "suggests"]

class ReviewLabel(BaseModel):
    informative: bool
    made_as_written: YN
    would_make_again: YN
    salt: L3; sweet: L3; sour: L3; spicy: L3
    richness: Literal["na", "bland", "ok", "rich"]
    greasy: YN
    texture_issues: list[Literal["dry","soggy","mushy","tough","rubbery","undercooked","overcooked","watery","dense","greasy","crumbly"]]
    timing: Literal["na", "too_short", "ok", "too_long"]
    difficulty: Literal["na", "easy", "moderate", "hard"]
    modifications: list[Modification]
    audience: list[Literal["kids","picky_eaters","guests","weeknight","meal_prep","healthy","budget","holiday"]]

JSON_SCHEMA = ReviewLabel.model_json_schema()

def to_v2(d: dict) -> dict:
    """convert a v1 few-shot answer dict to v2 shape"""
    yn = lambda v: "na" if v is None else ("yes" if v else "no")
    na = lambda v: "na" if v is None else v
    t = d["taste"]
    return {"informative": d["informative"], "made_as_written": yn(d["made_as_written"]), "would_make_again": yn(d["would_make_again"]),
            "salt": na(t["salt"]), "sweet": na(t["sweet"]), "sour": na(t["sour"]), "spicy": na(t["spicy"]),
            "richness": na(t["richness"]), "greasy": yn(t["greasy"]), "texture_issues": d["texture_issues"],
            "timing": na(d["timing"]), "difficulty": na(d["difficulty"]),
            "modifications": [{"action": m["action"], "ingredient": m["ingredient"], "amount": m["amount"] or "", "reason": m["reason"] or "", "kind": m["kind"]} for m in d["modifications"]],
            "audience": d["audience"]}
