"""Review -> structured aspect labels. Pydantic schema; also used for vLLM structured output."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

Level = Optional[Literal["too_little", "ok", "too_much"]]


class Taste(BaseModel):
    salt: Level = None
    sweet: Level = None
    sour: Level = None
    spicy: Level = None
    richness: Optional[Literal["bland", "ok", "rich"]] = None
    greasy: Optional[bool] = None


class Modification(BaseModel):
    action: Literal["reduce", "increase", "add", "remove", "substitute"]
    ingredient: str = Field(max_length=60)
    amount: Optional[str] = Field(default=None, max_length=30)
    reason: Optional[str] = Field(default=None, max_length=100)
    kind: Literal["did", "suggests"] = "did"


class ReviewLabel(BaseModel):
    informative: bool
    made_as_written: Optional[bool] = None
    would_make_again: Optional[bool] = None
    taste: Taste = Field(default_factory=Taste)
    texture_issues: list[
        Literal["dry", "soggy", "mushy", "tough", "rubbery", "undercooked",
                "overcooked", "watery", "dense", "greasy", "crumbly"]
    ] = Field(default_factory=list, max_length=4)
    timing: Optional[Literal["too_short", "ok", "too_long"]] = None
    difficulty: Optional[Literal["easy", "moderate", "hard"]] = None
    modifications: list[Modification] = Field(default_factory=list, max_length=6)
    audience: list[
        Literal["kids", "picky_eaters", "guests", "weeknight", "meal_prep",
                "healthy", "budget", "holiday"]
    ] = Field(default_factory=list, max_length=4)


JSON_SCHEMA = ReviewLabel.model_json_schema()

if __name__ == "__main__":
    import json
    print(json.dumps(JSON_SCHEMA, indent=1))
