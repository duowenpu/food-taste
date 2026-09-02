"""Synthetic review set for throughput benchmarking (no real data needed)."""
import json, random
random.seed(7)
RECIPES = [
    ("Creamy Tuscan Chicken", ["chicken breast", "heavy cream", "sun-dried tomatoes", "spinach", "garlic", "parmesan", "salt", "pepper"]),
    ("Beef and Broccoli Stir Fry", ["flank steak", "broccoli", "soy sauce", "brown sugar", "garlic", "ginger", "cornstarch", "sesame oil"]),
    ("Lemon Blueberry Muffins", ["flour", "sugar", "blueberries", "lemon zest", "eggs", "butter", "milk", "baking powder"]),
    ("Slow Cooker Pulled Pork", ["pork shoulder", "bbq sauce", "brown sugar", "paprika", "onion", "apple cider vinegar", "salt"]),
    ("Thai Red Curry", ["coconut milk", "red curry paste", "chicken", "bell pepper", "bamboo shoots", "fish sauce", "basil", "sugar"]),
    ("Classic Meatloaf", ["ground beef", "breadcrumbs", "egg", "onion", "ketchup", "worcestershire", "salt", "pepper"]),
    ("Shrimp Scampi Pasta", ["shrimp", "linguine", "butter", "garlic", "white wine", "lemon", "parsley", "red pepper flakes"]),
    ("Honey Garlic Salmon", ["salmon", "honey", "soy sauce", "garlic", "lemon", "olive oil"]),
]
OPEN = ["Made this last night.", "Second time making this.", "Tried this for a dinner party.", "Quick weeknight dinner.",
        "Followed the recipe exactly.", "Made a few changes.", "", "My husband requested this again."]
MID = [
    "Way too salty, I'd cut the {ing} by half.", "Needed more {ing}, it was a little bland.",
    "Perfect amount of heat.", "A bit too sweet for my taste, reduced the {ing}.",
    "The {ing2} came out dry, I'd bake 10 minutes less.", "Took almost double the stated time to cook through.",
    "I added a splash of lemon at the end and it really brightened it up.", "Substituted {ing} with what I had on hand.",
    "Skin got nice and crispy.", "Sauce was watery, I'd add more cornstarch.", "Rich and flavorful, restaurant quality.",
    "Kids loved it, even my picky eater.", "Easy enough for a beginner.", "Too greasy, drained the fat next time.",
    "Great for meal prep, reheats well.", "Doubled the garlic because we love garlic.",
]
CLOSE = ["Will make again!", "Definitely a keeper.", "Probably won't make again.", "Thanks for sharing.", "5 stars.", "", "Yum!"]
SHORT = ["Delicious!", "yum", "Great recipe", "Will try soon", "Thanks!!", "So good", "Loved it"]

rows = []
for i in range(3000):
    name, ings = random.choice(RECIPES)
    if random.random() < 0.3:
        text = random.choice(SHORT)
    else:
        parts = [random.choice(OPEN)] + [random.choice(MID).format(ing=random.choice(ings), ing2=ings[0]) for _ in range(random.randint(1, 3))] + [random.choice(CLOSE)]
        text = " ".join(p for p in parts if p)
    rows.append({"id": f"syn{i}", "recipe_name": name, "ingredients": ings, "rating": random.choice([0, 3, 4, 4, 5, 5, 5]), "text": text})
with open("data/synthetic.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "rows -> data/synthetic.jsonl")
