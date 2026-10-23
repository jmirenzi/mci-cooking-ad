"""CaptainCook4D words -> EPIC-100 classes, hand-made for the pre-check.

The step expansion (Stage 3) is the LLM's job; this table only lets the pre-check say how much of
each recipe the closed vocabulary can name, and what it cannot name at all. Entries follow EPIC's
own class instance lists (EPIC_100_noun_classes_v2.csv / EPIC_100_verb_classes.csv: e.g.
"tortilla" is an instance of `wrap`, "mug" of `cup`, "wipe" of `wash`) where those exist. None
means EPIC has no class for it. Phrases are matched longest first.
"""
import re

NOUNS = {
    # ingredients
    "baking powder": "powder", "baking soda": "powder", "banana": "banana", "basil": "basil",
    "bell pepper": "pepper", "black pepper": "pepper", "pepper powder": "pepper", "pepper": "pepper",
    "bread": "bread", "baguette": "bread", "english muffin": "bread", "broccoli": "broccoli",
    "butter": "butter", "nut butter": "spreads", "carrot": "carrot", "chaat masala": "spice",
    "cheddar": "cheese", "cheese": "cheese", "mozzarella": "cheese", "parmesan": "cheese",
    "cherry tomato": "tomato", "chocolate": "chocolate", "chocolate frosting": None,
    "cilantro": "coriander", "cinnamon": "cinnamon", "coffee": "coffee", "coffee bean": "coffee",
    "cooking spray": "oil", "corn": "corn", "cornstarch": "flour", "cucumber": "cucumber",
    "cumin": "cumin", "cumin seed": "cumin", "curd": "yoghurt", "egg": "egg", "flavour packet": "package",
    "flour": "flour", "garlic": "garlic", "ginger": "ginger", "green chilli": "chilli",
    "red chilli powder": "chilli", "red chili powder": "chilli", "honey": "spreads",
    "italian herb": "oregano", "italian seasoning": "oregano", "jelly": "spreads", "lettuce": "lettuce",
    "lime": "lime", "lime juice": "lime", "maple syrup": "syrup", "marinara sauce": "sauce",
    "mayonnaise": "sauce", "meatball": "meat", "milk": "milk", "mushroom": "mushroom",
    "mustard": "seed", "noodle": "noodle", "oil": "oil", "olive oil": "oil", "sesame oil": "oil",
    "vegetable oil": "oil", "onion": "onion", "oregano": "oregano", "salsa": "sauce", "salt": "salt",
    "scallion": "onion:spring", "shallot": "onion", "soy sauce": "sauce", "sriracha": "sauce",
    "strawberry": "strawberry", "sugar": "sugar", "sweet and sour sauce": "sauce",
    "sweet-and-sour sauce": "sauce", "tofu": "tofu", "tomato": "tomato", "tomato puree": "sauce",
    "tortilla": "wrap", "tuna": "tuna", "turmeric": "turmeric", "vanilla extract": "extract:vanilla",
    "balsamic vinegar": "vinegar", "water": "water", "zucchini": "courgette", "avocado": "avocado",
    # tools, containers, appliances
    "blender": "blender", "bowl": "bowl", "mixing bowl": "bowl", "ramekin": "bowl", "can": "can",
    "cup": "cup", "mug": "cup", "cutting board": "board:chopping", "dripper": "filter",
    "filter": "filter", "paper filter": "filter", "floss": None, "fork": "fork", "frying pan": "pan",
    "pan": "pan", "skillet": "pan", "jar": "jar", "kettle": "kettle", "knife": "knife",
    "butter knife": "knife", "lid": "lid", "microwave": "microwave", "paper towel": "towel:kitchen",
    "tea towel": "towel", "towel": "towel", "plate": "plate", "platter": "plate",
    "serving dish": "plate", "serving bowl": "bowl", "scissors": "scissors", "sieve": "colander",
    "spatula": "spatula", "fish slice": "spatula", "spiralizer": None, "spoon": "spoon",
    "wooden spoon": "spoon", "tongs": "tongs", "toothpick": "cocktail", "zip top bag": "bag",
    "bag": "bag", "cupcake liner": None, "paper liner": None, "package": "package",
}

# CC4D step verbs ("Verb-description") -> EPIC verb class. None: no class names it.
VERBS = {
    "add": "add", "allow": "wait", "beat": "mix", "blitz": "mix", "boil": "cook", "brush": "brush",
    "check": "check", "chop": "cut", "chop or grate": "cut", "clean": "wash", "coat": "apply",
    "combine": "mix", "continue cooking": "cook", "cook": "cook", "cover": "put",
    "crack": "break", "cross": "move", "cut": "cut", "cut or tear": "cut", "discard": "throw",
    "drain": "filter", "drizzle": "pour", "dry": "dry", "extract": "squeeze", "fill": "fill",
    "flip": "flip", "garnish": "sprinkle", "grind": "crush", "heat": "cook", "invert": "flip",
    "lay": "put", "let": "wait", "line": "put", "lower": "turn-down", "measure": "measure",
    "measure and add": "add", "melt": "cook", "microwave": "turn-on", "mince": "cut", "mix": "mix",
    "mixing": "mix", "open": "open", "pat": "pat", "peel": "peel", "place": "put", "pour": "pour",
    "prepare": "fold", "pull": "pull", "puree": "mix", "put": "put", "remove": "remove",
    "replace": "put", "return": "put", "rinse": "wash", "roll": "roll", "saute": "cook",
    "scoop": "scoop", "seal": "close", "season": "season", "secure": "insert", "serve": "serve",
    "set": "put", "simmer": "cook", "slice": "cut", "slicing": "cut", "slide": "slide",
    "splash": "pour", "spoon": "scoop", "spiralize": "cut", "spread": "apply", "sprinkle": "sprinkle",
    "squeeze": "squeeze", "stir": "mix", "stop": "finish", "take": "take", "thaw": "wash",
    "toast": "cook", "top": "put", "transfer": "pour", "trim": "cut", "turn": "turn-on",
    "wait": "wait", "weigh": "measure", "whisk": "mix",
}

_PHRASES = sorted(NOUNS, key=len, reverse=True)


def step_verb(description):
    """'Chop-Chop 1 strawberry' -> 'chop' (the CC4D step label before the first hyphen)."""
    return description.split("-", 1)[0].strip().lower()


def nouns_in(text):
    """-> (EPIC nouns named, CC4D phrases with no EPIC class), matched longest first so
    'olive oil' is consumed before 'oil'."""
    text = " " + re.sub(r"[^a-z\- ]", " ", text.lower()) + " "
    text = " ".join(_singular(w) for w in text.split()) + " "
    text = " " + text
    found, unmapped = [], []
    for phrase in _PHRASES:
        pat = f" {phrase} "
        while pat in text:
            text = text.replace(pat, " | ", 1)
            (found if NOUNS[phrase] else unmapped).append(NOUNS[phrase] or phrase)
    return found, unmapped


def _singular(word):
    """Crude, but enough for recipe text: tomatoes -> tomato, berries -> berry, cloves -> clove."""
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word
