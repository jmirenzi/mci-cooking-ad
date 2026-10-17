"""A toy world for the drift generator's tests (not a test module): two small recipes with
phases, hand-written expansions with sour and fatty variants, tags, a duration model and a
vocabulary. The knobs are the real configs/drift_synth/knobs.yaml."""
import math

from cook_ad.drift_synth.assets import Assets
from cook_ad.drift_synth.durations import DurationModel
from cook_ad.drift_synth.knobs import KNOBS, load_knobs
from cook_ad.drift_synth.types import RecipeGraph, Step, StepExpansion, Variant

V = Variant


def _graph(rid, steps, edges):
    return RecipeGraph(rid, {k: Step(k, d, ph) for k, (d, ph) in steps.items()}, edges)


OMELET = _graph("omelet", {
    "1": ("Rinse-rinse a tomato", "prep"),
    "2": ("Cut-cut the tomato", "prep"),
    "3": ("Heat-heat oil in a pan", "cook"),
    "4": ("Add-add the tomato to the pan", "cook"),
    "5": ("Crack-crack an egg into a bowl", "prep"),
    "6": ("Serve-serve on a plate with cheese", "serve"),
}, [("1", "2"), ("2", "4"), ("3", "4"), ("4", "6"), ("5", "6")])

SALAD = _graph("salad", {
    "1": ("Rinse-rinse the lettuce", "prep"),
    "2": ("Cut-cut the lettuce", "prep"),
    "3": ("Dress-dress the salad", "serve"),
}, [("1", "2"), ("2", "3")])

EXPANSIONS = {
    "omelet": {
        "1": StepExpansion("1", [("wash", "tomato")], []),
        "2": StepExpansion("2", [("cut", "tomato"), ("cut", "tomato")], []),
        "3": StepExpansion("3", [("turn-on", "hob"), ("pour", "oil")],
                           [V("fatty", "omit", "3", [], [1])]),
        "4": StepExpansion("4", [("add", "tomato"), ("mix", "tomato")],
                           [V("sour", "insert", "4", [("squeeze", "lemon")], None)]),
        "5": StepExpansion("5", [("break", "egg"), ("mix", "egg")], []),
        "6": StepExpansion("6", [("scoop", "mixture"), ("sprinkle", "cheese")],
                           [V("fatty", "substitute", "6", [("sprinkle", "basil")], [1])]),
    },
    "salad": {
        "1": StepExpansion("1", [("wash", "lettuce")], []),
        "2": StepExpansion("2", [("cut", "lettuce")], []),
        "3": StepExpansion("3", [("pour", "oil"), ("mix", "lettuce")],
                           [V("sour", "insert", "3", [("pour", "vinegar")], None),
                            V("fatty", "omit", "3", [], [0])]),
    },
}

TAGS = {
    "verbs": {"cut": ["fine_motor"], "break": ["fine_motor"], "mix": ["fine_motor"],
              "turn-on": ["heat"], "wash": ["cleaning"]},
    "nouns": {"tomato": ["ingredient"], "oil": ["fat", "ingredient"], "hob": ["heat"],
              "egg": ["ingredient"], "lemon": ["sour", "ingredient"],
              "vinegar": ["sour", "ingredient"], "cheese": ["fat", "ingredient"],
              "basil": ["ingredient"], "lettuce": ["ingredient"], "pan": ["container"],
              "bowl": ["container"], "plate": ["container"], "mixture": []},
}

VERBS = ["add", "break", "cut", "mix", "pour", "put", "scoop", "sprinkle", "squeeze", "take",
         "turn-on", "wash", "stall"]
NOUNS = ["basil", "bowl", "cheese", "egg", "hob", "lemon", "lettuce", "mixture", "oil", "pan",
         "plate", "tomato", "vinegar", "idle"]
VOCAB = {"verbs": {v: i for i, v in enumerate(VERBS)}, "nouns": {n: i for i, n in enumerate(NOUNS)}}

DURATIONS = DurationModel(source="toy", verb={"cut": [math.log(4.0), 0.4, 100]},
                          overall=[math.log(2.0), 0.5, 100], gap_zero=0.3,
                          gap=[math.log(1.0), 0.5, 100])

GEN = {
    "logistics_verbs": {"take": "take", "put": "put", "wash": "wash"},
    "person": {"repertoire": [6, 10], "dirichlet_alpha": 1.0, "p_two_sessions": 0.3,
               "theta0": [0.25, 0.75]},
    "day_noise": {"sigma": {"sour": 0.3, "sweet": 0.3, "salty": 0.3, "fatty": 0.3,
                            "mise_en_place": 0.3, "tidy": 0.3, "slow": 0.2, "fine_motor": 0.2},
                  "speed_rho": 0.7, "speed_sigma": 0.08},
    "recognizer": {"p_sub_verb": 0.03, "p_sub_noun": 0.05, "p_delete": 0.04, "p_insert": 0.03,
                   "jitter_s": 0.3, "min_event_s": 0.2},
    "export": {"tick_seconds": 0.5, "sil": ["stall", "idle"], "fps": 20},
}


def toy_assets():
    return Assets(graphs={"omelet": OMELET, "salad": SALAD}, expansions=EXPANSIONS, tags=TAGS,
                  knobs=load_knobs("configs/drift_synth/knobs.yaml"), durations=DURATIONS,
                  vocab=VOCAB, gen=GEN)


def theta(**over):
    t = {k: 0.5 for k in KNOBS}
    t.update(over)
    return t
