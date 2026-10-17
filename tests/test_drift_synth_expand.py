"""Stage 3 checks that need no LLM: reply parsing, validation, schema, exemplar lookup."""
from _drift_fixtures import OMELET, TAGS, VOCAB
from cook_ad.drift_synth.expand import (ExemplarIndex, parse, schema, user_prompt,
                                        validate_expansion)
from cook_ad.drift_synth.types import StepExpansion, Variant


def _reply(tokens, variants=()):
    pair = lambda t: {"verb": t[0], "noun": t[1]}
    return {"tokens": [pair(t) for t in tokens],
            "variants": [{"knob": k, "kind": kind, "tokens": [pair(t) for t in toks], "replaces": rep}
                         for k, kind, toks, rep in variants]}


def test_valid_reply_parses_and_passes():
    exp = parse("4", _reply([("add", "tomato"), ("pour", "oil")],
                            [("sour", "insert", [("squeeze", "lemon")], []),
                             ("fatty", "omit", [], [1])]))
    assert validate_expansion(exp, VOCAB, TAGS) == []
    assert exp.variants[0].replaces is None and exp.variants[1].replaces == [1]


def test_validation_catches_what_a_schema_cannot():
    bad = StepExpansion("4", [("take", "tomato"), ("pour", "oil"), ("fly", "kite")], [
        Variant("sour", "insert", "4", [("pour", "oil")], None),          # not sour-tagged
        Variant("fatty", "omit", "4", [], [0]),                # tomato is not fat
        Variant("fatty", "substitute", "4", [("pour", "oil")], [9]),  # out of range
        Variant("sour", "substitute", "4", [("pour", "oil")], [1]),        # wrong kind
    ])
    errs = " | ".join(validate_expansion(bad, VOCAB, TAGS))
    for needle in ("fetch logistics", "'fly kite' is not in the vocabulary", "sour-tagged",
                   "fat-tagged", "must be indices", "allows only"):
        assert needle in errs, needle
    assert validate_expansion(StepExpansion("1", [], []), VOCAB, TAGS)


def test_schema_enumerates_the_vocab_without_sil():
    s = schema(VOCAB)
    verbs = s["properties"]["tokens"]["items"]["properties"]["verb"]["enum"]
    assert "stall" not in verbs and "cut" in verbs


def test_prompt_names_neighbours_and_exemplars():
    text = user_prompt(OMELET.steps["4"], OMELET, VOCAB, [[("cut", "tomato"), ("pour", "oil")]])
    assert "Step (cook): Add-add the tomato to the pan" in text
    assert "Cut-cut the tomato" in text and "Serve-serve" in text
    assert "- cut tomato, pour oil" in text and "idle" not in text


def test_exemplar_index_prefers_overlap_and_distinct_videos():
    videos = {
        "a": [(i, i + 1, "cut", n) for i, n in enumerate(["tomato", "tomato", "onion", "pan"] * 3)],
        "b": [(i, i + 1, "wash", n) for i, n in enumerate(["cup", "plate"] * 6)],
        "c": [(i, i + 1, "pour", n) for i, n in enumerate(["oil", "pan", "tomato", "egg"] * 3)],
    }
    idx = ExemplarIndex(videos, width=4, stride=2)
    best = idx.best(["tomato", "pan", "egg"], k=2)
    assert len(best) == 2 and best[0][0] == ("pour", "oil")         # c shares 3 nouns
    assert all(a != b for a, b in zip(best[1], best[1][1:]))         # runs collapsed
    assert idx.best(["kite"]) == []


def test_core_ingredients_must_come_from_the_recipe():
    exp = StepExpansion("8", [("sprinkle", "basil"), ("mix", "mixture")], [])
    assert validate_expansion(exp, VOCAB, TAGS, ingredients=["basil", "tomato"]) == []
    errs = validate_expansion(exp, VOCAB, TAGS, ingredients=["tomato"])
    assert len(errs) == 1 and "not an ingredient of this recipe" in errs[0]
    var = Variant("sour", "insert", "8", [("squeeze", "lemon")], None)
    assert validate_expansion(StepExpansion("8", [("mix", "mixture")], [var]), VOCAB, TAGS,
                              ingredients=["tomato"]) == []
