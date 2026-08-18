"""Preprompt construction for the LLM anomaly-detection baseline.

Two variants, which is the whole point of the comparison:

  * **no-recipes** -- the model gets the vocabulary, the response grammar, and the definitions of
    the five error types, but nothing about what these particular recipes look like. Whatever it
    detects, it detects from general prior knowledge about cooking.
  * **with-recipes** -- the same, plus a data-derived description of each of the 10 Breakfast
    recipes.

CAVEAT, restated wherever a with-recipes number is reported: those descriptions are derived from
labels.json, which this repo documents everywhere as VALIDATION-ONLY and never feeds to training
(see docs/README.md's cross-cutting conventions). Handing it to the LLM's preprompt gives that arm
ground-truth task structure the HSMM never saw. The gap between the two variants is the
interesting measurement; the with-recipes number is NOT a like-for-like comparison against the
HSMM and must not be reported as one.
"""
import collections
import statistics

from cook_ad.synthetic import error_injection

# Definitions lifted from the corresponding synthetic/error_injection.py injector docstrings. The
# LLM is scored on type accuracy against the injector's ground-truth label, so it has to be judged
# against the definition the injector actually implements -- describing these any other way would
# make the type-confusion matrix measure prompt drift rather than model capability.
ERROR_TYPE_DEFINITIONS = {
    "substitution": (
        "the right action performed on the wrong object -- a single brief step whose object does "
        "not belong at that point in the task, e.g. spreading, but with mustard"
    ),
    "abandonment": (
        "a step that is dropped early -- it appears, but lasts far less time than that step "
        "normally takes, so it cannot have been completed"
    ),
    "omission": (
        "a step that is missing entirely -- the step before it runs straight into the step after "
        "it, skipping something that normally sits between them"
    ),
    "transposition": (
        "two adjacent steps performed in the wrong order -- both are present and both have normal "
        "durations, but they are swapped relative to the usual sequence"
    ),
    "repetition": (
        "a step that is done twice -- it either appears twice in a row, or appears once for "
        "roughly double its usual duration"
    ),
}

_TYPE_LIST = " | ".join(error_injection.ERROR_TYPES)


def vocab_block(vocab, sil_verb="stall", sil_noun="kitchen"):
    """The exact verb and noun vocabularies. Given verbatim so the model's proposed correction is
    guaranteed in-vocabulary and therefore exactly scoreable against the pre-injection step --
    a free-text correction would need fuzzy matching, which would put a tunable knob in the middle
    of the metric."""
    verbs = sorted(vocab["verbs"], key=lambda v: vocab["verbs"][v])
    nouns = sorted(vocab["nouns"], key=lambda n: vocab["nouns"][n])
    return (
        f"VERBS ({len(verbs)}): {', '.join(verbs)}\n"
        f"NOUNS ({len(nouns)}): {', '.join(nouns)}\n\n"
        f"'{sil_verb}' is not a real action and '{sil_noun}' is not a real object: they are the "
        f"placeholder tokens for 'nothing specific is happening'. A step reading "
        f"'{sil_verb} {sil_noun}' means the person is idle, and idle stretches at the start and "
        f"end of a task are normal, not anomalies."
    )


def task_block():
    """Protocol + the strict response grammar detect.parse_response is built against."""
    return f"""You are monitoring a person with mild cognitive impairment as they cook a
single breakfast recipe. Each step is one uninterrupted stretch of a single action on a single
object, written as:

    VERB NOUN for NUMBER seconds

You will be given a numbered list of the steps taken SO FAR, in order. The list stops at the
present moment: there are no future steps, and the task may not be finished.

Judge ONLY THE LAST step in the list. The earlier steps are context for that judgement -- do not
comment on them, and do not judge them again.

Reply with EXACTLY one line, in one of these two forms and nothing else -- no preamble, no
explanation, no markdown:

    No Anomaly

    <TYPE> Anomaly. Correct move would have been VERB NOUN for NUMBER seconds

where <TYPE> is exactly one of: {_TYPE_LIST}

and VERB and NOUN are drawn from the vocabularies below. The correction states what the person
should have done at this point instead of what they did.

The five anomaly types mean:

""" + "\n".join(
        f"  - {t}: {ERROR_TYPE_DEFINITIONS[t]}" for t in error_injection.ERROR_TYPES
    ) + """

Most steps in most trials are NORMAL. Answer 'No Anomaly' unless the LAST step genuinely looks
wrong.
"""


def _canonical_recipes(labels, top_k_variants=1):
    """Per recipe: the modal collapsed subtask-label sequence and each step's median duration.

    Derived from labels.json rather than written by hand, so the description matches the corpus
    the detector is scored on instead of an idealized recipe someone typed out. Consecutive
    repeats are collapsed first, matching how a trial renders into steps (textify's RLE).
    """
    seqs_by_recipe = collections.defaultdict(list)
    durs_by_step = collections.defaultdict(lambda: collections.defaultdict(list))

    for trial in labels:
        collapsed = []
        for label in trial["subtask_labels"]:
            if collapsed and collapsed[-1][0] == label:
                collapsed[-1][1] += 1
            else:
                collapsed.append([label, 1])
        recipe = trial["recipe_label"]
        seqs_by_recipe[recipe].append(tuple(s for s, _ in collapsed))
        for step, d in collapsed:
            durs_by_step[recipe][step].append(d)

    out = {}
    for recipe, seqs in seqs_by_recipe.items():
        counts = collections.Counter(seqs)
        modal, n_modal = counts.most_common(top_k_variants)[0]
        out[recipe] = {
            "steps": [
                {"label": s, "median_seconds": int(statistics.median(durs_by_step[recipe][s]))}
                for s in modal
            ],
            "n_trials": len(seqs),
            "n_modal": n_modal,
            "n_distinct_orders": len(counts),
        }
    return out


def _label_to_step_text(label, sil_verb="stall", sil_noun="kitchen"):
    """labels.json writes the idle class as the bare sentinel 'SIL', but the steps the model is
    actually shown render it as '<sil_verb> <sil_noun>' (textify uses lexicon.verb/noun, which do
    not collapse SIL). Translate here so the recipe descriptions and the observation stream name
    the same thing the same way -- otherwise the model is told to match a token it never sees."""
    return f"{sil_verb}_{sil_noun}" if label == "SIL" else label


def recipe_block(labels):
    """The with-recipes addendum. Reports how often the modal order actually occurs, so the model
    is told which recipes are rigidly ordered (cereals, tea, milk) and which vary a lot across
    participants (salat: 50 distinct orders in 52 trials) rather than being handed a single
    canonical order as if it were a rule."""
    canon = _canonical_recipes(labels)
    lines = [
        "The person is making exactly one of the following recipes; you are not told which, and "
        "you should infer it from the steps as they arrive. Each recipe is shown as its most "
        "common step order, with each step's typical duration in seconds.",
        "",
    ]
    for recipe in sorted(canon):
        info = canon[recipe]
        steps = ", ".join(
            f"{_label_to_step_text(s['label'])} (~{s['median_seconds']}s)" for s in info["steps"]
        )
        lines.append(f"  {recipe}: {steps}")
        lines.append(
            f"      (most common order in {info['n_modal']}/{info['n_trials']} recorded trials; "
            f"{info['n_distinct_orders']} distinct orders observed overall)"
        )
    lines += [
        "",
        "Step labels above are written verb_noun and use the same vocabulary as the steps you "
        "will be shown. Orders vary between people: treat the sequences above as typical, not "
        "mandatory.",
    ]
    return "\n".join(lines)


def build_system_prompt(vocab, with_recipes=False, labels=None):
    """The full preprompt. with_recipes=True requires labels (parsed labels.json) -- see this
    module's docstring for why that arm is not a like-for-like comparison against the HSMM."""
    if with_recipes and labels is None:
        raise ValueError("with_recipes=True requires labels (the parsed labels.json)")
    blocks = [task_block(), vocab_block(vocab)]
    if with_recipes:
        blocks.append(recipe_block(labels))
    return "\n\n".join(blocks).strip()


VARIANTS = ("no-recipes", "with-recipes")


def build_variant(variant, vocab, labels=None):
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant: {variant!r} (expected one of {VARIANTS})")
    return build_system_prompt(vocab, with_recipes=(variant == "with-recipes"), labels=labels)
