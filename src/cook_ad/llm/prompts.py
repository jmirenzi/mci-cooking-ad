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

from cook_ad.llm import textify
from cook_ad.synthetic import error_injection

# Definitions lifted from the corresponding synthetic/error_injection.py injector docstrings. The
# LLM is scored on type accuracy against the injector's ground-truth label, so it has to be judged
# against the definition the injector actually implements -- describing these any other way would
# make the type-confusion matrix measure prompt drift rather than model capability.
ERROR_TYPE_DEFINITIONS = {
    "substitution": (
        "one whole step done with the wrong object, or the wrong action, for its entire "
        "duration -- e.g. spreading mustard instead of jelly for the whole step, not just a "
        "brief flicker"
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


# How the steps arrive differs by protocol, so the sentence describing that has to differ too.
# The `incremental` text is FROZEN: it is part of the response-cache key, and changing a character
# invalidates every cached response collected under it.
_DELIVERY = {
    "incremental": """You will be given a numbered list of the steps taken SO FAR, in order. The list stops at the
present moment: there are no future steps, and the task may not be finished.

Judge ONLY THE LAST step in the list. The earlier steps are context for that judgement -- do not
comment on them, and do not judge them again.""",
    "conversational": """You will be given the steps ONE AT A TIME, in order, as the person performs them. There are no
future steps, the task may not be finished, and you cannot revise an answer you have already
given.

Judge ONLY the step you have just been given. Everything earlier in this conversation is context
for that judgement -- do not comment on it, and do not judge it again.""",
    "batch": """You will be given the complete numbered list of steps for one trial.

Judge EVERY step in the list, in order, one verdict per step.""",
}


def task_block(protocol="incremental"):
    """Protocol + the strict response grammar detect.parse_response is built against."""
    delivery = _DELIVERY.get(protocol, _DELIVERY["incremental"])
    return f"""You are monitoring a person with mild cognitive impairment as they cook a
single breakfast recipe. Each step is one uninterrupted stretch of a single action on a single
object, written as:

    VERB NOUN for NUMBER seconds

{delivery}

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


# --------------------------------------------------------------------------------------------
# the tick unit
# --------------------------------------------------------------------------------------------
#
# A SEPARATE template rather than a parameterised task_block, deliberately. The step wording above
# is frozen -- it is part of the response-cache key, so a template that produced it by
# substitution would risk invalidating ~30k collected responses over a whitespace change. The two
# blocks are meant to be read side by side and kept in sync by hand.
#
# What actually differs, and why:
#   * each line carries ELAPSED time in the run so far, and the tick that starts a new run also
#     reports how long the previous one lasted. The first version of this block printed only
#     'pour cereals' and left the model to count identical lines -- which gave it strictly less
#     than the step unit (`pour cereals for 19 seconds`) and less than the HSMM's duration
#     channels, and was measured producing false `repetition` verdicts inside long normal runs.
#     Both annotations look only BACKWARD, so the protocol stays causal; see
#     textify.render_tick_lines for why the completion note lands on the tick AFTER the run ends.
#   * the judged object is the LAST LINE, i.e. the current second, not a finished step. The
#     anomaly definitions are unchanged -- they describe the same five injector behaviours -- but
#     they are restated in terms of what the RECORD SO FAR shows, since at tick level a step is
#     still in progress when it is judged. An abandonment, in particular, is not yet visible at
#     its first tick, which is a real property of the online problem and not a prompt defect.
#   * the base-rate reminder is stronger. At this unit ~1 line in 20 is anomalous rather than
#     ~1 step in 7, so the prior the model should hold is further toward 'No Anomaly'.
_TICK_DELIVERY = {
    "incremental": """You will be given a numbered, second-by-second record of what the person has done SO FAR: line N
is what they were doing during second N. The record stops at the present moment: there is no
future, and the task may not be finished.

Judge ONLY THE LAST line -- the current second. The earlier lines are context for that judgement
-- do not comment on them, and do not judge them again.""",
    "conversational": """You will be given the person's actions ONE SECOND AT A TIME, in order, as they happen. There is
no future, the task may not be finished, and you cannot revise an answer you have already given.

Judge ONLY the second you have just been given. Everything earlier in this conversation is context
for that judgement -- do not comment on it, and do not judge it again.""",
    "batch": """You will be given the complete second-by-second record of one trial.

Judge EVERY line in the record, in order, one verdict per second.""",
}

_TICK_ANOMALY_MEANINGS = {
    "substitution": (
        "the wrong object, or the wrong action, for the whole of the action currently under way "
        "-- e.g. spreading mustard where the recipe wants jelly"
    ),
    "abandonment": (
        "the action that just ENDED stopped far sooner than that action normally takes, so it "
        "cannot have been completed -- judge this on a line whose bracket reports a finished "
        "action that ran too short"
    ),
    "omission": (
        "the action that just started should have been preceded by something that never "
        "appeared -- the record runs straight from one action into another, skipping a step that "
        "normally sits between them"
    ),
    "transposition": (
        "the action that just started and the one before it are in the wrong order -- both are "
        "present and both run for normal lengths, but they are swapped relative to the usual "
        "sequence"
    ),
    "repetition": (
        "the action under way has already been done once earlier in the trial and is being done "
        "again, or its counter has now passed roughly double the time that action normally "
        "takes. A counter that is merely still rising within the action's normal length is NOT "
        "a repetition"
    ),
}


def tick_task_block(protocol="incremental"):
    """The unit='tick' counterpart of task_block: one line per second, judged per second."""
    delivery = _TICK_DELIVERY.get(protocol, _TICK_DELIVERY["incremental"])
    return f"""You are monitoring a person with mild cognitive impairment as they cook a
single breakfast recipe. You see what they are doing once per second, written as:

    VERB NOUN (Ns)

Each line is exactly one second, so an action that lasts 19 seconds appears as 19 consecutive
lines, and the line number is the elapsed time in seconds since the trial began.

(Ns) is how long the action ON THAT LINE has been going so far, counting the current second.
It COUNTS UP while one action continues and RESETS TO (1s) when a different action begins:

    41. pour milk (3s)
    42. pour milk (4s)
    43. pour milk (5s)
    44. stall kitchen (1s)   [pour milk ended after 5s]
    45. stall kitchen (2s)

So consecutive lines with a rising counter are ONE ongoing action that has lasted that many
seconds -- NOT the action being done again and again. The action is still in progress and you do
not yet know how long it will last in total.

When an action ends, the FIRST line of whatever comes next also reports how long the finished
action ran, in square brackets, as line 44 does above. That bracket is the only place a completed
duration appears -- which means you find out that an action has ended one second after it did.

{delivery}

Reply with EXACTLY one line, in one of these two forms and nothing else -- no preamble, no
explanation, no markdown:

    No Anomaly

    <TYPE> Anomaly. Correct move would have been VERB NOUN for NUMBER seconds

where <TYPE> is exactly one of: {_TYPE_LIST}

and VERB and NOUN are drawn from the vocabularies below. The correction states what the person
should have been doing at this point instead, and for how long that action normally lasts.

The five anomaly types mean, judged at the current second:

""" + "\n".join(
        f"  - {t}: {_TICK_ANOMALY_MEANINGS[t]}" for t in error_injection.ERROR_TYPES
    ) + """

The overwhelming majority of seconds in every trial are NORMAL, including every second in the
middle of an action that is proceeding correctly. Answer 'No Anomaly' unless the CURRENT second
genuinely looks wrong.
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
    """labels.json label -> the same 'VERB NOUN' surface form the observation stream uses.

    Two translations, both so the recipe descriptions name things exactly as the steps do:
    'SIL' becomes '<sil_verb> <sil_noun>', and the underscore in 'stir_dough' becomes a space.

    The underscore matters more than it looks. An earlier version left labels as `stir_dough`, and
    models copied that into their corrections ("Correct move would have been stir_dough for 36
    seconds"), which the response grammar then rejected -- a 25% parse-failure rate on the
    with-recipes arm, caused entirely by the prompt teaching a format the parser refused. The
    parser is now tolerant of both, and this no longer teaches the wrong one.
    """
    if label == "SIL":
        return f"{sil_verb} {sil_noun}"
    return label.replace("_", " ", 1)


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


def build_system_prompt(vocab, with_recipes=False, labels=None, protocol="incremental",
                        unit="step"):
    """The full preprompt. with_recipes=True requires labels (parsed labels.json) -- see this
    module's docstring for why that arm is not a like-for-like comparison against the HSMM.

    `unit` selects task_block (run-length steps) or tick_task_block (one line per second). The
    vocabulary and recipe blocks are unit-independent: both describe the world, not the sampling
    of it. The step path is byte-for-byte what it was before the tick unit existed, which is what
    keeps the existing response cache valid.
    """
    if with_recipes and labels is None:
        raise ValueError("with_recipes=True requires labels (the parsed labels.json)")
    if unit not in textify.UNITS:
        raise ValueError(f"unknown unit: {unit!r} (expected one of {textify.UNITS})")
    head = task_block(protocol) if unit == "step" else tick_task_block(protocol)
    blocks = [head, vocab_block(vocab)]
    if with_recipes:
        blocks.append(recipe_block(labels))
    return "\n\n".join(blocks).strip()


VARIANTS = ("no-recipes", "with-recipes")


def build_variant(variant, vocab, labels=None, protocol="incremental", unit="step"):
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant: {variant!r} (expected one of {VARIANTS})")
    return build_system_prompt(vocab, with_recipes=(variant == "with-recipes"), labels=labels,
                               protocol=protocol, unit=unit)
