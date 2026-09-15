"""The predictive-disagreement arm (arm 3): the model PREDICTS, deterministic code DECIDES.

The two existing arms (llm/detect.py) ask for a verdict on the current step and read the anomaly
decision straight out of the reply. This one asks the model, at every step boundary, for two
things -- how long the step now starting will run, and which step will follow it -- and then flags
a disagreement between those predictions and what is actually observed. The architecture matches
PREGO; the factorization matches the HSMM's own P(Z'|Z) and P(d|Z).

Why it exists: PREGO has no duration model, and a duration model is the stated differentiator for
this project. A baseline that cannot predict durations ASSUMES that differentiator rather than
testing it, so this arm is given both halves and the two are scored as independent channels.

Deliberately NOT registered in detect.PROTOCOLS. Every driver in there returns Verdicts, and this
one returns tick-indexed Flags; keeping it out preserves that invariant rather than making
run_trial's return type depend on its argument.

Structured for K-sampling but NOT sampling today. The client call returns a LIST of parsed
predictions, length 1; each channel reduces that list to a (point estimate, score) pair, today
element 0 and a boolean cast to 0/1; and flags carry a float strength, thresholded at 0.5. Later,
`n=K` in one request gives an empirical distribution, the symbol score becomes 1 - P_hat(observed)
and the duration score an empirical tail probability, and the arm becomes threshold-sweepable --
removing the current asymmetry where only the HSMM has a sweep. None of those changes touch the
interfaces below.
"""
import re
from typing import NamedTuple

from cook_ad.eval.element_metrics import StepVerdict, duration_tolerance
from cook_ad.eval.span_metrics import Flag
from cook_ad.llm import detect, textify

PROTOCOL = "predictive"

# One channel name per failure mode, so per-type attribution survives into the diagnostics.
SYMBOL = "p_symbol"
DUR_LONG = "p_dur_long"
DUR_SHORT = "p_dur_short"
CHANNELS = (SYMBOL, DUR_LONG, DUR_SHORT)

# Mirrors element_metrics.CHANNEL_TO_TYPE so this arm's type_confusion is read against the HSMM's
# on the same axes: a symbol disagreement is the transition violation s_transition reports, an
# over-run is what s_temporal reports, and an under-run is s_dur_two's left tail.
CHANNEL_TO_TYPE = {SYMBOL: "omission", DUR_LONG: "repetition", DUR_SHORT: "abandonment"}

# Two labelled lines, parsed INDEPENDENTLY so a malformed line degrades one channel rather than
# both. Each is searched for anywhere in the reply (MULTILINE + search), since models prepend
# chatter, and each is anchored at both ends of its own line so a sentence mentioning a number
# cannot be read as an answer.
#
# The surface tolerances are detect._DURATION's, restated rather than reused: that clause requires
# either the word `for` or parentheses, both of which come from the correction grammar
# (`... would have been pour milk for 19 seconds`), and neither is present in the form THIS prompt
# asks for (`DURATION: 19 seconds`). Accepted here: `19`, `19 seconds`, `19s`, `19 secs`,
# `for 19 seconds`, `(19s)`. Same principle as detect.py's -- being strict about a format slip
# measures obedience instead of prediction.
_DUR_LINE = re.compile(
    r"^\s*DURATION\s*[:=-]?\s*\(?\s*(?:for\s+)?(?P<dur>\d+)\s*(?:seconds?|secs?|s)?\s*\)?"
    r"\s*[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# The optional trailing clause allows `NEXT: pour milk (12s)` and `NEXT: pour milk for 12 seconds`
# -- models copy the duration format they have just been shown -- without loosening the anchor
# into a scan that could pick two words out of a sentence.
_NEXT_LINE = re.compile(
    r"^\s*NEXT\s*[:=-]?\s*(?P<verb>\w+)[\s_]+(?P<noun>\w+)"
    r"\s*(?:\(.*?\)|for\s+\d+\s*\w*)?\s*[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _last(pattern, text):
    """The final match in the reply, or None."""
    m = None
    for m in pattern.finditer(text):
        pass
    return m


class Prediction(NamedTuple):
    duration: int | None                 # predicted total length of the step IN PROGRESS
    symbol: tuple | None                 # (verb, noun) predicted for the step AFTER it
    dur_parse_ok: bool
    next_parse_ok: bool
    raw: str


def parse_predictions(text, vocab):
    """One reply -> a LIST of Predictions, length 1 today.

    A list, not a Prediction, because that is the whole K-sampling seam: `n=K` in one request
    returns K completions and every consumer below already reduces a list.

    Each line is matched on its own against the whole reply, so a reply with a good DURATION and a
    garbled NEXT still yields a usable duration -- and records `next_parse_ok=False`, which is
    reported as a per-line parse failure rather than being silently rescued or silently dropped.
    """
    raw = (text or "").strip()

    # The LAST match, not the first. Without Auto-CoT there is only one of each so this is the
    # same thing; with it the model reasons before answering, and a reasoning sentence that
    # happens to be formatted like an answer line would otherwise be read as the answer.
    duration = None
    m = _last(_DUR_LINE, raw)
    if m:
        duration = int(m.group("dur"))

    symbol = None
    m = _last(_NEXT_LINE, raw)
    if m:
        # Reuses detect._canon_correction's in-vocabulary rule (with a throwaway duration): an
        # out-of-vocabulary prediction cannot be compared against an observed symbol without fuzzy
        # matching, and a fuzzy threshold inside a metric is a tunable knob in the middle of it.
        canon = detect._canon_correction(m.group("verb"), m.group("noun"), 0, vocab)
        if canon is not None:
            symbol = (canon[0], canon[1])

    return [Prediction(duration, symbol, duration is not None, symbol is not None, raw)]


# ---- the request ----------------------------------------------------------------------------

def predictive_prefix(steps, upto):
    """The user turn for the boundary at step `upto`: completed steps in full, then the step that
    has just STARTED, without its duration.

    The missing duration is the whole point. render_step prints `pour milk for 19 seconds`, and
    showing that for the step being asked about would hand the model the answer to the DURATION
    question. The current step therefore renders as `pour milk (in progress)`, which is exactly
    what is observable at the moment the request is issued: the (verb, noun) has been seen, the
    length has not.

    The consequence for caching, stated rather than discovered: request i's text is NOT a strict
    prefix of request i+1's, because line i changes from `(in progress)` to a completed step. The
    shared prefix is everything through line i-1, i.e. all but the last line, so prefix reuse is
    (n-1)/n rather than perfect. That is inherent to asking about a step before it ends, not a
    layout mistake -- and the layout is otherwise unchanged: one system turn, one user turn.
    """
    lines = [f"{j + 1}. {textify.render_step(steps[j])}" for j in range(upto)]
    cur = steps[upto]
    lines.append(f"{upto + 1}. {cur.verb} {cur.noun} (in progress)")
    return "\n".join(lines)


def predictive_messages(system_prompt, steps):
    """Every request for one trial, built upfront -- one per step boundary.

    Prefix-only, no self-feedback: the model's own earlier predictions are never fed back, so each
    boundary is an independent test and every request is a pure function of (trial, step index),
    which is what lets them be issued concurrently through complete_many.
    """
    return [
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": predictive_prefix(steps, i)}]
        for i in range(len(steps))
    ]


def request_cost(steps):
    """One request per step boundary. The last boundary's NEXT is unused (no step follows it) but
    its DURATION is not, so it is still issued."""
    return len(steps)


# ---- the channels ---------------------------------------------------------------------------
#
# Each takes the prediction LIST and reduces it to (point_estimate, score). Today: the point
# estimate is element 0 and the score is the disagreement boolean cast to 0/1. Under n=K sampling
# the same signature carries 1 - P_hat(observed) and an empirical tail probability, with no change
# at either call site.

def symbol_disagreement(predictions, observed):
    """(predicted_symbol, score) for the observed symbol against a prediction list."""
    point = predictions[0].symbol if predictions else None
    if point is None:
        return None, 0.0
    return point, float(point != observed)


def duration_disagreement(predictions, elapsed):
    """(predicted_duration, score) for an elapsed time against a prediction list.

    Tolerance is keyed off the PREDICTED duration -- the only reference available at flag time --
    through the same eval.element_metrics.duration_tolerance the correction metric uses.
    """
    point = predictions[0].duration if predictions else None
    if point is None:
        return None, 0.0
    return point, float(elapsed > point + duration_tolerance(point))


def run_predictive(client, system_prompt, steps, vocab):
    """One trial -> (flags, stats). `flags` are tick-indexed span_metrics.Flags.

    Two channels, independent BY CONSTRUCTION: the duration channel conditions on the OBSERVED
    symbol of the step it is timing, so a substitution cannot leak into the temporal channel and
    corrupt per-type attribution.

      symbol   -- at boundary i >= 1, flag if the observed (verb, noun) differs from the NEXT
                  returned at boundary i-1. Cannot fire at i = 0: no prior prediction exists.
                  Flagged at the first tick of step i, the moment the disagreement is visible.

      duration -- two-sided, and no LLM call for either side; the tick loop is arithmetic.
                  OVER-RUN is live: at each tick while step i runs, flag the FIRST tick where
                  elapsed exceeds the predicted duration by tolerance, then suppress until the
                  next boundary. UNDER-RUN is retrospective: at the boundary that closes step i,
                  flag if the step's observed length fell short of the prediction by the same
                  tolerance, at step i's last tick.

    The under-run side is not in the PREGO formulation and is here deliberately. Over-run alone
    can never catch an abandonment -- that injector makes a step SHORT, and the step that follows
    a truncated segment is the correct next symbol, so the symbol channel is silent too. An arm
    with only the upper tail would report ~0 recall on abandonment as a property of the metric
    rather than of the model, which is exactly the confusion this arm exists to remove. It mirrors
    the HSMM's own s_dur_short / s_dur_long split, and span_metrics' deadline (rather than window)
    matching rule exists to keep retrospective channels like it admissible.
    """
    replies = client.complete_many(predictive_messages(system_prompt, steps))
    preds = [parse_predictions(r, vocab) for r in replies]

    flags = []
    n_dur_fail = n_next_fail = 0
    for p in preds:
        n_dur_fail += int(not p[0].dur_parse_ok)
        n_next_fail += int(not p[0].next_parse_ok)

    for i, step in enumerate(steps):
        # ---- symbol: this step's observed symbol against the PREVIOUS boundary's prediction
        if i >= 1:
            _, score = symbol_disagreement(preds[i - 1], (step.verb, step.noun))
            if score:
                flags.append(Flag(int(step.tick_start), score, SYMBOL))

        # ---- duration, over-run: the first tick at which this step has run too long
        point = preds[i][0].duration
        if point is not None:
            limit = point + duration_tolerance(point)
            for t in range(step.tick_start, step.tick_end):
                elapsed = t - step.tick_start + 1
                if elapsed > limit:
                    flags.append(Flag(int(t), 1.0, DUR_LONG))
                    break  # one alarm per step; suppressed until the next boundary

            # ---- duration, under-run: visible only once the step closes
            if step.duration < point - duration_tolerance(point):
                flags.append(Flag(int(step.tick_end - 1), 1.0, DUR_SHORT))

    flags.sort()
    stats = {
        "n_boundaries": len(steps),
        "duration_parse_failures": n_dur_fail,
        "next_parse_failures": n_next_fail,
    }
    return flags, stats


# Which label survives when two channels fire on one element. Mirrors
# element_metrics.CHANNEL_PRIORITY's reasoning: the most SPECIFIC evidence first. A symbol
# disagreement names a transition that should not have happened; an under-run names a step that
# stopped early; an over-run is the weakest, since a step running long is also what every other
# structural error looks like once the stream stalls.
CHANNEL_PRIORITY = (SYMBOL, DUR_SHORT, DUR_LONG)


def verdicts_from_flags(flags, elements):
    """Tick flags -> one element_metrics.StepVerdict per element, so this arm also lands in the
    existing step/tick tables beside the other three rather than only in the span layer.

    An element is anomalous if it CONTAINS a flag tick. That is the same rule
    element_metrics.step_verdicts_from_flags applies to the HSMM's tick channels, so at
    unit='tick' the predictive arm's verdicts are directly comparable to both other arms'.

    `parse_ok` is always True here and that is deliberate, not a papered-over failure: these
    verdicts are DERIVED arithmetically from predictions, never parsed, and there are ~6.5
    boundaries to a trial against ~121 tick elements, so spreading a per-request parse failure
    over the elements it happens to cover would make evaluate_steps' parse_failure_rate a function
    of the unit. The real per-line failure rates are reported separately by run_predictive.
    """
    by_tick = {}
    for f in flags:
        prev = by_tick.get(f.tick)
        if prev is None or CHANNEL_PRIORITY.index(f.channel) < CHANNEL_PRIORITY.index(prev):
            by_tick[f.tick] = f.channel
    out = []
    for e in elements:
        fired = [by_tick[t] for t in range(e.tick_start, e.tick_end) if t in by_tick]
        if fired:
            channel = min(fired, key=CHANNEL_PRIORITY.index)
            out.append(StepVerdict(e.index, True, CHANNEL_TO_TYPE[channel], None,
                                   parse_ok=True, raw=",".join(sorted(set(fired)))))
        else:
            out.append(StepVerdict(e.index, False, None, None, parse_ok=True, raw=""))
    return out


# ---- Auto-CoT rationale generation (offline, once per exemplar) -------------------------------

# Zero-Shot-CoT, which is how Auto-CoT (Zhang et al., ICLR 2023) fills in each demonstration's
# reasoning without anyone hand-writing it. Run ONCE per exemplar when the prompt is built, not
# per request -- so unlike TI-PREGO's two-pass scheme this costs k calls for the whole sweep
# rather than doubling every one of them, and inference stays a single prefix-cacheable request.
RATIONALE_PROMPT = (
    "Below is the record of a person cooking, up to the present moment. The last line is the step "
    "they have just started.\n\n{prefix}\n\nIt turns out that step ran for {duration} seconds "
    "in total, and the step that followed it was '{next_verb} {next_noun}'.\n\n"
    "In ONE OR TWO SENTENCES, and writing in the present tense as if you did not yet know the "
    "outcome, explain what in the record above points to that duration and that next step. Refer "
    "only to actions and objects that appear in the record. Do not mention that you were told the "
    "answer, and do not restate it in any DURATION/NEXT format."
)


def build_rationales(client, exemplar_steps, boundary_fn):
    """One auto-generated rationale per exemplar, in exemplar order.

    Faithful to Auto-CoT in the part that matters -- nobody writes these by hand, and the
    diversity of the clustered exemplars is what the method leans on rather than the polish of any
    one rationale.

    DELIBERATE DEVIATION from Zhang et al.: the rationale is generated CONDITIONED on the
    exemplar's true outcome. Zhang keeps whatever Zero-Shot-CoT produced as a rationale/answer
    PAIR, so the two are at least self-consistent even when both are wrong. Generating the
    rationale unconditioned and then pairing it with the ground-truth answer -- which is what this
    function did first -- produces demonstrations that openly contradict themselves: measured on
    gemma3:27b, one argued for "around 20-30 seconds" beside an answer of 10, another for "around
    53 seconds" beside an answer of 9, and both invented out-of-vocabulary actions ("wash fruit",
    "slice tomato"). That teaches the model that its reasoning need not reach its answer, which is
    worse than giving it no rationale at all. Since the exemplars are healthy TRAINING trials we
    know the outcome, so conditioning on it costs nothing and keeps every demonstration both
    correct and coherent.
    """
    out = []
    for steps in exemplar_steps:
        b = boundary_fn(steps)
        reply = client.complete([{"role": "user", "content": RATIONALE_PROMPT.format(
            prefix=predictive_prefix(steps, b),
            duration=steps[b].duration,
            next_verb=steps[b + 1].verb,
            next_noun=steps[b + 1].noun,
        )}])
        text = " ".join((reply or "").split())
        out.append(text or "The record so far gives no strong signal.")
    return out
