"""Trajectory -> text, for the LLM comparison baseline.

A **step** is one maximal run of constant (verb, noun) over consecutive ticks -- exactly the
run-length encoding of the observation stream. This is the unit the LLM baseline reads, answers
about, and is scored on (eval/element_metrics.py), and it is deliberately NOT the model's latent
segmentation: z_star comes from Viterbi, which the LLM never sees. Two detectors reading two
different segmentations would not be comparable.

Measured before committing to RLE (both at K=64 on the full corpus): real Breakfast trials give a
median 6 runs/trial at median 11s, and synthetic ancestral samples give a median 7 runs at median
11s with only 12% of runs 1 tick long -- the fitted emissions are peaked enough that per-tick
i.i.d. sampling still yields clean runs. So the same rendering works on both healthy sources with
no special-casing of synthetic/generate.py.
"""
from itertools import groupby
from typing import NamedTuple

import numpy as np

# tick_seconds is 1.0 in both shipped configs, so a run length in ticks IS a duration in seconds.
# assert_tick_seconds() makes that dependency explicit rather than letting "seconds" silently
# become a lie if the binning ever changes.
TICK_SECONDS = 1.0


class Step(NamedTuple):
    index: int          # position in the trial's step list
    tick_start: int
    tick_end: int       # exclusive
    verb_id: int
    noun_id: int
    verb: str
    noun: str
    duration: int       # ticks == seconds (see TICK_SECONDS)


def assert_tick_seconds(config):
    """Guard: the rendered text says 'seconds', which is only true at 1s ticks."""
    ts = float(config.get("tick_seconds", TICK_SECONDS))
    if ts != TICK_SECONDS:
        raise ValueError(
            f"textify renders durations as seconds, which assumes tick_seconds == 1.0, got {ts}. "
            "Either rebin the dataset or teach render_step to scale ticks -> seconds."
        )


def steps_from_ids(verb_ids, noun_ids, lexicon):
    """Run-length encode (verb_ids, noun_ids) into Steps.

    Names come from lexicon.verb()/lexicon.noun(), NOT lexicon.phrase(): phrase() collapses the
    SIL sentinels ('stall'/'kitchen') to 'idle' or to a bare noun, which is right for a narrated
    query card but breaks the fixed 'VERB NOUN' template this baseline's response grammar is
    built on -- and an unparseable reply is scored as a parse failure, not as a verdict. SIL
    therefore renders literally as 'stall kitchen'; prompts.py explains the token instead.
    """
    verb_ids = np.asarray(verb_ids, dtype=np.int64)
    noun_ids = np.asarray(noun_ids, dtype=np.int64)
    steps = []
    pos = 0
    for (v, n), group in groupby(zip(verb_ids.tolist(), noun_ids.tolist())):
        d = sum(1 for _ in group)
        steps.append(Step(
            index=len(steps), tick_start=pos, tick_end=pos + d,
            verb_id=int(v), noun_id=int(n),
            verb=lexicon.verb(v), noun=lexicon.noun(n), duration=d,
        ))
        pos += d
    return steps


def steps_from_trajectory(traj, lexicon):
    """Same, for a trajectory dict from synthetic.generate or synthetic.error_injection. Both
    shapes carry verb_ids/noun_ids, so healthy and degraded trials render identically."""
    return steps_from_ids(traj["verb_ids"], traj["noun_ids"], lexicon)


def render_step(step):
    """'pour cereals for 19 seconds' / 'take bowl for 1 second'."""
    unit = "second" if step.duration == 1 else "seconds"
    return f"{step.verb} {step.noun} for {step.duration} {unit}"


def render_trial(steps):
    return [render_step(s) for s in steps]


def gt_steps_for_window(steps, window):
    """Tick-space injection window (t0, t1) inclusive -> the step indices it overlaps.

    This is the bridge between synthetic.error_injection's tick-space ground truth and the
    step-space unit both detectors are scored in. Checked against all five injectors:
    substitution's single retagged tick splits a run into three, so it becomes its own 1-second
    step; abandonment's truncated segment becomes a 1-second step; omission's window is the new
    boundary tick, i.e. the first tick of the following step; transposition's window spans both
    swapped runs; repetition's duplicate is adjacent-identical to its original and so MERGES into
    one double-length step -- which is the same 'Viterbi merges the copy' behavior docs/
    synthetic.md already documents for the tick-level path, not a new approximation.
    """
    t0, t1 = int(window[0]), int(window[1])
    return [s.index for s in steps if s.tick_start <= t1 and s.tick_end > t0]


def injection_touched_steps(steps, tick_map, edited_ticks, gt_steps):
    """Degraded steps that exist, or take the tick range they have, only because of the
    injection -- but are NOT themselves the ground-truth anomaly. Neither the injected anomaly
    nor a clean normal step: debris, to be excluded from false-positive scoring rather than
    counted either way (eval/element_metrics.py).

    Two ways a step can be debris:
      1. It borders an edited tick (is directly adjacent to, or contains, a tick in
         `edited_ticks`). Substitution retags exactly one tick inside a run, splitting what was
         one step into up to three; the two surviving fragments are shortened runs that exist
         only because the edit forced an RLE break next to them, even though neither fragment's
         own ticks were rewritten.
      2. Its own tick range is glued from two non-contiguous original positions -- consecutive
         degraded ticks i, i+1 inside the step with `tick_map[i+1] != tick_map[i] + 1`.
         Repetition's duplicate is adjacent-identical to its original and merges into one
         over-long run in the degraded RLE, spanning the splice.

    Splices that fall exactly on an existing step boundary (abandonment/omission/transposition:
    every splice a structural injector introduces lands where the RLE already breaks) are NOT
    debris by either rule -- confirmed against each injector's index arithmetic in
    synthetic/error_injection.py, and is why this function only ever flags something for the
    substitution and repetition injectors in practice.
    """
    tick_map = np.asarray(tick_map)
    edited = set(int(t) for t in edited_ticks)
    gt = set(gt_steps)

    touched = set()
    for s in steps:
        if s.index in gt:
            continue
        a, b = s.tick_start, s.tick_end
        if any(t in edited for t in range(a, b)):
            touched.add(s.index)
            continue
        if (a - 1) in edited or b in edited:
            touched.add(s.index)
            continue
        if any(tick_map[t + 1] != tick_map[t] + 1 for t in range(a, b - 1)):
            touched.add(s.index)
    return touched


def step_covering_tick(steps, tick):
    """The Step whose [tick_start, tick_end) contains `tick`, or None.

    Applied to the SOURCE (pre-injection) trial's steps at the degraded window's FIRST tick, this
    recovers the ground-truth "correct move" for every one of the five injectors with a single
    rule, because each injector only rewrites ticks at or after its window start and leaves every
    earlier tick untouched:

      substitution  -> the original run whose one tick was retagged
      abandonment   -> the same step at its FULL original duration (the duration is the correction)
      omission      -> the deleted step (window start is its first source tick)
      transposition -> the step that should have come first of the swapped pair
      repetition    -> the step that should have followed instead of the duplicate

    Verified against each injector's index arithmetic in synthetic/error_injection.py.
    """
    for s in steps:
        if s.tick_start <= tick < s.tick_end:
            return s
    return None
