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


def gt_steps_for_window(steps, window):
    """Tick-space injection window (t0, t1) inclusive -> the step indices it overlaps.

    This is the bridge between synthetic.error_injection's tick-space ground truth and the
    step-space unit both detectors are scored in. Checked against all five injectors:
    substitution's whole retagged segment keeps the segment's own boundaries, so it becomes one
    step of unchanged duration but a different (verb, noun) label; abandonment's truncated
    segment becomes a shorter step; omission's window is the new boundary tick, i.e. the first
    tick of the following step; transposition's window spans both swapped runs; repetition's
    duplicate is adjacent-identical to its original and so MERGES into one double-length step --
    which is the same 'Viterbi merges the copy' behavior docs/synthetic.md already documents for
    the tick-level path, not a new approximation.
    """
    t0, t1 = int(window[0]), int(window[1])
    return [s.index for s in steps if s.tick_start <= t1 and s.tick_end > t0]


def gt_steps_for_ticks(steps, anomaly_ticks):
    """The elements covering `degraded["anomaly_ticks"]` -- the injector's own list of the ticks
    that are actually anomalous (synthetic/error_injection._result), as opposed to the whole
    extent it disturbed.

    Supersedes gt_steps_for_window as the source of ground truth. The window is still what
    injection_touched_steps uses to decide the debris extent, so the two are used together:
    points are ground truth, window-minus-points is debris.
    """
    ticks = sorted({int(t) for t in anomaly_ticks})
    return [s.index for s in steps if any(s.tick_start <= t < s.tick_end for t in ticks)]


def injection_touched_steps(steps, tick_map, edited_ticks, gt_steps, window=None):
    """Degraded steps that exist, or take the tick range they have, only because of the
    injection -- but are NOT themselves the ground-truth anomaly. Neither the injected anomaly
    nor a clean normal step: debris, to be excluded from false-positive scoring rather than
    counted either way (eval/element_metrics.py).

    Three ways a step can be debris:
      0. It lies inside `window` -- the injection's whole disturbed extent -- without being one
         of the anomalous points. This is the rule that carries the weight once ground truth is
         points rather than ranges: a transposition's ground truth is three junction ticks, and
         the ~46 ticks between them are a correctly-executed run in the wrong place, which is
         neither a hit nor a false alarm. Pass `window=None` to disable it and recover the
         pre-points behaviour exactly.
      1. It contains an edited tick, or is directly adjacent to one (`edited_ticks`), without
         itself being the injected step.
      2. Its own tick range is glued from two non-contiguous original positions -- consecutive
         degraded ticks i, i+1 INSIDE the step with `tick_map[i+1] != tick_map[i] + 1`.

    Which of them fires, measured over 25 real trials x 5 injections at both units (the earlier
    version of this docstring asserted the exact opposite of each line below, having been written
    before substitution became a whole-segment edit):

    | injector      | rule 1: contains | rule 2: borders | rule 3: interior splice |
    |---------------|------------------|-----------------|-------------------------|
    | substitution  | 0                | 50 (2/trial)    | 0                       |
    | the other four| 0                | 0               | 0                       |

    So the ADJACENCY half of rule 1 is the only live rule, and only for substitution: that
    injector rewrites a whole segment, so the runs either side of it border an edited tick. Its
    own segment is the ground truth and is skipped before any rule runs.

    The interior-splice rule has never fired for any shipped injector at either unit, because
    every splice these five introduce lands exactly where the RLE already breaks -- and a splice
    ON a step boundary is deliberately not debris. At unit="tick" that rule cannot fire even in
    principle (`range(a, b - 1)` is empty when a step is one tick wide), which is the SAME policy
    on a finer grid rather than a lost case: a splice between two ticks is always on an element
    boundary. It is kept for a future injector that edits mid-run.

    The one thing that does change with the unit is the SIZE of substitution's excused halo:
    2 bordering steps (~22 ticks) at unit="step", 2 bordering ticks at unit="tick". The tick unit
    is therefore stricter about false positives near a substitution, which is one reason
    precision does not transfer between units.
    """
    tick_map = np.asarray(tick_map)
    edited = set(int(t) for t in edited_ticks)
    gt = set(gt_steps)

    touched = set()
    for s in steps:
        if s.index in gt:
            continue
        a, b = s.tick_start, s.tick_end
        if window is not None and a <= int(window[1]) and b > int(window[0]):
            touched.add(s.index)
            continue
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


# --------------------------------------------------------------------------------------------
# the tick unit
# --------------------------------------------------------------------------------------------
#
# Everything above encodes a trial as run-length STEPS, and every consumer -- the prompt builder,
# the ground-truth bridge, eval/element_metrics.py -- is written against Step's (index,
# tick_start, tick_end) triple rather than against the run-length property itself. So the second
# unit is not a second pipeline: it is the same Step type at duration 1, one per tick, with
# index == tick_start == the tick number.
#
# That is what makes the two units comparable. The HSMM emits a value at every tick and the step
# unit collapses those to one verdict per run (element_metrics.step_verdicts_from_flags), which
# is a handicap the step unit imposes on BOTH arms equally but which no longer has to be imposed
# at all once the LLM is asked per tick. Read the resulting numbers knowing what changed with it:
# a trial offers ~18x more chances to false-positive (12227 ticks vs 661 steps over the test
# split), so precision is not comparable ACROSS units even though it is comparable across arms
# WITHIN one. evaluate_steps reports chance_precision for exactly this reason -- quote it.
UNITS = ("step", "tick")


def _check_unit(unit):
    if unit not in UNITS:
        raise ValueError(f"unknown unit: {unit!r} (expected one of {UNITS})")


def ticks_from_ids(verb_ids, noun_ids, lexicon):
    """One Step per tick, in tick order: the identity segmentation.

    Deliberately the same NamedTuple rather than a parallel type. gt_steps_for_window,
    injection_touched_steps, step_covering_tick and element_metrics all read only tick_start /
    tick_end / index, so they work on this unchanged -- and a tick element's `index` IS its tick,
    which makes a tick-level ground-truth window map to itself.
    """
    verb_ids = np.asarray(verb_ids, dtype=np.int64)
    noun_ids = np.asarray(noun_ids, dtype=np.int64)
    return [
        Step(index=t, tick_start=t, tick_end=t + 1, verb_id=int(v), noun_id=int(n),
             verb=lexicon.verb(int(v)), noun=lexicon.noun(int(n)), duration=1)
        for t, (v, n) in enumerate(zip(verb_ids.tolist(), noun_ids.tolist()))
    ]


def ticks_from_trajectory(traj, lexicon):
    return ticks_from_ids(traj["verb_ids"], traj["noun_ids"], lexicon)


def elements_from_ids(verb_ids, noun_ids, lexicon, unit="step"):
    _check_unit(unit)
    fn = steps_from_ids if unit == "step" else ticks_from_ids
    return fn(verb_ids, noun_ids, lexicon)


def elements_from_trajectory(traj, lexicon, unit="step"):
    """The unit switch every caller goes through. `unit='step'` is the historical behaviour."""
    _check_unit(unit)
    fn = steps_from_trajectory if unit == "step" else ticks_from_trajectory
    return fn(traj, lexicon)


def render_tick_lines(ticks):
    """Tick elements -> one line each, carrying elapsed time in the CURRENT run and, on the tick
    where a run ends, how long the run that just ended lasted:

        5. pour milk (5s)
        6. stall kitchen (1s)   [pour milk ended after 5s]

    The elapsed counter exists for fairness, and its absence was a defect. The step unit says
    `pour cereals for 19 seconds` outright, and the HSMM's duration channels carry elapsed
    occupancy for free; a tick rendering that printed only `pour cereals` gave the LLM strictly
    LESS than either and made every tick-vs-step difference partly measure "the durations were
    deleted". It also forced the model to detect anomalies by counting identical lines, which is
    arithmetic rather than the capability under test -- measured on the first tick-unit smoke run
    as repeated false `repetition` verdicts fired inside long, normal runs.

    Both annotations are strictly BACKWARD-LOOKING, which is what keeps the protocol causal and
    the prefixes append-only. In particular the completion note lands on the FIRST tick of the
    next run, not the last tick of the finished one: at second 5 the model cannot yet know the
    step is over, and it learns so at second 6. Attaching it a tick earlier would leak one tick
    of the future onto exactly the type whose whole signal is "this step ended too early", and
    would inflate abandonment recall for free.
    """
    lines, elapsed, previous = [], 0, None
    for i, e in enumerate(ticks):
        note = ""
        if i and (e.verb, e.noun) == (ticks[i - 1].verb, ticks[i - 1].noun):
            elapsed += 1
        else:
            if i:
                previous = (ticks[i - 1].verb, ticks[i - 1].noun, elapsed)
                note = f"   [{previous[0]} {previous[1]} ended after {previous[2]}s]"
            elapsed = 1
        lines.append(f"{e.verb} {e.noun} ({elapsed}s){note}")
    return lines


def render_tick(step):
    """One tick with no run context -- `'pour cereals'`. render_tick_lines is what the prompts
    actually use; this stays for callers holding a single element."""
    return f"{step.verb} {step.noun}"


def render_element(step, unit="step"):
    _check_unit(unit)
    return render_step(step) if unit == "step" else render_tick(step)


def render_trial(steps, unit="step"):
    """One rendered line per element. The tick unit renders the LIST rather than each element
    independently, because a tick line reports how long the current run has been going, which no
    single element knows. Line i still depends only on elements <= i."""
    _check_unit(unit)
    if unit == "tick":
        return render_tick_lines(steps)
    return [render_step(s) for s in steps]
