import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm import params

# Every injection needs some segments before/after the perturbation for it to be a genuine
# anomaly (an out-of-order step is only out-of-order relative to context). Trajectories with
# fewer segments than this are skipped by the driver rather than injected into.
MIN_SEGMENTS = 4

ERROR_TYPES = ("substitution", "abandonment", "omission", "transposition", "repetition")


def _seg_bounds(segments):
    """[(state, d), ...] -> [(start, end_exclusive, state, d), ...] over tick indices."""
    bounds = []
    pos = 0
    for state, d in segments:
        bounds.append((pos, pos + d, state, d))
        pos += d
    return bounds


def _result(verb_ids, noun_ids, t0, t1, error_type):
    return {
        "verb_ids": np.asarray(verb_ids, dtype=np.int64),
        "noun_ids": np.asarray(noun_ids, dtype=np.int64),
        "window": (int(t0), int(t1)),
        "error_type": error_type,
    }


def inject_substitution(traj, rng, hsmm_params, select="random"):
    """Swap one tick's noun for a low-probability noun under that tick's subtask -- a clean
    item substitution ('spreading, but with mustard'). The segment is chosen randomly; the
    replacement is the state's least-likely noun so the perturbation is genuinely anomalous."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = _seg_bounds(traj["segments"])

    _, _, _, log_emit_n = params.normalize_categoricals(hsmm_params)
    noun_prob = np.asarray(jnp.exp(log_emit_n))

    i = _pick_segment(rng, len(bounds), lo=0, hi=len(bounds), select=select)
    start, end, state, _ = bounds[i]
    tick = (start + end) // 2
    new_noun = int(np.argmin(noun_prob[state]))
    if new_noun == noun_ids[tick]:
        new_noun = int(np.argsort(noun_prob[state])[1])
    noun_ids[tick] = new_noun
    return _result(verb_ids, noun_ids, tick, tick, "substitution")


def inject_abandonment(traj, rng, keep_ticks=1, select="random"):
    """Truncate an interior segment to ~1 tick (the user drops the step early). Only the
    retrospective left-tail duration channel can catch this; the live stall signal cannot."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = _seg_bounds(traj["segments"])

    i = _pick_segment(rng, len(bounds), lo=1, hi=len(bounds), select=select)  # non-first (need context)
    start, end, _, d = bounds[i]
    keep = min(keep_ticks, d)
    cut_lo, cut_hi = start + keep, end
    verb_ids = np.delete(verb_ids, np.arange(cut_lo, cut_hi))
    noun_ids = np.delete(noun_ids, np.arange(cut_lo, cut_hi))
    onset = start + keep - 1  # the segment's premature end tick (unshifted; cuts are after it)
    return _result(verb_ids, noun_ids, onset, onset, "abandonment")


def inject_omission(traj, rng, select="random"):
    """Drop an interior dependent segment entirely, so its predecessor transitions straight to
    its successor -- an out-of-order / missing-step transition anomaly."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = _seg_bounds(traj["segments"])

    i = _pick_segment(rng, len(bounds), lo=1, hi=len(bounds) - 1, select=select)  # interior
    start, end, _, _ = bounds[i]
    verb_ids = np.delete(verb_ids, np.arange(start, end))
    noun_ids = np.delete(noun_ids, np.arange(start, end))
    return _result(verb_ids, noun_ids, start, start, "omission")  # boundary now at `start`


def inject_transposition(traj, rng, select="random"):
    """Swap two adjacent interior segments -- steps done in the wrong order. Anomaly spans the
    swapped pair (two out-of-order boundaries)."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = _seg_bounds(traj["segments"])

    i = _pick_segment(rng, len(bounds), lo=1, hi=len(bounds) - 2, select=select)  # i and i+1 interior
    a_start, a_end, _, da = bounds[i]
    b_start, b_end, _, db = bounds[i + 1]

    def _swapped(arr):
        return np.concatenate([arr[:a_start], arr[b_start:b_end], arr[a_start:a_end], arr[b_end:]])

    verb_ids = _swapped(verb_ids)
    noun_ids = _swapped(noun_ids)
    return _result(verb_ids, noun_ids, a_start, a_start + da + db - 1, "transposition")


def inject_repetition(traj, rng, select="random"):
    """Duplicate an interior segment in place (the user repeats a step). Shows as an impossible
    re-entry transition or, once Viterbi merges the copy, an over-long stall."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = _seg_bounds(traj["segments"])

    i = _pick_segment(rng, len(bounds), lo=1, hi=len(bounds) - 1, select=select)  # interior
    start, end, _, d = bounds[i]

    def _duplicated(arr):
        return np.concatenate([arr[:end], arr[start:end], arr[end:]])

    verb_ids = _duplicated(verb_ids)
    noun_ids = _duplicated(noun_ids)
    return _result(verb_ids, noun_ids, end, end + d - 1, "repetition")  # the inserted copy


def _pick_segment(rng, n_segments, lo, hi, select):
    """Pick a segment index in [lo, hi). `hi` is exclusive. 'random' (honest recall) or
    'hardest' (leftmost valid -- a deterministic placeholder; a truly-adversarial pick would
    score each candidate's induced surprise, noted as future work)."""
    if hi <= lo:
        raise ValueError(f"trajectory has too few segments ({n_segments}) for this injection")
    if select == "random":
        return int(rng.integers(lo, hi))
    if select == "hardest":
        return lo
    raise ValueError(f"unknown select mode: {select!r}")


def inject(error_type, traj, rng, hsmm_params, select="random"):
    """Uniform dispatch. Only substitution needs the emission table (to pick a low-prob noun);
    the other four are purely structural."""
    if error_type == "substitution":
        return inject_substitution(traj, rng, hsmm_params, select=select)
    if error_type == "abandonment":
        return inject_abandonment(traj, rng, select=select)
    if error_type == "omission":
        return inject_omission(traj, rng, select=select)
    if error_type == "transposition":
        return inject_transposition(traj, rng, select=select)
    if error_type == "repetition":
        return inject_repetition(traj, rng, select=select)
    raise ValueError(f"unknown error type: {error_type!r}")
