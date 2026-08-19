"""Counterfactual pairing -- the real fix for blast radius (docs/eval.md, docs/llm.md).

The step-level metric window treats an injected anomaly as a point event with a crisp extent, but
a transposition genuinely disturbs three boundaries plus a belief-state decay tail that runs into
LATER steps. Anything downstream of the ground-truth window is charged as a false positive despite
being *caused by* the injection.

The fix does not widen the window (which just moves the arbitrary cutoff) -- it removes the need
for one. For every degraded trial, the SAME detector is also run on the matching unaltered
(healthy) trial. `tick_map` (synthetic.error_injection) says which degraded tick came from which
healthy tick, so the healthy run's flags can be projected into degraded tick-space and compared
directly:

  - flagged in BOTH the degraded run and its healthy counterfactual -> that detector's baseline
    noise on this trial, nothing to do with the injection;
  - flagged ONLY in the degraded run -> injection-attributable, wherever in the trial it lands.

No new detector runs are needed for this: `run_llm_eval.py` already computes healthy verdicts for
every trial pool, and every request is cached, so pairing is pure post-hoc arithmetic over
already-collected flags.
"""
import numpy as np

from cook_ad.anomaly import surprise
from cook_ad.eval import metrics

ALL_CHANNELS = surprise.CHANNELS
DEFAULT_LATENCY_TOL = 5


def project_flags(healthy_flags, tick_map):
    """healthy_flags: per-channel boolean flag dict over the UNALTERED trial's ticks (whatever
    length that trial is). tick_map: degraded tick -> healthy tick
    (synthetic.error_injection's `tick_map`). Returns a per-channel boolean dict over the
    DEGRADED trial's ticks: `projected[ch][i] == healthy_flags[ch][tick_map[i]]` -- what the
    detector said about that same underlying moment when nothing had been injected.

    Repetition's `tick_map` is many-to-one (the duplicate's ticks map to the same source ticks as
    the original), which is fine here: fancy indexing just reads the source flag twice.
    """
    tick_map = np.asarray(tick_map)
    return {ch: np.asarray(flags)[tick_map] for ch, flags in healthy_flags.items()}


def attributable(degraded_flags, projected):
    """Per-channel boolean dict: ticks flagged in the degraded run but NOT in the projected
    healthy counterfactual -- flags the injection is responsible for. `projected` is
    `project_flags`'s output, already aligned to the degraded trial's tick space."""
    return {
        ch: np.asarray(degraded_flags[ch]) & ~np.asarray(projected[ch])
        for ch in degraded_flags
    }


def _union(flags_by_channel, channels):
    mask = None
    for ch in channels:
        m = np.asarray(flags_by_channel[ch])
        mask = m.copy() if mask is None else (mask | m)
    return mask if mask is not None else np.zeros(0, dtype=bool)


def score_counterfactual(healthy_flags, degraded_flags, tick_map, window, channels=ALL_CHANNELS,
                         latency_tol=DEFAULT_LATENCY_TOL):
    """(detected, localized, latency, attributable_ticks) for one degraded trial paired with its
    healthy counterfactual.

    `detected` -- did the injection change the detector's output AT ALL: at least one
    attributable flag anywhere in the trial, in or out of the ground-truth window. This is what
    "blast radius" downstream disturbance now counts FOR instead of against.

    `localized` -- was the EARLIEST attributable flag inside [t0, t1 + latency_tol]. This is the
    only one of the two that window width can affect, which is why the two are reported
    separately (docs/llm.md) rather than collapsed into one recall number.

    `latency` -- earliest attributable tick minus t0, only meaningful when `localized`.
    """
    projected = project_flags(healthy_flags, tick_map)
    attrib = attributable(degraded_flags, projected)
    mask = _union(attrib, channels)
    ticks = np.flatnonzero(mask)
    if ticks.size == 0:
        return False, False, None, mask

    t0, t1 = window
    hi = t1 + latency_tol
    first = int(ticks[0])
    localized = t0 <= first <= hi
    latency = (first - t0) if localized else None
    return True, localized, latency, mask


def evaluate_counterfactual(healthy_flags_by_trial, degraded_by_type, channels=ALL_CHANNELS,
                            latency_tol=DEFAULT_LATENCY_TOL):
    """healthy_flags_by_trial: [per-channel flag dict, ...], one per usable healthy trial.
    degraded_by_type: {error_type: [(degraded_flags, tick_map, window), ...]}, each error type's
    list index-aligned with `healthy_flags_by_trial` (trial i's healthy flags pair with trial i's
    degraded flags for every error type, mirroring how run_llm_eval.py's shared pool is built).

    Returns detection_rate / localisation_rate / mean_latency per error type, and the healthy
    false-positive rate -- UNCHANGED from metrics.evaluate: a healthy trial is a false positive if
    its own flags contain any flagged tick, full stop. There is nothing to attribute on a trial with
    no injection, so the counterfactual comparison plays no role there.
    """
    fp_healthy = sum(1 for f in healthy_flags_by_trial if metrics._any_flag(f, channels))
    n_healthy = len(healthy_flags_by_trial)

    per_type = {}
    for etype, trials in degraded_by_type.items():
        n = len(trials)
        n_detected = n_localized = 0
        latencies = []
        for i, (degraded_flags, tick_map, window) in enumerate(trials):
            healthy_flags = healthy_flags_by_trial[i]
            detected, localized, latency, _ = score_counterfactual(
                healthy_flags, degraded_flags, tick_map, window, channels, latency_tol
            )
            n_detected += int(detected)
            n_localized += int(localized)
            if localized:
                latencies.append(latency)
        per_type[etype] = {
            "n": n,
            "detection_rate": n_detected / n if n else 0.0,
            "localisation_rate": n_localized / n if n else 0.0,
            "mean_latency": float(np.mean(latencies)) if latencies else float("nan"),
        }

    return {
        "per_type": per_type,
        "healthy": {"n": n_healthy, "false_positive_trials": fp_healthy,
                    "false_positive_rate": fp_healthy / n_healthy if n_healthy else 0.0},
        "channels": list(channels),
    }
