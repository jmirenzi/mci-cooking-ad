import jax.numpy as jnp
import numpy as np

from cook_ad.anomaly import surprise
from cook_ad.lifecycle import divergence

ALL_CHANNELS = surprise.CHANNELS
DEFAULT_LATENCY_TOL = 5


def detect(flags, channels=ALL_CHANNELS):
    """flags: the per-channel boolean dict from surprise.flag. Returns the sorted tick indices
    flagged by ANY channel in the selected subset. The subset is the (deferred) ablation knob."""
    mask = None
    for ch in channels:
        mask = flags[ch].copy() if mask is None else (mask | flags[ch])
    return np.flatnonzero(mask)


def _channels_in_window(flags, window, channels, latency_tol):
    t0, t1 = window
    hi = t1 + latency_tol
    return [ch for ch in channels if np.any(flags[ch][t0 : hi + 1])]


def _persistent_mask(flags, channels, min_run):
    """Union flag mask (as `detect`), collapsed to ticks that belong to a run of >= min_run
    CONSECUTIVE flagged ticks. At the per-state alpha=0.05 quantile calibration surprise.flag
    already provides, a single exceedance over a ~150-tick real trial is expected noise, not
    signal: measured directly on healthy full-scale trials, min_run=1 (the old behavior) gives
    a ~100% trial-level false-positive rate purely from that arithmetic (1-0.95^150 ~= 0.9994),
    for BOTH cascade and joint -- confirming it's a property of scoring every tick independently
    with no persistence requirement, not a per-channel/per-model calibration bug. min_run=10
    was chosen by sweeping the same healthy trials' longest-flagged-run distribution down to a
    ~10% trial-level rate for both models (cascade 10.0%, joint 6.2% at n=80), the smallest
    persistence requirement that gets both into a reasonable band.

    Used ONLY for false-positive determination (_any_flag, and score_trial's out_of_window) --
    never for in-window detection, which stays single-tick sensitive so genuine low-latency
    detections (e.g. substitution's observed 0-tick latency) aren't delayed by min_run ticks."""
    mask = None
    for ch in channels:
        mask = flags[ch].copy() if mask is None else (mask | flags[ch])
    if mask is None:
        return np.zeros(0, dtype=bool)
    if min_run <= 1:
        return mask
    out = np.zeros_like(mask)
    run_start = None
    for i, v in enumerate(mask):
        if v:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                out[run_start:i] = True
            run_start = None
    if run_start is not None and len(mask) - run_start >= min_run:
        out[run_start:] = True
    return out


def score_trial(flags, window, channels=ALL_CHANNELS, latency_tol=DEFAULT_LATENCY_TOL, min_run=1):
    """Returns (detected, latency, flagged_out_of_window, channels_hit). A detection is any
    flagged tick inside [t0, t1+latency_tol] (single-tick sensitive, unaffected by min_run);
    latency is the first such tick minus t0. Flags outside that window only count as a false
    positive once they form a run of >= min_run consecutive flagged ticks (see _persistent_mask)
    -- outside the window there's no injected anomaly to justify single-tick sensitivity, so
    the same persistence requirement that fixes healthy-trial false positives applies there too."""
    t0, t1 = window
    hi = t1 + latency_tol
    flagged = detect(flags, channels)
    in_window = flagged[(flagged >= t0) & (flagged <= hi)]
    detected = in_window.size > 0
    latency = int(in_window[0] - t0) if detected else None
    persistent = np.flatnonzero(_persistent_mask(flags, channels, min_run))
    out_of_window = bool(np.any((persistent < t0) | (persistent > hi)))
    channels_hit = _channels_in_window(flags, window, channels, latency_tol) if detected else []
    return detected, latency, out_of_window, channels_hit


def _any_flag(flags, channels, min_run=1):
    return bool(_persistent_mask(flags, channels, min_run).any())


def evaluate(healthy_flags, degraded_by_type, channels=ALL_CHANNELS, latency_tol=DEFAULT_LATENCY_TOL, min_run=1):
    """healthy_flags: list of flag-dicts for healthy control trials (no injected anomaly).
    degraded_by_type: {error_type: [(flags, window), ...]}. Returns a per-error-type report
    (recall, precision, mean latency, n) plus a channel x error-type attribution matrix.

    Precision pools each type's degraded trials with the shared healthy controls: a healthy
    trial that flags a persistent run (see _persistent_mask/min_run) is a false positive, as is
    a degraded trial with such a run outside its window; a degraded trial flagged in-window is a
    true positive regardless of run length (in-window detection is unaffected by min_run).
    """
    fp_healthy = sum(1 for f in healthy_flags if _any_flag(f, channels, min_run))
    n_healthy = len(healthy_flags)

    per_type = {}
    attribution = {}
    for error_type, trials in degraded_by_type.items():
        tp = 0
        fp_out = 0
        latencies = []
        channel_hits = dict.fromkeys(channels, 0)
        for flags, window in trials:
            detected, latency, out_of_window, hits = score_trial(flags, window, channels, latency_tol, min_run)
            if detected:
                tp += 1
                latencies.append(latency)
                for ch in hits:
                    channel_hits[ch] += 1
            elif out_of_window:
                fp_out += 1
        n = len(trials)
        fp = fp_out + fp_healthy
        per_type[error_type] = {
            "n": n,
            "recall": tp / n if n else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            # Precision with the shared healthy-trial pool excluded from the denominator: this
            # isolates the type-specific false-alarm component from fp_healthy, which is the
            # SAME constant pooled into every error type's `precision` above and therefore
            # partly guarantees the "constant floor across error types" pattern by construction.
            "precision_excl_healthy": tp / (tp + fp_out) if (tp + fp_out) else 0.0,
            "mean_latency": float(np.mean(latencies)) if latencies else float("nan"),
            "tp": tp,
            "fp_out_of_window": fp_out,
        }
        attribution[error_type] = {ch: (channel_hits[ch] / tp if tp else 0.0) for ch in channels}

    return {
        "per_type": per_type,
        "attribution": attribution,
        "healthy": {"n": n_healthy, "false_positive_trials": fp_healthy,
                    "false_positive_rate": fp_healthy / n_healthy if n_healthy else 0.0},
        "channels": list(channels),
    }


def kl_sanity(healthy_traj, degraded_traj, n_nouns, floor=1e-9):
    """Sanity check (NOT a detection metric): KL between the pooled noun-token histograms of the
    healthy vs degraded sets should be nonzero -- the perturbations moved the distribution. Reuses
    lifecycle.divergence.categorical_kl."""
    def _hist(trajs):
        counts = np.full(n_nouns, floor)
        for t in trajs:
            ids = np.asarray(t["noun_ids"])
            counts += np.bincount(ids, minlength=n_nouns)
        return counts / counts.sum()

    p = _hist(healthy_traj)
    q = _hist(degraded_traj)
    return float(divergence.categorical_kl(jnp.log(jnp.array(p)), jnp.log(jnp.array(q))))
