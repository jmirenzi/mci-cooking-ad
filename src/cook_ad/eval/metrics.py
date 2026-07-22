import jax.numpy as jnp
import numpy as np

from cook_ad.anomaly import surprise
from cook_ad.lifecycle import divergence

ALL_CHANNELS = tuple(surprise.DEFAULT_THRESHOLDS.keys())
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


def score_trial(flags, window, channels=ALL_CHANNELS, latency_tol=DEFAULT_LATENCY_TOL):
    """Returns (detected, latency, flagged_out_of_window, channels_hit). A detection is any
    flagged tick inside [t0, t1+latency_tol]; latency is the first such tick minus t0. Flags
    strictly before t0 or after the tolerance window are out-of-window (false-positive) flags."""
    t0, t1 = window
    hi = t1 + latency_tol
    flagged = detect(flags, channels)
    in_window = flagged[(flagged >= t0) & (flagged <= hi)]
    detected = in_window.size > 0
    latency = int(in_window[0] - t0) if detected else None
    out_of_window = bool(np.any((flagged < t0) | (flagged > hi)))
    channels_hit = _channels_in_window(flags, window, channels, latency_tol) if detected else []
    return detected, latency, out_of_window, channels_hit


def _any_flag(flags, channels):
    return detect(flags, channels).size > 0


def evaluate(healthy_flags, degraded_by_type, channels=ALL_CHANNELS, latency_tol=DEFAULT_LATENCY_TOL):
    """healthy_flags: list of flag-dicts for healthy control trials (no injected anomaly).
    degraded_by_type: {error_type: [(flags, window), ...]}. Returns a per-error-type report
    (recall, precision, mean latency, n) plus a channel x error-type attribution matrix.

    Precision pools each type's degraded trials with the shared healthy controls: a healthy
    trial that flags anywhere is a false positive, as is a degraded trial that flags only
    outside its window; a degraded trial flagged in-window is a true positive.
    """
    fp_healthy = sum(1 for f in healthy_flags if _any_flag(f, channels))
    n_healthy = len(healthy_flags)

    per_type = {}
    attribution = {}
    for error_type, trials in degraded_by_type.items():
        tp = 0
        fp_out = 0
        latencies = []
        channel_hits = dict.fromkeys(channels, 0)
        for flags, window in trials:
            detected, latency, out_of_window, hits = score_trial(flags, window, channels, latency_tol)
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
