import jax.numpy as jnp
import numpy as np
from scipy import stats

from cook_ad.hsmm import durations
from cook_ad.lifecycle import divergence

# `online_update.COUNT_FIELD` covers verb/noun/trans only: durations are NB point estimates
# (dur_r/dur_p), not a conjugate Dirichlet count family, so there is no live/frozen split and
# no PreferenceEvent-equivalent for duration. This module estimates drift as a standalone
# report; feeding it back into DualModel is future work (see the integration-gap note at the
# bottom of this file).

DEFAULT_R_FALLBACK = 5.0
DEFAULT_MIN_INSTANCES = 5
DEFAULT_MIN_DELTA_TICKS = 2.0
DEFAULT_ALPHA = 0.05


def _usable_segments(segments):
    """Drop each trial's final segment: it is the only structurally right-censored one
    (durations.impute_censored_histogram's docstring). Including it without imputation biases
    the mean downward; dropping it is the honest cheap fix for a lightweight drift report."""
    return segments[:-1] if len(segments) > 1 else []


def duration_histogram(segments_by_trial, k_subtask, d_max):
    """segments_by_trial: list of per-trial [(state, duration), ...]. Returns (K, d_max) counts,
    binned the same way durations.duration_tables indexes columns: column 0 = d=1, last column
    absorbs the tail (d >= d_max)."""
    hist = np.zeros((k_subtask, d_max), dtype=np.float64)
    for segments in segments_by_trial:
        for state, d in _usable_segments(segments):
            col = min(d, d_max) - 1
            hist[state, col] += 1.0
    return hist


def raw_durations(segments_by_trial, k_subtask):
    """segments_by_trial -> list of per-state raw duration lists (same right-censored-segment
    exclusion as duration_histogram), for the Mann-Whitney significance test."""
    out = [[] for _ in range(k_subtask)]
    for segments in segments_by_trial:
        for state, d in _usable_segments(segments):
            out[state].append(int(d))
    return out


def fit_nb(hist_row, d_max, r_fallback=DEFAULT_R_FALLBACK):
    """hist_row: (d_max,) counts for one state. Reuses durations.newton_update_r (which itself
    seeds from method_of_moments_r internally) + durations.update_p_given_r -- the exact M-step
    estimator, not a second duration-fitting code path. Returns (r, p, n_segments)."""
    n_hat = jnp.asarray(hist_row, dtype=jnp.float64).reshape(1, d_max)
    n_hat_total, s_hat = durations.duration_stats_from_histogram(n_hat, d_max)
    r_old = jnp.array([r_fallback], dtype=jnp.float64)
    r = durations.newton_update_r(n_hat, n_hat_total, s_hat, r_old, n_iters=5)
    p = durations.update_p_given_r(n_hat_total, s_hat, r)
    return float(r[0]), float(p[0]), int(round(float(n_hat_total[0])))


def _nb_mean(r, p):
    return 1.0 + r * (1.0 - p) / p


def duration_drift(recent_segments, frozen_segments, k_subtask, d_max,
                    min_instances=DEFAULT_MIN_INSTANCES, min_delta_ticks=DEFAULT_MIN_DELTA_TICKS,
                    alpha=DEFAULT_ALPHA):
    """Per state: mean_recent, mean_frozen, delta_mean, KL(recent||frozen), a one-sided
    Mann-Whitney p-value, and a `reportable` gate = (|delta| >= min_delta_ticks) AND (p < alpha)
    AND both windows have >= min_instances. KL alone is not a decision rule -- at ~5 sessions/
    week and a handful of instances per state, a KL of 0.3 says nothing about whether a shift
    is real versus sampling noise at that n.
    """
    recent_hist = duration_histogram(recent_segments, k_subtask, d_max)
    frozen_hist = duration_histogram(frozen_segments, k_subtask, d_max)
    recent_raw = raw_durations(recent_segments, k_subtask)
    frozen_raw = raw_durations(frozen_segments, k_subtask)

    rows = []
    for state in range(k_subtask):
        n_recent = len(recent_raw[state])
        n_frozen = len(frozen_raw[state])
        if n_recent == 0 or n_frozen == 0:
            continue

        r_recent, p_recent, _ = fit_nb(recent_hist[state], d_max)
        r_frozen, p_frozen, _ = fit_nb(frozen_hist[state], d_max)
        mean_recent = _nb_mean(r_recent, p_recent)
        mean_frozen = _nb_mean(r_frozen, p_frozen)
        delta_mean = mean_recent - mean_frozen

        log_pmf_recent, _ = durations.duration_tables(
            jnp.array([r_recent]), jnp.array([p_recent]), d_max
        )
        log_pmf_frozen, _ = durations.duration_tables(
            jnp.array([r_frozen]), jnp.array([p_frozen]), d_max
        )
        kl = float(divergence.categorical_kl(log_pmf_recent[0], log_pmf_frozen[0]))

        enough_data = n_recent >= min_instances and n_frozen >= min_instances
        if enough_data:
            alternative = "greater" if delta_mean > 0 else "less"
            _, p_value = stats.mannwhitneyu(
                recent_raw[state], frozen_raw[state], alternative=alternative, method="auto"
            )
            p_value = float(p_value)
        else:
            p_value = 1.0

        reportable = bool(enough_data and abs(delta_mean) >= min_delta_ticks and p_value < alpha)

        rows.append({
            "state": state,
            "n_recent": n_recent,
            "n_frozen": n_frozen,
            "mean_recent": mean_recent,
            "mean_frozen": mean_frozen,
            "delta_mean": delta_mean,
            "direction": "slower" if delta_mean > 0 else "faster",
            "kl": kl,
            "p_value": p_value,
            "reportable": reportable,
        })

    return rows


def narrate_drift(rows, lexicon, window_name="this week"):
    """Renders reportable rows using the same Lexicon as narrate.py, so subtask naming is
    identical between live queries and weekly-review queries."""
    reportable = sorted(
        (r for r in rows if r["reportable"]), key=lambda r: -abs(r["delta_mean"])
    )
    lines = []
    for r in reportable:
        name = lexicon.subtask(r["state"])
        lines.append(
            f"You've been {r['direction']} at {name} {window_name} -- about "
            f"{r['mean_recent']:.0f} ticks now vs your usual {r['mean_frozen']:.0f} "
            f"(p={r['p_value']:.3f})."
        )
    return lines


# Integration gap, out of scope for this pass: duration_drift produces a report and rendered
# strings but does not feed back into state_manager the way PreferenceEvent/handle_confirmation
# does for verb/noun/trans. There is no PreferenceEvent-equivalent for duration because
# HSMMParams.dur_r/dur_p are point estimates, not counts, so online_update.bounded_bump's
# pattern does not apply directly. Closing this (giving duration a live/frozen split inside
# DualModel itself, not just a standalone report) is future work, scoped separately. If it
# ever lands, divergence.model_divergence's docstring claim that duration KL is always 0 by
# construction becomes wrong and must change in the same commit.
