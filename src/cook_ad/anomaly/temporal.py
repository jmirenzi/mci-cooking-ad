import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm import durations

LOG2 = float(np.log(2.0))


def live_stall_surprise(segments, log_dur_survival, d_max):
    """segments: [(state, duration), ...] covering a full trial. Returns (T,) live 'stuck'
    surprise S(t) = -log P(D >= d_elapsed | state), where d_elapsed resets to 1 at each
    segment's first tick. This is the survival surprise: monotonically increasing within a
    segment (the longer you persist, the deeper into the upper tail you are), and -- because
    it IS a tail probability -- a single global threshold on it is automatically per-state
    calibrated (flagging when the elapsed duration passes that state's alpha upper-tail
    quantile). A right-censored, in-progress segment is exactly the case where survival, not
    pmf, is the statistically correct object. Values past d_max clamp to P(D >= d_max).
    """
    log_dur_survival = np.asarray(log_dur_survival)
    t_true = sum(d for _, d in segments)
    s_temporal = np.zeros(t_true, dtype=np.float64)

    pos = 0
    for state, d in segments:
        n = min(d, d_max)
        elapsed_idx = np.arange(n)  # d_elapsed = 1..n maps to survival column 0..n-1
        s_temporal[pos : pos + n] = -log_dur_survival[state, elapsed_idx]
        if d > d_max:
            s_temporal[pos + d_max : pos + d] = -log_dur_survival[state, d_max - 1]
        pos += d

    return s_temporal


def completed_segment_surprise(segments, dur_r, dur_p, final_censored=True):
    """Two-sided retrospective duration surprise, computed once a segment closes at its final
    observed duration d (no longer censored). Both tails are proper p-values, so they share a
    scale across states:
      s_long  = -log P(D >= d)   (stuck: overdue)
      s_short = -log P(D <= d)   (left too early: the live monotone signal structurally cannot
                                  catch this, since short durations always have low survival
                                  surprise)
      s_two   = -log( min(1, 2*min(P(D>=d), P(D<=d)) ) )   two-sided p-value surprise
    Returns per-segment arrays (s_long, s_short, s_two, attribution) where attribution is
    'stuck' if the right tail is the smaller (more surprising) one, else 'left_early'.

    `final_censored` (default True) leaves the LAST segment unscored -- all three values 0,
    attribution 'none'. That segment has not closed: observation stopped, the activity did not,
    which is the same right-censoring the E-step already models with survival rather than pmf
    (hsmm/durations.py, docs/README.md's cross-cutting conventions). Asking "did this end too
    early?" of a segment that has not ended is not a question the data can answer, and answering
    it anyway fires on the trial's final tick for any state whose fitted mean exceeds the
    remaining recording -- measured on 419 healthy real trials, that single flag accounts for
    ~62% of the residual healthy trial-level false-positive rate at tight alpha (0.100 -> 0.038).

    The right tail alone would still be valid for a censored segment (it HAS lasted at least d),
    but that is exactly what `live_stall_surprise` already computes, so there is nothing to
    recover here -- the information is in s_temporal, not lost.

    Pass False only when the last segment is known to have genuinely ended.
    """
    dur_r = np.asarray(dur_r)
    dur_p = np.asarray(dur_p)
    n = len(segments)
    s_long = np.zeros(n)
    s_short = np.zeros(n)
    s_two = np.zeros(n)
    attribution = np.full(n, "none", dtype=object)

    scored = n - 1 if (final_censored and n > 0) else n
    if scored > 0:
        # One vectorised call per tail rather than one per segment. nb_log_survival/nb_log_cdf
        # are elementwise, so this is the same arithmetic on the same inputs -- but a per-segment
        # loop pays a full JAX dispatch and device round-trip for each scalar, which at ~7
        # segments x hundreds of trials x six source groups is the dominant cost of every
        # evaluation sweep in the repo (measured: minutes per group, versus milliseconds here).
        states = np.array([state for state, _ in segments[:scored]], dtype=np.int64)
        ds = np.array([float(d) for _, d in segments[:scored]], dtype=np.float64)
        r_j = dur_r[states].astype(np.float64)
        p_j = dur_p[states].astype(np.float64)
        log_surv = durations.nb_log_survival_np(ds, r_j, p_j)
        log_cdf = durations.nb_log_cdf_np(ds, r_j, p_j)
        s_long[:scored] = -log_surv
        s_short[:scored] = -log_cdf
        s_two[:scored] = np.maximum(0.0, -(LOG2 + np.minimum(log_surv, log_cdf)))
        attribution[:scored] = np.where(log_surv < log_cdf, "stuck", "left_early")

    return s_long, s_short, s_two, attribution


def pit_coordinate(segments, dur_r, dur_p):
    """Mid-PIT coordinate per segment: F(d-1) + 0.5*P(D=d). For a well-fit duration model the
    PIT values of healthy segments are approximately Uniform[0,1] (mean ~0.5, flat histogram),
    so this is the calibration diagnostic: systematic deviation flags duration-model misfit,
    not user anomalies. Approximate because D is discrete (exact PIT-uniformity needs the
    randomized transform); mid-PIT is the standard discrete correction.
    """
    dur_r = np.asarray(dur_r)
    dur_p = np.asarray(dur_p)
    pit = np.zeros(len(segments))
    if not segments:
        return pit

    # Vectorised for the same reason completed_segment_surprise is -- see its comment.
    states = np.array([state for state, _ in segments], dtype=np.int64)
    d_np = np.array([float(d) for _, d in segments], dtype=np.float64)
    r_j = dur_r[states].astype(np.float64)
    p_j = dur_p[states].astype(np.float64)
    # nb_log_cdf's second beta shape arg must stay >= 1, so d=1 segments (whose F(d-1) is 0 by
    # definition) are evaluated at a safe dummy and masked out afterwards rather than passed in.
    d_below = np.maximum(d_np - 1.0, 1.0)
    cdf_below = np.exp(durations.nb_log_cdf_np(d_below, r_j, p_j))
    cdf_below = np.where(d_np <= 1.0, 0.0, cdf_below)
    pmf_here = np.exp(durations.nb_log_pmf_np(d_np, r_j, p_j))
    pit[:] = cdf_below + 0.5 * pmf_here
    return pit
