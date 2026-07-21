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


def completed_segment_surprise(segments, dur_r, dur_p):
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
    """
    dur_r = np.asarray(dur_r)
    dur_p = np.asarray(dur_p)
    n = len(segments)
    s_long = np.zeros(n)
    s_short = np.zeros(n)
    s_two = np.zeros(n)
    attribution = np.full(n, "none", dtype=object)

    for i, (state, d) in enumerate(segments):
        d_j = jnp.array(float(d))
        r_j, p_j = jnp.array(float(dur_r[state])), jnp.array(float(dur_p[state]))
        log_surv = float(durations.nb_log_survival(d_j, r_j, p_j))
        log_cdf = float(durations.nb_log_cdf(d_j, r_j, p_j))
        s_long[i] = -log_surv
        s_short[i] = -log_cdf
        s_two[i] = max(0.0, -(LOG2 + min(log_surv, log_cdf)))
        attribution[i] = "stuck" if log_surv < log_cdf else "left_early"

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

    for i, (state, d) in enumerate(segments):
        r_j, p_j = jnp.array(float(dur_r[state])), jnp.array(float(dur_p[state]))
        cdf_below = 0.0 if d <= 1 else float(jnp.exp(durations.nb_log_cdf(jnp.array(float(d - 1)), r_j, p_j)))
        pmf_here = float(jnp.exp(durations.nb_log_pmf(jnp.array(float(d)), r_j, p_j)))
        pit[i] = cdf_below + 0.5 * pmf_here

    return pit
