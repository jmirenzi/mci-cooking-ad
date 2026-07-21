import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm import durations


def live_stall_surprise(segments, dur_r, dur_p, d_max):
    """segments: [(state, duration), ...] from segmentize.segment_all, oldest-to-newest,
    covering the full trial. Returns (T,) S_temporal, the surprise of the segment NOT ending
    on this tick given it has already lasted this long: S_temporal(d) = -log(1 - h(d)) where
    h(d) = P(D=d|D>=d) is the hazard. This is the complement of the hazard itself -- h(d) is
    "how surprising is it that the segment ends now," so 1-h(d) is "how surprising is it that
    it does NOT end now," which is what should spike as a stall drags on. (Using -log h(d)
    directly would invert the signal: h(d) is small and -log h(d) large right at segment
    start, which is backwards.) Elapsed resets to 1 at each segment's first tick. Not the
    full-segment retrospective term (see `completed_segment_surprise`).
    """
    t_true = sum(d for _, d in segments)
    s_temporal = np.zeros(t_true, dtype=np.float64)

    pos = 0
    for state, d in segments:
        elapsed = jnp.arange(1, min(d, d_max) + 1, dtype=jnp.float64)
        log_hazard = durations.nb_log_hazard(elapsed, dur_r[state], dur_p[state])
        s = -jnp.log1p(-jnp.exp(log_hazard))
        s_temporal[pos : pos + elapsed.shape[0]] = np.asarray(s)
        if d > d_max:
            s_temporal[pos + d_max : pos + d] = s_temporal[pos + d_max - 1]
        pos += d

    return s_temporal


def completed_segment_surprise(segments, dur_r, dur_p):
    """Retrospective per-segment term -log P(D=d|state), known only once a segment ends --
    offline comparison point for the live hazard-based signal above, not used for detection."""
    return np.array(
        [
            float(-durations.nb_log_pmf(jnp.array(float(d)), dur_r[state], dur_p[state]))
            for state, d in segments
        ]
    )
