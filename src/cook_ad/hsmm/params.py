from typing import NamedTuple

import jax
import jax.numpy as jnp

from cook_ad.hsmm import durations

FLOOR = 1e-12


class HSMMParams(NamedTuple):
    init_counts: jnp.ndarray    # (K,)    Dirichlet pseudocounts (prior + data), initial-state dist
    trans_counts: jnp.ndarray   # (K,K)   Dirichlet pseudocounts per row; diagonal unused (self-transitions banned)
    verb_counts: jnp.ndarray    # (K,V)   Dirichlet pseudocounts per row
    noun_counts: jnp.ndarray    # (K,N)   Dirichlet pseudocounts per row
    dur_r: jnp.ndarray          # (K,)    NB dispersion, point estimate
    dur_p: jnp.ndarray          # (K,)    NB success-prob, point estimate


class HSMMLogProbs(NamedTuple):
    log_init: jnp.ndarray          # (K,)
    log_trans: jnp.ndarray         # (K,K), diagonal -inf
    log_emit_v: jnp.ndarray        # (K,V)
    log_emit_n: jnp.ndarray        # (K,N)
    log_dur_pmf: jnp.ndarray       # (K,D_max)
    log_dur_survival: jnp.ndarray  # (K,D_max)


def _dirichlet_counts(key, alpha, width, n_rows):
    concentration = jnp.full((n_rows, width), alpha / width)
    probs = jax.random.dirichlet(key, concentration)
    return alpha * probs


def init_weak_limit_params(
    key,
    k_subtask,
    vocab_verbs,
    vocab_nouns,
    d_max,
    alpha_init=0.5,
    alpha_trans=0.5,
    alpha_emit_v=None,
    alpha_emit_n=None,
    dur_mean_range=(3.0, 40.0),
    dur_r_range=(1.0, 20.0),
) -> HSMMParams:
    """Random draw per restart -- symmetry-breaking. A symmetric prior plus a symmetric
    likelihood never breaks symmetry on its own, which is why EM restarts matter.

    alpha/K<1 sparsity applies only to init/trans (distributions over K). verb/noun are
    ordinary closed-vocabulary categoricals and default to alpha=width (per-category
    alpha=1, safely log-concave, never needs the clipped-MAP floor at normalization time).
    """
    if alpha_emit_v is None:
        alpha_emit_v = float(vocab_verbs)
    if alpha_emit_n is None:
        alpha_emit_n = float(vocab_nouns)

    key_init, key_trans, key_verb, key_noun, key_mean, key_r = jax.random.split(key, 6)

    init_counts = alpha_init * jax.random.dirichlet(
        key_init, jnp.full((k_subtask,), alpha_init / k_subtask)
    )

    # Sample a full (K,K) Dirichlet draw per row, then zero the diagonal: self-transitions
    # are structurally banned, so the (small, ~1/K of the row's mass) diagonal draw is simply
    # discarded rather than re-normalized away -- negligible for a prior, avoids re-deriving
    # a K-1-wide Dirichlet just for this.
    trans_counts = _dirichlet_counts(key_trans, alpha_trans, k_subtask, k_subtask)
    trans_counts = trans_counts * (1.0 - jnp.eye(k_subtask))

    verb_counts = _dirichlet_counts(key_verb, alpha_emit_v, vocab_verbs, k_subtask)
    noun_counts = _dirichlet_counts(key_noun, alpha_emit_n, vocab_nouns, k_subtask)

    dur_mean = jax.random.uniform(
        key_mean, (k_subtask,), minval=dur_mean_range[0], maxval=dur_mean_range[1]
    )
    dur_r = jax.random.uniform(key_r, (k_subtask,), minval=dur_r_range[0], maxval=dur_r_range[1])
    dur_p = dur_r / (dur_r + dur_mean)

    return HSMMParams(init_counts, trans_counts, verb_counts, noun_counts, dur_r, dur_p)


def _row_normalize(counts, floor=FLOOR, mask_diag=False):
    numerator = jnp.maximum(counts - 1.0, floor)
    if mask_diag:
        k = counts.shape[0]
        numerator = numerator * (1.0 - jnp.eye(k))
    row_sum = jnp.maximum(jnp.sum(numerator, axis=-1, keepdims=True), floor)
    return jnp.log(numerator) - jnp.log(row_sum)


def normalize_categoricals(params: HSMMParams, floor=FLOOR):
    """This is where alpha/K<1 non-log-concavity becomes a live numerical hazard: the
    textbook Dirichlet-MAP closed form (counts-1)/(sum(counts)-K) is only valid when every
    per-category count is >=1; with alpha/K<1 a near-empty state can drive the numerator
    negative, which is invalid and would feed log() of a non-positive number. Floor before
    every log, never after.

    `counts` already represents the full posterior concentration (prior + accumulated data)
    at all times -- see em.py's m_step, which reconstructs it fresh each iteration as
    alpha/width + expected_data_counts. mask_diag=True additionally zeroes the diagonal's
    contribution before computing the row sum (not just after taking the log), so a
    structurally-banned self-transition doesn't quietly steal normalization mass from the
    K-1 real entries in that row.
    """
    log_init = _row_normalize(params.init_counts, floor)
    log_trans = _row_normalize(params.trans_counts, floor, mask_diag=True)
    log_emit_v = _row_normalize(params.verb_counts, floor)
    log_emit_n = _row_normalize(params.noun_counts, floor)
    return log_init, log_trans, log_emit_v, log_emit_n


def to_log_probs(params: HSMMParams, d_max: int) -> HSMMLogProbs:
    log_init, log_trans, log_emit_v, log_emit_n = normalize_categoricals(params)
    log_dur_pmf, log_dur_survival = durations.duration_tables(params.dur_r, params.dur_p, d_max)
    return HSMMLogProbs(log_init, log_trans, log_emit_v, log_emit_n, log_dur_pmf, log_dur_survival)


def save_params(params: HSMMParams, path):
    import numpy as np

    np.savez(path, **{name: np.asarray(value) for name, value in params._asdict().items()})


def load_params(path) -> HSMMParams:
    import numpy as np

    with np.load(path) as data:
        return HSMMParams(**{name: jnp.asarray(data[name]) for name in HSMMParams._fields})
