"""Joint HSMM parameters and the log tables inference runs on (docs/model.md §1)."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cook_ad.hsmm import durations

FLOOR = 1e-12  # smallest probability any table holds before its log


class JointHSMMParams(NamedTuple):
    """Dirichlet pseudocounts (prior + data) and NB durations. Emissions are shared by recipes."""
    init_counts: jnp.ndarray    # (K_R, K)
    trans_counts: jnp.ndarray   # (K_R, K, K) diagonal unused: no self-transitions
    verb_counts: jnp.ndarray    # (K, V)
    noun_counts: jnp.ndarray    # (K, N)
    dur_r: jnp.ndarray          # (K_R, K) NB dispersion
    dur_p: jnp.ndarray          # (K_R, K) NB success probability
    pi_counts: jnp.ndarray      # (K_R,)


class LogTables(NamedTuple):
    """Every table inference reads, in log space (docs/model.md §1)."""
    log_pi: jnp.ndarray          # (K_R,)       P(r)
    log_init: jnp.ndarray        # (K_R, K)     P(z_0 = i | r)
    log_A: jnp.ndarray           # (K_R, K, K)  P(z = i | previous k, r); -inf on the diagonal
    log_emit_v: jnp.ndarray      # (K, V)
    log_emit_n: jnp.ndarray      # (K, N)
    log_pmf: jnp.ndarray         # (K_R, K, D)  P(D = j+1); the last column holds P(D >= D_max)
    log_surv: jnp.ndarray        # (K_R, K, D)  P(D >= j+1)
    log_h: jnp.ndarray           # (K_R, K, D)  hazard P(D = j+1 | D >= j+1); 1 at D_max
    log_stay: jnp.ndarray        # (K_R, K, D)  P(D >= j+2 | D >= j+1); -inf at D_max
    log_A_marginal: jnp.ndarray  # (K, K)       sum_r P(r) A[r]


def _row_normalize(counts, floor=FLOOR, mask_diag=False):
    """Dirichlet MAP: log (counts - 1) / sum, floored before the log."""
    numerator = jnp.maximum(counts - 1.0, floor)
    if mask_diag:
        numerator = numerator * (1.0 - jnp.eye(counts.shape[0]))
    row_sum = jnp.maximum(jnp.sum(numerator, axis=-1, keepdims=True), floor)
    return jnp.log(numerator) - jnp.log(row_sum)


def hazard_tables(log_pmf, log_surv):
    """(log_h, log_stay) from the duration pmf and survival; a segment always ends at D_max."""
    pmf = np.asarray(log_pmf, dtype=np.float64)
    surv = np.asarray(log_surv, dtype=np.float64)
    finite = np.isfinite(surv)
    with np.errstate(invalid="ignore"):
        log_h = np.where(finite, pmf - surv, -np.inf)
        log_stay = np.full_like(surv, -np.inf)
        log_stay[..., :-1] = np.where(finite[..., :-1], surv[..., 1:] - surv[..., :-1], -np.inf)
    log_h[..., -1] = 0.0
    return jnp.asarray(log_h), jnp.asarray(log_stay)


def marginal_log_trans(log_pi, log_A):
    """(K, K) log sum_r P(r) A[r]."""
    marginal = jnp.einsum("r,rjk->jk", jnp.exp(log_pi), jnp.exp(log_A))
    return jnp.log(jnp.maximum(marginal, FLOOR))


def to_log_tables(params: JointHSMMParams, d_max: int) -> LogTables:
    log_pi = _row_normalize(params.pi_counts[None, :], FLOOR)[0]
    log_A = jax.vmap(lambda c: _row_normalize(c, FLOOR, mask_diag=True))(params.trans_counts)
    log_pmf, log_surv = jax.vmap(durations.duration_tables, in_axes=(0, 0, None))(
        params.dur_r, params.dur_p, d_max)
    return LogTables(
        log_pi, _row_normalize(params.init_counts, FLOOR), log_A,
        _row_normalize(params.verb_counts, FLOOR), _row_normalize(params.noun_counts, FLOOR),
        log_pmf, log_surv, *hazard_tables(log_pmf, log_surv), marginal_log_trans(log_pi, log_A))


class RecipeView(NamedTuple):
    """One recipe's parameters, with the shared emissions."""
    init_counts: jnp.ndarray    # (K,)
    trans_counts: jnp.ndarray   # (K, K)
    verb_counts: jnp.ndarray    # (K, V)
    noun_counts: jnp.ndarray    # (K, N)
    dur_r: jnp.ndarray          # (K,)
    dur_p: jnp.ndarray          # (K,)


def select_recipe(params: JointHSMMParams, r: int) -> RecipeView:
    return RecipeView(params.init_counts[r], params.trans_counts[r], params.verb_counts,
                      params.noun_counts, params.dur_r[r], params.dur_p[r])


def save(params: JointHSMMParams, path):
    np.savez(path, **{name: np.asarray(value) for name, value in params._asdict().items()})


def load(path) -> JointHSMMParams:
    """Raises on fields this model does not have (e.g. a noun tilt or emission kernel)."""
    with np.load(path) as data:
        extra = set(data.files) - set(JointHSMMParams._fields)
        if extra:
            raise ValueError(f"{path}: unsupported parameter arrays {sorted(extra)}")
        return JointHSMMParams(**{name: jnp.asarray(data[name]) for name in JointHSMMParams._fields})


# ---------------------------------------------------------------------------------------------
# Cascade (single-recipe) model. Still read by the old EM, synthetic generators and Lexicon;
# deleted in restructure Phase E.


class HSMMParams(NamedTuple):
    init_counts: jnp.ndarray    # (K,)    Dirichlet pseudocounts (prior + data), initial-state dist
    trans_counts: jnp.ndarray   # (K,K)   Dirichlet pseudocounts per row; diagonal unused (self-transitions banned)
    verb_counts: jnp.ndarray    # (K,V)   Dirichlet pseudocounts per row
    noun_counts: jnp.ndarray    # (K,N)   Dirichlet pseudocounts per row
    dur_r: jnp.ndarray          # (K,)    NB dispersion, point estimate
    dur_p: jnp.ndarray          # (K,)    NB success-prob, point estimate
    # Fixed similarity kernels for the latent-intended-token emission (kernel.py).
    # None == identity == the plain categorical emission.
    kernel_v: jnp.ndarray = None   # (V,V) row-stochastic, or None
    kernel_n: jnp.ndarray = None   # (N,N) row-stochastic, or None


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


def _log_kernel(kernel):
    """log S, with exact zeros kept at -inf rather than floored.

    S is a fixed known table, not an estimated distribution, so a structural zero in it is a
    real zero. Flooring it to FLOOR instead would put -27.6 on the identity's off-diagonal --
    above the -36 nats _row_normalize already gave unobserved cells -- so composing with the
    identity would change a model it must leave alone.
    """
    k = jnp.asarray(kernel)
    return jnp.where(k > 0, jnp.log(jnp.where(k > 0, k, 1.0)), -jnp.inf)


def compose_kernel(log_b, kernel):
    """log_b: (K,W) log P(intended token m | state k). kernel: (W,W) row-stochastic, or None.

    Returns log (B S)[k,n] = log sum_m B[k,m] S[m,n] -- the marginal emission after integrating
    out the latent intended token. `None` short-circuits to log_b.
    """
    if kernel is None:
        return log_b
    return logsumexp(log_b[:, :, None] + _log_kernel(kernel)[None, :, :], axis=1)


def latent_counts(observed_counts, log_b, kernel):
    """E-step counts over OBSERVED tokens -> expected counts over the LATENT intended token:

        Chat[k,m] = sum_v C[k,v] * R[k,m,v],   R[k,m,v] = B[k,m] S[m,v] / (B S)[k,v]

    R is P(m | v, k), independent of t, so C stays sufficient and the E-step needs no change.
    This is an exact latent-variable M-step, which is what keeps EM monotone; smoothing the
    counts directly (`counts @ S`) is not a generative model and would break that.
    """
    if kernel is None:
        return observed_counts
    log_s = _log_kernel(kernel)
    log_joint = log_b[:, :, None] + log_s[None, :, :]              # (K,M,V) log B[k,m] S[m,v]
    log_r = log_joint - logsumexp(log_joint, axis=1, keepdims=True)  # (K,M,V) log P(m|v,k)
    return jnp.einsum("kv,kmv->km", observed_counts, jnp.exp(log_r))


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
    log_emit_v = compose_kernel(_row_normalize(params.verb_counts, floor), params.kernel_v)
    log_emit_n = compose_kernel(_row_normalize(params.noun_counts, floor), params.kernel_n)
    return log_init, log_trans, log_emit_v, log_emit_n


def to_log_probs(params: HSMMParams, d_max: int) -> HSMMLogProbs:
    log_init, log_trans, log_emit_v, log_emit_n = normalize_categoricals(params)
    log_dur_pmf, log_dur_survival = durations.duration_tables(params.dur_r, params.dur_p, d_max)
    return HSMMLogProbs(log_init, log_trans, log_emit_v, log_emit_n, log_dur_pmf, log_dur_survival)


def save_params(params: HSMMParams, path):
    import numpy as np

    np.savez(path, **{name: np.asarray(value) for name, value in params._asdict().items()
                      if value is not None})


def load_params(path) -> HSMMParams:
    """Missing kernel_v/kernel_n read back as None (= identity), so pre-kernel .npz files load
    and score unchanged."""
    import numpy as np

    with np.load(path) as data:
        return HSMMParams(**{name: jnp.asarray(data[name]) for name in HSMMParams._fields
                             if name in data})
