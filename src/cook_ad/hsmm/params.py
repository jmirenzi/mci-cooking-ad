"""Joint HSMM parameters and the log tables inference runs on (docs/model.md §1)."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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
