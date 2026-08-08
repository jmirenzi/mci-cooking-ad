from typing import NamedTuple

import jax
import jax.numpy as jnp

from cook_ad.hsmm import durations, params
from cook_ad.hsmm.params import FLOOR, _row_normalize

HSMMParams = params.HSMMParams


class JointHSMMParams(NamedTuple):
    init_counts: jnp.ndarray    # (K_R,K)    Dirichlet pseudocounts, per-recipe initial-state dist
    trans_counts: jnp.ndarray   # (K_R,K,K)  Dirichlet pseudocounts per (recipe,row); diagonal unused
    verb_counts: jnp.ndarray    # (K,V)      Dirichlet pseudocounts per row -- SHARED across recipes
    noun_counts: jnp.ndarray    # (K,N)      Dirichlet pseudocounts per row -- SHARED across recipes
    dur_r: jnp.ndarray          # (K_R,K)    NB dispersion, point estimate
    dur_p: jnp.ndarray          # (K_R,K)    NB success-prob, point estimate
    pi_counts: jnp.ndarray      # (K_R,)     Dirichlet pseudocounts, recipe-mixture dist


class JointHSMMLogProbs(NamedTuple):
    log_pi: jnp.ndarray            # (K_R,)
    log_init: jnp.ndarray          # (K_R,K)
    log_trans: jnp.ndarray         # (K_R,K,K), diagonal -inf
    log_emit_v: jnp.ndarray        # (K,V)      SHARED
    log_emit_n: jnp.ndarray        # (K,N)      SHARED
    log_dur_pmf: jnp.ndarray       # (K_R,K,D_max)
    log_dur_survival: jnp.ndarray  # (K_R,K,D_max)


def to_log_probs_joint(joint_params: JointHSMMParams, d_max: int) -> JointHSMMLogProbs:
    """Per-recipe normalization mirroring params.to_log_probs, vmapped over the leading K_R
    axis wherever the underlying helper assumes rank matches a single recipe's params (trans's
    mask_diag builds jnp.eye(counts.shape[0]), which must see a (K,K) slice, not (K_R,K,K); the
    duration tables likewise assume 1-D dur_r/dur_p). init/emit_v/emit_n already normalize each
    row independently regardless of leading batch dims, so no vmap is needed there. pi is a
    single distribution, not a per-row family, so it's fed through as one (1,K_R) "row".
    """
    log_pi = _row_normalize(joint_params.pi_counts[None, :], FLOOR)[0]
    log_init = _row_normalize(joint_params.init_counts, FLOOR)
    log_trans = jax.vmap(lambda c: _row_normalize(c, FLOOR, mask_diag=True))(joint_params.trans_counts)
    log_emit_v = _row_normalize(joint_params.verb_counts, FLOOR)
    log_emit_n = _row_normalize(joint_params.noun_counts, FLOOR)
    log_dur_pmf, log_dur_survival = jax.vmap(durations.duration_tables, in_axes=(0, 0, None))(
        joint_params.dur_r, joint_params.dur_p, d_max
    )
    return JointHSMMLogProbs(
        log_pi, log_init, log_trans, log_emit_v, log_emit_n, log_dur_pmf, log_dur_survival
    )


def marginal_log_trans(log_probs: JointHSMMLogProbs) -> jnp.ndarray:
    """pi-weighted mixture of the per-recipe transition matrices: log(sum_r pi[r] *
    exp(log_trans[r])), (K,K). Used as the s_recipe_transition baseline -- "how surprising is
    this transition in general" against which a trial's recipe-conditioned transition surprise
    is compared. Floor-safe: every input is already a valid log-probability (>=0 after exp),
    so the mixture can only underflow toward 0, never go negative.
    """
    pi = jnp.exp(log_probs.log_pi)
    trans = jnp.exp(log_probs.log_trans)  # (K_R,K,K); diagonal is exp(-inf)=0, contributes nothing
    marginal = jnp.einsum("r,rjk->jk", pi, trans)
    return jnp.log(jnp.maximum(marginal, FLOOR))


def collapse_to_marginal(joint_params: JointHSMMParams) -> HSMMParams:
    """Recipe-agnostic HSMMParams built by pi-weighting the per-recipe families -- used
    wherever a caller genuinely needs a single recipe-agnostic model (the s_recipe_transition
    baseline, and synthetic.error_injection, which only reads emissions/durations and has no
    concept of a per-trial recipe latent). Emission counts are copied verbatim since they are
    already shared/recipe-agnostic. `pi` here is a plain mixing weight (not the Dirichlet-MAP
    posterior probs to_log_probs_joint produces), since this is aggregation, not a distribution
    that itself needs the floor-safe log.
    """
    pi = joint_params.pi_counts / jnp.sum(joint_params.pi_counts)
    init_counts = jnp.einsum("r,rk->k", pi, joint_params.init_counts)
    trans_counts = jnp.einsum("r,rjk->jk", pi, joint_params.trans_counts)
    dur_r = jnp.einsum("r,rk->k", pi, joint_params.dur_r)
    dur_p = jnp.einsum("r,rk->k", pi, joint_params.dur_p)
    return HSMMParams(
        init_counts, trans_counts, joint_params.verb_counts, joint_params.noun_counts, dur_r, dur_p
    )


def select_recipe(joint_params: JointHSMMParams, r_hat: int) -> HSMMParams:
    """Recipe-r_hat-conditioned HSMMParams: that recipe's own init/trans/duration tables, paired
    with the shared emission counts. Unlike collapse_to_marginal's pi-weighted average, this is
    the exact per-trial model assemble_trace_joint/quantile.threshold_tables_joint score
    against for trial r_hat -- a narrate.Lexicon built from this (not the marginal) reports
    expected durations consistent with the duration surprise actually computed for the trial.
    """
    return HSMMParams(
        joint_params.init_counts[r_hat], joint_params.trans_counts[r_hat],
        joint_params.verb_counts, joint_params.noun_counts,
        joint_params.dur_r[r_hat], joint_params.dur_p[r_hat],
    )


def save_params(joint_params: JointHSMMParams, path):
    import numpy as np

    np.savez(path, **{name: np.asarray(value) for name, value in joint_params._asdict().items()})


def load_params(path) -> JointHSMMParams:
    import numpy as np

    with np.load(path) as data:
        return JointHSMMParams(**{name: jnp.asarray(data[name]) for name in JointHSMMParams._fields})
