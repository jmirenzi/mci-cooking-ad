import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

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
    # See params.HSMMParams -- emissions are shared across recipes, so the kernels are too.
    kernel_v: jnp.ndarray = None   # (V,V) row-stochastic, or None == identity
    kernel_n: jnp.ndarray = None   # (N,N) row-stochastic, or None == identity
    # Rank-1 per-recipe noun modulation: P_r(n|k) = softmax_n(log_emit_n[k,n] + noun_tilt[r,n]).
    # None == no modulation (every recipe sees the shared noun table unchanged). This is the
    # recipe latent's only DIRECT access to content -- everything else (init/trans/duration)
    # only reaches content indirectly, via which states a recipe favours. See tilt_terms.
    noun_tilt: jnp.ndarray = None  # (K_R,N), or None == no modulation


class JointHSMMLogProbs(NamedTuple):
    log_pi: jnp.ndarray            # (K_R,)
    log_init: jnp.ndarray          # (K_R,K)
    log_trans: jnp.ndarray         # (K_R,K,K), diagonal -inf
    log_emit_v: jnp.ndarray        # (K,V)      SHARED
    log_emit_n: jnp.ndarray        # (K,N)      SHARED, pre-tilt
    log_dur_pmf: jnp.ndarray       # (K_R,K,D_max)
    log_dur_survival: jnp.ndarray  # (K_R,K,D_max)
    # noun_tilt=None reads back as an all-zero (K_R,N)/(K_R,K) pair here (see tilt_terms), so a
    # caller adding log_tilt[r, n] - log_tilt_norm[r, k] to log_emit_n always gets a valid
    # per-(recipe,state) distribution over nouns, tilted or not.
    log_tilt: jnp.ndarray = None       # (K_R,N)
    log_tilt_norm: jnp.ndarray = None  # (K_R,K)


def tilt_terms(log_emit_n, noun_tilt, k_r):
    """log_emit_n: (K,N) shared, pre-tilt log P(n|k). noun_tilt: (K_R,N) or None. k_r: needed to
    shape the zero fallback when noun_tilt is None (there is otherwise no K_R anywhere in a bare
    (K,N) table).

    Returns (log_tilt (K_R,N), log_tilt_norm (K_R,K)) such that

        log_emit_n[k,n] + log_tilt[r,n] - log_tilt_norm[r,k]

    is a valid log-probability over n for every (r,k): log_tilt_norm is the per-(recipe,state)
    log-normalizer log sum_n exp(log_emit_n[k,n] + noun_tilt[r,n]). `None` short-circuits to an
    all-zero pair, making the add-and-subtract above a no-op -- the inert-tilt case tested by
    test_noun_tilt_zero_matches_none_in_e_step.
    """
    if noun_tilt is None:
        n = log_emit_n.shape[1]
        k = log_emit_n.shape[0]
        return jnp.zeros((k_r, n)), jnp.zeros((k_r, k))
    log_joint = log_emit_n[None, :, :] + noun_tilt[:, None, :]  # (K_R,K,N)
    log_norm = logsumexp(log_joint, axis=-1)  # (K_R,K)
    return noun_tilt, log_norm


def log_emit_n_recipe(log_probs: JointHSMMLogProbs, r: int) -> jnp.ndarray:
    """(K,N) recipe-r-conditioned observed-noun table P_r(n|k), post-tilt. Equal to
    log_probs.log_emit_n exactly when no tilt is active (log_tilt/log_tilt_norm are then zero)."""
    return log_probs.log_emit_n + log_probs.log_tilt[r][None, :] - log_probs.log_tilt_norm[r][:, None]


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
    log_emit_v = params.compose_kernel(_row_normalize(joint_params.verb_counts, FLOOR),
                                      joint_params.kernel_v)
    log_emit_n = params.compose_kernel(_row_normalize(joint_params.noun_counts, FLOOR),
                                      joint_params.kernel_n)
    log_dur_pmf, log_dur_survival = jax.vmap(durations.duration_tables, in_axes=(0, 0, None))(
        joint_params.dur_r, joint_params.dur_p, d_max
    )
    k_r = joint_params.pi_counts.shape[0]
    log_tilt, log_tilt_norm = tilt_terms(log_emit_n, joint_params.noun_tilt, k_r)
    return JointHSMMLogProbs(
        log_pi, log_init, log_trans, log_emit_v, log_emit_n, log_dur_pmf, log_dur_survival,
        log_tilt, log_tilt_norm,
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

    Drops any noun_tilt: a tilt is a per-recipe reweighting and cannot be expressed as counts,
    so a tilted model collapsed this way silently loses its recipe-specific noun content. Warns
    rather than raising -- the marginal is still a legitimate recipe-agnostic baseline, just
    not one that reflects the tilt.
    """
    if joint_params.noun_tilt is not None:
        warnings.warn(
            "collapse_to_marginal drops noun_tilt -- the returned HSMMParams' noun emission is "
            "the untilted shared table, not any recipe's tilted view of it"
        )
    pi = joint_params.pi_counts / jnp.sum(joint_params.pi_counts)
    init_counts = jnp.einsum("r,rk->k", pi, joint_params.init_counts)
    trans_counts = jnp.einsum("r,rjk->jk", pi, joint_params.trans_counts)
    dur_r = jnp.einsum("r,rk->k", pi, joint_params.dur_r)
    dur_p = jnp.einsum("r,rk->k", pi, joint_params.dur_p)
    return HSMMParams(
        init_counts, trans_counts, joint_params.verb_counts, joint_params.noun_counts, dur_r, dur_p,
        joint_params.kernel_v, joint_params.kernel_n,
    )


def select_recipe(joint_params: JointHSMMParams, r_hat: int) -> HSMMParams:
    """Recipe-r_hat-conditioned HSMMParams: that recipe's own init/trans/duration tables, paired
    with the shared emission counts. Unlike collapse_to_marginal's pi-weighted average, this is
    the exact per-trial model assemble_trace_joint/quantile.threshold_tables_joint score
    against for trial r_hat -- a narrate.Lexicon built from this (not the marginal) reports
    expected durations consistent with the duration surprise actually computed for the trial.

    Drops any noun_tilt for the same reason as collapse_to_marginal: the returned HSMMParams
    has no per-recipe axis to hold it. Warns -- a tilted model's anomaly scoring is silently
    missing the tilt's contribution to the noun emission until surprise/quantile/narrate are
    updated to consume log_emit_n_recipe directly (out of scope here).
    """
    if joint_params.noun_tilt is not None:
        warnings.warn(
            "select_recipe drops noun_tilt -- the returned HSMMParams' noun emission is the "
            f"untilted shared table, not recipe {r_hat}'s tilted view of it"
        )
    return HSMMParams(
        joint_params.init_counts[r_hat], joint_params.trans_counts[r_hat],
        joint_params.verb_counts, joint_params.noun_counts,
        joint_params.dur_r[r_hat], joint_params.dur_p[r_hat],
        joint_params.kernel_v, joint_params.kernel_n,
    )


def save_params(joint_params: JointHSMMParams, path):
    import numpy as np

    np.savez(path, **{name: np.asarray(value) for name, value in joint_params._asdict().items()
                      if value is not None})


def load_params(path) -> JointHSMMParams:
    import numpy as np

    with np.load(path) as data:
        return JointHSMMParams(**{name: jnp.asarray(data[name]) for name in JointHSMMParams._fields
                                  if name in data})
