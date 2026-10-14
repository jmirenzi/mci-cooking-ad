"""Live (causal) surprise channels -- the design in docs/live.md, as an opt-in path.

Nothing here touches the shipped hindsight detector (surprise.py / eval.batch). Every channel
below reads only o_{0:t} (or o_{0:t+L} for an explicit lag L), never the whole-trial Viterbi
decode z_star or the whole-trial recipe r_hat.

The state is ONE exact filter over (recipe, subtask, elapsed): b_t(r,i,j) = P(R=r, z_t=i,
e_t=j+1 | o_{0:t}), with j the 0-based elapsed index (column j means "j+1 ticks in"). It is the
joint model's own forward pass read out per tick, so sum_t log P(o_t|o_<t) equals
joint_em.infer_recipe's trial_ll (tests/test_live.py).

Segments end with certainty at D_max, matching messages.forward_pass: the duration table's last
pmf column holds P(D>=D_max) (durations.duration_tables), so the hazard there is 1. There is no
absorbing column.

Two kinds of output, per docs/live.md rule R1:

  * emission channels (verb / noun / emit): a predictive p-value of the observation under the
    forecast mixture -- alpha-free, flagged as p <= alpha;
  * hidden-event channels (stall / early exit / transition / recipe): the posterior probability
    that a bad event happened, flagged as prob > kappa. The "bad" sets are the SAME alpha-tail
    sets the hindsight detector uses (quantile.py), so with a point-mass belief on the hindsight
    answer every live flag reduces to the current one (rule R2).

Hidden-event probabilities are returned for a whole alpha grid at once. Each cell of a bad set
is binned by the smallest grid alpha at which it enters the set (the sets are nested in alpha),
so one segment-sum per tick gives every alpha.
"""
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cook_ad.anomaly import quantile
from cook_ad.hsmm import durations, emissions, joint_params

EVENT_CHANNELS = ("stall", "early", "trans", "recipe")

# The live channels reuse the hindsight channel names so every downstream scorer
# (element_metrics.CHANNEL_TO_TYPE, run_detect_eval's trial_loc) reads them unchanged.
CHANNEL_NAMES = {
    "p_emit": "s_emit",
    "p_verb": "s_verb",
    "p_noun": "s_noun",
    "stall": "s_temporal",
    "early": "s_dur_two",
    "trans": "s_transition",
    "recipe": "s_recipe_transition",
}

# Ties in the forecast mixture are decided in log space with this absolute tolerance: the
# mixture is a logsumexp over K states, so two outcomes that are exactly tied under every
# component can differ by an ULP after mixing. Far below any surprise gap this repo treats as
# meaningful; at a point-mass belief the mixture is exact and no tie is broken by it.
TIE_ATOL = 1e-10


class LiveTables(NamedTuple):
    log_init0: jnp.ndarray   # (K_R,K)    log P(R=r) + log pi_init[r,i] (the prior is swappable)
    log_init: jnp.ndarray    # (K_R,K)    log pi_init[r,i] alone
    log_trans: jnp.ndarray   # (K_R,K,K)
    log_emit_v: jnp.ndarray  # (K,V)
    log_emit_n: jnp.ndarray  # (K,N)
    log_haz: jnp.ndarray     # (K_R,K,D)  log P(D=j+1 | D>=j+1)
    log_stay: jnp.ndarray    # (K_R,K,D)  log P(D>=j+2 | D>=j+1); -inf at the last column
    stall_bin: jnp.ndarray   # (K_R,K,D)  int, first alpha index at which "past the upper tail"
    early_bin: jnp.ndarray   # (K_R,K,D)  int, ... "ended in the short alpha/2 tail"
    trans_bin: jnp.ndarray   # (K_R,K,K)  int, ... "k -> i in the alpha-tail of A[r,k,:]"
    recipe_bin: jnp.ndarray  # (K_R,K,K)  int, ... "k -> i in r's excess alpha-tail"
    alphas: np.ndarray       # (G,) ascending; bin value G means "never bad"


def _first_true(masks):
    """masks: (G, ...) bool, nested (masks[g] implies masks[g+1]). -> int index of the first True
    along axis 0, or G where none is."""
    masks = np.asarray(masks)
    g = masks.shape[0]
    if not np.all(masks[:-1] <= masks[1:]):
        raise ValueError("bad sets are not nested in alpha -- the binning would be wrong")
    return np.where(masks.any(axis=0), masks.argmax(axis=0), g).astype(np.int32)


def build_tables(jhp, d_max, alphas):
    """Precompute every table the live filter and its event channels need, for one fitted
    joint model and one alpha grid. `alphas` may be in any order; it is sorted ascending."""
    if jhp.noun_tilt is not None:
        raise NotImplementedError("live channels do not yet carry noun_tilt (docs/live.md §1)")
    alphas = np.sort(np.asarray(alphas, dtype=np.float64))
    lp = joint_params.to_log_probs_joint(jhp, d_max)
    ltm = joint_params.marginal_log_trans(lp)

    pmf = np.asarray(lp.log_dur_pmf, dtype=np.float64)
    surv = np.asarray(lp.log_dur_survival, dtype=np.float64)
    finite = np.isfinite(surv)
    with np.errstate(invalid="ignore"):
        log_haz = np.where(finite, pmf - surv, -np.inf)
        log_stay = np.full_like(surv, -np.inf)
        log_stay[..., :-1] = np.where(finite[..., :-1], surv[..., 1:] - surv[..., :-1], -np.inf)
    log_haz[..., -1] = 0.0  # the last pmf column IS the survival: certain end at D_max

    # Stall: elapsed j+1 with P(D >= j+1) < alpha  <=>  s_temporal = -log S > -log alpha.
    stall_bin = np.searchsorted(alphas, np.exp(surv), side="right").astype(np.int32)

    # Early exit: P(D <= d) < alpha/2, the short side of s_dur_two's two-sided test, on the same
    # untruncated NB tail temporal.completed_segment_surprise uses.
    d = np.arange(1, d_max + 1, dtype=np.float64)
    dur_r = np.asarray(jhp.dur_r, dtype=np.float64)[..., None]
    dur_p = np.asarray(jhp.dur_p, dtype=np.float64)[..., None]
    log_cdf = durations.nb_log_cdf_np(d[None, None, :], dur_r, dur_p)
    early_bin = np.searchsorted(alphas, 2.0 * np.exp(log_cdf), side="right").astype(np.int32)

    log_trans = np.asarray(lp.log_trans, dtype=np.float64)
    ltm_np = np.asarray(ltm, dtype=np.float64)
    k_r, k, _ = log_trans.shape
    trans_masks = np.zeros((len(alphas), k_r, k, k), dtype=bool)
    recipe_masks = np.zeros_like(trans_masks)
    with np.errstate(invalid="ignore"):
        excess = ltm_np[None] - log_trans  # (K_R,K,K)
    real = np.isfinite(log_trans)
    for g, a in enumerate(alphas):
        for r in range(k_r):
            t_tau = quantile.transition_quantile_threshold(log_trans[r], a)
            trans_masks[g, r] = real[r] & (-log_trans[r] > t_tau[:, None])
            r_tau = quantile.recipe_excess_thresholds(lp, log_trans[r], ltm_np, a)
            recipe_masks[g, r] = real[r] & (excess[r] > r_tau[:, None])

    return LiveTables(
        log_init0=jnp.asarray(np.asarray(lp.log_pi)[:, None] + np.asarray(lp.log_init)),
        log_init=jnp.asarray(lp.log_init),
        log_trans=jnp.asarray(log_trans),
        log_emit_v=jnp.asarray(lp.log_emit_v),
        log_emit_n=jnp.asarray(lp.log_emit_n),
        log_haz=jnp.asarray(log_haz),
        log_stay=jnp.asarray(log_stay),
        stall_bin=jnp.asarray(stall_bin),
        early_bin=jnp.asarray(early_bin),
        trans_bin=jnp.asarray(_first_true(trans_masks)),
        recipe_bin=jnp.asarray(_first_true(recipe_masks)),
        alphas=alphas,
    )


def with_recipe_prior(tb, log_prior):
    """Replace the fitted recipe mixture weights P(R=r) with a per-trial prior (K_R,) in log space,
    e.g. a strong external hint about which recipe this session is. Must put mass on >= 1 recipe."""
    log_prior = jnp.asarray(log_prior, dtype=tb.log_init.dtype)
    log_prior = log_prior - logsumexp(log_prior)
    return tb._replace(log_init0=log_prior[:, None] + tb.log_init)


def pin_recipe(tb, r):
    """The known-recipe case: slice every per-recipe table to recipe r (K_R becomes 1, prior 1).
    The bad sets keep their meaning -- recipe_bin[r] was built against the marginal over ALL
    recipes, so "ordinary in general, wrong for this recipe" is unchanged. With the recipe known,
    the recipe belief cannot move, so §4.7's circularity does not arise."""
    sl = slice(int(r), int(r) + 1)
    return tb._replace(
        log_init0=tb.log_init[sl], log_init=tb.log_init[sl], log_trans=tb.log_trans[sl],
        log_haz=tb.log_haz[sl], log_stay=tb.log_stay[sl], stall_bin=tb.stall_bin[sl],
        early_bin=tb.early_bin[sl], trans_bin=tb.trans_bin[sl], recipe_bin=tb.recipe_bin[sl],
    )


# --------------------------------------------------------------------------------------------
# The filter


def _predict(logb_prev, tb):
    """One predict step from the filtered table at t-1. Returns (bp, end, m):
    bp (K_R,K,D) the predictive table at t; end[r,k,j] = log P(step k ends after j+1 ticks at
    t-1, r, ...); m[r,k] = logsumexp_j end."""
    end = logb_prev + tb.log_haz
    m = logsumexp(end, axis=-1)                                  # (K_R,K)
    new = logsumexp(m[:, :, None] + tb.log_trans, axis=1)       # (K_R,K): into column 0
    cont = logb_prev[..., :-1] + tb.log_stay[..., :-1]
    bp = jnp.concatenate([new[..., None], cont], axis=-1)
    return bp, end, m


def _initial_table(tb):
    k_r, k, d = tb.log_haz.shape
    first = tb.log_init0[..., None]
    rest = jnp.full((k_r, k, d - 1), -jnp.inf, dtype=first.dtype)
    return jnp.concatenate([first, rest], axis=-1)


@jax.jit
def run_filter(tb, loglik, mask):
    """loglik: (T,K) per-tick log P(o_t | z=k); mask: (T,) bool prefix of real ticks.
    Returns (logb_all (T,K_R,K,D) filtered tables, log_norm (T,) log P(o_t|o_<t),
    log_pi_tilde (T,K) log P(z_t=k | o_<t), log_stay (T,K), log_switch (T,K)). The last two
    split the predictive table into "the step continued" (columns >= 1) and "a step began at
    t" (column 0) -- unnormalised, they sum to log_pi_tilde. Padded ticks repeat the last real
    table and contribute log_norm 0."""
    b0 = _initial_table(tb)

    def step(carry, inp):
        logb_prev, is_first = carry
        e, real = inp
        bp_pred, _, _ = _predict(logb_prev, tb)
        bp = jnp.where(is_first, b0, bp_pred)
        joint = bp + e[None, :, None]
        ln = logsumexp(joint)
        logb = joint - ln
        log_pt = logsumexp(bp, axis=(0, 2))
        log_stay = logsumexp(bp[..., 1:], axis=(0, 2))
        log_switch = logsumexp(bp[..., 0], axis=0)
        logb = jnp.where(real, logb, logb_prev)
        ln = jnp.where(real, ln, 0.0)
        new_first = jnp.where(real, False, is_first)
        return (logb, new_first), (logb, ln, log_pt, log_stay, log_switch)

    init = (b0, jnp.array(True))
    _, (logb_all, log_norm, log_pt, log_stay, log_switch) = jax.lax.scan(step, init, (loglik, mask))
    return logb_all, log_norm, log_pt, log_stay, log_switch


def _backstep(beta_next, e_next, tb):
    """beta_t(r,i,j) = log P(o_{t+1:...} | r, z_t=i, e_t=j+1) from beta_{t+1} and o_{t+1}."""
    stay = tb.log_stay[..., :-1] + e_next[None, :, None] + beta_next[..., 1:]
    stay = jnp.concatenate([stay, jnp.full(stay.shape[:-1] + (1,), -jnp.inf, stay.dtype)], axis=-1)
    into = logsumexp(tb.log_trans + (e_next[None, :] + beta_next[..., 0])[:, None, :], axis=-1)
    leave = tb.log_haz + into[..., None]
    return jnp.logaddexp(stay, leave)


@jax.jit
def full_beta(tb, loglik, mask):
    """lag = infinity: the ordinary backward pass, beta_t = log P(o_{t+1:T_true-1} | r, z_t, e_t).
    A reverse scan, so it compiles once per length bucket however long the trial is."""
    shape = tuple(tb.log_haz.shape)
    nxt_real = jnp.concatenate([mask[1:], jnp.zeros((1,), bool)])
    e_next = jnp.concatenate([loglik[1:], jnp.zeros_like(loglik[:1])], axis=0)

    def step(beta_next, inp):
        e, real = inp
        beta = jnp.where(real, _backstep(beta_next, e, tb), jnp.zeros(shape, loglik.dtype))
        return beta, beta

    _, betas = jax.lax.scan(step, jnp.zeros(shape, loglik.dtype), (e_next, nxt_real), reverse=True)
    return betas


@functools.partial(jax.jit, static_argnames=("lag",))
def fixed_lag_beta(tb, loglik, t_true, lag):
    """(T,K_R,K,D) log backward messages over a window of `lag` ticks, for every t at once:
    beta_t = log P(o_{t+1 : min(t+lag, T_true-1)} | r, z_t, e_t). lag=0 is all zeros. Past
    the trial end there is nothing left to explain, so the window simply truncates (the final
    step is right-censored, matching the forward pass's survival weighting)."""
    t_max = loglik.shape[0]
    shape = (t_max,) + tuple(tb.log_haz.shape)
    beta = jnp.zeros(shape, dtype=loglik.dtype)
    if lag == 0:
        return beta
    has_next = (jnp.arange(t_max) + 1) < t_true
    e_next = jnp.concatenate([loglik[1:], jnp.zeros_like(loglik[:1])], axis=0)
    for _ in range(lag):
        beta_next = jnp.concatenate([beta[1:], jnp.zeros_like(beta[:1])], axis=0)
        stepped = jax.vmap(_backstep, in_axes=(0, 0, None))(beta_next, e_next, tb)
        beta = jnp.where(has_next[:, None, None, None], stepped, 0.0)
    return beta


def _binned_mass(log_cells, bins, n_bins):
    """Sum exp(log_cells) per bin, then cumulate: out[g] = mass of cells bad at alphas[g]."""
    per_bin = jax.ops.segment_sum(jnp.exp(log_cells).ravel(), bins.ravel(), num_segments=n_bins + 1)
    return jnp.cumsum(per_bin)[:n_bins]


def _events_one_tick(logb_prev, logb_t, beta_t, e_t, is_first, tb):
    """Hidden-event posteriors at one tick, for every alpha in the grid. beta_t = 0 is the L=0
    filter; a lag-L beta_t makes every probability condition on o_{0:t+L}."""
    g = tb.alphas.shape[0]
    bp, end, m = _predict(logb_prev, tb)
    bp = jnp.where(is_first, _initial_table(tb), bp)
    z = logsumexp(bp + e_t[None, :, None] + beta_t)

    # Stall: posterior over (r,i,j) at t, past the upper tail.
    post = logb_t + beta_t
    post = post - logsumexp(post)
    stall = _binned_mass(post, tb.stall_bin, g)

    # The switch posterior q_t(r,k,j,i), never materialised: step k ended at t-1 after j+1 ticks,
    # step i began at t. Its pieces: m (summed over j), the arrival term into i at elapsed 1.
    arrive = e_t[None, :] + beta_t[..., 0]                                  # (K_R,K) over i
    trans_cells = m[:, :, None] + tb.log_trans + arrive[:, None, :] - z      # (K_R,K,K)
    gk = logsumexp(tb.log_trans + arrive[:, None, :], axis=-1)              # (K_R,K) over k
    early_cells = end + gk[..., None] - z                                    # (K_R,K,D)

    # Recipe: held at the pre-transition recipe belief w_{t-1}(r), so the transition under test
    # cannot move the recipe toward one that explains it (docs/live.md §4.7).
    w_prev = logsumexp(bp, axis=(1, 2))                                      # (K_R,)
    z_r = logsumexp(bp + e_t[None, :, None] + beta_t, axis=(1, 2))           # (K_R,)
    # A recipe the prior rules out has w_prev = z_r = -inf; keep it at -inf, not NaN.
    lead = jnp.where(jnp.isfinite(w_prev), w_prev - z_r, -jnp.inf)
    recipe_cells = lead[:, None, None] + m[:, :, None] + tb.log_trans + arrive[:, None, :]

    neg = jnp.full((g,), 0.0, dtype=stall.dtype)
    trans = jnp.where(is_first, neg, _binned_mass(trans_cells, tb.trans_bin, g))
    early = jnp.where(is_first, neg, _binned_mass(early_cells, tb.early_bin, g))
    recipe = jnp.where(is_first, neg, _binned_mass(recipe_cells, tb.recipe_bin, g))
    switch = jnp.where(is_first, 0.0, jnp.exp(logsumexp(trans_cells)))
    return stall, early, trans, recipe, switch


@jax.jit
def event_posteriors(tb, logb_all, beta, loglik, mask):
    """(T,G) posterior probability of each bad hidden event per tick and alpha, plus (T,) total
    switch probability. Padded ticks are zero."""
    t_max = loglik.shape[0]
    logb_prev = jnp.concatenate([logb_all[:1], logb_all[:-1]], axis=0)
    is_first = jnp.arange(t_max) == 0

    def body(xs):
        lp_, lb_, be_, e_, f_ = xs
        return _events_one_tick(lp_, lb_, be_, e_, f_, tb)

    stall, early, trans, recipe, switch = jax.lax.map(
        body, (logb_prev, logb_all, beta, loglik, is_first)
    )
    keep = mask[:, None]
    return (jnp.where(keep, stall, 0.0), jnp.where(keep, early, 0.0),
            jnp.where(keep, trans, 0.0), jnp.where(keep, recipe, 0.0),
            jnp.where(mask, switch, 0.0))


# --------------------------------------------------------------------------------------------
# Emission p-values


def _pvalue(log_mix, obs_idx):
    """log_mix: (T, X) forecast log-probabilities; obs_idx: (T,). p = sum of the forecast's
    mass on outcomes no more probable than the observed one, ties included."""
    obs = jnp.take_along_axis(log_mix, obs_idx[:, None], axis=1)
    keep = log_mix <= obs + TIE_ATOL
    return jnp.sum(jnp.where(keep, jnp.exp(log_mix), 0.0), axis=1)


@jax.jit
def emission_pvalues(log_pi_tilde, log_emit_v, log_emit_n, verb_ids, noun_ids):
    """(p_emit, p_verb, p_noun), each (T,): the forecast-tail p-value of the observation under
    the live mixture sum_k pi_tilde_t(k) P(x | k). alpha-free; flag as p <= alpha."""
    lv = logsumexp(log_pi_tilde[:, :, None] + log_emit_v[None], axis=1)        # (T,V)
    ln = logsumexp(log_pi_tilde[:, :, None] + log_emit_n[None], axis=1)        # (T,N)
    joint = logsumexp(
        log_pi_tilde[:, :, None, None] + log_emit_v[None, :, :, None] + log_emit_n[None, :, None, :],
        axis=1,
    )                                                                          # (T,V,N)
    n_noun = log_emit_n.shape[1]
    t_max = joint.shape[0]
    p_emit = _pvalue(joint.reshape(t_max, -1), verb_ids * n_noun + noun_ids)
    return p_emit, _pvalue(lv, verb_ids), _pvalue(ln, noun_ids)


# A hypothesis carrying less predictive mass than this is treated as absent: it cannot excuse a
# token, so it drops out of the max (p counted as 0). At a point-mass belief the absent
# hypothesis has exactly zero mass, so R2 is unaffected.
HYPOTHESIS_FLOOR = 1e-12


def split_emission_pvalues(log_stay, log_switch, log_emit_v, log_emit_n, verb_ids, noun_ids):
    """The emission p-values with step TIMING taken out (docs/live.md §4.1, as measured).

    The plain forecast mixes "the step continued" with "a new step began now", weighted by the
    hazard. A healthy switch that comes earlier than the duration model expected therefore gets
    a small forecast probability and is flagged as a wrong token -- duration surprise counted a
    second time, in the emission channel. Measured on 39 healthy Breakfast test trials: every one
    of the 28 live s_emit flags at alpha=5e-3 sat on a step change, after a step that had run a
    median 0.52 of its expected length.

    So score the token under each hypothesis separately and keep the larger p-value: a token is
    flagged only if it is in the alpha-tail whether or not a step change happened at t. Timing is
    left to the duration channels and order to the transition channel."""
    def norm(x):
        tot = logsumexp(x, axis=1, keepdims=True)
        return x - tot, jnp.exp(tot[:, 0]) > HYPOTHESIS_FLOOR

    stay, has_stay = norm(log_stay)
    switch, has_switch = norm(log_switch)
    ps = emission_pvalues(stay, log_emit_v, log_emit_n, verb_ids, noun_ids)
    pw = emission_pvalues(switch, log_emit_v, log_emit_n, verb_ids, noun_ids)
    return tuple(
        jnp.maximum(jnp.where(has_stay, a, 0.0), jnp.where(has_switch, b, 0.0))
        for a, b in zip(ps, pw)
    )


# --------------------------------------------------------------------------------------------
# One trial end to end


class LiveTrace(NamedTuple):
    p_emit: np.ndarray        # (T,) timing-split p-value (the default emission flag)
    p_verb: np.ndarray        # (T,)
    p_noun: np.ndarray        # (T,)
    p_mix: tuple              # (p_emit, p_verb, p_noun) against the plain forecast mixture
    events: dict              # lag -> {"stall"|"early"|"trans"|"recipe": (T,G)}
    switch: dict              # lag -> (T,) total switch probability (diagnostic)
    log_norm: np.ndarray      # (T,) log P(o_t | o_<t)
    log_pi_tilde: np.ndarray  # (T,K)


def _bucket(t, multiple=64):
    return int(-(-t // multiple) * multiple)


INF_LAG = "inf"


def compute_live_trace(tb, verb_ids, noun_ids, lags=(0,), recipe=None):
    """Every live channel for one trial, at every lag in `lags` (emission p-values are a forecast
    of o_t and have no lag). A lag of "inf" conditions on the whole trial (the smoother) -- the
    reference that separates "the future helps" from "Viterbi's single path differs from the
    posterior". `recipe`: None (infer it live from the fitted mixture), an int (known recipe,
    pin_recipe), or a (K_R,) log prior (with_recipe_prior). Trials are padded to a multiple of 64
    ticks so jit compiles once per length bucket, not once per trial."""
    if recipe is not None:
        tb = pin_recipe(tb, recipe) if np.ndim(recipe) == 0 else with_recipe_prior(tb, recipe)
    t_true = len(verb_ids)
    t_max = _bucket(t_true)
    verb = np.zeros(t_max, dtype=np.int32)
    noun = np.zeros(t_max, dtype=np.int32)
    verb[:t_true] = verb_ids
    noun[:t_true] = noun_ids
    mask = np.zeros(t_max, dtype=bool)
    mask[:t_true] = True
    verb, noun, mask = jnp.asarray(verb), jnp.asarray(noun), jnp.asarray(mask)

    loglik = emissions.sequence_loglik(verb, noun, tb.log_emit_v, tb.log_emit_n, mask)
    logb_all, log_norm, log_pt, log_stay, log_switch = run_filter(tb, loglik, mask)
    p_mix = emission_pvalues(log_pt, tb.log_emit_v, tb.log_emit_n, verb, noun)
    p_emit, p_verb, p_noun = split_emission_pvalues(
        log_stay, log_switch, tb.log_emit_v, tb.log_emit_n, verb, noun)

    events, switch = {}, {}
    for lag in lags:
        key = INF_LAG if lag == INF_LAG else int(lag)
        beta = full_beta(tb, loglik, mask) if key == INF_LAG else fixed_lag_beta(tb, loglik, t_true, key)
        stall, early, trans, rec, sw = event_posteriors(tb, logb_all, beta, loglik, mask)
        events[key] = {
            "stall": np.asarray(stall)[:t_true], "early": np.asarray(early)[:t_true],
            "trans": np.asarray(trans)[:t_true], "recipe": np.asarray(rec)[:t_true],
        }
        switch[key] = np.asarray(sw)[:t_true]
        del beta

    return LiveTrace(
        p_emit=np.asarray(p_emit)[:t_true], p_verb=np.asarray(p_verb)[:t_true],
        p_noun=np.asarray(p_noun)[:t_true],
        p_mix=tuple(np.asarray(x)[:t_true] for x in p_mix), events=events, switch=switch,
        log_norm=np.asarray(log_norm)[:t_true], log_pi_tilde=np.asarray(log_pt)[:t_true],
    )


def flags(trace, alphas, alpha, kappa, lag=0, emission="split"):
    """Per-channel (T,) bool flags at one operating point, keyed by the hindsight channel names.
    `alphas` is the ascending grid the tables were built on; `alpha` must be one of its values.
    emission="split" (default) or "mix" picks the emission p-value variant."""
    g = int(np.flatnonzero(np.isclose(alphas, alpha))[0])
    ev = trace.events[lag]
    pe, pv, pn = (trace.p_emit, trace.p_verb, trace.p_noun) if emission == "split" else trace.p_mix
    return {
        "s_emit": pe <= alpha,
        "s_verb": pv <= alpha,
        "s_noun": pn <= alpha,
        "s_temporal": ev["stall"][:, g] > kappa,
        "s_dur_two": ev["early"][:, g] > kappa,
        "s_transition": ev["trans"][:, g] > kappa,
        "s_recipe_transition": ev["recipe"][:, g] > kappa,
    }
