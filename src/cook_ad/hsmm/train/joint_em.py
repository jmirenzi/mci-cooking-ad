"""Soft (Baum-Welch) EM for the joint HSMM: the E-step's expected counts under every recipe,
weighted by each trial's recipe responsibility, then a Dirichlet-MAP M-step with the shrunk
duration fit. docs/training.md.
"""
import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from tqdm.auto import tqdm

from cook_ad.hsmm.kernels import forward_pass, sequence_loglik
from cook_ad.hsmm.params import JointHSMMParams, to_log_tables
from cook_ad.hsmm.train import durations_fit, messages

jax.config.update("jax_enable_x64", True)


def pad_batch(sequences, t_max=None):
    """sequences: list of {"verb_ids": [...], "noun_ids": [...]} (Phase 1's sequences.json).
    Returns (verb_ids, noun_ids: (N,T_max) int32, mask: (N,T_max) bool). Pads with the
    valid dummy id 0 -- never read meaningfully, gated out everywhere via `mask`.
    `t_max` is derived from the data if not given, never hardcoded.
    """
    lengths = [len(seq["verb_ids"]) for seq in sequences]
    if t_max is None:
        t_max = max(lengths)

    n = len(sequences)
    verb_ids = np.zeros((n, t_max), dtype=np.int32)
    noun_ids = np.zeros((n, t_max), dtype=np.int32)
    mask = np.zeros((n, t_max), dtype=bool)
    for i, seq in enumerate(sequences):
        length = len(seq["verb_ids"])
        verb_ids[i, :length] = seq["verb_ids"]
        noun_ids[i, :length] = seq["noun_ids"]
        mask[i, :length] = True

    return jnp.array(verb_ids), jnp.array(noun_ids), jnp.array(mask)


# Chunks are trimmed to their own longest sequence, rounded up to the next power of two.
# pad_batch pads everything to the batch's global T_max, so on a corpus with a wide length
# spread most chunks are mostly padding the mask then discards. Powers of two rather than a
# fixed stride because each distinct trimmed length is a separate XLA compilation, and one
# compilation here costs tens of seconds -- a finer stride spends more on compiling than the
# trimming saves.
TICK_BUCKET_MIN = 256


def _chunk_ticks(mask_chunk, t_max):
    """Ticks this chunk needs: its longest true length, rounded up to a power of two."""
    used = int(jnp.max(jnp.sum(mask_chunk, axis=-1)))
    bucket = TICK_BUCKET_MIN
    while bucket < used:
        bucket *= 2
    return int(min(bucket, t_max))


def _length_order(mask):
    """Sequence indices sorted by true length, so chunks are length-homogeneous. Safe to
    reorder: e_step accumulates sums over sequences and returns nothing per-sequence."""
    return jnp.argsort(jnp.sum(mask, axis=-1))


def make_lam_schedule(spec):
    """Parse a lam-schedule spec string into `iteration -> float`, for `run_joint_em`'s
    `lam_schedule` argument. `iteration` is that loop's own 0-based counter.

        "const:X"        -- lam = X always (equivalent to passing lam=X directly; exists so a
                             CLI can express both the constant and moving cases the same way).
        "geom:X0,ratio"   -- lam = X0 * ratio**iteration (geometric decay).
        "linear:X0,n"     -- lam = X0 * max(0, 1 - iteration/n) (reaches exactly 0 at n, clamped
                             non-negative after).
        "freeze:X0,n"     -- lam = X0 for iteration < n, else 0 (freeze the assignment for the
                             first n iterations, then release it to the likelihood alone).
    """
    kind, _, rest = spec.partition(":")
    if kind == "const":
        x = float(rest)
        return lambda it: x
    if kind == "geom":
        x0_s, ratio_s = rest.split(",")
        x0, ratio = float(x0_s), float(ratio_s)
        return lambda it: x0 * (ratio ** it)
    if kind == "linear":
        x0_s, n_s = rest.split(",")
        x0, n = float(x0_s), float(n_s)
        return lambda it: max(0.0, x0 * (1.0 - it / n))
    if kind == "freeze":
        x0_s, n_s = rest.split(",")
        x0, n = float(x0_s), float(n_s)
        return lambda it: x0 if it < n else 0.0
    raise ValueError(f"unknown lam schedule: {spec!r} (expected const/geom/linear/freeze)")


@functools.partial(jax.jit, static_argnames=("d_max",))
def _e_step_chunk(log_pi, log_init, log_A, log_emit_v, log_emit_n, log_pmf, log_surv,
                  log_prior, lam, verb_ids, noun_ids, mask, d_max):
    """Joint E-step for one chunk of trials. Emissions are shared across recipes, so `loglik`
    is computed once per trial and fed into K_R forward-backward passes via a nested vmap: inner
    over recipes (the four recipe-indexed tables), outer over the trial chunk.

    `log_prior` (chunk,K_R) / `lam`: the anchoring dial, added to log_rho as `lam * log_prior`
    (see e_step's docstring for the caller-side slicing this depends on getting right). `lam` is
    a traced scalar, not a static jit argument, so a schedule that changes it every iteration
    does not retrigger compilation.
    """
    loglik = jax.vmap(sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_emit_v, log_emit_n, mask
    )  # (chunk,T,K) -- recipe-independent

    def _combine_recipe(loglik_i, mask_i, log_init_r, log_A_r, log_pmf_r, log_surv_r):
        ll = jnp.where(mask_i[:, None], loglik_i, 0.0)
        return messages.combine_sufficient_stats(
            ll, mask_i, log_init_r, log_A_r, log_pmf_r, log_surv_r, d_max
        )

    combine_over_r = jax.vmap(_combine_recipe, in_axes=(None, None, 0, 0, 0, 0))
    combine_over_trials = jax.vmap(combine_over_r, in_axes=(0, 0, None, None, None, None))
    xi_trans, xi_dur, cens, gamma, log_z = combine_over_trials(
        loglik, mask, log_init, log_A, log_pmf, log_surv
    )
    # xi_trans (chunk,K_R,K,K); xi_dur,cens (chunk,K_R,K,D); gamma (chunk,K_R,T,K); log_z (chunk,K_R)

    log_rho = log_pi[None, :] + log_z + lam * log_prior   # (chunk,K_R)
    # (chunk,) the joint objective this E-step maximizes; equals marginal log P(obs_i) only
    # at lam=0 -- otherwise it is log P(obs_i) tilted by the external prior, still a valid EM
    # objective for a FIXED lam (see run_joint_em's docstring for why a moving lam breaks this).
    trial_ll = logsumexp(log_rho, axis=-1)
    rho = jnp.exp(log_rho - trial_ll[:, None])  # (chunk,K_R), sums to 1 over r -- never exp(raw log_z)

    gamma_masked = jnp.where(mask[:, None, :, None], gamma, 0.0)  # (chunk,K_R,T,K)

    # Shared emission stats: collapse the recipe axis via rho-weighting BEFORE the tick/state
    # einsum.
    gamma_bar = jnp.einsum("nr,nrtk->ntk", rho, gamma_masked)  # (chunk,T,K)
    n_verb, n_noun = log_emit_v.shape[1], log_emit_n.shape[1]
    verb_onehot = jax.nn.one_hot(verb_ids, n_verb, dtype=gamma.dtype)
    noun_onehot = jax.nn.one_hot(noun_ids, n_noun, dtype=gamma.dtype)

    stats = {
        "pi_counts": jnp.sum(rho, axis=0),                                              # (K_R,)
        "init_counts": jnp.sum(rho[:, :, None] * gamma_masked[:, :, 0, :], axis=0),      # (K_R,K)
        "trans_counts": jnp.sum(rho[:, :, None, None] * xi_trans, axis=0),               # (K_R,K,K)
        "verb_counts": jnp.einsum("ntk,ntv->kv", gamma_bar, verb_onehot),                # (K,V) shared
        "noun_counts": jnp.einsum("ntk,ntv->kv", gamma_bar, noun_onehot),                # (K,N) shared
        "xi_dur": jnp.sum(rho[:, :, None, None] * xi_dur, axis=0),                       # (K_R,K,D)
        "cens": jnp.sum(rho[:, :, None, None] * cens, axis=0),                           # (K_R,K,D)
    }
    return stats, jnp.sum(trial_ll)


def e_step(params, verb_ids, noun_ids, mask, d_max, temperature=1.0, chunk_size=8,
           recipe_log_prior=None, lam=0.0):
    """Chunked E-step. Peak memory scales K_R-fold over a single-recipe E-step (the
    (chunk,K_R,T,K) gamma intermediate), so keep chunk_size small.

    `recipe_log_prior`: (N_trials,K_R) external per-trial recipe log-prior, or None (treated as
    all-zero -- a no-op regardless of `lam`). `lam`: the anchoring dial's strength.

    THE TRAP: trials are reordered by length before chunking (`_length_order`) so that chunks are
    length-homogeneous -- `verb_ids`/`noun_ids`/`mask` are all sliced via `[idx]`, never
    `[start:end]`. `recipe_log_prior` is a PER-TRIAL array in the original trial order, so it must
    be sliced with that exact same `idx` or the prior silently applies to the wrong trial.
    """
    lp = to_log_tables(params, d_max)
    inv_temp = 1.0 / temperature
    log_emit_v = lp.log_emit_v * inv_temp
    log_emit_n = lp.log_emit_n * inv_temp
    log_pmf = lp.log_pmf * inv_temp
    log_surv = lp.log_surv * inv_temp

    n = verb_ids.shape[0]
    k_r = lp.log_pi.shape[0]
    if recipe_log_prior is None:
        recipe_log_prior = jnp.zeros((n, k_r))
    lam = jnp.asarray(lam, dtype=lp.log_pi.dtype)

    total_stats = None
    total_trial_ll = 0.0
    t_max = verb_ids.shape[1]
    order = _length_order(mask)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        idx = order[start:end]  # THE TRAP above: recipe_log_prior must use this same idx
        t_used = _chunk_ticks(mask[idx], t_max)
        chunk_stats, chunk_ll = _e_step_chunk(
            lp.log_pi, lp.log_init, lp.log_A, log_emit_v, log_emit_n, log_pmf, log_surv,
            recipe_log_prior[idx], lam,
            verb_ids[idx, :t_used], noun_ids[idx, :t_used], mask[idx, :t_used], d_max,
        )
        if total_stats is None:
            total_stats = chunk_stats
        else:
            total_stats = {name: total_stats[name] + chunk_stats[name] for name in total_stats}
        total_trial_ll = total_trial_ll + chunk_ll

    return total_stats, total_trial_ll


@functools.partial(jax.jit, static_argnames=("d_max", "global_damping"))
def m_step(params, stats, alpha_pi, alpha_init, alpha_trans, alpha_emit_v, alpha_emit_n, kappa,
           d_max, prev_global_r=None, prev_global_p=None, global_damping=0.0,
           emit_prior_v=None, emit_prior_n=None):
    """Dirichlet MAP per recipe for init/trans + pi, shared Dirichlet MAP for emissions, and the
    shrunk duration fit (durations_fit.fit_durations_shrunk).

    `prev_global_r`/`prev_global_p`/`global_damping`: passed through to fit_durations_shrunk to
    damp the pooled global per-state duration fit across calls. Returns (params, global_r,
    global_p) -- the caller threads global_r/global_p back in on the next call.

    `emit_prior_v`/`emit_prior_n`: optional (K,V)/(K,N) Dirichlet pseudocount matrices replacing
    the flat `alpha_emit_*/width` prior -- needed when the initialisation has already given each
    state a meaning to keep (lexical_init.py).
    """
    k_r = params.pi_counts.shape[0]
    k = params.init_counts.shape[1]
    n_verb = params.verb_counts.shape[1]
    n_noun = params.noun_counts.shape[1]

    new_pi_counts = alpha_pi / k_r + stats["pi_counts"]
    new_init_counts = alpha_init / k + stats["init_counts"]
    new_trans_counts = (alpha_trans / k + stats["trans_counts"]) * (1.0 - jnp.eye(k))[None, :, :]
    prior_v = alpha_emit_v / n_verb if emit_prior_v is None else emit_prior_v
    prior_n = alpha_emit_n / n_noun if emit_prior_n is None else emit_prior_n
    new_verb_counts = prior_v + stats["verb_counts"]
    new_noun_counts = prior_n + stats["noun_counts"]

    dur_r, dur_p, global_r, global_p = durations_fit.fit_durations_shrunk(
        stats["xi_dur"], stats["cens"], params.dur_r, params.dur_p, d_max, kappa,
        prev_global_r=prev_global_r, prev_global_p=prev_global_p, global_damping=global_damping,
    )
    new_params = JointHSMMParams(
        new_init_counts, new_trans_counts, new_verb_counts, new_noun_counts, dur_r, dur_p,
        new_pi_counts,
    )
    return new_params, global_r, global_p


def run_joint_em(
    init_params,
    sequences,
    d_max,
    alpha_pi=1.0,
    alpha_init=0.5,
    alpha_trans=0.5,
    alpha_emit_v=None,
    alpha_emit_n=None,
    kappa=5.0,
    max_iters=100,
    tol=1e-4,
    chunk_size=8,
    progress=False,
    start_iteration=0,
    init_history=None,
    init_prev_obj=None,
    on_checkpoint=None,
    checkpoint_every=5,
    global_damping=0.0,
    emit_prior_v=None,
    emit_prior_n=None,
    recipe_log_prior=None,
    lam=0.0,
    lam_schedule=None,
):
    """Single deterministic EM run from `init_params` (the lexical warm start).

    The joint objective sum_i logsumexp_r(log_pi[r] + logz_ir) must be non-decreasing across
    iterations; a decrease beyond `tol` is logged as a warning (not raised), since the
    shrinkage duration fit is not an exact M-step and can legitimately dip slightly.

    Resumable: `start_iteration`/`init_history`/`init_prev_obj` let a caller continue a run
    from a previously checkpointed state (`init_params` is then the CHECKPOINTED params, not
    the original warm start -- the caller's job). `on_checkpoint(iteration, params, history)`,
    if given, is called every `checkpoint_every` iterations AFTER that iteration's M-step, so
    `params` is always the state to resume FROM (the next E-step will use exactly this). This
    function does no file I/O itself -- `on_checkpoint` owns persistence, keeping this module
    pure-computation.

    Returns (p, obj, history, converged). `converged` is True only if the tol-based stopping
    criterion actually fired (not merely reaching max_iters) -- a caller resuming a checkpoint
    should keep calling this (with a higher max_iters if needed) until `converged` is True,
    not until `history` stops growing, since hitting max_iters without converging means there
    was more work requested than budget allowed, not that the fit is done. If `start_iteration
    >= max_iters` (resuming a run already at or past this call's iteration budget), the loop
    body never executes and this returns immediately with `converged=False` and `history`/`p`
    unchanged from the input -- cheap and safe to call unconditionally.

    `emit_prior_v`/`emit_prior_n`: passed through to `m_step` -- see its docstring.

    `recipe_log_prior`/`lam`/`lam_schedule`: the anchoring dial (see `e_step`'s docstring for
    the term itself, and `make_lam_schedule` for building a schedule from a spec string).
    `lam_schedule`, if given, is a callable `iteration -> float` called each iteration in place
    of the constant `lam` -- e.g. `make_lam_schedule("freeze:1e6,15")` freezes the assignment to
    `recipe_log_prior` for 15 iterations, then releases it (lam=0) for the rest of the run,
    making "freeze then release" a schedule rather than separate machinery. `iteration` is this
    loop's own counter (the same index `start_iteration` resumes at), so a schedule survives a
    checkpoint resume unless the caller re-derives it. The decrease-warning above is suppressed
    on any iteration where lam actually changed value: the objective ITSELF changes when lam
    moves, so a "decrease" there reflects a different objective, not a broken M-step.

    `global_damping`: EMA damping factor (0 = off, the default) for the duration M-step's
    pooled global per-state fit -- see fit_durations_shrunk's docstring for why a near-empty
    state's global fit can swing by an order of magnitude between calls and drag every recipe's
    copy of that state with it. The EMA's running state lives in this function's local
    `prev_global_r`/`prev_global_p`, NOT in `p` (JointHSMMParams's on-disk schema is
    unchanged), so it resets to undamped on every call -- including a checkpoint resume. That's
    fine within one long run (it warms up again in a few iterations) but means damping is not
    itself something a checkpoint remembers across a restart.
    """
    if alpha_emit_v is None:
        alpha_emit_v = float(init_params.verb_counts.shape[1])
    if alpha_emit_n is None:
        alpha_emit_n = float(init_params.noun_counts.shape[1])

    verb_ids, noun_ids, mask = pad_batch(sequences)

    p = init_params
    prev_obj = jnp.asarray(-jnp.inf) if init_prev_obj is None else jnp.asarray(init_prev_obj)
    obj = prev_obj
    history = list(init_history) if init_history else []
    converged = False  # stays False if the range is empty (e.g. resuming a run already at max_iters)
    prev_global_r, prev_global_p = None, None  # duration EMA state -- see global_damping docstring above
    prev_lam = None  # set on the first iteration; see the decrease-warning suppression below

    iter_bar = tqdm(
        range(start_iteration, max_iters), desc="joint EM",
        initial=start_iteration, total=max_iters, disable=not progress,
    )
    for iteration in iter_bar:
        cur_lam = lam_schedule(iteration) if lam_schedule is not None else lam
        stats, obj = e_step(
            p, verb_ids, noun_ids, mask, d_max, chunk_size=chunk_size,
            recipe_log_prior=recipe_log_prior, lam=cur_lam,
        )
        obj_value = float(obj)
        lam_moved = prev_lam is not None and cur_lam != prev_lam
        if obj_value < float(prev_obj) - tol and not lam_moved:
            warnings.warn(
                f"joint EM objective decreased at iteration {iteration}: "
                f"{float(prev_obj):.4f} -> {obj_value:.4f}"
            )
        history.append(obj_value)
        iter_bar.set_postfix(obj=f"{obj_value:.1f}", lam=f"{cur_lam:.3g}")
        converged = abs(obj_value - float(prev_obj)) < tol and not lam_moved
        prev_obj = obj
        prev_lam = cur_lam
        p, prev_global_r, prev_global_p = m_step(
            p, stats, alpha_pi, alpha_init, alpha_trans, alpha_emit_v, alpha_emit_n, kappa, d_max,
            prev_global_r=prev_global_r, prev_global_p=prev_global_p, global_damping=global_damping,
            emit_prior_v=emit_prior_v, emit_prior_n=emit_prior_n,
        )
        if on_checkpoint is not None and ((iteration + 1) % checkpoint_every == 0 or converged):
            on_checkpoint(iteration + 1, p, history)
        if converged:
            break
    iter_bar.close()

    return p, obj, history, converged


@functools.partial(jax.jit, static_argnames=("d_max",))
def _recipe_logz_chunk(log_pi, log_init, log_A, log_pmf, log_surv, log_emit_v, log_emit_n,
                       log_prior, lam, verb_ids, noun_ids, mask, d_max):
    """Recipe responsibilities for one chunk: one forward pass per (trial, recipe)."""
    loglik = jax.vmap(sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_emit_v, log_emit_n, mask
    )

    def forward_logz(loglik_i, mask_i, log_init_r, log_A_r, log_pmf_r, log_surv_r):
        ll = jnp.where(mask_i[:, None], loglik_i, 0.0)
        log_norm, _ = forward_pass(ll, log_init_r, log_A_r, log_pmf_r, log_surv_r, mask_i, d_max)
        return jnp.sum(log_norm)

    over_r = jax.vmap(forward_logz, in_axes=(None, None, 0, 0, 0, 0))
    over_trials = jax.vmap(over_r, in_axes=(0, 0, None, None, None, None))
    log_z = over_trials(loglik, mask, log_init, log_A, log_pmf, log_surv)  # (chunk,K_R)

    log_rho = log_pi[None, :] + log_z + lam * log_prior
    trial_ll = logsumexp(log_rho, axis=-1)
    rho = jnp.exp(log_rho - trial_ll[:, None])
    r_hat = jnp.argmax(log_rho, axis=-1)
    return r_hat, rho, trial_ll


def infer_recipe(params, verb_ids, noun_ids, mask, d_max, chunk_size=8, recipe_log_prior=None,
                 lam=0.0):
    """r_hat_i = argmax_r(log_pi[r] + log P(o_i | r) + lam * log_prior_ir), with the posterior rho
    and log P(o_i). Training's version of hsmm.batch.infer_recipe, with the anchoring prior.
    Unlike e_step this does NOT reorder trials, so `recipe_log_prior` is sliced `[start:end]`."""
    lp = to_log_tables(params, d_max)
    n = verb_ids.shape[0]
    k_r = lp.log_pi.shape[0]
    if recipe_log_prior is None:
        recipe_log_prior = jnp.zeros((n, k_r))
    lam = jnp.asarray(lam, dtype=lp.log_pi.dtype)

    out = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out.append(_recipe_logz_chunk(
            lp.log_pi, lp.log_init, lp.log_A, lp.log_pmf, lp.log_surv, lp.log_emit_v,
            lp.log_emit_n, recipe_log_prior[start:end], lam,
            verb_ids[start:end], noun_ids[start:end], mask[start:end], d_max,
        ))
    return tuple(jnp.concatenate(x) for x in zip(*out))
