import functools
import warnings

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from tqdm.auto import tqdm

from cook_ad.hsmm import durations, emissions, joint_params, messages
from cook_ad.hsmm.em import _chunk_ticks, _length_order, pad_batch  # noqa: F401 -- pad_batch re-exported
from cook_ad.hsmm.joint_params import JointHSMMParams

jax.config.update("jax_enable_x64", True)


@functools.partial(jax.jit, static_argnames=("d_max",))
def _e_step_chunk(log_pi, log_init, log_trans, log_emit_v, log_emit_n, log_dur_pmf, log_dur_survival,
                   verb_ids, noun_ids, mask, d_max):
    """Joint E-step for one chunk of trials. Emissions are shared across recipes, so `loglik`
    is computed exactly once per trial (em.py's single-recipe pattern) and then fed into K_R
    forward-backward passes via a nested vmap: inner over recipes (the four recipe-indexed
    tables), outer over the trial chunk (loglik/mask vary per trial, tables broadcast).
    """
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_emit_v, log_emit_n, mask
    )  # (chunk,T,K) -- recipe-independent

    combine_over_r = jax.vmap(
        messages.combine_sufficient_stats, in_axes=(None, None, 0, 0, 0, 0, None)
    )
    combine_over_trials = jax.vmap(
        combine_over_r, in_axes=(0, 0, None, None, None, None, None)
    )
    xi_trans, xi_dur, cens, gamma, log_z = combine_over_trials(
        loglik, mask, log_init, log_trans, log_dur_pmf, log_dur_survival, d_max
    )
    # xi_trans (chunk,K_R,K,K); xi_dur,cens (chunk,K_R,K,D); gamma (chunk,K_R,T,K); log_z (chunk,K_R)

    log_rho = log_pi[None, :] + log_z          # (chunk,K_R)
    trial_ll = logsumexp(log_rho, axis=-1)     # (chunk,) marginal log P(obs_i), the joint objective
    rho = jnp.exp(log_rho - trial_ll[:, None])  # (chunk,K_R), sums to 1 over r -- never exp(raw log_z)

    gamma_masked = jnp.where(mask[:, None, :, None], gamma, 0.0)  # (chunk,K_R,T,K)

    # Shared emission stats: collapse the recipe axis via rho-weighting BEFORE the tick/state
    # einsum, so the accumulation is shape-identical to em.py's single-recipe version.
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


def e_step(joint_hsmm_params, verb_ids, noun_ids, mask, d_max, temperature=1.0, chunk_size=8):
    """Chunked driver mirroring hsmm.em.e_step. Peak memory scales an extra K_R-fold over the
    single-recipe E-step (the (chunk,K_R,T,K) gamma intermediate), so callers should shrink
    chunk_size roughly by K_R relative to the cascade EM's setting.
    """
    log_probs = joint_params.to_log_probs_joint(joint_hsmm_params, d_max)
    inv_temp = 1.0 / temperature
    log_emit_v = log_probs.log_emit_v * inv_temp
    log_emit_n = log_probs.log_emit_n * inv_temp
    log_dur_pmf = log_probs.log_dur_pmf * inv_temp
    log_dur_survival = log_probs.log_dur_survival * inv_temp

    n = verb_ids.shape[0]
    total_stats = None
    total_trial_ll = 0.0
    t_max = verb_ids.shape[1]
    order = _length_order(mask)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        idx = order[start:end]
        t_used = _chunk_ticks(mask[idx], t_max)
        chunk_stats, chunk_ll = _e_step_chunk(
            log_probs.log_pi, log_probs.log_init, log_probs.log_trans,
            log_emit_v, log_emit_n, log_dur_pmf, log_dur_survival,
            verb_ids[idx, :t_used], noun_ids[idx, :t_used], mask[idx, :t_used], d_max,
        )
        if total_stats is None:
            total_stats = chunk_stats
        else:
            total_stats = {name: total_stats[name] + chunk_stats[name] for name in total_stats}
        total_trial_ll = total_trial_ll + chunk_ll

    return total_stats, total_trial_ll


@functools.partial(jax.jit, static_argnames=("d_max", "global_damping"))
def m_step(joint_hsmm_params, stats, alpha_pi, alpha_init, alpha_trans, alpha_emit_v, alpha_emit_n, kappa, d_max,
           prev_global_r=None, prev_global_p=None, global_damping=0.0,
           emit_prior_v=None, emit_prior_n=None):
    """Dirichlet MAP per recipe for init/trans + pi, shared Dirichlet MAP for emissions (as
    today), plus the shrinkage duration fit (durations.fit_durations_shrunk) in place of
    em.py's plain censoring-imputation + Newton.

    `prev_global_r`/`prev_global_p`/`global_damping`: passed straight through to
    fit_durations_shrunk to damp the pooled global per-state duration fit across calls (see
    its docstring for why this matters -- shared instability across all K_R recipes' copies of
    a near-empty state, not per-cell noise). Returns (params, global_r, global_p) instead of
    just params -- the caller (run_joint_em) threads global_r/global_p back in as
    prev_global_r/prev_global_p on the next call to continue the EMA.

    `emit_prior_v`/`emit_prior_n`: optional (K,V)/(K,N) Dirichlet pseudocount matrices replacing
    the flat `alpha_emit_*/width` prior -- needed when the initialisation has already given each
    state a meaning to keep (recipe/lexical_init.py).
    """
    k_r = joint_hsmm_params.pi_counts.shape[0]
    k = joint_hsmm_params.init_counts.shape[1]
    n_verb = joint_hsmm_params.verb_counts.shape[1]
    n_noun = joint_hsmm_params.noun_counts.shape[1]

    new_pi_counts = alpha_pi / k_r + stats["pi_counts"]
    new_init_counts = alpha_init / k + stats["init_counts"]
    new_trans_counts = (alpha_trans / k + stats["trans_counts"]) * (1.0 - jnp.eye(k))[None, :, :]
    prior_v = alpha_emit_v / n_verb if emit_prior_v is None else emit_prior_v
    prior_n = alpha_emit_n / n_noun if emit_prior_n is None else emit_prior_n
    # Latent-intended-token mapping, as in em.m_step; a no-op without a kernel.
    log_b_v = joint_params.params._row_normalize(joint_hsmm_params.verb_counts)
    log_b_n = joint_params.params._row_normalize(joint_hsmm_params.noun_counts)
    verb_stats = joint_params.params.latent_counts(
        stats["verb_counts"], log_b_v, joint_hsmm_params.kernel_v)
    noun_stats = joint_params.params.latent_counts(
        stats["noun_counts"], log_b_n, joint_hsmm_params.kernel_n)
    new_verb_counts = prior_v + verb_stats
    new_noun_counts = prior_n + noun_stats

    dur_r, dur_p, global_r, global_p = durations.fit_durations_shrunk(
        stats["xi_dur"], stats["cens"], joint_hsmm_params.dur_r, joint_hsmm_params.dur_p, d_max, kappa,
        prev_global_r=prev_global_r, prev_global_p=prev_global_p, global_damping=global_damping,
    )

    new_params = JointHSMMParams(
        new_init_counts, new_trans_counts, new_verb_counts, new_noun_counts, dur_r, dur_p,
        new_pi_counts, joint_hsmm_params.kernel_v, joint_hsmm_params.kernel_n,
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
):
    """Single deterministic EM run from `init_params` (the cascade warm start, per spec --
    no restart loop here; random-init fallback restarts are the runner's concern, mirroring
    how hsmm.em.run_em owns restarts while its e_step/m_step stay restart-agnostic).

    The joint objective sum_i logsumexp_r(log_pi[r] + logz_ir) must be non-decreasing across
    iterations; a decrease beyond `tol` is logged as a warning (not raised), since the
    shrinkage duration fit is not an exact M-step and can legitimately dip slightly.

    Resumable: `start_iteration`/`init_history`/`init_prev_obj` let a caller continue a run
    from a previously checkpointed state (`init_params` is then the CHECKPOINTED params, not
    the original warm start -- the caller's job). `on_checkpoint(iteration, params, history)`,
    if given, is called every `checkpoint_every` iterations AFTER that iteration's M-step, so
    `params` is always the state to resume FROM (the next E-step will use exactly this). This
    function does no file I/O itself -- `on_checkpoint` owns persistence, keeping this module
    pure-computation (see run_joint.py for the actual save/resume implementation).

    Returns (p, obj, history, converged). `converged` is True only if the tol-based stopping
    criterion actually fired (not merely reaching max_iters) -- a caller resuming a checkpoint
    should keep calling this (with a higher max_iters if needed) until `converged` is True,
    not until `history` stops growing, since hitting max_iters without converging means there
    was more work requested than budget allowed, not that the fit is done. If `start_iteration
    >= max_iters` (resuming a run already at or past this call's iteration budget), the loop
    body never executes and this returns immediately with `converged=False` and `history`/`p`
    unchanged from the input -- cheap and safe to call unconditionally.

    `emit_prior_v`/`emit_prior_n`: passed through to `m_step` -- see its docstring.

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

    iter_bar = tqdm(
        range(start_iteration, max_iters), desc="joint EM",
        initial=start_iteration, total=max_iters, disable=not progress,
    )
    for iteration in iter_bar:
        stats, obj = e_step(p, verb_ids, noun_ids, mask, d_max, chunk_size=chunk_size)
        obj_value = float(obj)
        if obj_value < float(prev_obj) - tol:
            warnings.warn(
                f"joint EM objective decreased at iteration {iteration}: "
                f"{float(prev_obj):.4f} -> {obj_value:.4f}"
            )
        history.append(obj_value)
        iter_bar.set_postfix(obj=f"{obj_value:.1f}")
        converged = abs(obj_value - float(prev_obj)) < tol
        prev_obj = obj
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
def _recipe_logz_chunk(log_pi, log_init, log_trans, log_dur_pmf, log_dur_survival,
                        log_emit_v, log_emit_n, verb_ids, noun_ids, mask, d_max):
    """Decode-only per-chunk kernel: recipe responsibilities need only each (trial,recipe)'s
    total log-likelihood, not the full sufficient-stats bundle -- so this uses
    messages.forward_pass directly (an existing function, not a reimplementation) rather than
    the heavier combine_sufficient_stats, skipping the backward pass and the xi/gamma scans
    entirely. log_z here is bit-identical to combine_sufficient_stats's log_z (both are
    jnp.sum(forward_pass(...)[0])), so this cannot diverge numerically from training's E-step.
    """
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_emit_v, log_emit_n, mask
    )

    def forward_logz(loglik_i, mask_i, log_init_r, log_trans_r, log_dur_pmf_r, log_dur_survival_r):
        log_norm, _ = messages.forward_pass(loglik_i, log_init_r, log_trans_r, log_dur_pmf_r, log_dur_survival_r, mask_i, d_max)
        return jnp.sum(log_norm)

    over_r = jax.vmap(forward_logz, in_axes=(None, None, 0, 0, 0, 0))
    over_trials = jax.vmap(over_r, in_axes=(0, 0, None, None, None, None))
    log_z = over_trials(loglik, mask, log_init, log_trans, log_dur_pmf, log_dur_survival)  # (chunk,K_R)

    log_rho = log_pi[None, :] + log_z
    trial_ll = logsumexp(log_rho, axis=-1)
    rho = jnp.exp(log_rho - trial_ll[:, None])
    r_hat = jnp.argmax(log_rho, axis=-1)
    return r_hat, rho, trial_ll


def infer_recipe(joint_hsmm_params, verb_ids, noun_ids, mask, d_max, chunk_size=8):
    """r_hat_i = argmax_r(log_pi[r] + logz_ir); also returns the full posterior rho and the
    per-trial marginal log-likelihood. Chunked the same way as e_step.
    """
    log_probs = joint_params.to_log_probs_joint(joint_hsmm_params, d_max)

    n = verb_ids.shape[0]
    r_hat_chunks, rho_chunks, trial_ll_chunks = [], [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        r_hat, rho, trial_ll = _recipe_logz_chunk(
            log_probs.log_pi, log_probs.log_init, log_probs.log_trans,
            log_probs.log_dur_pmf, log_probs.log_dur_survival,
            log_probs.log_emit_v, log_probs.log_emit_n,
            verb_ids[start:end], noun_ids[start:end], mask[start:end], d_max,
        )
        r_hat_chunks.append(r_hat)
        rho_chunks.append(rho)
        trial_ll_chunks.append(trial_ll)

    return jnp.concatenate(r_hat_chunks), jnp.concatenate(rho_chunks), jnp.concatenate(trial_ll_chunks)
