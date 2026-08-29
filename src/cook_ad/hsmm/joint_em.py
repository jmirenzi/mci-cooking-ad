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


def _tilt_tick(log_tilt, noun_ids, mask):
    """log_tilt: (K_R,N) per-recipe additive noun term. noun_ids, mask: (chunk,T). Returns
    (chunk,K_R,T): log_tilt gathered at each tick's observed noun, zeroed on padded ticks so it
    composes safely with sequence_loglik's own masked output (a nonzero tilt on a padded tick
    would otherwise leak into what must stay an exact no-op)."""
    tick = jnp.transpose(log_tilt[:, noun_ids], (1, 0, 2))  # (chunk,K_R,T)
    return jnp.where(mask[:, None, :], tick, 0.0)


@functools.partial(jax.jit, static_argnames=("d_max",))
def _e_step_chunk(log_pi, log_init, log_trans, log_emit_v, log_emit_n, log_dur_pmf, log_dur_survival,
                   log_tilt, log_tilt_norm, log_prior, lam, verb_ids, noun_ids, mask, d_max):
    """Joint E-step for one chunk of trials. Emissions are shared across recipes, so `loglik`
    is computed exactly once per trial (em.py's single-recipe pattern) and then fed into K_R
    forward-backward passes via a nested vmap: inner over recipes (the four recipe-indexed
    tables, plus the per-(trial,recipe) tilt tick and the per-recipe tilt normalizer), outer
    over the trial chunk (loglik/mask/tilt_tick vary per trial, tables broadcast).

    `log_tilt`/`log_tilt_norm`: the rank-1 recipe modulation of the noun emission (see
    joint_params.tilt_terms). Adding log_tilt[r, n_t] - log_tilt_norm[r, k] to the shared
    `loglik` inside the recipe vmap is the ENTIRE cost of the modulation -- `loglik` itself is
    still computed once per trial, not once per (trial, recipe): the naive K_R-fold increase in
    emissions.sequence_loglik calls that a per-recipe emission table would otherwise cost.

    `log_prior` (chunk,K_R) / `lam`: the anchoring dial, added to log_rho as `lam * log_prior`
    (see e_step's docstring for the caller-side slicing this depends on getting right). `lam` is
    a traced scalar, not a static jit argument, so a schedule that changes it every iteration
    does not retrigger compilation.
    """
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_emit_v, log_emit_n, mask
    )  # (chunk,T,K) -- recipe-independent
    tilt_tick = _tilt_tick(log_tilt, noun_ids, mask)  # (chunk,K_R,T)

    def _combine_recipe(loglik_i, tilt_tick_ir, mask_i, log_init_r, log_trans_r,
                         log_dur_pmf_r, log_dur_survival_r, log_tilt_norm_r):
        ll = jnp.where(mask_i[:, None], loglik_i + tilt_tick_ir[:, None] - log_tilt_norm_r[None, :], 0.0)
        return messages.combine_sufficient_stats(
            ll, mask_i, log_init_r, log_trans_r, log_dur_pmf_r, log_dur_survival_r, d_max
        )

    combine_over_r = jax.vmap(
        _combine_recipe, in_axes=(None, 0, None, 0, 0, 0, 0, 0)
    )
    combine_over_trials = jax.vmap(
        combine_over_r, in_axes=(0, 0, 0, None, None, None, None, None)
    )
    xi_trans, xi_dur, cens, gamma, log_z = combine_over_trials(
        loglik, tilt_tick, mask, log_init, log_trans, log_dur_pmf, log_dur_survival, log_tilt_norm
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
    # einsum, so the accumulation is shape-identical to em.py's single-recipe version.
    gamma_bar = jnp.einsum("nr,nrtk->ntk", rho, gamma_masked)  # (chunk,T,K)
    n_verb, n_noun = log_emit_v.shape[1], log_emit_n.shape[1]
    verb_onehot = jax.nn.one_hot(verb_ids, n_verb, dtype=gamma.dtype)
    noun_onehot = jax.nn.one_hot(noun_ids, n_noun, dtype=gamma.dtype)
    noun_onehot_masked = jnp.where(mask[:, :, None], noun_onehot, 0.0)  # (chunk,T,N)

    stats = {
        "pi_counts": jnp.sum(rho, axis=0),                                              # (K_R,)
        "init_counts": jnp.sum(rho[:, :, None] * gamma_masked[:, :, 0, :], axis=0),      # (K_R,K)
        "trans_counts": jnp.sum(rho[:, :, None, None] * xi_trans, axis=0),               # (K_R,K,K)
        "verb_counts": jnp.einsum("ntk,ntv->kv", gamma_bar, verb_onehot),                # (K,V) shared
        "noun_counts": jnp.einsum("ntk,ntv->kv", gamma_bar, noun_onehot),                # (K,N) shared
        "xi_dur": jnp.sum(rho[:, :, None, None] * xi_dur, axis=0),                       # (K_R,K,D)
        "cens": jnp.sum(rho[:, :, None, None] * cens, axis=0),                           # (K_R,K,D)
        # rho-weighted per-recipe noun bag and state occupancy -- the tilt's own sufficient
        # stats, consumed by m_step's GIS update (a no-op when tilt_steps=0).
        "tilt_noun_counts": jnp.einsum("nr,ntv->rv", rho, noun_onehot_masked),           # (K_R,N)
        "occ": jnp.einsum("nr,nrtk->rk", rho, gamma_masked),                             # (K_R,K)
    }
    return stats, jnp.sum(trial_ll)


def e_step(joint_hsmm_params, verb_ids, noun_ids, mask, d_max, temperature=1.0, chunk_size=8,
           recipe_log_prior=None, lam=0.0):
    """Chunked driver mirroring hsmm.em.e_step. Peak memory scales an extra K_R-fold over the
    single-recipe E-step (the (chunk,K_R,T,K) gamma intermediate), so callers should shrink
    chunk_size roughly by K_R relative to the cascade EM's setting.

    `recipe_log_prior`: (N_trials,K_R) external per-trial recipe log-prior, or None (treated as
    all-zero -- a no-op regardless of `lam`). `lam`: the anchoring dial's strength, added to
    log_rho as `lam * recipe_log_prior[trial]` inside `_e_step_chunk`.

    THE TRAP: this function reorders trials by length before chunking (`_length_order`, below)
    so that chunks are length-homogeneous -- `verb_ids`/`noun_ids`/`mask` are all sliced via
    `[idx]`, never `[start:end]` directly. `recipe_log_prior` is a PER-TRIAL array in the same
    original trial order those three start in, so it must be sliced with that exact same `idx`
    or the prior silently applies to the wrong trial once chunking reorders. Contrast
    `infer_recipe` below, which does NOT reorder and slices `[start:end]` directly -- see its
    docstring for why that asymmetry is deliberate, not a second bug waiting to happen.
    """
    log_probs = joint_params.to_log_probs_joint(joint_hsmm_params, d_max)
    inv_temp = 1.0 / temperature
    log_emit_v = log_probs.log_emit_v * inv_temp
    log_emit_n = log_probs.log_emit_n * inv_temp
    log_dur_pmf = log_probs.log_dur_pmf * inv_temp
    log_dur_survival = log_probs.log_dur_survival * inv_temp
    # Tempered as a unit with the emission it modulates -- an approximation when annealing is
    # on (log_tilt_norm was computed in to_log_probs_joint against the UNtempered log_emit_n),
    # but every config in the repo has annealing off, and no test exercises tilt+annealing.
    log_tilt = log_probs.log_tilt * inv_temp
    log_tilt_norm = log_probs.log_tilt_norm * inv_temp

    n = verb_ids.shape[0]
    k_r = log_probs.log_pi.shape[0]
    if recipe_log_prior is None:
        recipe_log_prior = jnp.zeros((n, k_r))
    lam = jnp.asarray(lam, dtype=log_probs.log_pi.dtype)

    total_stats = None
    total_trial_ll = 0.0
    t_max = verb_ids.shape[1]
    order = _length_order(mask)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        idx = order[start:end]  # THE TRAP above: recipe_log_prior must use this same idx
        t_used = _chunk_ticks(mask[idx], t_max)
        chunk_stats, chunk_ll = _e_step_chunk(
            log_probs.log_pi, log_probs.log_init, log_probs.log_trans,
            log_emit_v, log_emit_n, log_dur_pmf, log_dur_survival,
            log_tilt, log_tilt_norm, recipe_log_prior[idx], lam,
            verb_ids[idx, :t_used], noun_ids[idx, :t_used], mask[idx, :t_used], d_max,
        )
        if total_stats is None:
            total_stats = chunk_stats
        else:
            total_stats = {name: total_stats[name] + chunk_stats[name] for name in total_stats}
        total_trial_ll = total_trial_ll + chunk_ll

    return total_stats, total_trial_ll


@functools.partial(jax.jit, static_argnames=("d_max", "global_damping", "tilt_steps"))
def m_step(joint_hsmm_params, stats, alpha_pi, alpha_init, alpha_trans, alpha_emit_v, alpha_emit_n, kappa, d_max,
           prev_global_r=None, prev_global_p=None, global_damping=0.0,
           emit_prior_v=None, emit_prior_n=None,
           tilt_steps=0, tilt_step_size=1.0, tilt_alpha=1.0, tilt_max=5.0):
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

    `tilt_steps`/`tilt_step_size`/`tilt_alpha`/`tilt_max`: the noun_tilt GIS update (see
    joint_params.tilt_terms and this module's `_e_step_chunk` docstring). `tilt_steps=0` (the
    default) or `joint_hsmm_params.noun_tilt is None` leaves noun_tilt untouched -- the default
    path is exactly today's M-step. When active, each of `tilt_steps` generalized-iterative-
    scaling steps nudges `a_r` so the model's predicted per-recipe noun marginal matches the
    rho-weighted noun bag (`stats["tilt_noun_counts"]`), using the SAME `new_noun_counts` this
    call just produced -- so the shared-emission update and the tilt update are a coordinate
    step on the two count families, not a joint solve. That is why this M-step is only
    approximately monotone (like the shrinkage duration fit): holding `a` fixed, the shared
    noun_counts closed form ignores the log_tilt_norm[r,k] coupling the tilt introduces.
    `tilt_alpha`/N is a Dirichlet-style pseudocount stopping a zero-count noun from driving `a`
    to -inf; `tilt_max` bounds the tilt magnitude for the same reason; centering `a` around its
    own mean each step removes the shift non-identifiability (log_tilt_norm absorbs any
    constant shift, so only the SHAPE of `a` is identified).
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

    new_noun_tilt = joint_hsmm_params.noun_tilt
    if tilt_steps > 0 and new_noun_tilt is not None:
        log_emit_n_cur = joint_params.params.compose_kernel(
            joint_params.params._row_normalize(new_noun_counts), joint_hsmm_params.kernel_n
        )
        occ = stats["occ"]  # (K_R,K)
        log_occ = jnp.log(jnp.maximum(
            occ / jnp.maximum(jnp.sum(occ, axis=-1, keepdims=True), joint_params.FLOOR),
            joint_params.FLOOR,
        ))[:, :, None]  # (K_R,K,1)
        smoothed = stats["tilt_noun_counts"] + tilt_alpha / n_noun  # (K_R,N)
        log_c = jnp.log(smoothed) - jnp.log(jnp.sum(smoothed, axis=-1, keepdims=True))

        a = new_noun_tilt
        for _ in range(tilt_steps):
            log_p_r = log_emit_n_cur[None, :, :] + a[:, None, :]                   # (K_R,K,N)
            log_p_r = log_p_r - logsumexp(log_p_r, axis=-1, keepdims=True)         # P_r(n|k)
            log_q = logsumexp(log_occ + log_p_r, axis=1)                           # (K_R,N)
            a = a + tilt_step_size * (log_c - log_q)
            a = a - jnp.mean(a, axis=-1, keepdims=True)
            a = jnp.clip(a, -tilt_max, tilt_max)
        new_noun_tilt = a

    new_params = JointHSMMParams(
        new_init_counts, new_trans_counts, new_verb_counts, new_noun_counts, dur_r, dur_p,
        new_pi_counts, joint_hsmm_params.kernel_v, joint_hsmm_params.kernel_n, new_noun_tilt,
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
    tilt_steps=0,
    tilt_step_size=1.0,
    tilt_alpha=1.0,
    tilt_max=5.0,
    recipe_log_prior=None,
    lam=0.0,
    lam_schedule=None,
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

    `tilt_steps`/`tilt_step_size`/`tilt_alpha`/`tilt_max`: passed through to `m_step` on every
    iteration -- see its docstring for the GIS update. `tilt_steps=0` (the default) makes this
    call identical to before noun_tilt existed, whether or not `init_params.noun_tilt` is set.

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
            tilt_steps=tilt_steps, tilt_step_size=tilt_step_size, tilt_alpha=tilt_alpha, tilt_max=tilt_max,
        )
        if on_checkpoint is not None and ((iteration + 1) % checkpoint_every == 0 or converged):
            on_checkpoint(iteration + 1, p, history)
        if converged:
            break
    iter_bar.close()

    return p, obj, history, converged


@functools.partial(jax.jit, static_argnames=("d_max",))
def _recipe_logz_chunk(log_pi, log_init, log_trans, log_dur_pmf, log_dur_survival,
                        log_emit_v, log_emit_n, log_tilt, log_tilt_norm, log_prior, lam,
                        verb_ids, noun_ids, mask, d_max):
    """Decode-only per-chunk kernel: recipe responsibilities need only each (trial,recipe)'s
    total log-likelihood, not the full sufficient-stats bundle -- so this uses
    messages.forward_pass directly (an existing function, not a reimplementation) rather than
    the heavier combine_sufficient_stats, skipping the backward pass and the xi/gamma scans
    entirely. log_z here is bit-identical to combine_sufficient_stats's log_z (both are
    jnp.sum(forward_pass(...)[0])), so this cannot diverge numerically from training's E-step --
    including the tilt term, which must be applied identically here or r_hat stops matching the
    rho the E-step actually converged to (see _e_step_chunk's docstring for the same term).
    """
    loglik = jax.vmap(emissions.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb_ids, noun_ids, log_emit_v, log_emit_n, mask
    )
    tilt_tick = _tilt_tick(log_tilt, noun_ids, mask)  # (chunk,K_R,T)

    def forward_logz(loglik_i, tilt_tick_ir, mask_i, log_init_r, log_trans_r,
                      log_dur_pmf_r, log_dur_survival_r, log_tilt_norm_r):
        ll = jnp.where(mask_i[:, None], loglik_i + tilt_tick_ir[:, None] - log_tilt_norm_r[None, :], 0.0)
        log_norm, _ = messages.forward_pass(ll, log_init_r, log_trans_r, log_dur_pmf_r, log_dur_survival_r, mask_i, d_max)
        return jnp.sum(log_norm)

    over_r = jax.vmap(forward_logz, in_axes=(None, 0, None, 0, 0, 0, 0, 0))
    over_trials = jax.vmap(over_r, in_axes=(0, 0, 0, None, None, None, None, None))
    log_z = over_trials(
        loglik, tilt_tick, mask, log_init, log_trans, log_dur_pmf, log_dur_survival, log_tilt_norm
    )  # (chunk,K_R)

    log_rho = log_pi[None, :] + log_z + lam * log_prior
    trial_ll = logsumexp(log_rho, axis=-1)
    rho = jnp.exp(log_rho - trial_ll[:, None])
    r_hat = jnp.argmax(log_rho, axis=-1)
    return r_hat, rho, trial_ll


def infer_recipe(joint_hsmm_params, verb_ids, noun_ids, mask, d_max, chunk_size=8,
                  recipe_log_prior=None, lam=0.0):
    """r_hat_i = argmax_r(log_pi[r] + logz_ir + lam*log_prior_ir); also returns the full
    posterior rho and the per-trial marginal log-likelihood. Chunked the same way as e_step --
    but, UNLIKE e_step, this does NOT reorder trials by length first (there is no length-
    homogeneous-chunking win here: forward_pass alone, not the heavier combine_sufficient_stats,
    is cheap enough that the reorder isn't worth the bookkeeping). So `recipe_log_prior` is
    sliced plain `[start:end]`, matching `verb_ids`/`noun_ids`/`mask` -- using `_length_order`
    here would be the bug, not the fix.
    """
    log_probs = joint_params.to_log_probs_joint(joint_hsmm_params, d_max)

    n = verb_ids.shape[0]
    k_r = log_probs.log_pi.shape[0]
    if recipe_log_prior is None:
        recipe_log_prior = jnp.zeros((n, k_r))
    lam = jnp.asarray(lam, dtype=log_probs.log_pi.dtype)

    r_hat_chunks, rho_chunks, trial_ll_chunks = [], [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        r_hat, rho, trial_ll = _recipe_logz_chunk(
            log_probs.log_pi, log_probs.log_init, log_probs.log_trans,
            log_probs.log_dur_pmf, log_probs.log_dur_survival,
            log_probs.log_emit_v, log_probs.log_emit_n,
            log_probs.log_tilt, log_probs.log_tilt_norm,
            recipe_log_prior[start:end], lam,
            verb_ids[start:end], noun_ids[start:end], mask[start:end], d_max,
        )
        r_hat_chunks.append(r_hat)
        rho_chunks.append(rho)
        trial_ll_chunks.append(trial_ll)

    return jnp.concatenate(r_hat_chunks), jnp.concatenate(rho_chunks), jnp.concatenate(trial_ll_chunks)
