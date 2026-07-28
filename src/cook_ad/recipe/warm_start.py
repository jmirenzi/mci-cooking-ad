import functools

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm import durations, em
from cook_ad.hsmm.joint_params import JointHSMMParams
from cook_ad.recipe import recipe_hmm, segmentize

# A recipe with fewer trials than this cannot support a stable hard init/trans histogram (a
# single trial's segment sequence is not a distribution), so it falls back to the cascade's
# recipe-agnostic tables instead of a near-empty, noise-dominated one of its own.
MIN_TRIALS_PER_RECIPE = 2
FALLBACK_NOISE_SCALE = 0.05


def _hard_duration_histogram(segments_by_trial, k_subtask, d_max):
    """segments_by_trial: list of [(state,duration),...] (one list per trial). Returns
    (K,D_max) hard counts; durations >= d_max are clamped into the last bin, mirroring the
    survival-absorbing convention duration_tables uses for the model's own pmf table."""
    hist = np.zeros((k_subtask, d_max))
    for segments in segments_by_trial:
        for state, d in segments:
            idx = min(int(d), d_max) - 1
            hist[int(state), idx] += 1.0
    return hist


def _init_trans_hard_counts(segments_by_trial, k_subtask):
    """Hard init/trans histograms from Viterbi segment sequences. Zero-diagonal by
    construction: consecutive segments always differ in state, since the underlying HSMM's
    trans matrix already has the diagonal banned (-inf) at segmentation time."""
    init_counts = np.zeros(k_subtask)
    trans_counts = np.zeros((k_subtask, k_subtask))
    for segments in segments_by_trial:
        if not segments:
            continue
        init_counts[int(segments[0][0])] += 1.0
        for (s0, _), (s1, _) in zip(segments[:-1], segments[1:]):
            trans_counts[int(s0), int(s1)] += 1.0
    return init_counts, trans_counts


def cascade_to_joint(hsmm_params, recipe_params, sequences, d_max, k_recipe, kappa, seed=0):
    """Build the joint model's iteration-0 JointHSMMParams from the fitted cascade artifacts
    (a recipe-agnostic HSMM + the flat stage-2 recipe HMM), per spec section 4.

    Recovers per-trial cascade decisions (Viterbi segments, decoded recipe id), copies shared
    emissions verbatim, and builds hard per-recipe init/trans/duration histograms so the K_R
    recipes start genuinely differentiated -- required for EM to break its own
    recipe-permutation symmetry (see joint_em.run_joint_em's docstring). A recipe with fewer
    than MIN_TRIALS_PER_RECIPE assigned trials gets the cascade's own (recipe-agnostic)
    init/trans as a fallback, with small noise added for symmetry-breaking, rather than a
    near-empty histogram of its own -- applied per recipe, not as an all-or-nothing switch for
    the whole warm start.

    Durations reuse the cascade's own fitted (dur_r,dur_p) directly as the global per-state
    shape for shrinkage (no separate global refit needed -- that IS the cascade's global fit),
    and skip censoring imputation (Viterbi segments carry no censoring info; the trial's final
    segment is treated as exactly observed, a one-time optimism the first real joint M-step
    corrects).
    """
    k_subtask = hsmm_params.verb_counts.shape[0]
    verb_ids, noun_ids, mask = em.pad_batch(sequences)

    seg_results = segmentize.segment_all(hsmm_params, verb_ids, noun_ids, mask, d_max)
    segments_by_trial = [r["segments"] for r in seg_results]
    seg_sequences = [[state for state, _ in segs] for segs in segments_by_trial]

    obs_ids, seg_mask = recipe_hmm.pad_segment_batch(seg_sequences)
    r_i = np.asarray(recipe_hmm.decode_recipe(recipe_params, obs_ids, seg_mask))

    verb_counts = hsmm_params.verb_counts
    noun_counts = hsmm_params.noun_counts

    pi_counts = np.zeros(k_recipe)
    for r in range(k_recipe):
        pi_counts[r] = float(np.sum(r_i == r))

    fallback_init = np.asarray(hsmm_params.init_counts)
    fallback_trans = np.asarray(hsmm_params.trans_counts)

    init_counts = np.zeros((k_recipe, k_subtask))
    trans_counts = np.zeros((k_recipe, k_subtask, k_subtask))
    dur_hist = np.zeros((k_recipe, k_subtask, d_max))

    rng = np.random.default_rng(seed)
    for r in range(k_recipe):
        trial_idx = np.where(r_i == r)[0]
        if len(trial_idx) < MIN_TRIALS_PER_RECIPE:
            init_counts[r] = fallback_init + rng.normal(scale=FALLBACK_NOISE_SCALE, size=k_subtask)
            trans_counts[r] = fallback_trans
            # No segments assigned -- dur_hist[r] stays all-zero, so the shrinkage step below
            # falls back entirely to the global per-state shape for this recipe.
            continue

        trial_segments = [segments_by_trial[i] for i in trial_idx]
        init_r, trans_r = _init_trans_hard_counts(trial_segments, k_subtask)
        init_counts[r] = init_r
        trans_counts[r] = trans_r
        dur_hist[r] = _hard_duration_histogram(trial_segments, k_subtask, d_max)

    init_counts = jnp.asarray(init_counts)
    trans_counts = jnp.asarray(trans_counts) * (1.0 - jnp.eye(k_subtask))[None, :, :]
    dur_hist = jnp.asarray(dur_hist)

    log_pmf_global, _ = durations.duration_tables(hsmm_params.dur_r, hsmm_params.dur_p, d_max)
    pmf_global = jnp.exp(log_pmf_global)  # (K,D): the cascade's own fitted global per-state shape
    dur_r_old = jnp.tile(hsmm_params.dur_r[None, :], (k_recipe, 1))

    n_hat_shrunk = dur_hist + kappa * pmf_global[None, :, :]
    stats_over_r = jax.vmap(durations.duration_stats_from_histogram, in_axes=(0, None))
    n_tot, s_tot = stats_over_r(n_hat_shrunk, d_max)
    newton_over_r = jax.vmap(functools.partial(durations.newton_update_r, n_iters=5), in_axes=(0, 0, 0, 0))
    dur_r = newton_over_r(n_hat_shrunk, n_tot, s_tot, dur_r_old)
    p_over_r = jax.vmap(durations.update_p_given_r, in_axes=(0, 0, 0))
    dur_p = p_over_r(n_tot, s_tot, dur_r)

    return JointHSMMParams(
        init_counts=init_counts,
        trans_counts=trans_counts,
        verb_counts=verb_counts,
        noun_counts=noun_counts,
        dur_r=dur_r,
        dur_p=dur_p,
        pi_counts=jnp.asarray(pi_counts),
    )
