"""Viterbi (hard-assignment) EM for the joint model -- an alternative optimiser to joint_em's
soft EM, maximising over the MAP segmentation rather than marginalising it out:

    max_theta max_z log P_theta(o, z)   in place of   max_theta log sum_z P_theta(o, z)

The wrong estimator for calibrated posterior uncertainty, the right one here because every
hindsight channel scores against z_star -- so the counts the model is fit from and the decode
the detector reads become the same object.

One iteration: recipe-conditioned Viterbi decode -> hard init/transition counts and
per-(recipe, state) duration histograms -> the same Dirichlet-MAP renormalisation and shrunk NB
duration fit the soft M-step uses. Each trial's final segment is right-censored and imputed as
in joint_em. Emissions stay fixed. docs/training.md.
"""
import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm import batch
from cook_ad.hsmm.params import JointHSMMParams, to_log_tables
from cook_ad.hsmm.train import durations_fit


def hard_counts(segments, r_hat, k_recipe, k_subtask, d_max):
    """Per-recipe init/transition counts and per-(recipe,state) duration histograms from decoded
    segmentations [(state, duration)]. Returns (init, trans, xi_dur, cens) -- `cens` holding
    only each trial's final, still-open segment."""
    init = np.zeros((k_recipe, k_subtask))
    trans = np.zeros((k_recipe, k_subtask, k_subtask))
    xi = np.zeros((k_recipe, k_subtask, d_max))
    cens = np.zeros((k_recipe, k_subtask, d_max))
    for segs, r in zip(segments, r_hat):
        if not segs:
            continue
        r = int(r)
        init[r, segs[0][0]] += 1.0
        for (u, _), (v, _) in zip(segs[:-1], segs[1:]):
            if u != v:
                trans[r, u, v] += 1.0
        for k, d in segs[:-1]:
            xi[r, k, min(int(d), d_max) - 1] += 1.0
        k, d = segs[-1]
        cens[r, k, min(int(d), d_max) - 1] += 1.0
    return init, trans, xi, cens


def hard_em_step(p, verb_ids, noun_ids, mask, d_max, kappa, alpha_init, alpha_trans, alpha_pi,
                 chunk_size=32, r_given=None):
    """One hard-EM iteration. `r_given`: decode each trial under this recipe (the dish label)
    instead of the inferred one. Returns (new params, inferred r_hat, segments, log P(o) per trial)."""
    k_recipe, k_subtask = p.init_counts.shape
    tables = to_log_tables(p, d_max)
    r_hat, _rho, trial_ll = batch.infer_recipe(tables, verb_ids, noun_ids, mask, d_max, chunk_size)
    inferred = np.asarray(r_hat)
    if r_given is not None:
        r_hat = jnp.asarray(r_given)
    segments = batch.viterbi(tables, r_hat, verb_ids, noun_ids, mask, d_max)

    init_c, trans_c, xi, cens = hard_counts(segments, np.asarray(r_hat), k_recipe, k_subtask, d_max)
    pi_c = np.bincount(np.asarray(r_hat), minlength=k_recipe).astype(float)
    dur_r, dur_p, _gr, _gp = durations_fit.fit_durations_shrunk(
        jnp.asarray(xi), jnp.asarray(cens), p.dur_r, p.dur_p, d_max, kappa
    )
    new = JointHSMMParams(
        init_counts=jnp.asarray(init_c + alpha_init / k_subtask),
        trans_counts=jnp.asarray(trans_c + alpha_trans / k_subtask)
        * (1.0 - jnp.eye(k_subtask))[None, :, :],
        verb_counts=p.verb_counts,
        noun_counts=p.noun_counts,
        dur_r=dur_r,
        dur_p=dur_p,
        pi_counts=jnp.asarray(pi_c + alpha_pi / k_recipe),
    )
    return new, inferred, segments, trial_ll
