"""Refit only the duration parameters of a fitted joint model from its own Viterbi decode -- a
hard-assignment duration M-step, sweepable in `kappa` without a full EM run. docs/training.md.

The last segment of each trial is right-censored and imputed, as the real M-step does; dropping
it would bias every duration short.
"""
import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm.train import durations_fit


def duration_histograms(trajectories, k_recipe, k_subtask, d_max):
    """(xi_dur, cens), each (K_R, K, D): closed segments, and each trial's open last segment."""
    xi = np.zeros((k_recipe, k_subtask, d_max))
    cens = np.zeros((k_recipe, k_subtask, d_max))
    for t in trajectories:
        r, segs = t["recipe_id"], t["segments"]
        for k, d in segs[:-1]:
            xi[r, k, min(int(d), d_max) - 1] += 1.0
        if segs:
            k, d = segs[-1]
            cens[r, k, min(int(d), d_max) - 1] += 1.0
    return xi, cens


def refit_durations(jp, trajectories, d_max, kappa):
    """jp with (dur_r, dur_p) refitted from the decoded `trajectories`."""
    k_recipe, k_subtask = jp.init_counts.shape
    xi, cens = duration_histograms(trajectories, k_recipe, k_subtask, d_max)
    dur_r, dur_p, _g_r, _g_p = durations_fit.fit_durations_shrunk(
        jnp.asarray(xi), jnp.asarray(cens), jp.dur_r, jp.dur_p, d_max, kappa
    )
    return jp._replace(dur_r=dur_r, dur_p=dur_p), xi
