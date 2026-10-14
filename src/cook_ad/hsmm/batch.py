"""Whole-trial inference for many trials at once: HSMM.infer_recipe / viterbi and the one-recipe
predictive occupancy, vmapped over a padded batch so each op compiles once. Performance
plumbing for HindsightHSMMDetector.detect_many; the maths is hsmm/kernels.py.

Trials are padded to ONE global length (the longest in the list) and processed in chunks of
`chunk_size`, exactly as the eval has always done, so results are bitwise those of the
pre-restructure eval.batch.compute_traces_joint.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cook_ad.hsmm import kernels


def pad(verbs_list, nouns_list, t_max=None):
    """-> (verbs, nouns, mask), each (N, t_max), zero-padded."""
    t_max = t_max or max(len(v) for v in verbs_list)
    n = len(verbs_list)
    verb = np.zeros((n, t_max), dtype=np.int32)
    noun = np.zeros((n, t_max), dtype=np.int32)
    mask = np.zeros((n, t_max), dtype=bool)
    for i, (v, w) in enumerate(zip(verbs_list, nouns_list)):
        verb[i, : len(v)], noun[i, : len(v)], mask[i, : len(v)] = v, w, True
    return jnp.asarray(verb), jnp.asarray(noun), jnp.asarray(mask)


def _loglik(verb, noun, log_emit_v, log_emit_n, mask):
    return jax.vmap(kernels.sequence_loglik, in_axes=(0, 0, None, None, 0))(
        verb, noun, log_emit_v, log_emit_n, mask)


@functools.partial(jax.jit, static_argnames=("d_max",))
def _recipe_posterior_chunk(log_pi, log_init, log_A, log_pmf, log_surv, log_emit_v, log_emit_n,
                            verb, noun, mask, d_max):
    loglik = _loglik(verb, noun, log_emit_v, log_emit_n, mask)

    def log_z(e, m, li, la, lp, ls):
        return jnp.sum(kernels.forward_pass(e, li, la, lp, ls, m, d_max)[0])

    over_r = jax.vmap(log_z, in_axes=(None, None, 0, 0, 0, 0))
    log_rho = log_pi[None, :] + jax.vmap(over_r, in_axes=(0, 0, None, None, None, None))(
        loglik, mask, log_init, log_A, log_pmf, log_surv)
    log_p = logsumexp(log_rho, axis=-1)
    return jnp.argmax(log_rho, axis=-1), jnp.exp(log_rho - log_p[:, None]), log_p


def infer_recipe(tables, verb, noun, mask, d_max, chunk_size=8):
    """-> (r_hat (N,), P(r | o) (N, K_R), log P(o) (N,)), argmax_r P(r) P(o | r) per trial."""
    t = tables
    out = [_recipe_posterior_chunk(t.log_pi, t.log_init, t.log_A, t.log_pmf, t.log_surv,
                                   t.log_emit_v, t.log_emit_n, verb[s : s + chunk_size],
                                   noun[s : s + chunk_size], mask[s : s + chunk_size], d_max)
           for s in range(0, verb.shape[0], chunk_size)]
    return tuple(jnp.concatenate(x) for x in zip(*out))


@functools.partial(jax.jit, static_argnames=("d_max",))
def occupancy(verb, noun, mask, log_init, log_A, log_emit_v, log_emit_n, log_pmf, log_surv,
              d_max):
    """(N, T, K) log pi_tilde under each trial's own recipe tables (axis 0 of the dynamics)."""
    loglik = _loglik(verb, noun, log_emit_v, log_emit_n, mask)
    return jax.vmap(kernels.predictive_occupancy, in_axes=(0, 0, 0, 0, 0, 0, None))(
        loglik, log_init, log_A, log_pmf, log_surv, mask, d_max)


@functools.partial(jax.jit, static_argnames=("d_max",))
def _viterbi(log_init, log_A, log_pmf, log_surv, loglik, mask, d_max):
    return jax.vmap(kernels.viterbi_decode, in_axes=(0, 0, 0, 0, 0, 0, None))(
        loglik, log_init, log_A, log_pmf, log_surv, mask, d_max)


def viterbi(tables, r_hat, verb, noun, mask, d_max):
    """MAP segments [(subtask, duration)] per trial, each under its own recipe r_hat[i]."""
    t = tables
    loglik = _loglik(verb, noun, t.log_emit_v, t.log_emit_n, mask)
    _, dur_bp, asurv, dur_bp_surv, prev_bp = _viterbi(
        t.log_init[r_hat], t.log_A[r_hat], t.log_pmf[r_hat], t.log_surv[r_hat], loglik, mask, d_max)
    dur_bp, asurv, dur_bp_surv, prev_bp, mask = (
        np.asarray(x) for x in (dur_bp, asurv, dur_bp_surv, prev_bp, mask))
    out = []
    for i in range(mask.shape[0]):
        n = int(mask[i].sum())
        k_star = int(np.argmax(asurv[i, n - 1, :]))
        out.append(kernels.traceback(n, k_star, dur_bp[i], dur_bp_surv[i], prev_bp[i]))
    return out
