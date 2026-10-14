"""class HSMM: the joint (recipe, subtask, duration) model at inference (docs/model.md)."""
import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cook_ad.hsmm import kernels
from cook_ad.hsmm.params import JointHSMMParams, load, to_log_tables

BUCKET = 64  # trials are padded to a multiple of this many ticks, so jit compiles once per bucket


class View(NamedTuple):
    """The tables one trial runs on: every recipe, one pinned recipe, or a reweighted prior."""
    log_b0: jnp.ndarray     # (K_R', K) log P(r) + log P(z_0 = i | r)
    log_A: jnp.ndarray      # (K_R', K, K)
    log_h: jnp.ndarray      # (K_R', K, D) hazard
    log_stay: jnp.ndarray   # (K_R', K, D) log P(D >= j+2 | D >= j+1)
    recipes: slice          # which recipes of the full model this view keeps


class Forecast(NamedTuple):
    """The predictive table for tick t, before o_t is seen (docs/model.md §3.1)."""
    b_pred: jnp.ndarray     # (K_R', K, D)
    log_end: jnp.ndarray    # (K_R', K, D) subtask ended at t-1 after j+1 ticks
    log_m: jnp.ndarray      # (K_R', K)

    @property
    def pi_tilde(self):
        return kernels.occupancy(self.b_pred)[0]

    @property
    def stay(self):
        return kernels.occupancy(self.b_pred)[1]

    @property
    def switch(self):
        return kernels.occupancy(self.b_pred)[2]


class FilterResult(NamedTuple):
    """Whole-trial filter output, padded to the bucket; `n` is the true length."""
    log_b: jnp.ndarray       # (T', K_R', K, D)
    log_norm: jnp.ndarray    # (T',) log P(o_t | o_<t)
    pi_tilde: jnp.ndarray    # (T', K)
    log_stay: jnp.ndarray    # (T', K) occupancy with the subtask continuing
    log_switch: jnp.ndarray  # (T', K) occupancy with a subtask starting at t
    loglik: jnp.ndarray      # (T', K) log P(o_t | i)
    mask: jnp.ndarray        # (T',)
    n: int
    view: View

    @property
    def occupancy(self):
        return np.asarray(self.pi_tilde)[: self.n]

    @property
    def total_loglik(self):
        return float(np.sum(np.asarray(self.log_norm)[: self.n]))


_predict = jax.jit(kernels.predict)
_update = jax.jit(kernels.update)


class HSMM:
    """Parameters, their log tables, and the current belief b_t(r, i, j)."""

    def __init__(self, params: JointHSMMParams, d_max: int):
        self.params, self.d_max = params, d_max
        self.tables = to_log_tables(params, d_max)
        self.k_recipe = int(self.tables.log_pi.shape[0])
        self.reset()

    @classmethod
    def load(cls, path, d_max) -> "HSMM":
        return cls(load(path), d_max)

    # ---- the per-trial view

    def view(self, recipe=None) -> View:
        """None: fitted P(r). int: that recipe, with probability 1. Array: a (K_R,) log prior."""
        t = self.tables
        if recipe is None:
            log_b0 = jnp.asarray(np.asarray(t.log_pi)[:, None] + np.asarray(t.log_init))
            return View(log_b0, t.log_A, t.log_h, t.log_stay, slice(None))
        if np.ndim(recipe) == 0:
            r = slice(int(recipe), int(recipe) + 1)
            return View(t.log_init[r], t.log_A[r], t.log_h[r], t.log_stay[r], r)
        log_prior = jnp.asarray(recipe, dtype=t.log_init.dtype)
        log_prior = log_prior - logsumexp(log_prior)
        return View(log_prior[:, None] + t.log_init, t.log_A, t.log_h, t.log_stay, slice(None))

    # ---- one tick at a time

    def reset(self, recipe=None):
        self.current = self.view(recipe)
        self._b = None
        self._forecast = None

    def predict(self) -> Forecast:
        """b_{t-1} -> the forecast for tick t (docs/model.md §3.1)."""
        v = self.current
        if self._b is None:
            b0 = kernels.initial_belief(v.log_b0, self.d_max)
            none = jnp.full_like(b0, -jnp.inf)
            self._forecast = Forecast(b0, none, none[..., 0])
        else:
            self._forecast = Forecast(*_predict(self._b, v.log_A, v.log_h, v.log_stay))
        return self._forecast

    def update(self, verb, noun) -> float:
        """Condition on o_t; returns log P(o_t | o_<t) (docs/model.md §3.2)."""
        forecast = self._forecast or self.predict()
        e = self.tables.log_emit_v[:, verb] + self.tables.log_emit_n[:, noun]
        self._b, log_norm = _update(forecast.b_pred, e)
        self._forecast = None
        return float(log_norm)

    @property
    def belief(self):
        """b_t (K_R', K, D) in log space; None before the first update."""
        return self._b

    @property
    def recipe_belief(self):
        """log P(r | o_0..o_t), (K_R',)."""
        return logsumexp(self._b, axis=(1, 2))

    @property
    def occupancy(self):
        """log pi_tilde_t(i) = log P(z_t = i | o_<t) for the tick last predicted."""
        return (self._forecast or self.predict()).pi_tilde

    # ---- whole trial

    def pad(self, verbs, nouns):
        """-> (verbs, nouns, mask) padded to a multiple of BUCKET ticks."""
        n = len(verbs)
        t_max = -(-n // BUCKET) * BUCKET
        v, w, m = (np.zeros(t_max, np.int32) for _ in range(3))
        v[:n], w[:n], m[:n] = verbs, nouns, 1
        return jnp.asarray(v), jnp.asarray(w), jnp.asarray(m.astype(bool))

    def loglik(self, verbs, nouns, mask):
        """(T, K) log P(o_t | i)."""
        return kernels.sequence_loglik(verbs, nouns, self.tables.log_emit_v,
                                       self.tables.log_emit_n, mask)

    def filter(self, verbs, nouns, recipe=None) -> FilterResult:
        """The same predict + update kernels, scanned over the trial (docs/model.md §3)."""
        v = self.view(recipe)
        verb, noun, mask = self.pad(verbs, nouns)
        e = self.loglik(verb, noun, mask)
        out = kernels.filter_scan(v.log_b0, v.log_A, v.log_h, v.log_stay, e, mask)
        return FilterResult(*out, loglik=e, mask=mask, n=len(verbs), view=v)

    def smooth(self, fr: FilterResult, L):
        """Backward messages beta_t over o_{t+1..t+L} (L = math.inf: to the end), docs/model.md §5."""
        v = fr.view
        if L == math.inf:
            return kernels.full_beta(v.log_A, v.log_h, v.log_stay, fr.loglik, fr.mask)
        return kernels.fixed_lag_beta(v.log_A, v.log_h, v.log_stay, fr.loglik, fr.n, int(L))

    def infer_recipe(self, verbs, nouns):
        """r_hat = argmax_r P(r) P(o | r) -> (r_hat, log P(r | o), log P(o)), docs/model.md §6."""
        verb, noun, mask = self.pad(verbs, nouns)
        t = self.tables
        e = self.loglik(verb, noun, mask)
        log_z = _recipe_log_z(e, t.log_init, t.log_A, t.log_pmf, t.log_surv, mask, self.d_max)
        log_rho = t.log_pi + log_z
        log_p = logsumexp(log_rho)
        return int(jnp.argmax(log_rho)), np.asarray(log_rho - log_p), float(log_p)

    def viterbi(self, verbs, nouns, recipe: int):
        """MAP segments [(subtask, duration)] under one recipe, docs/model.md §7."""
        verb, noun, mask = self.pad(verbs, nouns)
        t, r = self.tables, int(recipe)
        e = self.loglik(verb, noun, mask)
        _, dur_bp, asurv, dur_bp_surv, prev_bp = _viterbi(
            e, t.log_init[r], t.log_A[r], t.log_pmf[r], t.log_surv[r], mask, self.d_max)
        n = len(verbs)
        k_star = int(np.argmax(np.asarray(asurv)[n - 1]))
        return kernels.traceback(n, k_star, np.asarray(dur_bp), np.asarray(dur_bp_surv),
                                 np.asarray(prev_bp))


@functools.partial(jax.jit, static_argnames=("d_max",))
def _recipe_log_z(e, log_init, log_A, log_pmf, log_surv, mask, d_max):
    """(K_R,) log P(o | r), one forward pass per recipe."""
    def one(li, la, lp, ls):
        return jnp.sum(kernels.forward_pass(e, li, la, lp, ls, mask, d_max)[0])
    return jax.vmap(one)(log_init, log_A, log_pmf, log_surv)


_viterbi = jax.jit(kernels.viterbi_decode, static_argnames=("d_max",))
