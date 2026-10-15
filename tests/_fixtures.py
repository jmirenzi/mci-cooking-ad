"""Small random models for the tests (not a test module)."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from cook_ad.hsmm import durations
from cook_ad.hsmm.params import FLOOR, JointHSMMParams, RecipeView, _row_normalize


def _dirichlet_counts(key, alpha, width, n_rows):
    return alpha * jax.random.dirichlet(key, jnp.full((width,), alpha / width), shape=(n_rows,))


def weak_limit_params(key, k, v, n, d_max, alpha_init=0.5, alpha_trans=0.5, alpha_emit_v=None,
                      alpha_emit_n=None, dur_mean_range=(3.0, 40.0), dur_r_range=(1.0, 20.0)):
    """A random one-recipe model (Dirichlet weak-limit draw), as a RecipeView."""
    alpha_emit_v = float(v) if alpha_emit_v is None else alpha_emit_v
    alpha_emit_n = float(n) if alpha_emit_n is None else alpha_emit_n
    ki, kt, kv, kn, km, kr = jax.random.split(key, 6)
    init = alpha_init * jax.random.dirichlet(ki, jnp.full((k,), alpha_init / k))
    trans = _dirichlet_counts(kt, alpha_trans, k, k) * (1.0 - jnp.eye(k))
    mean = jax.random.uniform(km, (k,), minval=dur_mean_range[0], maxval=dur_mean_range[1])
    r = jax.random.uniform(kr, (k,), minval=dur_r_range[0], maxval=dur_r_range[1])
    return RecipeView(init, trans, _dirichlet_counts(kv, alpha_emit_v, v, k),
                      _dirichlet_counts(kn, alpha_emit_n, n, k), r, r / (r + mean))


def stack_recipes(views, pi_counts=None):
    """JointHSMMParams from one RecipeView per recipe (emissions taken from the first)."""
    return JointHSMMParams(
        init_counts=jnp.stack([p.init_counts for p in views]),
        trans_counts=jnp.stack([p.trans_counts for p in views]),
        verb_counts=views[0].verb_counts, noun_counts=views[0].noun_counts,
        dur_r=jnp.stack([p.dur_r for p in views]), dur_p=jnp.stack([p.dur_p for p in views]),
        pi_counts=jnp.ones(len(views)) if pi_counts is None else jnp.asarray(pi_counts))


def one_recipe_tables(view, d_max):
    """Log tables of a single RecipeView: log_init, log_A, log_emit_v, log_emit_n, log_pmf, log_surv."""
    log_pmf, log_surv = durations.duration_tables(view.dur_r, view.dur_p, d_max)
    return SimpleNamespace(
        log_init=_row_normalize(view.init_counts, FLOOR),
        log_A=_row_normalize(view.trans_counts, FLOOR, mask_diag=True),
        log_emit_v=_row_normalize(view.verb_counts, FLOOR),
        log_emit_n=_row_normalize(view.noun_counts, FLOOR),
        log_pmf=log_pmf, log_surv=log_surv)
