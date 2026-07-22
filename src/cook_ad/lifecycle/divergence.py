import jax.numpy as jnp

from cook_ad.hsmm import params


def categorical_kl(log_p, log_q):
    """KL(p || q) = sum_w p_w (log p_w - log q_w) along the last axis, from log-prob arrays.
    Cells where p_w == 0 contribute 0 (masked before the multiply, so a banned/floored entry
    with log_q = -inf does not produce a 0 * -inf = NaN)."""
    p = jnp.exp(log_p)
    return jnp.sum(jnp.where(p > 0, p * (log_p - log_q), 0.0), axis=-1)


def model_divergence(live, frozen):
    """KL(live || frozen) over the categorical PDFs the model exposes, broken down by family
    and by state so drift is *localized* -- exactly what the weekly consolidation review needs
    ("find the large differences, then query the user about them"). Duration is omitted: it is
    never updated online, so live and frozen durations are identical and its KL is always 0.

    Returns a dict:
      init      scalar   KL of the initial-state distribution
      trans     (K,)     per-from-state KL of P(Z'|Z)
      verb      (K,)     per-state KL of P(v|Z)
      noun      (K,)     per-state KL of P(n|Z)
      per_state (K,)     trans + verb + noun, the drift heat map over subtasks
      total     scalar   init + sum(per_state)
    """
    li, lt, lv, ln = params.normalize_categoricals(live)
    fi, ft, fv, fn = params.normalize_categoricals(frozen)

    kl_init = categorical_kl(li, fi)
    kl_trans = categorical_kl(lt, ft)
    kl_verb = categorical_kl(lv, fv)
    kl_noun = categorical_kl(ln, fn)
    per_state = kl_trans + kl_verb + kl_noun

    return {
        "init": kl_init,
        "trans": kl_trans,
        "verb": kl_verb,
        "noun": kl_noun,
        "per_state": per_state,
        "total": kl_init + jnp.sum(per_state),
    }
