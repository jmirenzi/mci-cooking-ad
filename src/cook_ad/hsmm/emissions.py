import jax.numpy as jnp


def sequence_loglik(verb_ids, noun_ids, log_emit_v, log_emit_n, mask):
    """verb_ids, noun_ids: (T,) int32. log_emit_v: (K,V). log_emit_n: (K,N). mask: (T,) bool.

    Returns (T,K): log P(v_t|Z=k) + log P(n_t|Z=k), matching the locked conditional-
    independence assumption P(v,n|Z) = P(v|Z)*P(n|Z).

    Padded ticks (verb_ids/noun_ids filled with the valid dummy id 0, never -1 -- JAX
    gather wraps negative/out-of-range indices rather than erroring) are zeroed via `mask`
    so they contribute log-lik 0 (a no-op) regardless of what dummy id was used.
    """
    loglik = log_emit_v[:, verb_ids].T + log_emit_n[:, noun_ids].T
    return jnp.where(mask[:, None], loglik, 0.0)
