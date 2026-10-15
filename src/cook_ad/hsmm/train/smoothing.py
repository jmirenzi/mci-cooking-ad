"""Post-fit transition smoothing, so the Dirichlet-MAP mode stops erasing singleton
transitions. docs/training.md.

`params._row_normalize` takes numerator max(c - 1, floor), so a bigram observed exactly once
floors -- and at K_R=16 over ~400 trials most of a recipe's legal transitions are singletons.
`shifted_mode_counts(strength=s)` stores c + s, making the numerator c - 1 + s. Useful range is
0 < s < 1: a singleton clears the floor, a never-observed cell stays on it. s = 1 (the posterior
mean) is worse -- it also lifts never-seen transitions from ~32 nats to ~9.

`backoff_counts(tau)` mixes each recipe's rows toward the pooled-over-recipes row first. Both
are regularisation: select them on held-out data (scripts/data/make_dev_split.py), not
in-sample. Emissions are untouched -- their prior is already 1 per category, so the mode is the
plain data frequency, and shifting it would give unobserved tokens the mass the substitution
channel needs. Post-fit only: the M-step rebuilds trans_counts every iteration.
"""
import numpy as np


def backoff_counts(jp, tau):
    """Shrink each recipe's transition rows toward the pooled-over-recipes row for that state.
    A transition unseen everywhere stays maximally surprising; one merely unseen in THIS recipe
    inherits the pooled row's mass and becomes mildly so."""
    trans = np.asarray(jp.trans_counts)
    k = trans.shape[-1]
    off_diagonal = 1.0 - np.eye(k)
    pooled = trans.sum(axis=0) * off_diagonal          # (K,K)
    pooled = pooled / np.maximum(pooled.sum(axis=-1, keepdims=True), 1e-12)
    return jp._replace(trans_counts=trans + tau * pooled[None, :, :])


def shifted_mode_counts(jp, strength=1.0):
    """Add `strength` to every off-diagonal transition cell and to init."""
    trans = np.asarray(jp.trans_counts)
    init = np.asarray(jp.init_counts)
    k = trans.shape[-1]
    # off-diagonal only: self-transitions are structurally banned and must stay at zero count
    off_diagonal = (1.0 - np.eye(k))[None, :, :]
    return jp._replace(trans_counts=trans + strength * off_diagonal, init_counts=init + strength)
