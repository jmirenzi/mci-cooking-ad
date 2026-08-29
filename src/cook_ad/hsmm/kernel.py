"""Fixed semantic-neighbourhood kernel S for the latent-intended-token emission.

The emission gains one latent variable per tick: the subtask INTENDS token m ~ Cat(B[k]),
which surfaces as the observed token n ~ Cat(S[m]). Marginally P(n|k) = (B S)[k,n], a matrix
product with a FIXED S -- so this is standard latent-variable EM, not count smoothing bolted
onto the M-step, and EM's monotonicity is preserved (docs/hsmm.md tracks it).

    S = (1 - lam) * I + lam * rownorm(softmax_offdiag(cos(e_i, e_j) / tau))

Deliberately NOT a single softmax over the full row. The diagonal's cosine is 1.0 by
construction while the best off-diagonal is ~0.6, so one softmax couples two quantities that
must move independently: any tau sharp enough to tell water from knife leaves 99.9% of the row
on the self term (S is the identity), and any tau loose enough to leak real mass gives a
near-uniform row -- the undiscriminating mass-shifting smooth_params.py objects to.

Split, `lam` is how MUCH mass the intended token leaks and `tau` is how SEMANTICALLY that mass
is shaped, with no influence on its size. lam = 0 is the identity for any tau; tau -> inf
spreads it uniformly, which is the ablation isolating semantics from mass alone.

`eps` guarantees strict positivity so no observation is ever -inf. Both lam and tau are
regularisation: select them on held-out data (make_dev_split.py), never in-sample.
"""
import numpy as np

MIN_TAU = 1e-6


def similarity_kernel(embeddings, tau, lam, epsilon=1e-4, uniform=False):
    """embeddings: (W,d), rows need not be unit-norm (they are renormalised here).
    Returns (W,W) row-stochastic, strictly positive.

    `uniform=True` ignores the embeddings and spreads `lam` evenly over the other W-1 tokens:
    the ablation that separates "semantic neighbourhood helped" from "any leaked mass helped".
    """
    e = np.asarray(embeddings, dtype=np.float64)
    w = e.shape[0]
    eye = np.eye(w)
    if lam <= 0.0:
        return eye

    if uniform:
        off = (1.0 - eye) / (w - 1)
    else:
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        logits = (e @ e.T) / max(tau, MIN_TAU)
        logits = np.where(eye.astype(bool), -np.inf, logits)   # diagonal out of the softmax
        logits -= logits.max(axis=1, keepdims=True)
        off = np.exp(logits)
        off /= off.sum(axis=1, keepdims=True)

    s = (1.0 - lam) * eye + lam * off
    s = (1.0 - epsilon) * s + epsilon / w
    return s / s.sum(axis=1, keepdims=True)


def load_embeddings(path):
    """-> (verbs (V,d), nouns (N,d)) from tools_embed_vocab.py's .npz."""
    with np.load(path, allow_pickle=True) as d:
        return np.asarray(d["verbs"]), np.asarray(d["nouns"])


def kernels_from_embeddings(path, tau, lam, epsilon=1e-4, lam_verb=None, uniform=False):
    """(S_v, S_n) for one embeddings.npz. `lam_verb` defaults to `lam`, but 0 is the honest
    setting: tools_embed_vocab.py's nearest-neighbour gate shows clean noun neighbourhoods and
    near-noise verb ones (crack -> squeeze, butter -> stirfry), so 0 uses only the half that
    passed."""
    verbs, nouns = load_embeddings(path)
    lv = lam if lam_verb is None else lam_verb
    return (similarity_kernel(verbs, tau, lv, epsilon, uniform),
            similarity_kernel(nouns, tau, lam, epsilon, uniform))


def nearest_neighbours(embeddings):
    """(W,) int: each token's most similar OTHER token -- the replacement table for
    error_injection's select='near'."""
    e = np.asarray(embeddings, dtype=np.float64)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    return np.argmax(sim, axis=1)
