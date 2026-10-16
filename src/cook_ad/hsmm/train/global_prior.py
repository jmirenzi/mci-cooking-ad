"""A population ("global") prior for one person's model (docs/training.md §3, docs/drift_synth.md §9.2).

The global prior is per-recipe counts of first runs and of run-to-run transitions, read off many
people's tick streams by run-length encoding, keyed by the (verb, noun) pair. States are pairs
(lexical init), so no population EM is needed and the counts project onto any person's state
inventory, whatever its K.

`apply_global_prior` is post-fit, like smoothing.backoff_counts: each transition row of the
person's fitted model gets lam x the population's row for that recipe and pair, renormalised
over the person's states; init likewise. Cells are first lifted to count 1 so that, under the
Dirichlet-MAP mode (numerator c - 1), a cell the person never produced is worth exactly
lam x P_global, and lam = 0 returns the person model unchanged bit for bit (cells below 1 floor
either way). A transition nobody in the population produced stays on the floor.

`person_counts` + `n0` make the prior fade: a row the person has left n times gets
lam / (1 + n / n0), so rarely seen actions lean on the population and well-observed ones keep
the person's own habits (an unseen-by-the-person transition there stays rare enough for a
transition-channel alpha to flag it; eval/harness.hindsight_split_grid).

Durations are untouched (person only).
"""
import numpy as np


def run_length(verb_ids, noun_ids):
    """Tick stream -> [(verb, noun, ticks)] runs of constant pair."""
    runs = []
    for v, n in zip(verb_ids, noun_ids):
        v, n = int(v), int(n)
        if runs and runs[-1][0] == v and runs[-1][1] == n:
            runs[-1][2] += 1
        else:
            runs.append([v, n, 1])
    return [tuple(r) for r in runs]


def _key(v, n):
    return f"{v}_{n}"


def count_global(sequences, recipes):
    """sequences.json entries and each one's recipe name -> {"init": {recipe: {pair: c}},
    "trans": {recipe: {pair: {pair: c}}}, "n_trials": {recipe: n}}; pairs keyed "verb_noun"."""
    init, trans, n = {}, {}, {}
    for seq, r in zip(sequences, recipes):
        runs = run_length(seq["verb_ids"], seq["noun_ids"])
        if not runs:
            continue
        n[r] = n.get(r, 0) + 1
        a = _key(*runs[0][:2])
        row0 = init.setdefault(r, {})
        row0[a] = row0.get(a, 0) + 1
        rows = trans.setdefault(r, {})
        for (v0, n0, _), (v1, n1, _) in zip(runs[:-1], runs[1:]):
            row = rows.setdefault(_key(v0, n0), {})
            b = _key(v1, n1)
            row[b] = row.get(b, 0) + 1
    return {"init": init, "trans": trans, "n_trials": n}


def state_pairs(params, min_ratio=2.0):
    """Each state's (verb, noun) by the argmax of its emission counts, or None for a headroom
    state (a near-flat row: the lexical init had fewer pairs than K)."""
    v = np.asarray(params.verb_counts)
    n = np.asarray(params.noun_counts)
    out = []
    for k in range(v.shape[0]):
        anchored = v[k].max() > min_ratio * v[k].mean() and n[k].max() > min_ratio * n[k].mean()
        out.append((int(v[k].argmax()), int(n[k].argmax())) if anchored else None)
    return out


def _prior_rows(global_counts, recipe, pairs):
    """(K,) init and (K, K) transition probabilities over the person's states for one recipe."""
    k = len(pairs)
    keys = [None if p is None else _key(*p) for p in pairs]
    gi = global_counts["init"].get(recipe, {})
    gt = global_counts["trans"].get(recipe, {})
    p_init = np.array([gi.get(c, 0.0) if c else 0.0 for c in keys], dtype=float)
    p_trans = np.zeros((k, k))
    for i, a in enumerate(keys):
        if a is None:
            continue
        row = gt.get(a, {})
        p_trans[i] = [row.get(b, 0.0) if (b and j != i) else 0.0 for j, b in enumerate(keys)]
    p_init /= max(p_init.sum(), 1e-300) if p_init.sum() else 1.0
    s = p_trans.sum(axis=1, keepdims=True)
    p_trans = np.divide(p_trans, s, out=np.zeros_like(p_trans), where=s > 0)
    return p_init, p_trans


def _row_strength(lam, n0, person_counts, recipe, pairs):
    """(K,) prior strength per transition row, and the init row's: lam, or lam / (1 + n / n0)
    with n the person's observed exits from that pair (init: their trials of the recipe)."""
    k = len(pairs)
    if person_counts is None or not n0:
        return np.full(k, float(lam)), float(lam)
    rows = person_counts["trans"].get(recipe, {})
    n = np.array([sum(rows.get(_key(*p), {}).values()) if p else 0 for p in pairs], dtype=float)
    n_trials = person_counts["n_trials"].get(recipe, 0)
    return lam / (1.0 + n / n0), lam / (1.0 + n_trials / n0)


def apply_global_prior(params, global_counts, recipe_names, lam, lam_init=None, replace=False,
                       person_counts=None, n0=None):
    """The person model with the global prior added at strength `lam` (pseudocounts per row).
    `recipe_names[r]` names the person model's recipe r. `replace=True` drops the person's own
    init/transition counts first: the global-only model on the person's state space.
    `person_counts` (count_global on the person's TRAINING sequences) and `n0` fade the prior
    per row (module docstring)."""
    lam_init = lam if lam_init is None else lam_init
    pairs = state_pairs(params)
    init = np.asarray(params.init_counts, dtype=float).copy()
    trans = np.asarray(params.trans_counts, dtype=float).copy()
    k = trans.shape[-1]
    off = 1.0 - np.eye(k)
    if replace:
        init[:] = 0.0
        trans[:] = 0.0
    if lam == 0 and lam_init == 0 and not replace:
        return params
    for r, name in enumerate(recipe_names):
        p_init, p_trans = _prior_rows(global_counts, name, pairs)
        lam_rows, lam_i = _row_strength(lam, n0, person_counts, name, pairs)
        if lam_init is not None and person_counts is None:
            lam_i = lam_init
        init[r] = np.maximum(init[r], 1.0) + lam_i * p_init
        trans[r] = (np.maximum(trans[r], 1.0) + lam_rows[:, None] * p_trans) * off
    return params._replace(init_counts=init, trans_counts=trans)
