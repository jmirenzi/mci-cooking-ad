"""Iteration-0 parameters for the joint model built from the observations alone: one subtask
state per distinct observed (verb, noun) pair, recipes seeded by bag-of-pairs k-means.

Drop-in alternative to `warm_start.cascade_to_joint`, needing no cascade artifacts. The tick
stream is coarse action labels, so a maximal run of constant (v,n) is one action instance and
the distinct pairs are the action inventory -- recoverable from `sequences.json` alone.
`labels.json` is not read here and must not be.

Full rationale and the measurements behind the defaults: docs/recipe.md 4.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.hsmm import durations
from cook_ad.hsmm.joint_params import JointHSMMParams

# A pair covering fewer ticks than this gets no state of its own: its duration fit and
# transition row would be pure noise.
MIN_PAIR_TICKS = 5

# Pseudocount on a state's own tokens, and the mass spread over the rest of each vocabulary.
# Background is nonzero so an off-pair observation has finite, calibratable surprise, not -inf.
ANCHOR_MASS = 50.0
BACKGROUND_MASS = 1.0


def observed_pairs(sequences, min_ticks=MIN_PAIR_TICKS):
    """Distinct (verb_id, noun_id) pairs in `sequences`, most frequent first, keeping only
    those covering at least `min_ticks` ticks. Returns (pairs, tick_counts)."""
    counts = {}
    for seq in sequences:
        for v, n in zip(seq["verb_ids"], seq["noun_ids"]):
            key = (int(v), int(n))
            counts[key] = counts.get(key, 0) + 1
    kept = [(p, c) for p, c in counts.items() if c >= min_ticks]
    kept.sort(key=lambda x: -x[1])
    return [p for p, _ in kept], [c for _, c in kept]


def anchored_emission_counts(pairs, k_subtask, vocab_verbs, vocab_nouns,
                             anchor=ANCHOR_MASS, background=BACKGROUND_MASS):
    """(K,V) / (K,N) Dirichlet pseudocount matrices: `anchor` on state k's own pair tokens,
    `background` spread over the rest. Weak-limit headroom states get a flat row. Usable both as
    the iteration-0 value and as `joint_em.m_step`'s `emit_prior_v`/`emit_prior_n`."""
    verb = np.full((k_subtask, vocab_verbs), background / vocab_verbs)
    noun = np.full((k_subtask, vocab_nouns), background / vocab_nouns)
    for k, (v, n) in enumerate(pairs[:k_subtask]):
        verb[k, v] += anchor
        noun[k, n] += anchor
    return jnp.asarray(verb), jnp.asarray(noun)


def hard_segments(sequences, pairs):
    """Run-length-encode each trial's (v,n) stream into [(state_id, duration), ...]. A run whose
    pair was dropped by `min_ticks` is folded into a neighbour rather than discarded, so the
    segmentation still tiles the trial exactly -- every consumer assumes sum(d) == T.
    """
    index = {p: i for i, p in enumerate(pairs)}
    out = []
    for seq in sequences:
        runs = []
        prev = None
        for v, n in zip(seq["verb_ids"], seq["noun_ids"]):
            key = (int(v), int(n))
            if key == prev:
                runs[-1][1] += 1
            else:
                runs.append([key, 1])
                prev = key
        merged = []
        for key, d in runs:
            state = index.get(key)
            if state is None:
                if merged:
                    merged[-1][1] += d
                else:
                    merged.append([None, d])  # placeholder, absorbed by the next real run below
                continue
            if merged and merged[-1][0] is None:
                merged[-1] = [state, merged[-1][1] + d]
            elif merged and merged[-1][0] == state:
                merged[-1][1] += d
            else:
                merged.append([state, d])
        if merged and merged[0][0] is None:  # whole trial was dropped pairs -- keep it as state 0
            merged[0][0] = 0
        out.append([(int(s), int(d)) for s, d in merged])
    return out


def cluster_recipes(sequences, k_recipe, seed=0, n_init=20):
    """Trial-level recipe clusters by spherical k-means on L1-normalised (v,n)-pair histograms.
    A recipe is characterised by which actions it contains, which the bag of pairs states
    directly. Returns (assignments (N,), centroids). Empty clusters are re-seeded at the trial
    furthest from its own centroid, so the weak-limit prior -- not this function -- is what kills
    a recipe off.
    """
    pairs, _ = observed_pairs(sequences, min_ticks=1)
    index = {p: i for i, p in enumerate(pairs)}
    x = np.zeros((len(sequences), len(pairs)))
    for i, seq in enumerate(sequences):
        for v, n in zip(seq["verb_ids"], seq["noun_ids"]):
            x[i, index[(int(v), int(n))]] += 1.0
    x = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    rng = np.random.default_rng(seed)
    best_assign, best_obj, best_c = None, -np.inf, None
    for _ in range(n_init):
        centroids = x[rng.choice(len(x), size=k_recipe, replace=False)].copy()
        assign = None
        for _ in range(100):
            sim = x @ centroids.T
            new_assign = np.argmax(sim, axis=1)
            if assign is not None and np.array_equal(new_assign, assign):
                break
            assign = new_assign
            for r in range(k_recipe):
                members = x[assign == r]
                if len(members) == 0:
                    far = int(np.argmin(np.max(x @ centroids.T, axis=1)))
                    centroids[r] = x[far]
                    continue
                c = members.mean(axis=0)
                centroids[r] = c / max(np.linalg.norm(c), 1e-12)
        obj = float(np.sum(np.max(x @ centroids.T, axis=1)))
        if obj > best_obj:
            best_assign, best_obj, best_c = assign, obj, centroids.copy()
    return best_assign, best_c


def _init_trans_dur_counts(segments_by_trial, assign, k_subtask, k_recipe, d_max):
    init_counts = np.zeros((k_recipe, k_subtask))
    trans_counts = np.zeros((k_recipe, k_subtask, k_subtask))
    dur_hist = np.zeros((k_recipe, k_subtask, d_max))
    for segs, r in zip(segments_by_trial, assign):
        if not segs:
            continue
        r = int(r)
        init_counts[r, segs[0][0]] += 1.0
        for (s0, _), (s1, _) in zip(segs[:-1], segs[1:]):
            if s0 != s1:
                trans_counts[r, s0, s1] += 1.0
        for s, d in segs:
            dur_hist[r, s, min(int(d), d_max) - 1] += 1.0
    return init_counts, trans_counts, dur_hist


def lexical_to_joint(sequences, k_subtask, k_recipe, d_max, vocab_verbs, vocab_nouns, kappa,
                     seed=0, min_ticks=MIN_PAIR_TICKS, anchor=ANCHOR_MASS,
                     background=BACKGROUND_MASS, alpha_init=0.5, alpha_trans=0.5, alpha_pi=1.0,
                     init_prior_scale=1.0):
    """Build a `JointHSMMParams` whose states are the observed (v,n) pairs and whose per-recipe
    dynamics come from the bag-of-pairs clustering. Drop-in replacement for
    `warm_start.cascade_to_joint` -- same return type, and it needs no cascade artifacts at all.

    Returns (params, info); `info` carries the pair list, the cluster assignment and the emission
    prior matrices, so a caller can hand the same anchors to `joint_em.run_joint_em`.

    `init_prior_scale` scales the Dirichlet prior on the iteration-0 counts only. 1.0 is coherent
    and is the default, but **0.0 is what the best-measured detector uses** -- it ends at a lower
    objective and a worse subtask ARI, and at a better trial_loc accuracy (0.556 vs 0.518). Do
    not "fix" it without re-measuring; docs/recipe.md 4.
    """
    pairs, pair_ticks = observed_pairs(sequences, min_ticks=min_ticks)
    if len(pairs) > k_subtask:
        pairs, pair_ticks = pairs[:k_subtask], pair_ticks[:k_subtask]

    verb_counts, noun_counts = anchored_emission_counts(
        pairs, k_subtask, vocab_verbs, vocab_nouns, anchor=anchor, background=background
    )
    segments_by_trial = hard_segments(sequences, pairs)
    assign, _ = cluster_recipes(sequences, k_recipe, seed=seed)

    init_counts, trans_counts, dur_hist = _init_trans_dur_counts(
        segments_by_trial, assign, k_subtask, k_recipe, d_max
    )
    pi_counts = np.bincount(assign, minlength=k_recipe).astype(float)

    # The same Dirichlet prior joint_em.m_step adds on every subsequent iteration. Without it
    # these are raw hard counts, and params._row_normalize's MAP numerator max(c - 1, floor)
    # sends a bigram observed EXACTLY ONCE to the floor -- indistinguishable from one that never
    # happened. Roughly half the bigrams in a 25-trial recipe cluster are singletons, so leaving
    # the prior off does not merely start EM slightly off: it declares half the model's own
    # legal transitions impossible, which shows up directly as an s_transition false-positive
    # rate that no longer responds to alpha (docs/eval.md 6's signature of an ungated channel).
    init_counts = init_counts + init_prior_scale * alpha_init / k_subtask
    trans_counts = trans_counts + init_prior_scale * alpha_trans / k_subtask
    pi_counts = pi_counts + init_prior_scale * alpha_pi / k_recipe

    # Pooled-over-recipes per-state duration shape, used as the shrinkage target exactly the
    # way warm_start.cascade_to_joint uses the cascade's own global fit: a (r,k) cell with two
    # segments in it should look like that state does in general, not like those two segments.
    pooled = dur_hist.sum(axis=0)  # (K,D)
    pooled = pooled + 1e-3
    pmf_global = pooled / pooled.sum(axis=-1, keepdims=True)  # (K,D)

    n_hat = jnp.asarray(dur_hist) + kappa * jnp.asarray(pmf_global)[None, :, :]
    stats_over_r = jax.vmap(durations.duration_stats_from_histogram, in_axes=(0, None))
    n_tot, s_tot = stats_over_r(n_hat, d_max)
    r_old = jnp.full((k_recipe, k_subtask), 2.0)
    newton_over_r = jax.vmap(functools.partial(durations.newton_update_r, n_iters=8), in_axes=(0, 0, 0, 0))
    dur_r = newton_over_r(n_hat, n_tot, s_tot, r_old)
    p_over_r = jax.vmap(durations.update_p_given_r, in_axes=(0, 0, 0))
    dur_p = p_over_r(n_tot, s_tot, dur_r)

    jp = JointHSMMParams(
        init_counts=jnp.asarray(init_counts),
        trans_counts=jnp.asarray(trans_counts) * (1.0 - jnp.eye(k_subtask))[None, :, :],
        verb_counts=verb_counts,
        noun_counts=noun_counts,
        dur_r=dur_r,
        dur_p=dur_p,
        pi_counts=jnp.asarray(pi_counts),
    )
    info = {
        "pairs": pairs,
        "pair_ticks": pair_ticks,
        "assign": np.asarray(assign),
        "emit_prior_v": verb_counts,
        "emit_prior_n": noun_counts,
        "segments_by_trial": segments_by_trial,
    }
    return jp, info
