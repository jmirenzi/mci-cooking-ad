"""recipe/lexical_init.py -- the observation-derived warm start for the joint model."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from cook_ad.hsmm import joint_params
from cook_ad.recipe import lexical_init

V, N = 5, 6


def _trial(pairs_and_durations):
    verbs, nouns = [], []
    for (v, n), d in pairs_and_durations:
        verbs += [v] * d
        nouns += [n] * d
    return {"verb_ids": verbs, "noun_ids": nouns}


@pytest.fixture
def sequences():
    # two "recipes": A uses pairs (0,0)->(1,1)->(2,2); B uses (3,3)->(4,4)->(1,1)
    a = [_trial([((0, 0), 6), ((1, 1), 5), ((2, 2), 7)]) for _ in range(6)]
    b = [_trial([((3, 3), 4), ((4, 4), 8), ((1, 1), 6)]) for _ in range(6)]
    return a + b


def test_observed_pairs_respects_min_ticks(sequences):
    pairs, counts = lexical_init.observed_pairs(sequences, min_ticks=1)
    assert set(pairs) == {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)}
    assert sorted(counts, reverse=True) == counts, "pairs must come back most-frequent first"

    # (4,4) covers 8*6 = 48 ticks, (3,3) covers 24; a cut above 24 must drop (3,3) and keep (4,4)
    pairs_cut, _ = lexical_init.observed_pairs(sequences, min_ticks=30)
    assert (3, 3) not in pairs_cut and (4, 4) in pairs_cut


def test_hard_segments_are_the_run_length_encoding(sequences):
    pairs, _ = lexical_init.observed_pairs(sequences, min_ticks=1)
    index = {p: i for i, p in enumerate(pairs)}
    segs = lexical_init.hard_segments(sequences, pairs)
    assert segs[0] == [(index[(0, 0)], 6), (index[(1, 1)], 5), (index[(2, 2)], 7)]
    for seq, s in zip(sequences, segs):
        assert sum(d for _, d in s) == len(seq["verb_ids"]), "segments must tile the trial exactly"


def test_dropped_pair_is_folded_into_a_neighbour_not_lost():
    """A run whose pair fell below min_ticks has no state to go to. It is absorbed by an
    adjacent run rather than dropped, because dropping it would leave the segmentation no
    longer tiling the trial -- and every downstream consumer assumes sum(durations) == T."""
    seqs = [_trial([((0, 0), 10), ((1, 1), 1), ((0, 0), 10)]) for _ in range(3)]
    pairs, _ = lexical_init.observed_pairs(seqs, min_ticks=5)
    assert (1, 1) not in pairs
    for seq, s in zip(seqs, lexical_init.hard_segments(seqs, pairs)):
        assert sum(d for _, d in s) == len(seq["verb_ids"])


def test_emission_anchor_puts_each_state_on_its_own_pair(sequences):
    pairs, _ = lexical_init.observed_pairs(sequences, min_ticks=1)
    verb, noun = lexical_init.anchored_emission_counts(pairs, 8, V, N, anchor=50.0, background=1.0)
    verb, noun = np.asarray(verb), np.asarray(noun)
    for k, (v, n) in enumerate(pairs):
        assert int(np.argmax(verb[k])) == v
        assert int(np.argmax(noun[k])) == n
    # headroom states beyond len(pairs) stay flat -- nothing claimed for them
    assert np.allclose(verb[len(pairs)], verb[len(pairs)][0])
    # no zero entries anywhere: an off-pair token must have finite, calibratable surprise
    assert (verb > 0).all() and (noun > 0).all()


def test_cluster_recipes_separates_the_two_action_inventories(sequences):
    assign, _ = lexical_init.cluster_recipes(sequences, k_recipe=2, seed=0)
    assert len(set(assign[:6])) == 1 and len(set(assign[6:])) == 1
    assert assign[0] != assign[6]


def test_lexical_to_joint_is_a_usable_joint_model(sequences):
    jp, info = lexical_init.lexical_to_joint(
        sequences, k_subtask=8, k_recipe=2, d_max=20, vocab_verbs=V, vocab_nouns=N, kappa=5.0
    )
    assert isinstance(jp, joint_params.JointHSMMParams)
    lp = joint_params.to_log_probs_joint(jp, 20)
    for name in ("log_init", "log_trans", "log_emit_v", "log_emit_n"):
        arr = np.asarray(getattr(lp, name))
        assert not np.isnan(arr).any(), f"{name} has NaN"
    assert np.allclose(np.exp(np.asarray(lp.log_trans)).sum(-1), 1.0, atol=1e-6)
    # structurally banned self-transitions survive the prior addition
    assert np.all(np.asarray(lp.log_trans)[:, np.arange(8), np.arange(8)] == -np.inf)
    assert set(info) >= {"pairs", "assign", "emit_prior_v", "emit_prior_n"}


def test_prior_keeps_a_once_seen_transition_above_a_never_seen_one(sequences):
    """The reason the Dirichlet prior is added at iteration 0 rather than left to the first
    M-step: params._row_normalize's MAP numerator max(c - 1, floor) sends a raw count of 1 to
    the floor, which would make a bigram seen once indistinguishable from one never seen."""
    extra = [_trial([((0, 0), 6), ((2, 2), 5), ((1, 1), 7)])]  # a one-off 0->2 ordering
    seqs = sequences + extra
    jp, info = lexical_init.lexical_to_joint(
        seqs, k_subtask=8, k_recipe=2, d_max=20, vocab_verbs=V, vocab_nouns=N, kappa=5.0
    )
    pairs = info["pairs"]
    s0, s1, s2 = (pairs.index(p) for p in ((0, 0), (1, 1), (2, 2)))
    r = int(info["assign"][-1])
    log_trans = np.asarray(joint_params.to_log_probs_joint(jp, 20).log_trans)[r]
    once_seen = log_trans[s0, s2]
    never_seen = log_trans[s1, s0]
    assert once_seen > never_seen + 1.0, (
        f"once-seen {once_seen:.2f} must beat never-seen {never_seen:.2f} by a real margin"
    )
