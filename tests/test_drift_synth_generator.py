"""The synthetic drift generator on the toy world (tests/_drift_fixtures.py): the spec's nine
tests (docs/drift_synth.md §7) plus unit checks on each stage."""
import filecmp
import math
from collections import Counter

import numpy as np
import pytest
from scipy.stats import chi2, chisquare, ks_2samp, norm

from _drift_fixtures import OMELET, TAGS, VOCAB, theta, toy_assets
from cook_ad.drift_synth import ground_truth as GT
from cook_ad.drift_synth.durations import fit_from_intervals
from cook_ad.drift_synth.export import export, to_ticks
from cook_ad.drift_synth.graphs import is_topological
from cook_ad.drift_synth.kitchen_state import insert_logistics
from cook_ad.drift_synth.knobs import apply_variants, sample_topological_order
from cook_ad.drift_synth.noise import NoiseCfg, recognizer_noise
from cook_ad.drift_synth.sampler import ConditionCfg, generate_person, rng_for
from cook_ad.drift_synth.schedule import DriftRegime, ramp, theta_at
from cook_ad.drift_synth.types import Event, RecipeGraph, Step

ASSETS = toy_assets()
N_MC = 400


@pytest.fixture(scope="module")
def stats_cache():
    cache = {}

    def get(recipe, th, seed):
        key = (recipe, tuple(sorted(th.items())), seed)
        if key not in cache:
            cache[key] = GT.observable_stats(recipe, th, N_MC, seed, ASSETS)
        return cache[key]
    return get


# ---- 1. vocab compliance ---------------------------------------------------------------------

def test_every_emitted_token_is_in_vocab():
    cond = ConditionCfg("c", DriftRegime("sudden", 10), {"sour": 0.4, "calorie_conscious": 0.4},
                        n_persons=3, n_days=21)
    for i in range(cond.n_persons):
        _, sessions, _ = generate_person(i, cond, 0, ASSETS)
        for s in sessions:
            for e in s.events:
                assert e.verb in VOCAB["verbs"] and e.noun in VOCAB["nouns"], (e, s.recipe_id)


# ---- 2. graph validity -----------------------------------------------------------------------

@pytest.mark.parametrize("mise", [0.0, 0.5, 1.0])
def test_sampled_orders_are_topological(mise):
    rng = np.random.default_rng(0)
    for _ in range(300):
        assert is_topological(OMELET, sample_topological_order(OMELET, mise, 4.0, rng))


# ---- 3. determinism --------------------------------------------------------------------------

def test_same_seed_same_bytes(tmp_path):
    cond = ConditionCfg("c", DriftRegime("gradual", 5, 7), {"slow": 0.3}, n_days=14)
    for out in ("a", "b"):
        _, sessions, gt = generate_person(1, cond, 3, ASSETS)
        export(sessions, tmp_path / out, VOCAB, gt, ASSETS.gen["export"])
    cmp = filecmp.dircmp(tmp_path / "a", tmp_path / "b")
    assert not cmp.diff_files and not cmp.left_only and not cmp.right_only
    sub = filecmp.dircmp(tmp_path / "a" / "sessions", tmp_path / "b" / "sessions")
    assert not sub.diff_files and len(sub.common_files) > 10


# ---- 4. knob isolation -----------------------------------------------------------------------

def _linked_tokens(knob, tokens):
    """Tokens a knob may change the per-session count of."""
    def noun_in(nouns):
        return {t for t in tokens if t.split(" ")[1] in nouns}
    if knob == "sour":
        return noun_in({"lemon", "vinegar"})
    if knob == "calorie_conscious":
        return noun_in({"oil", "cheese", "basil"})
    if knob == "tidy":
        return {t for t in tokens if t.split(" ")[0] in ("put", "wash")}
    if knob == "fine_motor":
        return {t for t in tokens if TAGS["verbs"].get(t.split(" ")[0]) == ["fine_motor"]}
    return set()   # slow and mise_en_place move no counts


def _linked_durations(knob, tokens):
    if knob == "slow":
        return set(tokens)
    if knob == "fine_motor":
        return {t for t in tokens if TAGS["verbs"].get(t.split(" ")[0]) == ["fine_motor"]}
    if knob == "calorie_conscious":
        return {t for t in tokens if "fat" in TAGS["nouns"].get(t.split(" ")[1], [])}
    return set()


@pytest.mark.parametrize("knob", ["sour", "calorie_conscious", "mise_en_place", "tidy", "slow",
                                  "fine_motor"])
def test_knob_isolation(knob, stats_cache):
    th0, th1 = theta(), theta(**{knob: 0.95})
    for recipe in ("omelet", "salad"):
        s0, s1 = stats_cache(recipe, th0, 11), stats_cache(recipe, th1, 12)
        tokens = set(s0.counts) | set(s1.counts)
        z = GT.count_z(s0, s1)
        crit = norm.ppf(1 - 0.001 / (2 * len(z)))
        free = tokens - _linked_tokens(knob, tokens)
        moved = {t: round(z[t], 2) for t in free if abs(z[t]) > crit}
        assert not moved, f"{knob} moved unlinked counts in {recipe}: {moved}"
        common = set(s0.logdur) & set(s1.logdur)
        free_d = common - _linked_durations(knob, common)
        bad = {t for t in free_d if ks_2samp(s0.logdur[t], s1.logdur[t]).pvalue < 0.001 / len(common)}
        assert not bad, f"{knob} moved unlinked durations in {recipe}: {bad}"
        if knob == "mise_en_place" and recipe == "omelet":
            null_js = GT.js_divergence(s0.bigram, stats_cache(recipe, th0, 13).bigram)
            assert GT.js_divergence(s0.bigram, s1.bigram) > 3 * null_js


def test_knobs_move_their_own_features(stats_cache):
    base = stats_cache("omelet", theta(), 11)
    up = {k: stats_cache("omelet", theta(**{k: 0.95}), 12) for k in ("sour", "tidy", "slow")}
    assert GT.count_z(base, up["sour"])["squeeze lemon"] > 5
    assert GT.count_z(base, up["tidy"])["put tomato"] > 5
    assert GT.drift_magnitude(base, up["slow"])["duration_w1_mean"] > 0.3


# ---- 5. null equivalence ---------------------------------------------------------------------

def test_null_equivalence(stats_cache):
    th = theta(sour=0.3, tidy=0.7)
    pvals = []
    for recipe in ("omelet", "salad"):
        s0, s1 = stats_cache(recipe, th, 21), stats_cache(recipe, th, 22)
        pvals += [2 * norm.sf(abs(z)) for z in GT.count_z(s0, s1).values() if math.isfinite(z)]
        pvals += [ks_2samp(s0.logdur[t], s1.logdur[t]).pvalue for t in set(s0.logdur) & set(s1.logdur)]
    assert min(pvals) > 0.01 / len(pvals), "a null comparison is significant after Bonferroni"


# ---- 6. dose-response ------------------------------------------------------------------------

@pytest.mark.parametrize("knob, metric", [("sour", "marginal_js"), ("slow", "duration_w1_mean")])
def test_dose_response_is_monotone(knob, metric, stats_cache):
    base = stats_cache("omelet", theta(**{knob: 0.1}), 31)
    mags = [GT.drift_magnitude(base, stats_cache("omelet", theta(**{knob: 0.1 + d}), 32))[metric]
            for d in (0.0, 0.2, 0.4, 0.6, 0.8)]
    assert all(b >= a for a, b in zip(mags, mags[1:])), mags


# ---- 7. noise rates --------------------------------------------------------------------------

def test_noise_rates_match_config():
    cfg = NoiseCfg(p_sub_verb=0.05, p_sub_noun=0.08, p_delete=0.06, p_insert=0.04, jitter_s=0.3)
    events = [Event(2.0 * i, 2.0 * i + 1.5, "cut", "tomato") for i in range(10_000)]
    pool = (["cut", "mix", "pour"], ["tomato", "egg", "oil"])
    out, counts = recognizer_noise(events, cfg, pool, np.random.default_rng(0))
    for key, rate in (("sub_verb", 0.05), ("sub_noun", 0.08), ("deleted", 0.06), ("inserted", 0.04)):
        assert abs(counts[key] / 10_000 - rate) < 0.1 * rate, (key, counts[key])
    assert sum(e.verb != "cut" for e in out) > 0 and all(e.t_end > e.t_start for e in out)
    assert all(a.t_end <= b.t_start for a, b in zip(out, out[1:]))


# ---- 8. ordering symmetry --------------------------------------------------------------------

def _orders_graph():
    """a -> b -> d, c -> d: 3 orders. A softmax over available steps would give c-first 1/2."""
    steps = {"1": Step("1", "a", "prep"), "2": Step("2", "b", "cook"), "3": Step("3", "c", "prep"),
             "4": Step("4", "d", "serve")}
    return RecipeGraph("sym", steps, [("1", "2"), ("2", "4"), ("3", "4")])


def test_order_is_uniform_at_half():
    g = _orders_graph()
    rng = np.random.default_rng(0)
    c = Counter(tuple(sample_topological_order(g, 0.5, 4.0, rng)) for _ in range(3000))
    assert len(c) == 3
    assert chisquare(list(c.values())).pvalue > 0.01, c


def test_mise_en_place_pulls_prep_forward():
    g = _orders_graph()
    rng = np.random.default_rng(0)
    prep_first = Counter(sample_topological_order(g, m, 4.0, rng).index("3") <
                         sample_topological_order(g, m, 4.0, rng).index("2") for m in [1.0] * 1000)
    late = Counter(sample_topological_order(g, 0.0, 4.0, rng).index("3") <
                   sample_topological_order(g, 0.0, 4.0, rng).index("2") for _ in range(1000))
    assert prep_first[True] > 900 and late[True] < 500


# ---- 9. schedule -----------------------------------------------------------------------------

def test_sudden_switches_exactly_at_onset():
    reg = DriftRegime("sudden", 28)
    t0, t1 = theta(sour=0.2), theta(sour=0.7)
    rng = np.random.default_rng(0)
    for d in range(70):
        th, which = theta_at(d, t0, t1, reg, rng)
        assert (which, th["sour"]) == ((0, 0.2) if d < 28 else (1, 0.7))


def test_gradual_tracks_its_ramp():
    reg = DriftRegime("gradual", 28, 21)
    t0, t1 = theta(sour=0.2), theta(sour=0.7)
    for week in range(10):
        days = range(7 * week, 7 * week + 7)
        frac = np.mean([theta_at(d, t0, t1, reg, rng_for(r, d))[1] == 1
                        for d in days for r in range(400)])
        assert abs(frac - np.mean([ramp(d, reg) for d in days])) < 0.04, week


def test_incremental_and_recurring():
    t0, t1 = theta(slow=0.2), theta(slow=0.8)
    rng = np.random.default_rng(0)
    inc = DriftRegime("incremental", 10, 20)
    assert theta_at(20, t0, t1, inc, rng)[0]["slow"] == pytest.approx(0.5)
    assert theta_at(40, t0, t1, inc, rng)[0]["slow"] == pytest.approx(0.8)
    rec = DriftRegime("recurring", 10, period_days=7)
    assert [theta_at(d, t0, t1, rec, rng)[1] for d in (9, 10, 16, 17, 23, 24)] == [0, 1, 1, 0, 0, 1]


# ---- stage units -----------------------------------------------------------------------------

def test_logistics_take_put_wash():
    toks = [("cut", "tomato"), ("pour", "oil"), ("mix", "tomato")]
    tags = {"nouns": {"tomato": ["ingredient"], "oil": ["ingredient"], "pan": ["container"]}}
    out, prov = insert_logistics(toks + [("mix", "pan")], tags, 1.0, 1.0, np.random.default_rng(0))
    assert out.index(("take", "tomato")) < out.index(("cut", "tomato"))
    assert out.index(("put", "tomato")) > out.index(("mix", "tomato"))
    assert out.index(("wash", "pan")) < out.index(("put", "pan"))
    assert ("wash", "tomato") not in out and prov.count("core") == 4
    none, _ = insert_logistics(toks, tags, 0.0, 0.0, np.random.default_rng(0))
    assert [t for t in none if t[0] == "put"] == []
    assert Counter(none)[("take", "tomato")] == 1


def test_logistic_counts_do_not_depend_on_order():
    tags = {"nouns": {"a": ["ingredient"], "b": ["tool"]}}
    toks = [("cut", "a"), ("mix", "b"), ("cut", "a")]
    c1 = Counter(insert_logistics(toks, tags, 1.0, 1.0, np.random.default_rng(0))[0])
    c2 = Counter(insert_logistics(toks[::-1], tags, 1.0, 1.0, np.random.default_rng(0))[0])
    assert c1 == c2


def test_variants_apply_by_theta():
    exp = toy_assets().expansions["salad"]["3"]
    rng = np.random.default_rng(0)
    full, _ = apply_variants(exp, theta(sour=1.0, calorie_conscious=1.0), ASSETS.knobs, rng)
    assert full == [("mix", "lettuce"), ("pour", "vinegar")]
    none, used = apply_variants(exp, theta(sour=0.0, calorie_conscious=0.0), ASSETS.knobs, rng)
    assert none == exp.tokens and used == []


def test_duration_fit_backs_off():
    rng = np.random.default_rng(0)
    videos = {"v": []}
    t = 0.0
    for i in range(300):
        verb, noun = ("cut", "onion") if i % 3 else ("take", "cup" if i % 2 else "bowl")
        d = float(np.exp(rng.normal(1.5 if verb == "cut" else 0.2, 0.3)))
        videos["v"].append((t, t + d, verb, noun))
        t += d + (0.0 if i % 4 == 0 else 1.0)
    m = fit_from_intervals(videos)
    assert m.params(("cut", "onion"))[0] == pytest.approx(1.5, abs=0.1)
    assert "take|cup" in m.pair and m.params(("take", "jar")) == tuple(m.verb["take"][:2])
    assert m.params(("stir", "pot")) == tuple(m.overall[:2])
    assert m.gap_zero == pytest.approx(0.25, abs=0.02)


def test_export_ticks():
    events = [Event(0.0, 2.0, "cut", "tomato"), Event(3.0, 4.0, "mix", "egg")]
    v, n = to_ticks(events, VOCAB, ("stall", "idle"), 20, 0.5)
    cut, mix, stall = VOCAB["verbs"]["cut"], VOCAB["verbs"]["mix"], VOCAB["verbs"]["stall"]
    assert v == [cut] * 4 + [stall] * 2 + [mix] * 2
    assert n[4] == VOCAB["nouns"]["idle"]


def test_null_and_drift_conditions_share_pre_onset_days():
    null = ConditionCfg("null", n_days=20)
    drift = ConditionCfg("d", DriftRegime("sudden", 10), {"tidy": 0.4}, n_days=20)
    _, a, ga = generate_person(2, null, 0, ASSETS)
    _, b, gb = generate_person(2, drift, 0, ASSETS)
    pre = lambda ss: [(s.day, [(e.verb, e.noun, e.t_start) for e in s.events]) for s in ss if s.day < 10]
    assert pre(a) == pre(b)
    assert ga["theta0"] == gb["theta0"] and gb["delta"] == {"tidy": pytest.approx(
        gb["theta1"]["tidy"] - gb["theta0"]["tidy"])}


def test_conditions_file_expands_to_the_spec_set():
    import sys
    sys.path.insert(0, "scripts/drift_synth")
    from generate import conditions
    conds, seeds = conditions("configs/drift_synth/conditions.yaml")
    assert len(conds) == 31 and seeds == [0, 1, 2]   # 2 + 6x3 + 1 + 2x5
    assert conds["null"].regime.kind == "none" and conds["mix_shift_only"].mix_shift
    assert conds["slow_incremental"].regime.duration_days == 21
    assert conds["pref_plus_capability"].moving == {"sour": 0.4, "slow": 0.4}
