"""The synthetic drift generator on the toy world (tests/_drift_fixtures.py): the spec's nine
tests (docs/drift_synth.md §7) plus unit checks on each stage."""
import filecmp
import math
from collections import Counter

import numpy as np
import pytest
from scipy.stats import chisquare, ks_2samp, norm

from _drift_fixtures import OMELET, SALAD, TAGS, VOCAB, V, theta, toy_assets
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
    cond = ConditionCfg("c", DriftRegime("sudden", 10), {"sour": 0.4, "fatty": 0.4},
                        n_persons=3, n_sessions=21)
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
    cond = ConditionCfg("c", DriftRegime("gradual", 5, 7), {"slow": 0.3}, n_sessions=14)
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
    if knob == "fatty":
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
    if knob == "fatty":
        return {t for t in tokens if "fat" in TAGS["nouns"].get(t.split(" ")[1], [])}
    return set()


@pytest.mark.parametrize("knob", ["sour", "sweet", "salty", "fatty", "mise_en_place", "tidy",
                                  "slow", "fine_motor"])
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
    assert GT.count_z(base, up["tidy"])["put oil"] > 5
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
    rec = DriftRegime("recurring", 10, period=7)
    assert [theta_at(d, t0, t1, rec, rng)[1] for d in (9, 10, 16, 17, 23, 24)] == [0, 1, 1, 0, 0, 1]


# ---- stage units -----------------------------------------------------------------------------

def test_logistics_take_put_wash():
    toks = [("cut", "egg"), ("pour", "oil"), ("mix", "egg")]
    tags = {"nouns": {"egg": ["ingredient"], "oil": ["ingredient"], "pan": ["container"]}}
    out, prov = insert_logistics(toks + [("mix", "pan")], tags, 1.0, 1.0, np.random.default_rng(0))
    assert out.index(("take", "egg")) < out.index(("cut", "egg"))
    assert out.index(("put", "egg")) > out.index(("mix", "egg"))
    assert out.index(("wash", "pan")) < out.index(("put", "pan"))
    assert ("wash", "egg") not in out and prov.count("core") == 4
    none, _ = insert_logistics(toks, tags, 0.0, 0.0, np.random.default_rng(0))
    assert [t for t in none if t[0] == "put"] == []
    assert Counter(none)[("take", "egg")] == 1


def test_a_vessel_is_in_use_for_its_steps_not_just_where_a_token_names_it():
    tags = {"nouns": {"pan": ["container"], "garlic": ["ingredient"], "plate": ["container"]}}
    toks = [("put", "pan"), ("wait", "pan"), ("add", "garlic"), ("cook", "garlic"), ("put", "plate")]
    in_use = [["pan"], ["pan"], ["pan"], ["pan"], ["plate"]]
    rng = lambda: np.random.default_rng(0)
    named, _ = insert_logistics(toks, tags, 1.0, 1.0, rng())
    assert named.index(("wash", "pan")) < named.index(("add", "garlic"))     # named uses only
    held, _ = insert_logistics(toks, tags, 1.0, 1.0, rng(), in_use=in_use)
    assert held.index(("wash", "pan")) > held.index(("cook", "garlic"))
    first, _ = insert_logistics([("add", "garlic")], tags, 0.0, 0.0, rng(), in_use=[["pan"]])
    assert first == [("take", "garlic"), ("take", "pan"), ("add", "garlic")]   # fetched before use
    served, _ = insert_logistics(toks, tags, 1.0, 1.0, rng(), in_use=in_use, serves_in="pan")
    assert ("wash", "pan") not in served and ("put", "pan") == toks[0]
    assert [t for t in served if t == ("put", "pan")] == [("put", "pan")]     # only the recipe's own


def test_consumed_and_made_items_are_not_put_back_and_water_is_not_fetched():
    tags = {"nouns": {"garlic": ["ingredient"], "noodle": ["ingredient"], "water": ["ingredient"],
                      "salt": ["ingredient"]}}
    toks = [("peel", "garlic"), ("add", "noodle"), ("add", "water"), ("add", "salt")]
    out, _ = insert_logistics(toks, tags, 1.0, 1.0, np.random.default_rng(0))
    assert ("take", "garlic") in out and ("put", "garlic") not in out      # consumed
    assert not any(n in ("noodle", "water") for v, n in out if v in ("take", "put"))
    assert ("take", "salt") in out and ("put", "salt") in out              # stock


def test_logistic_counts_do_not_depend_on_order():
    tags = {"nouns": {"a": ["ingredient"], "b": ["tool"]}}
    toks = [("cut", "a"), ("mix", "b"), ("cut", "a")]
    c1 = Counter(insert_logistics(toks, tags, 1.0, 1.0, np.random.default_rng(0))[0])
    c2 = Counter(insert_logistics(toks[::-1], tags, 1.0, 1.0, np.random.default_rng(0))[0])
    assert c1 == c2


def test_flavour_knobs_are_two_sided_around_half():
    exp = toy_assets().expansions["salad"]["3"]   # sour insert (vinegar), fatty omit (oil)
    rng = np.random.default_rng(0)
    full, _ = apply_variants(exp, theta(sour=1.0, fatty=0.0), ASSETS.knobs, rng)
    assert full == [("mix", "lettuce"), ("pour", "vinegar")]
    for th in (theta(), theta(sour=0.0, fatty=1.0)):   # as written: 0.5, or the other two sides
        same, used = apply_variants(exp, th, ASSETS.knobs, rng)
        assert same == exp.tokens and used == []


def test_affinity_scales_inserts_only():
    exp = toy_assets().expansions["salad"]["3"]
    n = 4000

    def rate(th, aff, tok):
        rng = np.random.default_rng(1)
        return sum(tok in apply_variants(exp, th, ASSETS.knobs, rng, aff)[0] for _ in range(n)) / n
    assert rate(theta(sour=1.0), {"sour": 0.1}, ("pour", "vinegar")) == pytest.approx(0.1, abs=0.02)
    assert rate(theta(sour=0.75), {"sour": 1.0}, ("pour", "vinegar")) == pytest.approx(0.5, abs=0.03)
    assert rate(theta(fatty=0.0), {"fatty": 0.1}, ("pour", "oil")) == 0.0   # the omit ignores it


def test_flavour_insert_is_one_draw_per_recipe():
    """Two sour sites: the dish gets sour at most once a session, with P = 2 (theta - 0.5) * affinity."""
    from cook_ad.drift_synth.knobs import pick_inserts
    from cook_ad.drift_synth.types import StepExpansion
    exps = {"1": StepExpansion("1", [("cut", "tomato")], [V("sour", "insert", "1", [("add", "lemon")], None)]),
            "2": StepExpansion("2", [("mix", "tomato")], [V("sour", "insert", "2", [("add", "vinegar")], None)])}
    rng = np.random.default_rng(0)
    picks = [pick_inserts(exps, theta(sour=0.75), ASSETS.knobs, rng, {"sour": 0.4}) for _ in range(5000)]
    n_sites = [sum(len(v) for v in p.values()) for p in picks]
    assert max(n_sites) == 1
    assert np.mean(n_sites) == pytest.approx(0.2, abs=0.02)
    assert np.mean([("1" in p) for p in picks if p]) == pytest.approx(0.5, abs=0.05)
    assert pick_inserts(exps, theta(sour=0.4), ASSETS.knobs, rng, {"sour": 1.0}) == {}


def test_healthy_fine_motor_never_repeats():
    from cook_ad.drift_synth.knobs import keep_side_of_hinge, repetition_prob
    kn = ASSETS.knobs
    assert repetition_prob(("cut", "tomato"), theta(fine_motor=0.5), kn, TAGS) == 0.0
    assert repetition_prob(("cut", "tomato"), theta(fine_motor=0.9), kn, TAGS) == pytest.approx(0.24)
    base = theta(fine_motor=0.45)
    assert keep_side_of_hinge(theta(fine_motor=0.7), base, kn)["fine_motor"] == 0.5
    assert keep_side_of_hinge(theta(fine_motor=0.3), theta(fine_motor=0.8), kn)["fine_motor"] == 0.5
    a = toy_assets()
    a.gen = dict(a.gen, person=dict(a.gen["person"], theta0_by_knob={"fine_motor": [0.3, 0.5]}),
                 session_noise=dict(a.gen["session_noise"],
                                    sigma=dict(a.gen["session_noise"]["sigma"], fine_motor=2.0)))
    for i in range(4):
        _, sessions, _ = generate_person(i, ConditionCfg("null", n_sessions=40), 0, a)
        assert sum(s.meta["repetitions"] for s in sessions) == 0


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


def test_export_ticks():
    events = [Event(0.0, 2.0, "cut", "tomato"), Event(3.0, 4.0, "mix", "egg")]
    v, n = to_ticks(events, VOCAB, ("stall", "idle"), 20, 0.5)
    cut, mix, stall = VOCAB["verbs"]["cut"], VOCAB["verbs"]["mix"], VOCAB["verbs"]["stall"]
    assert v == [cut] * 4 + [stall] * 2 + [mix] * 2
    assert n[4] == VOCAB["nouns"]["idle"]


def test_null_and_drift_conditions_share_pre_onset_sessions():
    null = ConditionCfg("null", n_sessions=20)
    drift = ConditionCfg("d", DriftRegime("sudden", 10), {"tidy": 0.4}, n_sessions=20)
    _, a, ga = generate_person(2, null, 0, ASSETS)
    _, b, gb = generate_person(2, drift, 0, ASSETS)
    def pre(ss):
        return [(s.index, [(e.verb, e.noun, e.t_start) for e in s.events]) for s in ss if s.index < 10]
    assert pre(a) == pre(b)
    assert ga["theta0"] == gb["theta0"] and gb["delta"] == {"tidy": pytest.approx(
        gb["theta1"]["tidy"] - gb["theta0"]["tidy"])}


def test_conditions_file_expands_to_the_spec_set():
    import sys
    sys.path.insert(0, "scripts/drift_synth")
    from generate import conditions
    conds, seeds = conditions("configs/drift_synth/conditions.yaml")
    assert len(conds) == 37 and seeds == [0, 1, 2]   # 2 + 8x3 + 1 + 2x5
    assert conds["null"].regime.kind == "none" and conds["mix_shift_only"].mix_shift
    assert conds["slow_incremental"].regime.duration == 27
    assert conds["pref_plus_capability"].moving == {"sour": 0.4, "slow": 0.4}


def test_step_labels_and_repeats_are_logged():
    from cook_ad.drift_synth.export import step_labels
    from cook_ad.drift_synth.noise import NoiseCfg
    from cook_ad.drift_synth.sampler import sample_session
    s = sample_session(OMELET, ASSETS.expansions["omelet"], theta(fine_motor=1.0), 1.0, ASSETS,
                       NoiseCfg().off(), rng_for(5, 1))
    clean = s.meta["clean_events"]
    assert s.meta["repetitions"] == len(s.meta["repeats"]) > 0
    for j in s.meta["repeats"]:      # a repeat duplicates the token just before it, after a pause
        assert clean[j][2:4] == clean[j - 1][2:4]
        assert clean[j][0] - clean[j - 1][1] >= ASSETS.knobs["fine_motor"]["repeat_gap_s"] - 1e-6
    v, _ = to_ticks(s.events, VOCAB, ("stall", "idle"), 20, 0.5)
    labels = step_labels(s, len(v), 0.5)
    assert len(labels) == len(v) and set(labels) <= {0, 1, 2, 3, 4, 5, 6}
    assert {1, 2, 3, 4, 5, 6} <= set(labels)    # every step's core tokens are labelled somewhere


def test_habit_concentrates_orders_and_zero_is_a_no_op():
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    pref = ["5", "3", "1", "2", "4", "6"]
    for _ in range(50):     # habit 0: identical draws to the plain sampler
        assert (sample_topological_order(OMELET, 0.5, 4.0, rng_a, pref, 0.0)
                == sample_topological_order(OMELET, 0.5, 4.0, rng_b))
    rng = np.random.default_rng(0)
    free = Counter(tuple(sample_topological_order(OMELET, 0.5, 4.0, rng)) for _ in range(500))
    tight = Counter(tuple(sample_topological_order(OMELET, 0.5, 4.0, rng, pref, 3.0))
                    for _ in range(500))
    assert tight[tuple(pref)] > 250 and free[tuple(pref)] < 100
    assert all(is_topological(OMELET, list(o)) for o in tight)


def test_per_knob_ranges_and_habit_range():
    from cook_ad.drift_synth.sampler import make_person
    a = toy_assets()
    base = make_person(0, 1, a)
    a.gen = dict(a.gen, person=dict(a.gen["person"], theta0_by_knob={"slow": [0.4, 0.6]},
                                    habit=[0.0, 1.0]))
    p = make_person(0, 1, a)
    assert 0.4 <= p.theta0["slow"] <= 0.6 and 0.0 <= p.habit <= 1.0
    assert p.repertoire == base.repertoire and p.habit_orders == base.habit_orders
    assert {k: v for k, v in p.theta0.items() if k != "slow"} == \
        {k: v for k, v in base.theta0.items() if k != "slow"}


def _salad_times(mean_s):
    """StepTimes with salad step 2 (one action, `cut lettuce`) at `mean_s` seconds on average."""
    from cook_ad.drift_synth.durations import StepTimes
    sigma = 0.2
    return StepTimes({"salad": {"2": [math.log(mean_s) - sigma ** 2 / 2, 5]}}, sigma)


def test_step_totals_come_from_cc4d_and_logistics_from_epic():
    from cook_ad.drift_synth.noise import NoiseCfg
    from cook_ad.drift_synth.sampler import sample_session
    a = toy_assets()
    a.step_times = _salad_times(60.0)
    core2, logi = [], []
    for i in range(300):
        s = sample_session(SALAD, a.expansions["salad"], theta(), 1.0, a, NoiseCfg().off(), rng_for(1, i))
        ev = s.meta["clean_events"]
        core2.append(sum(e[1] - e[0] for e in ev if e[4] == "2"))
        logi += [e[1] - e[0] for e in ev if e[4] is None and e[2] == "take"]
        # no gaps: every event starts where the previous one ended
        assert all(abs(b[0] - x[1]) < 2e-3 for x, b in zip(ev, ev[1:]))
    assert np.mean(core2) == pytest.approx(60.0, rel=0.1)      # CC4D's step total
    assert np.median(logi) == pytest.approx(2.0, rel=0.2)      # EPIC overall row (toy): 2 s


def test_an_added_action_adds_its_own_share():
    """A flavour insert takes time on top of the step's CC4D total; it does not shrink the rest."""
    from cook_ad.drift_synth.noise import NoiseCfg
    from cook_ad.drift_synth.sampler import sample_session
    a = toy_assets()
    sp = lambda th: [sum(e[1] - e[0] for e in sample_session(  # noqa: E731
        SALAD, a.expansions["salad"], th, 1.0, a, NoiseCfg().off(), rng_for(2, i)).meta["clean_events"]
        if e[4] == "3" and e[3] != "vinegar") for i in range(300)]
    plain, sour = sp(theta()), sp(theta(sour=1.0))
    assert np.mean(sour) == pytest.approx(np.mean(plain), rel=0.1)


def test_an_action_is_capped_and_a_written_timer_ignores_slow():
    """Salad step 2 (`cut lettuce`, EPIC 4 s) at 60 s in CC4D: capped at 3 x EPIC. Step 3 says
    "for 2 minutes": its last timer-verb token lasts 120 s, at any slow."""
    from cook_ad.drift_synth.durations import mean_duration
    from cook_ad.drift_synth.noise import NoiseCfg
    from cook_ad.drift_synth.sampler import sample_session
    a = toy_assets()
    a.step_times = _salad_times(60.0)
    a.gen = dict(a.gen, durations={"active_unit_cap": 3, "timer_verbs": ["mix"], "timer_sigma": 0.0})
    timed = RecipeGraph("salad", dict(SALAD.steps, **{"3": Step("3", "Dress-dress for 2 minutes", "serve")}),
                        SALAD.edges)

    def per_step(th, step, verb):
        return [sum(e[1] - e[0] for e in sample_session(
            timed, a.expansions["salad"], th, 1.0, a, NoiseCfg().off(), rng_for(3, i)).meta["clean_events"]
            if e[4] == step and e[2] == verb) for i in range(300)]
    assert np.mean(per_step(theta(), "2", "cut")) == pytest.approx(
        3 * mean_duration(a.durations, ("cut", "lettuce")), rel=0.1)
    assert per_step(theta(slow=0.0), "3", "mix") == pytest.approx([120.0] * 300)
    assert per_step(theta(slow=1.0), "3", "mix") == pytest.approx([120.0] * 300)


@pytest.mark.parametrize("text, seconds", [
    ("Microwave-Microwave for 3 minutes, stirring in between", 180),
    ("Microwave-Microwave the corn for 3 more minutes", 180),
    ("cook-cook for 20-30 seconds more", 25),
    ("Cook-Cook for 1 min or until the tops start to bubble", 60),
    ("Melt-Melt the cheese by microwaving cup for 30 sec. (Check after 30 seconds and microwave "
     "for 10 seconds more if needed)", 30),
    ("Cook-Cook garlic until fragrant (about 1 minutes). Be careful not to burn garlic", 60),
    ("Add-Add 1 large minced garlic cloves to the pan", None),
    ("Add-1/2 tsp baking powder to a blender", None),
])
def test_written_seconds(text, seconds):
    from cook_ad.drift_synth.durations import written_seconds
    assert written_seconds(text) == seconds


def test_sessions_are_the_only_clock():
    cond = ConditionCfg("c", n_persons=1, n_sessions=12)
    _, sessions, gt = generate_person(0, cond, 0, ASSETS)
    assert [s.index for s in sessions] == list(range(12))
    assert [x["session"] for x in gt["sessions"]] == list(range(12))
    from cook_ad.drift_synth.export import trial_id
    assert trial_id(sessions[3]) == "p00_s0003"


def test_no_put_back_right_after_the_recipes_own_put_and_no_fetching_dishes():
    tags = {"nouns": {"bowl": ["container"], "pancake": ["ingredient"], "egg": ["ingredient"]}}
    toks = [("mix", "egg"), ("put", "bowl"), ("cook", "pancake")]
    out, _ = insert_logistics(toks, tags, 1.0, 0.0, np.random.default_rng(0))
    assert Counter(out)[("put", "bowl")] == 1                         # the recipe's own
    assert not any(n == "pancake" for v, n in out if v in ("take", "put"))
    assert ("take", "egg") in out and ("put", "egg") in out
    washed, _ = insert_logistics(toks, tags, 1.0, 1.0, np.random.default_rng(0))
    i = washed.index(("put", "bowl"))       # washed in between, so putting it away is kept
    assert washed[i:i + 3] == [("put", "bowl"), ("wash", "bowl"), ("put", "bowl")]


def test_space_inserts_moves_a_same_ingredient_addition_off_its_twin():
    from cook_ad.drift_synth.knobs import space_inserts
    salt, lime = ("add", "salt"), ("add", "lime")
    blocks = [["1", [("core", ("cut", "tomato"), None)], []],
              ["2", [("core", salt, None)], [("ins", salt, "salty")]],
              ["3", [("core", lime, None)], [("ins", lime, "sour")]],
              ["4", [("core", ("mix", "tomato"), None)], []]]
    space_inserts(blocks)
    flat = [x[1] for _, core, ins in blocks for x in core + ins]
    assert all(a != b for a, b in zip(flat, flat[1:])), flat
    assert flat.count(salt) == 2 and flat.count(lime) == 2


def test_back_to_back_identical_actions_are_one_event():
    from cook_ad.drift_synth.sampler import sample_session
    for seed in range(5):
        s = sample_session(OMELET, ASSETS.expansions["omelet"], theta(), 1.0, ASSETS,
                           NoiseCfg().off(), rng_for(seed, 2))
        ev = s.meta["clean_events"]
        assert all(a[2:4] != b[2:4] for a, b in zip(ev, ev[1:]))
    assert s.meta["merged"] >= 1     # omelet step 2 is "cut tomato, cut tomato"


def test_narrowed_keeps_the_mean():
    from cook_ad.drift_synth.durations import narrowed
    mu, sigma = narrowed(1.0, 0.8, 0.5)
    assert sigma == pytest.approx(0.4)
    assert math.exp(mu + sigma ** 2 / 2) == pytest.approx(math.exp(1.0 + 0.8 ** 2 / 2))
