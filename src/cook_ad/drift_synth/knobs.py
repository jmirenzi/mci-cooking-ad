"""Stage 6: knobs. theta maps each knob to [0, 1]; what each knob does is declared in
configs/drift_synth/knobs.yaml and implemented here. Every random choice takes the caller's rng."""
import math
from pathlib import Path

import numpy as np
import yaml

from cook_ad.drift_synth.graphs import linear_extension_counter, parents

KNOBS = ("sour", "calorie_conscious", "mise_en_place", "tidy", "slow", "fine_motor")


def load_knobs(path):
    knobs = yaml.safe_load(Path(path).read_text())["knobs"]
    if set(knobs) != set(KNOBS):
        raise ValueError(f"{path}: knobs {sorted(knobs)} != {sorted(KNOBS)}")
    return knobs


def _lerp(lo_hi, theta):
    lo, hi = lo_hi
    return lo + (hi - lo) * theta


# ---- inclusion and substitution --------------------------------------------------------------

def apply_variants(exp, theta, knobs, rng):
    """One step's tokens after its knob variants. Each variant is drawn independently with
    P = theta[its knob] (only kinds the knob declares). Substitute/omit act on base-token indices;
    a variant whose indices another applied variant already took is skipped. Inserts go after
    the step's last token. Returns (tokens, [applied variant indices]).
    """
    replaced = {}      # base index -> replacement tokens ([] for omit)
    inserts = []
    applied = []
    for i, v in enumerate(exp.variants):
        draw = rng.random()  # always drawn, so later draws do not depend on which knobs moved
        knob = knobs.get(v.knob)
        if knob is None or v.kind not in knob.get("variant_kinds", ()):
            continue
        if draw >= theta[v.knob]:
            continue
        if v.kind == "insert":
            inserts.extend(v.tokens)
        else:
            idx = list(v.replaces or [])
            if any(j in replaced for j in idx):
                continue
            for j in idx:
                replaced[j] = []
            if v.kind == "substitute" and idx:
                replaced[idx[0]] = list(v.tokens)
        applied.append(i)
    out = []
    for j, tok in enumerate(exp.tokens):
        out.extend(replaced.get(j, [tok]))
    return out + inserts, applied


# ---- ordering --------------------------------------------------------------------------------

_COUNTERS = {}


def _ordering_tables(graph):
    """The completion counter, memoised per graph structure (it is the expensive part)."""
    key = (graph.recipe_id, tuple(sorted(graph.steps)), tuple(sorted(graph.edges)))
    if key not in _COUNTERS:
        _COUNTERS[key] = linear_extension_counter(graph)
    return _COUNTERS[key]


def sample_topological_order(graph, mise_en_place, beta, rng):
    """A topological order of `graph`. Each available step is weighted by the number of orders
    that complete it (so at mise_en_place = 0.5 the result is uniform over all topological
    orders) times exp(beta * (2 * mise_en_place - 1) * is_prep). Steps must carry phases."""
    index, count = _ordering_tables(graph)
    par = parents(graph)
    pos = {k: i for i, k in enumerate(index)}
    par_mask = [sum(1 << pos[p] for p in par[k]) for k in index]
    is_prep = [graph.steps[k].phase == "prep" for k in index]
    bias = beta * (2.0 * mise_en_place - 1.0)
    mask = (1 << len(index)) - 1
    order = []
    while mask:
        cands = [i for i in range(len(index)) if mask >> i & 1 and not par_mask[i] & mask]
        logw = np.array([math.log(count(mask & ~(1 << i))) + bias * is_prep[i] for i in cands])
        w = np.exp(logw - logw.max())
        i = cands[int(rng.choice(len(cands), p=w / w.sum()))]
        order.append(index[i])
        mask &= ~(1 << i)
    return order


# ---- duration scales and repetition ----------------------------------------------------------

def is_tagged(tags, kind, item, tag_list):
    return any(t in tags[kind].get(item, ()) for t in tag_list)


def slow_multiplier(theta, knobs):
    return math.exp(knobs["slow"]["a"] * (theta["slow"] - 0.5))


def token_multiplier(token, theta, knobs, tags):
    """Per-token duration multiplier from token-specific knobs (fine_motor, calorie_conscious)."""
    verb, noun = token
    m = 1.0
    fm = knobs["fine_motor"]
    if is_tagged(tags, "verbs", verb, fm["target_tags"]["verbs"]):
        m *= math.exp(fm["a"] * (theta["fine_motor"] - 0.5))
    cc = knobs["calorie_conscious"]
    if is_tagged(tags, "nouns", noun, cc["target_tags"]["nouns"]):
        m *= math.exp(-cc["fat_duration_k"] * theta["calorie_conscious"])
    return m


def repetition_prob(token, theta, knobs, tags):
    fm = knobs["fine_motor"]
    if not is_tagged(tags, "verbs", token[0], fm["target_tags"]["verbs"]):
        return 0.0
    return fm["repeat_base"] + theta["fine_motor"] * fm["repeat_max"]


def tidy_rates(theta, knobs):
    return _lerp(knobs["tidy"]["put_back"], theta["tidy"]), _lerp(knobs["tidy"]["wash"], theta["tidy"])

