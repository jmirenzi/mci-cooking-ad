"""Stage 6: knobs. theta maps each knob to [0, 1]; what each knob does is declared in
configs/drift_synth/knobs.yaml and implemented here. Every random choice takes the caller's rng."""
import math
from pathlib import Path

import numpy as np
import yaml

from cook_ad.drift_synth.graphs import linear_extension_counter, parents

KNOBS = ("sour", "sweet", "salty", "fatty", "mise_en_place", "tidy", "slow", "fine_motor")
FLAVOUR = ("sour", "sweet", "salty", "fatty")
INSERT_KINDS, REMOVE_KINDS = ("insert",), ("substitute", "omit")


def load_knobs(path):
    knobs = yaml.safe_load(Path(path).read_text())["knobs"]
    if set(knobs) != set(KNOBS):
        raise ValueError(f"{path}: knobs {sorted(knobs)} != {sorted(KNOBS)}")
    return knobs


def _lerp(lo_hi, theta):
    lo, hi = lo_hi
    return lo + (hi - lo) * theta


# ---- inclusion and substitution --------------------------------------------------------------

def variant_prob(knob, kind, theta, knobs, affinity=None):
    """P(apply one variant). A flavour knob is two-sided around 0.5: its inserts apply with
    2 (theta - 0.5) * affinity above it, its substitutes and omits with 2 (0.5 - theta) below it."""
    cfg = knobs.get(knob)
    if cfg is None or cfg.get("operator") != "flavour":
        return 0.0
    t = theta[knob]
    if kind in INSERT_KINDS:
        return 2.0 * max(0.0, t - 0.5) * (1.0 if affinity is None else affinity.get(knob, 1.0))
    if kind in REMOVE_KINDS:
        return 2.0 * max(0.0, 0.5 - t)
    return 0.0


def pick_inserts(expansions, theta, knobs, rng, affinity=None):
    """The session's flavour additions, decided once per recipe: for each flavour knob, with
    P = variant_prob(insert) the dish gets that flavour added, at ONE of its insert sites picked
    uniformly. So affinity is the chance a flavour-lover (theta = 1) changes this dish in a session,
    whatever the number of sites. -> {step_id: {variant index}}. Two draws per knob, always."""
    from cook_ad.drift_synth.graphs import _id_key
    sites = {}
    for sid in sorted(expansions, key=_id_key):
        for i, v in enumerate(expansions[sid].variants):
            if v.kind in INSERT_KINDS:
                sites.setdefault(v.knob, []).append((sid, i))
    chosen = {}
    for knob in FLAVOUR:
        u, pick = rng.random(), rng.random()
        opts = sites.get(knob, [])
        if opts and u < variant_prob(knob, "insert", theta, knobs, affinity):
            sid, i = opts[int(pick * len(opts))]
            chosen.setdefault(sid, set()).add(i)
    return chosen


def apply_variants(exp, theta, knobs, rng, affinity=None, chosen_inserts=None):
    """One step's tokens after its knob variants. Each substitute/omit is drawn independently
    with P = variant_prob. Inserts: the ones in `chosen_inserts` (this step's share of
    pick_inserts); None = each insert drawn on its own with P = variant_prob (affinity: this
    recipe's {knob: affinity}, None = 1). Substitute/omit act on base-token indices; a variant
    whose indices another applied variant already took is skipped. Inserts go after the step's
    last token. Returns (tokens, [applied variant indices]).
    """
    replaced = {}      # base index -> replacement tokens ([] for omit)
    inserts = []
    applied = []
    for i, v in enumerate(exp.variants):
        draw = rng.random()  # always drawn, so later draws do not depend on which knobs moved
        if v.kind in INSERT_KINDS and chosen_inserts is not None:
            if i not in chosen_inserts:
                continue
        elif draw >= variant_prob(v.knob, v.kind, theta, knobs, affinity):
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


def variant_layout(exp, used):
    """The step's tokens as apply_variants built them from the applied variants `used`, with
    provenance: [(kind, token, knob)], kind core / sub / ins / removed (removed tokens are not
    emitted; they are kept for the record)."""
    replaced, inserts = {}, []
    for i in used:
        v = exp.variants[i]
        if v.kind in INSERT_KINDS:
            inserts += [("ins", t, v.knob) for t in v.tokens]
            continue
        idx = list(v.replaces or [])
        for j in idx:
            replaced[j] = [("removed", exp.tokens[j], v.knob)]
        if v.kind == "substitute" and idx:
            replaced[idx[0]] += [("sub", t, v.knob) for t in v.tokens]
    out = []
    for j, tok in enumerate(exp.tokens):
        out += replaced.get(j, [("core", tok, None)])
    return out + inserts


def space_inserts(blocks):
    """blocks: [[step, [items], [insert items]]] in cooking order (items as variant_layout).
    An addition next to the same action (more salt right after, or right before, the recipe's own
    salt) would show as one action twice. It moves to the end of the nearest step, later ones
    first, where neither neighbour is the same action: a second addition, later in the cook. One
    with nowhere to go stays (the export then merges the pair)."""
    def seq():
        return [(b, kind, x) for b, (_, core, ins) in enumerate(blocks)
                for kind, x in [("core", x) for x in core] + [("ins", x) for x in ins]]

    def clash_at_end(tok, t):
        f = seq()
        upto = [i for i, (b, _, _) in enumerate(f) if b <= t]
        before = f[upto[-1]][2][1] if upto else None
        nxt = upto[-1] + 1 if upto else 0
        after = f[nxt][2][1] if nxt < len(f) else None
        return tok in (before, after)

    stuck = 0
    for _ in range(4 * sum(len(ins) for _, _, ins in blocks)):
        f = seq()
        bad = [i for i, (_, kind, x) in enumerate(f) if kind == "ins" and
               ((i > 0 and f[i - 1][2][1] == x[1]) or (i + 1 < len(f) and f[i + 1][2][1] == x[1]))]
        if len(bad) <= stuck:
            break
        b, _, item = f[bad[stuck]]
        blocks[b][2].remove(item)
        cands = list(range(b + 1, len(blocks))) + list(range(b - 1, -1, -1))
        target = next((t for t in cands if not clash_at_end(item[1], t)), None)
        if target is None:
            blocks[b][2].append(item)
            stuck += 1
        else:
            blocks[target][2].append(item)
    return blocks


# ---- ordering --------------------------------------------------------------------------------

_COUNTERS = {}


def _ordering_tables(graph):
    """The completion counter, memoised per graph structure (it is the expensive part)."""
    key = (graph.recipe_id, tuple(sorted(graph.steps)), tuple(sorted(graph.edges)))
    if key not in _COUNTERS:
        _COUNTERS[key] = linear_extension_counter(graph)
    return _COUNTERS[key]


def sample_topological_order(graph, mise_en_place, beta, rng, preferred=None, habit=0.0):
    """A topological order of `graph`. Each available step is weighted by the number of orders
    that complete it (so at mise_en_place = 0.5 and habit = 0 the result is uniform over all
    topological orders) times exp(beta * (2 * mise_en_place - 1) * is_prep). Steps must carry
    phases.

    `preferred` (a topological order) and `habit` >= 0 model a person's routine: a candidate
    also gets exp(-habit * r), r = its rank among the available steps by position in
    `preferred`. That is a sequential Mallows-style concentration around the habitual order;
    habit = 0 leaves the draw, and the rng stream, exactly as without it."""
    index, count = _ordering_tables(graph)
    par = parents(graph)
    pos = {k: i for i, k in enumerate(index)}
    par_mask = [sum(1 << pos[p] for p in par[k]) for k in index]
    is_prep = [graph.steps[k].phase == "prep" for k in index]
    bias = beta * (2.0 * mise_en_place - 1.0)
    pref_pos = {k: j for j, k in enumerate(preferred)} if preferred and habit else None
    mask = (1 << len(index)) - 1
    order = []
    while mask:
        cands = [i for i in range(len(index)) if mask >> i & 1 and not par_mask[i] & mask]
        logw = np.array([math.log(count(mask & ~(1 << i))) + bias * is_prep[i] for i in cands])
        if pref_pos is not None:
            ranks = np.argsort(np.argsort([pref_pos[index[i]] for i in cands]))
            logw = logw - habit * ranks
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
    """Per-token duration multiplier from token-specific knobs (fine_motor, fatty)."""
    verb, noun = token
    m = 1.0
    fm = knobs["fine_motor"]
    if is_tagged(tags, "verbs", verb, fm["target_tags"]["verbs"]):
        m *= math.exp(fm["a"] * (theta["fine_motor"] - 0.5))
    ft = knobs["fatty"]
    if is_tagged(tags, "nouns", noun, ft["target_tags"]["nouns"]):
        m *= math.exp(ft["fat_duration_k"] * (theta["fatty"] - 0.5))
    return m


def repetition_prob(token, theta, knobs, tags):
    """base + max * max(0, theta - hinge) for a fine-motor token, else 0."""
    fm = knobs["fine_motor"]
    if not is_tagged(tags, "verbs", token[0], fm["target_tags"]["verbs"]):
        return 0.0
    return fm["repeat_base"] + fm["repeat_max"] * max(0.0, theta["fine_motor"] - fm["hinge"])


def keep_side_of_hinge(theta_session, theta_base, knobs):
    """Session noise may not carry a hinged knob across its hinge: a base theta at or below the
    hinge keeps the session value at or below it (and above stays above), so a healthy person
    never repeats in a noisy session."""
    out = dict(theta_session)
    for k, cfg in knobs.items():
        h = cfg.get("hinge")
        if h is not None and k in out:
            out[k] = min(out[k], h) if theta_base[k] <= h else max(out[k], h)
    return out


def tidy_rates(theta, knobs):
    return _lerp(knobs["tidy"]["put_back"], theta["tidy"]), _lerp(knobs["tidy"]["wash"], theta["tidy"])

