"""Stage 9: one session, one person.

Time is the session index n. Random streams: a person's traits come from (seed, person); session
n's state (theta noise, speed, recipe) from (seed, person, n), tag _STATE; its content from
(seed, person, n), tag _SESSION. None includes the condition, so two conditions with the same
seed share everything before onset (common random numbers), and a drift condition differs from
`null` only through theta.
"""
from dataclasses import dataclass, field

import math

import numpy as np

from cook_ad.drift_synth import knobs as K
from cook_ad.drift_synth.durations import mean_duration, sample_duration, sample_step_total
from cook_ad.drift_synth.kitchen_state import insert_logistics
from cook_ad.drift_synth.noise import NoiseCfg, recognizer_noise, speed_factor, theta_for_session
from cook_ad.drift_synth.schedule import DriftRegime, MixShift, PersonCfg, recipe_for_session, theta_at
from cook_ad.drift_synth.types import Event, Session

_PERSON, _STATE, _SESSION, _MIX = 1, 2, 3, 4  # stream tags


def rng_for(seed, *keys):
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, keys)]))


@dataclass(frozen=True)
class ConditionCfg:
    name: str
    regime: DriftRegime = DriftRegime()
    moving: dict = field(default_factory=dict)   # knob -> |delta theta|
    mix_shift: bool = False
    n_persons: int = 8
    n_sessions: int = 90


def step_owners(steps):
    """Per event, the recipe step its time belongs to: a core token's own step; a logistics token
    (None) goes to the nearest core step before it, else the first one after it."""
    out, last = [], None
    for k, st in enumerate(steps):
        if st is not None:
            last = st
        out.append(last if last is not None else next((x for x in steps[k:] if x is not None), None))
    return out


def step_units(recipe, expansions, times, durations):
    """{step: seconds per unit of expected EPIC time}: the step's CC4D total over its written core
    actions' expected EPIC durations. A step CC4D never recorded gets the recipe's median unit
    (1.0 with no CC4D data at all)."""
    units = {}
    for s, exp in expansions.items():
        p = times.params(recipe.recipe_id, s) if times is not None else None
        base = sum(mean_duration(durations, t) for t in exp.tokens)
        if p is not None and base > 0:
            units[s] = math.exp(p[0] + 0.5 * p[1] ** 2) / base
    fallback = float(np.median(list(units.values()))) if units else 1.0
    return units, fallback


def sample_session(recipe, expansions, theta, speed, assets, noise, rng,
                   person_id="", index=0, preferred=None, habit=0.0):
    """The spec's order: step order, variants, logistics, repetition, durations, noise."""
    kn, tags, gen = assets.knobs, assets.tags, assets.gen
    affinity = (getattr(assets, "affinity", None) or {}).get(recipe.recipe_id)
    order = K.sample_topological_order(recipe, theta["mise_en_place"], kn["mise_en_place"]["beta"],
                                       rng, preferred, habit)

    chosen = K.pick_inserts(expansions, theta, kn, rng, affinity)
    blocks, applied, skipped, removed = [], {}, [], []
    for s in order:
        exp = expansions.get(s)
        if exp is None:
            skipped.append(s)
            continue
        _, used = K.apply_variants(exp, theta, kn, rng, affinity, chosen.get(s, set()))
        items = K.variant_layout(exp, used)
        removed += [[s, t[0], t[1], knob] for kind, t, knob in items if kind == "removed"]
        blocks.append([s, [x for x in items if x[0] not in ("ins", "removed")],
                       [x for x in items if x[0] == "ins"]])
        if used:
            applied[s] = used
    K.space_inserts(blocks)
    tokens, steps, kinds = [], [], []
    for s, core, ins in blocks:
        for kind, tok, knob in core + ins:
            tokens.append(tok)
            steps.append(s)
            kinds.append("core" if kind == "core" else f"+{knob}")

    put_p, wash_p = K.tidy_rates(theta, kn)
    with_log, prov = insert_logistics(tokens, tags, put_p, wash_p, rng, gen["logistics_verbs"])
    it, ik = iter(steps), iter(kinds)
    step_of = [next(it) if p == "core" else None for p in prov]
    kind_of = [next(ik) if p == "core" else "logistics" for p in prov]
    # Two identical actions back to back (two sauces that are both EPIC "sauce") are one run to a
    # camera: emit one event. Repetitions are added after this, on purpose.
    keep = [i for i in range(len(with_log)) if i == 0 or with_log[i] != with_log[i - 1]]
    merged = len(with_log) - len(keep)
    with_log = [with_log[i] for i in keep]
    step_of = [step_of[i] for i in keep]
    kind_of = [kind_of[i] for i in keep]

    final, final_step, final_kind = [], [], []
    repeats = []   # clean-event indices that are fine_motor repetitions (ground-truth spans)
    for tok, s, kd in zip(with_log, step_of, kind_of):
        u = rng.random()
        final.append(tok)
        final_step.append(s)
        final_kind.append(kd)
        if u < K.repetition_prob(tok, theta, kn, tags):
            final.append(tok)
            final_step.append(s)
            final_kind.append("repeat")
            repeats.append(len(final) - 1)

    # Durations (durations.py). A step's total is drawn from CC4D; each of its actions gets
    # (its EPIC draw) x (the step's total / its written actions' expected EPIC time), so an added
    # or removed action adds or removes its own share. Take / put / wash keep EPIC time. Then the
    # person's multipliers. No gaps: the model reads idle-free streams. A fine-motor repeat follows
    # a pause of repeat_gap_s, so it shows as a second run before scrubbing.
    slow = K.slow_multiplier(theta, kn)
    cfg = gen.get("durations") or {}
    cap = cfg.get("max_event_s")                  # None = uncapped
    k_sigma = cfg.get("sigma_scale", 1.0)
    times = getattr(assets, "step_times", None)
    units, fallback = step_units(recipe, expansions, times, assets.durations)
    step_total = {}                               # step -> this session's CC4D draw / expected
    rep_next = {i - 1 for i in repeats}           # events followed by their own repeat
    rep_gap = kn["fine_motor"].get("repeat_gap_s", 0.0)
    t = 0.0
    clean = []
    for i, tok in enumerate(final):
        st = final_step[i]
        x = sample_duration(assets.durations, tok, 1.0, rng, k_sigma)
        if st is None:
            m = 1.0
        else:
            if st not in step_total:
                p = times.params(recipe.recipe_id, st) if times is not None else None
                draw = sample_step_total(times, recipe.recipe_id, st, rng, k_sigma)
                step_total[st] = 1.0 if draw is None else draw / math.exp(p[0] + 0.5 * p[1] ** 2)
            m = units.get(st, fallback) * step_total[st]
        d = x * m * speed * slow * K.token_multiplier(tok, theta, kn, tags)
        d = min(d, cap) if cap else d
        clean.append(Event(round(t, 3), round(t + d, 3), tok[0], tok[1]))
        t += d + (rep_gap if i in rep_next else 0.0)

    pool = ([e.verb for e in clean], [e.noun for e in clean])
    events, counts = recognizer_noise(clean, noise, pool, rng)
    return Session(
        person_id=person_id, index=index, recipe_id=recipe.recipe_id,
        events=events, theta=dict(theta),
        meta={"order": order, "variants_applied": applied, "skipped_steps": skipped,
              "repetitions": len(repeats), "repeats": repeats, "speed": round(speed, 6), "noise": counts,
              "removed": removed, "merged": merged, "kinds": final_kind,
              "clean_events": [[e.t_start, e.t_end, e.verb, e.noun, s] for e, s in zip(clean, final_step)]},
    )


def make_person(seed, idx, assets):
    """A person's traits, independent of the condition."""
    rng = rng_for(seed, _PERSON, idx)
    p = assets.gen["person"]
    recipes = sorted(assets.graphs)
    lo, hi = p["repertoire"]
    n = int(rng.integers(min(lo, len(recipes)), min(hi, len(recipes)) + 1))
    rep = tuple(recipes[i] for i in sorted(rng.choice(len(recipes), size=n, replace=False)))
    w = rng.dirichlet(np.full(n, p["dirichlet_alpha"]))
    ranges = {k: p.get("theta0_by_knob", {}).get(k, p["theta0"]) for k in K.KNOBS}
    theta0 = {k: float(rng.uniform(*ranges[k])) for k in K.KNOBS}   # one draw per knob, always
    # Drawn after every other trait, so adding them changed none of the earlier draws.
    habit_orders = {r: K.sample_topological_order(assets.graphs[r], 0.5, 0.0, rng) for r in rep}
    habit = p.get("habit", 0.0)
    habit = float(rng.uniform(*habit)) if isinstance(habit, (list, tuple)) else float(habit)
    return PersonCfg(f"p{idx:02d}", rep, tuple(float(x) for x in w), theta0, habit, habit_orders)


def theta1_for(theta0, moving):
    """theta0 moved by |delta| on each moving knob: up if that fits in [0, 1], else down."""
    out = dict(theta0)
    for k, delta in sorted(moving.items()):
        out[k] = theta0[k] + delta if theta0[k] + delta <= 1.0 else theta0[k] - delta
        if not 0.0 <= out[k] <= 1.0:
            raise ValueError(f"delta {delta} on {k} does not fit theta0 {theta0[k]:.3f}")
    return out


def mix_shift_for(seed, idx, person, condition):
    if not condition.mix_shift:
        return None
    w1 = rng_for(seed, _MIX, idx).dirichlet(np.full(len(person.repertoire), 1.0))
    r = condition.regime
    return MixShift(r.onset, r.duration, tuple(float(x) for x in w1))


def generate_person(idx, condition, seed, assets):
    """-> (person, sessions, ground_truth dict)."""
    person = make_person(seed, idx, assets)
    theta0 = person.theta0
    theta1 = theta1_for(theta0, condition.moving)
    mix = mix_shift_for(seed, idx, person, condition)
    regime = condition.regime
    sn = assets.gen["session_noise"]
    noise = NoiseCfg.from_dict(assets.gen["recognizer"])

    sessions, log = [], []
    speed = 1.0
    for n in range(condition.n_sessions):
        rng = rng_for(seed, _STATE, idx, n)
        base, which = theta_at(n, theta0, theta1, regime, rng)
        theta = K.keep_side_of_hinge(theta_for_session(base, sn["sigma"], rng), base, assets.knobs)
        speed = speed_factor(speed, sn["speed_rho"], sn["speed_sigma"], rng)
        rid = recipe_for_session(person, n, mix, rng)
        log.append({"session": n, "which": which, "theta_base": _r(base), "theta": _r(theta),
                    "speed": round(speed, 6), "recipe": rid})
        sessions.append(sample_session(
            assets.graphs[rid], assets.expansions[rid], theta, speed, assets, noise,
            rng_for(seed, _SESSION, idx, n), person.person_id, n,
            person.habit_orders.get(rid), person.habit))

    gt = {
        "condition": condition.name, "seed": seed, "person_id": person.person_id,
        "regime": {"kind": regime.kind, "onset": regime.onset, "duration": regime.duration,
                   "period": regime.period, "unit": "session"},
        "moving_knobs": sorted(condition.moving), "delta": {k: round(theta1[k] - theta0[k], 6)
                                                           for k in sorted(condition.moving)},
        "theta0": _r(theta0), "theta1": _r(theta1),
        "repertoire": list(person.repertoire), "habit": person.habit, "weights0": [round(w, 6) for w in person.weights],
        "mix_shift": None if mix is None else {"onset": mix.onset, "duration": mix.duration,
                                                "weights1": [round(w, 6) for w in mix.weights1]},
        "sessions": log,
    }
    return person, sessions, gt


def _r(theta):
    return {k: round(v, 6) for k, v in sorted(theta.items())}
