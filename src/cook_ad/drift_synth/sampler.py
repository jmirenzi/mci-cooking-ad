"""Stage 9: one session, one person.

Random streams. A person's traits come from (seed, person); each day's draws from (seed, person,
day); each session's from (seed, person, day, session). None includes the condition, so two
conditions with the same seed share everything before onset (common random numbers), and a
drift condition differs from `null` only through theta.
"""
from dataclasses import dataclass, field

import numpy as np

from cook_ad.drift_synth import knobs as K
from cook_ad.drift_synth.durations import sample_duration, sample_gap
from cook_ad.drift_synth.kitchen_state import insert_logistics
from cook_ad.drift_synth.noise import NoiseCfg, day_speed_factor, recognizer_noise, theta_for_day
from cook_ad.drift_synth.schedule import DriftRegime, MixShift, PersonCfg, recipes_for_day, theta_at
from cook_ad.drift_synth.types import Event, Session

_PERSON, _DAY, _SESSION, _MIX = 1, 2, 3, 4  # stream tags


def rng_for(seed, *keys):
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, keys)]))


@dataclass(frozen=True)
class ConditionCfg:
    name: str
    regime: DriftRegime = DriftRegime()
    moving: dict = field(default_factory=dict)   # knob -> |delta theta|
    mix_shift: bool = False
    n_persons: int = 8
    n_days: int = 70


def sample_session(recipe, expansions, theta, speed, assets, noise, rng,
                   person_id="", day=0, session_idx=0):
    """The spec's order: step order, variants, logistics, repetition, durations, noise."""
    kn, tags, gen = assets.knobs, assets.tags, assets.gen
    order = K.sample_topological_order(recipe, theta["mise_en_place"], kn["mise_en_place"]["beta"], rng)

    tokens, steps, applied, skipped = [], [], {}, []
    for s in order:
        exp = expansions.get(s)
        if exp is None:
            skipped.append(s)
            continue
        toks, used = K.apply_variants(exp, theta, kn, rng)
        tokens += toks
        steps += [s] * len(toks)
        if used:
            applied[s] = used

    put_p, wash_p = K.tidy_rates(theta, kn)
    with_log, prov = insert_logistics(tokens, tags, put_p, wash_p, rng, gen["logistics_verbs"])
    it = iter(steps)
    step_of = [next(it) if p == "core" else None for p in prov]

    final, final_step = [], []
    n_rep = 0
    for tok, s in zip(with_log, step_of):
        u = rng.random()
        final.append(tok)
        final_step.append(s)
        if u < K.repetition_prob(tok, theta, kn, tags):
            final.append(tok)
            final_step.append(s)
            n_rep += 1

    slow = K.slow_multiplier(theta, kn)
    t = 0.0
    clean = []
    for i, tok in enumerate(final):
        d = sample_duration(assets.durations, tok, speed * slow * K.token_multiplier(tok, theta, kn, tags), rng)
        gap = sample_gap(assets.durations, speed * slow, rng)
        clean.append(Event(round(t, 3), round(t + d, 3), tok[0], tok[1]))
        t += d + (gap if i + 1 < len(final) else 0.0)

    pool = ([e.verb for e in clean], [e.noun for e in clean])
    events, counts = recognizer_noise(clean, noise, pool, rng)
    return Session(
        person_id=person_id, day=day, session_idx=session_idx, recipe_id=recipe.recipe_id,
        events=events, theta_day=dict(theta),
        meta={"order": order, "variants_applied": applied, "skipped_steps": skipped,
              "repetitions": n_rep, "speed": round(speed, 6), "noise": counts,
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
    t_lo, t_hi = p["theta0"]
    theta0 = {k: float(rng.uniform(t_lo, t_hi)) for k in K.KNOBS}
    return PersonCfg(f"p{idx:02d}", rep, tuple(float(x) for x in w), p["p_two_sessions"], theta0)


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
    return MixShift(r.onset_day, r.duration_days, tuple(float(x) for x in w1))


def generate_person(idx, condition, seed, assets):
    """-> (person, sessions, ground_truth dict)."""
    person = make_person(seed, idx, assets)
    theta0 = person.theta0
    theta1 = theta1_for(theta0, condition.moving)
    mix = mix_shift_for(seed, idx, person, condition)
    regime = condition.regime
    dn = assets.gen["day_noise"]
    noise = NoiseCfg.from_dict(assets.gen["recognizer"])

    sessions, days = [], []
    speed = 1.0
    for d in range(condition.n_days):
        rng = rng_for(seed, _DAY, idx, d)
        base, which = theta_at(d, theta0, theta1, regime, rng)
        theta = theta_for_day(base, dn["sigma"], rng)
        speed = day_speed_factor(speed, dn["speed_rho"], dn["speed_sigma"], rng)
        recipes = recipes_for_day(person, d, mix, rng)
        days.append({"day": d, "which": which, "theta_base": _r(base), "theta_day": _r(theta),
                     "speed": round(speed, 6), "recipes": recipes})
        for s, rid in enumerate(recipes):
            sessions.append(sample_session(
                assets.graphs[rid], assets.expansions[rid], theta, speed, assets, noise,
                rng_for(seed, _SESSION, idx, d, s), person.person_id, d, s))

    gt = {
        "condition": condition.name, "seed": seed, "person_id": person.person_id,
        "regime": {"kind": regime.kind, "onset_day": regime.onset_day,
                   "duration_days": regime.duration_days, "period_days": regime.period_days},
        "moving_knobs": sorted(condition.moving), "delta": {k: round(theta1[k] - theta0[k], 6)
                                                           for k in sorted(condition.moving)},
        "theta0": _r(theta0), "theta1": _r(theta1),
        "repertoire": list(person.repertoire), "weights0": [round(w, 6) for w in person.weights],
        "mix_shift": None if mix is None else {"onset_day": mix.onset_day,
                                                "duration_days": mix.duration_days,
                                                "weights1": [round(w, 6) for w in mix.weights1]},
        "days": days,
    }
    return person, sessions, gt


def _r(theta):
    return {k: round(v, 6) for k, v in sorted(theta.items())}
