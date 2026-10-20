"""Stage 7a: factors the knobs do not model. Session-to-session wobble in theta, a speed that
drifts over sessions as an AR(1), and recognizer noise on the emitted events (off in
generator.yaml; the code path stays for later)."""
import math
from dataclasses import dataclass, replace

import numpy as np

from cook_ad.drift_synth.types import Event

EPS = 1e-4


def _logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def theta_for_session(theta_base, sigma, rng):
    """Each knob's session value: sigmoid(logit(theta) + N(0, sigma[knob])), in sorted knob order."""
    out = {}
    for k in sorted(theta_base):
        z = rng.standard_normal()
        out[k] = 1.0 / (1.0 + math.exp(-(_logit(theta_base[k]) + sigma.get(k, 0.0) * z)))
    return out


def speed_factor(prev, rho, sigma, rng):
    """log s_n = rho * log s_{n-1} + N(0, sigma): a multiplier on every duration that session."""
    return math.exp(rho * math.log(prev) + sigma * rng.standard_normal())


@dataclass(frozen=True)
class NoiseCfg:
    p_sub_verb: float = 0.0
    p_sub_noun: float = 0.0
    p_delete: float = 0.0
    p_insert: float = 0.0
    jitter_s: float = 0.0
    min_event_s: float = 0.2

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def off(self):
        return replace(self, p_sub_verb=0.0, p_sub_noun=0.0, p_delete=0.0, p_insert=0.0, jitter_s=0.0)


def recognizer_noise(events, cfg, pool, rng):
    """events: clean, time-ordered, disjoint. pool: (verbs, nouns) a confusion or insertion is
    drawn from (the session's own tokens: plausible confusions, all in the closed vocab).
    Per clean event one categorical draw: delete, swap the verb, swap the noun, or keep. Then,
    independently (deleted or not), a spurious token takes the tail of the event's span with
    P = p_insert, so each rate is per clean event as configured.
    Finally every boundary moves by N(0, jitter_s), keeping order and a minimum length.
    Returns (noisy events, counts).
    """
    verbs, nouns = sorted(set(pool[0])), sorted(set(pool[1]))
    p = np.array([cfg.p_delete, cfg.p_sub_verb, cfg.p_sub_noun])
    keep_p = 1.0 - p.sum()
    if keep_p < 0:
        raise ValueError("noise rates sum past 1")
    probs = np.append(p, keep_p)
    counts = {"clean": len(events), "deleted": 0, "sub_verb": 0, "sub_noun": 0, "inserted": 0}
    out = []
    for e in events:
        what = int(rng.choice(4, p=probs))
        u_ins, u_frac = rng.random(), rng.random()
        alt_v, alt_n = int(rng.integers(1 << 30)), int(rng.integers(1 << 30))
        if what == 1:
            e = replace(e, verb=_other(verbs, e.verb, alt_v))
            counts["sub_verb"] += 1
        elif what == 2:
            e = replace(e, noun=_other(nouns, e.noun, alt_n))
            counts["sub_noun"] += 1
        kept = [] if what == 0 else [e]
        counts["deleted"] += what == 0
        # Insertion is independent of deletion: a spurious detection can fall inside a missed one.
        if u_ins < cfg.p_insert and e.t_end - e.t_start >= 2 * cfg.min_event_s:
            cut = e.t_start + (e.t_end - e.t_start) * (0.5 + 0.3 * u_frac)
            spurious = Event(round(cut, 3), e.t_end, verbs[alt_v % len(verbs)], nouns[alt_n % len(nouns)])
            kept = [replace(k, t_end=round(cut, 3)) for k in kept] + [spurious]
            counts["inserted"] += 1
        out += kept
    if cfg.jitter_s > 0:
        out = _jitter(out, cfg, rng)
    return out, counts


def _other(options, current, r):
    choices = [o for o in options if o != current]
    return choices[r % len(choices)] if choices else current


def _jitter(events, cfg, rng):
    out = []
    prev_end = 0.0
    for i, e in enumerate(events):
        nxt = events[i + 1].t_start if i + 1 < len(events) else math.inf
        s = max(e.t_start + cfg.jitter_s * rng.standard_normal(), prev_end)
        t = min(e.t_end + cfg.jitter_s * rng.standard_normal(), nxt)
        if t - s < cfg.min_event_s:      # a jitter that would erase the event leaves it alone
            s, t = max(e.t_start, prev_end), min(e.t_end, nxt)
        out.append(replace(e, t_start=round(s, 3), t_end=round(t, 3)))
        prev_end = out[-1].t_end
    return out
