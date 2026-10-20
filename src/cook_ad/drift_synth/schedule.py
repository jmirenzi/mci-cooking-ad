"""Stages 7b and 8: what a person cooks each session (and its nuisance shift), and how theta
moves under a drift regime. Time is the session index n = 0, 1, ...: one recipe per session,
no days (a day layer, e.g. mealtimes, can go on top later)."""
from dataclasses import dataclass, field

import numpy as np

REGIMES = ("none", "sudden", "incremental", "gradual", "recurring")


@dataclass(frozen=True)
class MixShift:
    """Recipe-mix change with theta untouched: weights move linearly from the person's w0 to
    `weights1` over sessions [onset, onset + duration]."""
    onset: int
    duration: int
    weights1: tuple


@dataclass(frozen=True)
class DriftRegime:
    kind: str = "none"
    onset: int = 0              # first session of the drift
    duration: int = 1           # incremental: ramp length; gradual: sessions for p(n) to reach 1
    period: int = 9             # recurring: sessions in each theta0 / theta1 block

    def __post_init__(self):
        if self.kind not in REGIMES:
            raise ValueError(f"unknown regime {self.kind!r}")


@dataclass(frozen=True)
class PersonCfg:
    person_id: str
    repertoire: tuple            # recipe ids
    weights: tuple               # Dirichlet draw over the repertoire
    theta0: dict = field(default_factory=dict)
    habit: float = 0.0           # strength of the routine step order (knobs.sample_topological_order)
    habit_orders: dict = field(default_factory=dict)   # recipe id -> preferred topological order


def mix_weights(person, n, mix_shift):
    w0 = np.asarray(person.weights, dtype=float)
    if mix_shift is None or n < mix_shift.onset:
        return w0
    frac = min((n - mix_shift.onset) / max(mix_shift.duration, 1), 1.0)
    w = (1 - frac) * w0 + frac * np.asarray(mix_shift.weights1, dtype=float)
    return w / w.sum()


def recipe_for_session(person, n, mix_shift, rng):
    """Session n's recipe, drawn from the mix weights."""
    return person.repertoire[int(rng.choice(len(person.repertoire), p=mix_weights(person, n, mix_shift)))]


def recipe_schedule(person, n_sessions, mix_shift, rng):
    return [recipe_for_session(person, n, mix_shift, rng) for n in range(n_sessions)]


def ramp(n, regime):
    """p(n) for gradual, the interpolation fraction for incremental."""
    return float(np.clip((n - regime.onset) / max(regime.duration, 1), 0.0, 1.0)) \
        if n >= regime.onset else 0.0


def theta_at(n, theta0, theta1, regime, rng):
    """Theta for session n before session noise. Returns (theta, which), which in {0, 1, "mix"}.

    incremental interpolates theta linearly. Every duration-scale knob is a multiplier
    exp(a * (theta - 0.5)), so linear in theta is linear in log duration, as the spec asks.
    gradual draws one Bernoulli per session (always, so the rng stream is regime-independent)."""
    u = rng.random()
    k = regime.kind
    if k == "none" or n < regime.onset:
        return dict(theta0), 0
    if k == "sudden":
        return dict(theta1), 1
    if k == "incremental":
        f = ramp(n, regime)
        return {n: theta0[n] + f * (theta1[n] - theta0[n]) for n in theta0}, ("mix" if f < 1 else 1)
    if k == "gradual":
        return (dict(theta1), 1) if u < ramp(n, regime) else (dict(theta0), 0)
    block = (n - regime.onset) // max(regime.period, 1)
    return (dict(theta1), 1) if block % 2 == 0 else (dict(theta0), 0)
