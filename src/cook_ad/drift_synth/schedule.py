"""Stages 7b and 8: what a person cooks each day (and its nuisance shift), and how theta moves
under a drift regime."""
from dataclasses import dataclass, field

import numpy as np

REGIMES = ("none", "sudden", "incremental", "gradual", "recurring")


@dataclass(frozen=True)
class MixShift:
    """Recipe-mix change with theta untouched: weights move linearly from the person's w0 to
    `weights1` over [onset_day, onset_day + duration_days]."""
    onset_day: int
    duration_days: int
    weights1: tuple


@dataclass(frozen=True)
class DriftRegime:
    kind: str = "none"
    onset_day: int = 0
    duration_days: int = 1      # incremental: ramp length; gradual: days for p(t) to reach 1
    period_days: int = 7        # recurring: length of each theta0 / theta1 block

    def __post_init__(self):
        if self.kind not in REGIMES:
            raise ValueError(f"unknown regime {self.kind!r}")


@dataclass(frozen=True)
class PersonCfg:
    person_id: str
    repertoire: tuple            # recipe ids
    weights: tuple               # Dirichlet draw over the repertoire
    p_two_sessions: float
    theta0: dict = field(default_factory=dict)


def mix_weights(person, day, mix_shift):
    w0 = np.asarray(person.weights, dtype=float)
    if mix_shift is None or day < mix_shift.onset_day:
        return w0
    frac = min((day - mix_shift.onset_day) / max(mix_shift.duration_days, 1), 1.0)
    w = (1 - frac) * w0 + frac * np.asarray(mix_shift.weights1, dtype=float)
    return w / w.sum()


def recipes_for_day(person, day, mix_shift, rng):
    """1 or 2 sessions; recipes drawn from the day's mix weights."""
    two = rng.random() < person.p_two_sessions
    picks = rng.choice(len(person.repertoire), size=2, p=mix_weights(person, day, mix_shift))
    return [person.repertoire[int(i)] for i in picks[: 2 if two else 1]]


def recipe_schedule(person, n_days, mix_shift, rng):
    return [recipes_for_day(person, d, mix_shift, rng) for d in range(n_days)]


def ramp(day, regime):
    """p(t) for gradual, the interpolation fraction for incremental."""
    return float(np.clip((day - regime.onset_day) / max(regime.duration_days, 1), 0.0, 1.0)) \
        if day >= regime.onset_day else 0.0


def theta_at(day, theta0, theta1, regime, rng):
    """Theta for `day` before day-level noise. Returns (theta, which) with which in {0, 1, "mix"}.

    incremental interpolates theta linearly. Every duration-scale knob is a multiplier
    exp(a * (theta - 0.5)), so linear in theta is linear in log duration, as the spec asks.
    gradual draws one Bernoulli per day (always, so the rng stream is regime-independent)."""
    u = rng.random()
    k = regime.kind
    if k == "none" or day < regime.onset_day:
        return dict(theta0), 0
    if k == "sudden":
        return dict(theta1), 1
    if k == "incremental":
        f = ramp(day, regime)
        return {n: theta0[n] + f * (theta1[n] - theta0[n]) for n in theta0}, ("mix" if f < 1 else 1)
    if k == "gradual":
        return (dict(theta1), 1) if u < ramp(day, regime) else (dict(theta0), 0)
    block = (day - regime.onset_day) // max(regime.period_days, 1)
    return (dict(theta1), 1) if block % 2 == 0 else (dict(theta0), 0)
