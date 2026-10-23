"""The types every detector shares: a trial goes in, alarms come out (docs/detectors.md §1)."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Span:
    """An annotated or injected anomaly over ticks start..end (inclusive)."""
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class Trial:
    """One session as (verb, noun) ids per tick, with its ground truth if known."""
    trial_id: str
    verbs: np.ndarray            # (T,) int
    nouns: np.ndarray            # (T,) int
    recipe: int | np.ndarray | None = None   # known dish, a (K_R,) log prior, or None = infer
    spans: tuple[Span, ...] = ()
    group: str = "healthy"       # "healthy", or the error type / error group


@dataclass(frozen=True)
class Alarm:
    """One flag. `tick` is what the metrics score; `known_at` is when the detector could raise it."""
    tick: int
    channel: str
    score: float
    known_at: int


@dataclass
class Detection:
    alarms: list[Alarm]
    trace: object | None = None  # detector-specific, e.g. HindsightTrace for narrate()


@dataclass(frozen=True)
class OperatingPoint:
    """alpha: one tail mass for every channel, or per channel. kappa: posterior cutoff (Live/Lag)."""
    alpha: float | dict[str, float]
    kappa: float | None = None

    def alpha_for(self, channel: str, default: float | None = None) -> float:
        """The channel's alpha; with a per-channel dict lacking it, `default` (None: KeyError)."""
        if not isinstance(self.alpha, dict):
            return self.alpha
        if channel in self.alpha or default is None:
            return self.alpha[channel]
        return default


class AnomalyDetector(ABC):
    """Base class. Causal detectors implement step(); the rest override detect_trial()."""
    name: str = "detector"
    causal: bool = True

    def fit(self, train: list[Trial]) -> None:
        """Default: nothing to fit (parameters are loaded at construction)."""

    @abstractmethod
    def reset(self, recipe: int | np.ndarray | None = None) -> None:
        """Start a new trial."""

    def step(self, verb: int, noun: int) -> list[Alarm]:
        """Consume one tick; return the alarms that became known at it."""
        raise NotImplementedError(f"{self.name} is not causal: use detect_trial()")

    def finish(self) -> list[Alarm]:
        """Flush alarms still pending at the end of the trial."""
        return []

    def detect_trial(self, trial: Trial) -> Detection:
        self.reset(trial.recipe)
        alarms = [a for v, n in zip(trial.verbs, trial.nouns) for a in self.step(int(v), int(n))]
        return Detection(alarms + self.finish())

    def detect_many(self, trials: list[Trial]) -> list[Detection]:
        return [self.detect_trial(t) for t in trials]


def alarms_from_flags(flags: dict[str, np.ndarray], scores: dict[str, np.ndarray] | None = None,
                      delay: int = 0) -> list[Alarm]:
    """{channel: (T,) bool} -> alarms sorted by (tick, channel order); known_at = tick + delay."""
    out = []
    for c, f in flags.items():
        s = None if scores is None else scores.get(c)
        out += [Alarm(int(t), c, float("nan") if s is None else float(s[t]), int(t) + delay)
                for t in np.flatnonzero(f)]
    order = {c: i for i, c in enumerate(flags)}
    return sorted(out, key=lambda a: (a.tick, order[a.channel]))


__all__ = ["Span", "Trial", "Alarm", "Detection", "OperatingPoint", "AnomalyDetector",
           "alarms_from_flags"]
