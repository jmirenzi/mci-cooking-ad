"""The generator's data types (the spec's, with `phase` optional until Stage 1 assigns it)."""
from dataclasses import dataclass, field
from typing import Literal

Token = tuple[str, str]  # (verb, noun), both in the closed vocab
Phase = Literal["prep", "cook", "serve", "cleanup"]


@dataclass(frozen=True)
class Step:
    step_id: str
    description: str
    phase: Phase | None = None


@dataclass(frozen=True)
class RecipeGraph:
    recipe_id: str
    steps: dict[str, Step]
    edges: list[tuple[str, str]]  # (parent, child): parent must come first


@dataclass(frozen=True)
class Variant:
    knob: str
    kind: Literal["insert", "substitute", "omit"]
    anchor_step: str
    tokens: list[Token]               # for insert or substitute
    replaces: list[int] | None        # token indices in the base expansion, for substitute or omit


@dataclass(frozen=True)
class StepExpansion:
    step_id: str
    tokens: list[Token]               # core actions only, no fetch/put of ingredients
    variants: list[Variant]


@dataclass(frozen=True)
class Event:
    t_start: float
    t_end: float
    verb: str
    noun: str


@dataclass
class Session:
    person_id: str
    index: int                        # session number n, the generator's only clock
    recipe_id: str
    events: list[Event]
    theta: dict[str, float]           # this session's theta (base + session noise)
    meta: dict = field(default_factory=dict)  # regime, noise draws, clean events before noise
