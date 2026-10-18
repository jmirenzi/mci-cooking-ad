"""Step-level error injection on a synthetic session's clean events (docs/drift_synth.md §9.6).

The tick-stream injectors (synthetic/error_injection.py) pick a segment of the model's own
segmentation; on Breakfast that is a recipe step, here a third of segments are idle gaps and a
third take/put/wash, so two thirds of their "errors" are normal behaviour (§9.5). These edit the
generator's ground truth instead: the clean events, each labelled with its CaptainCook4D step and
its kind (recipe action, flavour addition, take/put/wash). Every other event keeps its duration and
the gap after it; later events shift in time, no gap is created or removed.

  omission       every recipe action of one step is dropped (its fetches stay: the salt is taken
                 and never added)
  transposition  a step moves to just before a step it depends on in the task graph, so the
                 order breaks a precedence the recipe requires
  repetition     a step's recipe actions are done again right after the next step
  abandonment    a recipe action is cut to 5-20% of its length and the rest of its step is skipped
  substitution   a recipe action's verb or noun is replaced by one the person never uses in this
                 recipe

`inject` returns the edited events and the anomalous events, or None when the session has no
valid place for that error. Each span is ONE event, where the error starts (the successor of an
omitted step; the first action of the moved step and of the step it jumped; the first action of
the repeat; the cut action; the substituted action), as the tick injectors' one-segment spans:
metric G counts a span found from its start to DEADLINE ticks after its end, so a whole-step span
would be far easier to hit.
"""
from dataclasses import dataclass

from cook_ad.drift_synth.graphs import parents
from cook_ad.drift_synth.types import Event

ERROR_TYPES = ("substitution", "abandonment", "omission", "transposition", "repetition")
ACTION_KINDS = ("core",)            # plus "+<knob>" flavour additions
ABANDON_KEEP = (0.05, 0.20)
MIN_ABANDON_S = 8.0                 # an action shorter than this cannot be visibly cut
MIN_KEEP_S = 1.5                    # the kept part stays at least this long (visible at 1 s ticks)
MIN_GAP_S = 1.0                     # a moved or repeated block is set off by at least this


@dataclass
class Item:
    verb: str
    noun: str
    step: str | None
    kind: str
    dur: float
    gap: float                      # to the next event
    mark: int = 0                   # > 0: anomalous; equal ids form one span


def items_from_session(session_json):
    ev = session_json["meta"]["clean_events"]
    kinds = session_json["meta"]["kinds"]
    out = []
    for i, (t0, t1, v, n, st) in enumerate(ev):
        gap = ev[i + 1][0] - t1 if i + 1 < len(ev) else 0.0
        out.append(Item(v, n, st, kinds[i], t1 - t0, max(gap, 0.0)))
    return out


def is_action(it):
    return it.step is not None and (it.kind in ACTION_KINDS or it.kind.startswith("+"))


def to_events(items):
    """-> (events, [(index, span id) of marked items])."""
    t, out, marked = 0.0, [], []
    for i, it in enumerate(items):
        out.append(Event(round(t, 3), round(t + it.dur, 3), it.verb, it.noun))
        if it.mark:
            marked.append((i, it.mark))
        t += it.dur + (it.gap if i + 1 < len(items) else 0.0)
    return out, marked


def _blocks(items, order):
    """step -> indices of its recipe actions, for steps that have any, in cooking order."""
    b = {}
    for i, it in enumerate(items):
        if is_action(it):
            b.setdefault(it.step, []).append(i)
    return {s: b[s] for s in order if s in b}


def _copy(it, **kw):
    d = dict(it.__dict__)
    d.update(kw)
    return Item(**d)


def omission(items, order, graph, rng, **_):
    blocks = _blocks(items, order)
    steps = list(blocks)[:-1]                       # the last step has no successor to point at
    if not steps:
        return None
    s = steps[int(rng.integers(len(steps)))]
    drop = set(blocks[s])
    out = [_copy(it) for i, it in enumerate(items) if i not in drop]
    nxt = min(blocks[list(blocks)[list(blocks).index(s) + 1]])     # the successor step's first action
    succ = nxt - sum(1 for j in drop if j < nxt)
    out[succ].mark = 1                              # the step that now runs in its place
    return out


def transposition(items, order, graph, rng, **_):
    blocks = _blocks(items, order)
    par = parents(graph)
    pos = {s: k for k, s in enumerate(blocks)}
    # (a, b): a must come before b, and both have actions; move b to just before a
    pairs = [(a, b) for b in blocks for a in par.get(b, ()) if a in blocks and pos[a] < pos[b]]
    if not pairs:
        return None
    a, b = pairs[int(rng.integers(len(pairs)))]
    moved = [_copy(items[i]) for i in blocks[b]]
    moved[0].mark = 1                               # B begins where A was required
    moved[-1].gap = max(moved[-1].gap, MIN_GAP_S)
    rest = [_copy(it) for i, it in enumerate(items) if i not in set(blocks[b])]
    a_first = min(blocks[a]) - sum(1 for j in blocks[b] if j < min(blocks[a]))
    rest[a_first].mark = 2                          # A begins after B
    if a_first > 0:
        rest[a_first - 1].gap = max(rest[a_first - 1].gap, MIN_GAP_S)
    return rest[:a_first] + moved + rest[a_first:]


def repetition(items, order, graph, rng, **_):
    blocks = _blocks(items, order)
    steps = list(blocks)[:-1]
    if not steps:
        return None
    s = steps[int(rng.integers(len(steps)))]
    nxt = list(blocks)[list(blocks).index(s) + 1]
    at = max(blocks[nxt])                            # after the next step's last action
    copy = [_copy(items[i]) for i in blocks[s]]
    copy[0].mark = 1                                # the re-entry
    out = [_copy(it) for it in items]
    copy[-1].gap = max(out[at].gap, MIN_GAP_S)
    out[at].gap = max(out[at].gap, MIN_GAP_S)
    return out[:at + 1] + copy + out[at + 1:]


def abandonment(items, order, graph, rng, **_):
    blocks = _blocks(items, order)
    cands = [i for s in list(blocks)[:-1] for i in blocks[s] if items[i].dur >= MIN_ABANDON_S]
    if not cands:
        return None
    i = cands[int(rng.integers(len(cands)))]
    frac = float(rng.uniform(*ABANDON_KEEP))
    s = items[i].step
    skip = {j for j in blocks[s] if j > i}
    out = []
    for j, it in enumerate(items):
        if j in skip:
            continue
        out.append(_copy(it, dur=max(it.dur * frac, MIN_KEEP_S), mark=1) if j == i else _copy(it))
    return out


def substitution(items, order, graph, rng, unseen_verbs=(), unseen_nouns=(), **_):
    acts = [i for i, it in enumerate(items) if is_action(it)]
    if not acts or not (unseen_verbs or unseen_nouns):
        return None
    i = acts[int(rng.integers(len(acts)))]
    use_noun = bool(unseen_nouns) and (not unseen_verbs or rng.random() < 0.5)
    pool = sorted(unseen_nouns if use_noun else unseen_verbs)
    new = pool[int(rng.integers(len(pool)))]
    out = [_copy(it) for it in items]
    out[i] = _copy(out[i], mark=1, **({"noun": new} if use_noun else {"verb": new}))
    return out


INJECTORS = {"substitution": substitution, "abandonment": abandonment, "omission": omission,
             "transposition": transposition, "repetition": repetition}


def inject(error_type, session_json, graph, rng, **kw):
    """-> (events, marked event indices) or None."""
    items = items_from_session(session_json)
    out = INJECTORS[error_type](items, session_json["meta"]["order"], graph, rng, **kw)
    return None if out is None else to_events(out)


def event_spans(events, marked, tick_seconds):
    """Marked events [(index, span id)] -> closed tick spans [(a, b)], one per maximal run of
    consecutive events with the same span id, on the export's tick grid (ticks start at the
    first event)."""
    if not marked:
        return []
    t0 = events[0].t_start
    groups, cur = [], [marked[0]]
    for m in marked[1:]:
        if m[0] == cur[-1][0] + 1 and m[1] == cur[-1][1]:
            cur.append(m)
        else:
            groups.append(cur)
            cur = [m]
    groups = [[i for i, _ in g] for g in groups + [cur]]
    out = []
    for g in groups:
        a = int(round((events[g[0]].t_start - t0) / tick_seconds))     # the tick vote's grid
        b = max(a, int(round((events[g[-1]].t_end - t0) / tick_seconds)) - 1)
        out.append((a, b))
    return out
