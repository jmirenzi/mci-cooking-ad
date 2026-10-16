"""Stage 9 output: per person, one JSON per session (seconds, string tokens) plus the loader
format every other corpus uses (docs/data.md §1): sequences.json / labels.json / vocab.json at
0.5 s ticks with gaps as the SIL pair, and ground_truth.json.

Byte-identical for the same inputs: keys sorted, floats rounded, lists in trial order.
"""
import json
from pathlib import Path

from cook_ad.data.parse_epic import to_frame_segments
from cook_ad.data.tick_expansion import expand_verb_noun_to_ticks


def trial_id(session):
    return f"{session.person_id}_d{session.day:03d}_s{session.session_idx}"


def _dump(path, obj):
    Path(path).write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")


def to_ticks(events, vocab, sil, fps, tick_seconds):
    """Events in seconds -> (verb_ids, noun_ids), via parse_epic's frame grid and tick vote."""
    v, n = vocab["verbs"], vocab["nouns"]
    narr = [(e.t_start, e.t_end, v[e.verb], n[e.noun]) for e in events if e.t_end > e.t_start]
    if not narr:
        return [], []
    segments, _ = to_frame_segments(narr, (v[sil[0]], n[sil[1]]), fps=fps)
    return expand_verb_noun_to_ticks(segments, fps, tick_seconds)


def step_labels(session, n_ticks, tick_seconds):
    """Per tick, the CC4D step (int id) of the clean event covering the tick's midpoint, 0 for
    logistics and gaps. Ticks start at the first emitted event, as `to_ticks`. Validation only
    (labels.json's `subtask_labels`, which training reads to score segmentations)."""
    origin = session.events[0].t_start if session.events else 0.0
    clean = session.meta["clean_events"]
    out, j = [], 0
    for i in range(n_ticks):
        t = origin + (i + 0.5) * tick_seconds
        while j < len(clean) and clean[j][1] <= t:
            j += 1
        step = clean[j][4] if j < len(clean) and clean[j][0] <= t else None
        out.append(int(step) if step is not None and str(step).isdigit() else 0)
    return out


def export(sessions, out_dir, vocab, ground_truth, export_cfg):
    out = Path(out_dir)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    sessions = sorted(sessions, key=trial_id)
    seqs, labels = [], []
    for s in sessions:
        tid = trial_id(s)
        _dump(out / "sessions" / f"{tid}.json", {
            "trial_id": tid, "person_id": s.person_id, "day": s.day, "session_idx": s.session_idx,
            "recipe_id": s.recipe_id, "theta_day": {k: round(x, 6) for k, x in s.theta_day.items()},
            "events": [[e.t_start, e.t_end, e.verb, e.noun] for e in s.events], "meta": s.meta})
        verb_ids, noun_ids = to_ticks(s.events, vocab, export_cfg["sil"], export_cfg["fps"],
                                      export_cfg["tick_seconds"])
        seqs.append({"trial_id": tid, "verb_ids": verb_ids, "noun_ids": noun_ids,
                     "terminal_idle_ticks": 0})
        labels.append({"trial_id": tid, "recipe_label": s.recipe_id, "person_id": s.person_id,
                       "day": s.day, "session_idx": s.session_idx,
                       "subtask_labels": step_labels(s, len(verb_ids), export_cfg["tick_seconds"])})
    _dump(out / "sequences.json", seqs)
    _dump(out / "labels.json", labels)
    _dump(out / "vocab.json", vocab)
    _dump(out / "ground_truth.json", ground_truth)
