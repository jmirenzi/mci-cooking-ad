"""Scrub idle observations from a tick stream (docs/drift_synth.md §7).

In gap-heavy streams (two actions in three separated by a (stall, idle) run in the synthetic
data) the HSMM's transitions go A -> idle -> B, and idle -> anything is common, so a repeated,
moved or skipped step leaves no surprising transition (§9.7). Dropping the idle ticks before the
model sees the stream makes the transition table be between actions. Consequences, by choice:

- "A, pause, A" becomes one run of A (Jack, 2026-10-19): an action redone after a pause is one
  longer action;
- the pause itself is not lost: `kept` maps every scrubbed tick to its original tick, so gap
  lengths and real time can be recovered;
- every per-tick array that travels with the stream (step labels) is scrubbed with the same mask,
  and ground-truth spans move with `map_spans`.
"""
import numpy as np


def scrub_idle(verb_ids, noun_ids, sil):
    """-> (verb_ids, noun_ids, kept): the ticks whose (verb, noun) is not the idle pair `sil`
    (ids), and their original indices."""
    v, n = np.asarray(verb_ids), np.asarray(noun_ids)
    kept = np.flatnonzero(~((v == sil[0]) & (n == sil[1])))
    return v[kept], n[kept], kept


def map_spans(spans, kept):
    """Closed original-tick spans -> closed spans on the scrubbed ticks, keeping the kept ticks
    inside each span; a span with none (it covered only idle) collapses onto the first kept tick
    after it. Spans past the last kept tick are dropped."""
    kept = np.asarray(kept)
    out = []
    for a, b in spans:
        lo = int(np.searchsorted(kept, a, side="left"))
        hi = int(np.searchsorted(kept, b, side="right")) - 1
        if lo >= len(kept):
            continue
        out.append((lo, max(lo, hi)))
    return out


def scrub_sequence(seq, sil, labels=None):
    """A sequences.json entry (and optionally its labels.json entry) without idle ticks.
    -> (seq, labels, kept)."""
    v, n, kept = scrub_idle(seq["verb_ids"], seq["noun_ids"], sil)
    out = dict(seq, verb_ids=v.tolist(), noun_ids=n.tolist())
    if labels is not None and "subtask_labels" in labels:
        labels = dict(labels, subtask_labels=np.asarray(labels["subtask_labels"])[kept].tolist())
    return out, labels, kept
