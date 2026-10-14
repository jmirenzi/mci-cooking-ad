"""The trial pools every detector is scored on (docs/data.md §5).

Breakfast pools are healthy test trials plus copies with one injected error each; the injectors
place errors in the Viterbi segmentation of a placement model. EPIC-Tent pools are the real
sessions with their annotated errors, scored per error group.
"""
import json

import numpy as np

from cook_ad.data import split as split_mod
from cook_ad.data.parse_epic_tent import clean_fragments
from cook_ad.synthetic import error_injection, generate

ERROR_TYPES = error_injection.ERROR_TYPES


def usable(trajectories):
    """Indices of trajectories with enough segments for every injector."""
    return [i for i, t in enumerate(trajectories)
            if len(t["segments"]) >= error_injection.MIN_SEGMENTS]


def injected_per_type(params, sequences, d_max, seed=0, chunk_size=8):
    """{group: [trajectory]}: "healthy" plus one copy per error type, injected type by type over
    all trials (one rng). Placement uses `params`. -> (streams, kept indices into `sequences`)."""
    traj = generate.trajectories_from_real_joint(params, sequences, d_max, chunk_size=chunk_size)
    keep = usable(traj)
    healthy = [traj[i] for i in keep]
    rng = np.random.default_rng(seed)
    streams = {"healthy": healthy}
    for et in ERROR_TYPES:
        streams[et] = [error_injection.inject(et, t, rng, params, select="random") for t in healthy]
    return streams, keep


def injected_per_trial(params, sequences, d_max, seed=0, chunk_size=16):
    """[(healthy, {type: injected})]: every type injected into each trial in turn (one rng).
    Placement uses `params`. This is the pool the cached LLM responses were collected on.
    -> (pool, kept indices into `sequences`)."""
    traj = generate.trajectories_from_real_joint(params, sequences, d_max, chunk_size=chunk_size)
    rng = np.random.default_rng(seed)
    keep = usable(traj)
    return [(traj[i], {et: error_injection.inject(et, traj[i], rng, params) for et in ERROR_TYPES})
            for i in keep], keep


def per_trial_to_streams(pool):
    """[(healthy, {type: injected})] -> {group: [trajectory]}."""
    streams = {"healthy": [h for h, _ in pool]}
    for et in ERROR_TYPES:
        streams[et] = [d[et] for _, d in pool]
    return streams


def split_sequences(sequences_path, split_path, part, limit=None):
    seqs = json.load(open(sequences_path))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(split_path), part)
    return seqs[:limit] if limit else seqs


# ---- EPIC-Tent: real annotations, scored per error group ------------------------------------


def group_truth(trial_truth, group):
    """One session's truth with every type outside `group` REMOVED -> (typed spans, fragments).

    Removed means relabelled as normal, which is PREGO's treatment of the non-procedural types:
    an out-of-group span drops out of the ground truth and its ticks rejoin the healthy
    population. The healthy fragments are therefore recomputed per group."""
    spans = [(tuple(r["window"]), r["error_type"]) for r in trial_truth["records"]
             if r["error_type"] in group]
    return spans, clean_fragments([span for span, _ in spans], trial_truth["n_ticks"])


def slice_flags(flags, fragment):
    """The flags inside a closed tick interval (ticks are not re-based)."""
    start, end = fragment
    return [f for f in flags if start <= (f if isinstance(f, (int, np.integer)) else f.tick) <= end]


def trim_at_first_mistake(sequences, truth, group):
    """-> [(sequence, (t0, t1) or None)]: each session cut after the last tick of its FIRST
    in-group mistake (earliest onset). Sessions without one are kept whole."""
    out = []
    for seq, trial in zip(sequences, truth):
        spans, _ = group_truth(trial, group)
        if not spans:
            out.append((seq, None))
            continue
        (t0, t1), _ = min(spans, key=lambda s: s[0])
        end = t1 + 1
        out.append(({"trial_id": seq["trial_id"], "verb_ids": seq["verb_ids"][:end],
                     "noun_ids": seq["noun_ids"][:end]}, (t0, t1)))
    return out
