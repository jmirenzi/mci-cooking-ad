"""The trial pools every detector is scored on (docs/data.md §5).

Breakfast: the CANONICAL pools, built once by scripts/data/build_pools.py and read from
dataset/processed/breakfast/pools/{train,test}.pkl. Each usable trial (enough segments for every
injector) appears healthy and once per error type; the errors are placed in the Viterbi
segmentation of ONE fixed placement model (joint_params.npz), every type injected into a trial
before the next trial is touched (one rng). The test pool is exactly the pool the cached LLM
responses were collected on.

EPIC-Tent: the real sessions with their annotated errors, scored per error group (pools are the
split files plus anomalies.json; nothing is sampled).
"""
import json
import pickle
from pathlib import Path

import numpy as np

from cook_ad.data import split as split_mod
from cook_ad.data.parse_epic_tent import clean_fragments
from cook_ad.synthetic import error_injection, generate

ERROR_TYPES = error_injection.ERROR_TYPES


def usable(trajectories):
    """Indices of trajectories with enough segments for every injector."""
    return [i for i, t in enumerate(trajectories)
            if len(t["segments"]) >= error_injection.MIN_SEGMENTS]


PLACEMENT = {"breakfast": "dataset/processed/breakfast/joint_params.npz"}


def pool_path(corpus, part):
    return Path("dataset/processed") / corpus / "pools" / f"{part}.pkl"


def build_pool(sequences, placement_params, d_max, seed=0, chunk_size=16):
    """The canonical pool: [{"trial_id", "healthy", "injected": {type: trajectory}}]."""
    pool, keep = injected_per_trial(placement_params, sequences, d_max, seed, chunk_size)
    return [{"trial_id": sequences[i]["trial_id"], "healthy": h, "injected": inj}
            for i, (h, inj) in zip(keep, pool)]


def save_pool(path, entries, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"meta": meta, "entries": entries}, fh)


def load_pool(path):
    """-> (entries, meta) as written by save_pool."""
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    return payload["entries"], payload["meta"]


def to_streams(entries):
    """Canonical pool entries -> {group: [trajectory]} ("healthy", then each error type)."""
    streams = {"healthy": [e["healthy"] for e in entries]}
    for et in ERROR_TYPES:
        streams[et] = [e["injected"][et] for e in entries]
    return streams


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


def epic_tent_pool_path(split_file, part):
    """dataset/processed/epic_tent/pools/<split stem>_<part>.pkl"""
    return Path("dataset/processed/epic_tent/pools") / f"{Path(split_file).stem}_{part}.pkl"


def freeze_epic_tent_pool(sequences_path, anomalies_path, split_file, part):
    """Write one split's sessions and their annotation records to epic_tent_pool_path."""
    seqs = split_mod.filter_sequences(json.load(open(sequences_path)),
                                      split_mod.load_split(split_file), part)
    truth = {t["trial_id"]: t for t in json.load(open(anomalies_path))}
    entries = [{"sequence": s, "truth": truth[s["trial_id"]]} for s in seqs]
    path = epic_tent_pool_path(split_file, part)
    save_pool(path, entries, {"corpus": "epic_tent", "split_file": str(split_file), "part": part,
                              "n_sessions": len(entries)})
    return path, len(entries)


def load_epic_tent_pool(path, limit=None):
    """-> (sequences, truth records), in split order."""
    entries, _ = load_pool(path)
    entries = entries[:limit] if limit else entries
    return [e["sequence"] for e in entries], [e["truth"] for e in entries]


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
