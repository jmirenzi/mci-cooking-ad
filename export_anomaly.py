"""Export the best-detected example of each rollout scenario (substitution, abandonment,
omission, transposition, repetition, stall, healthy) for one calibrated user, mirroring
run_rollout_demo.py's own pipeline (surprise.compute_trace -> flag -> narrate) exactly --
this is a visual version of that script's output, not a new detector.

Sweeps holdout trial x injection `select` mode x seed, scores each with eval.metrics.score_trial,
and keeps the clearest true positive per error type (detected, >=1 in-window narrated query,
fewest out-of-window flags). The winning (trial_id, select, seed) is recorded so the choice is
reproducible, not a lucky run.
"""
import argparse
import json
import os

import jax
import numpy as np

from cook_ad.anomaly import narrate, surprise
from cook_ad.eval import metrics
from cook_ad.hsmm import params
from cook_ad.recipe import recipe_hmm
from cook_ad.synthetic import error_injection, generate

import run_rollout_demo as rollout  # reuse calibrate_to_user / inject_stall / load_user_trials

jax.config.update("jax_enable_x64", True)

D_MAX = 200
SCENARIOS = (*error_injection.ERROR_TYPES, "stall")
SEEDS = range(3)
SELECTS = ("random", "hardest")


def _combos():
    """(select, seed) pairs to try. 'hardest' ignores rng for segment choice (_pick_segment
    returns the leftmost valid index unconditionally) and every remaining rng use in the
    injectors is itself deterministic given the picked segment, so seed has no effect under
    'hardest' -- only try seed=0 there instead of repeating identical work SEEDS times."""
    return [("hardest", 0)] + [("random", s) for s in SEEDS]


def _runs(verb_ids, noun_ids, lexicon):
    from itertools import groupby

    runs = []
    pos = 0
    for (v, n), group in groupby(zip(verb_ids.tolist(), noun_ids.tolist())):
        length = sum(1 for _ in group)
        runs.append({
            "verb": lexicon.verb(v), "noun": lexicon.noun(n), "phrase": lexicon.phrase(v, n),
            "start": pos, "end": pos + length, "n": length,
        })
        pos += length
    return runs


def _segments(segments, lexicon):
    out = []
    for state, start, end in segments:
        out.append({
            "z": int(state), "start": int(start), "end": int(end), "name": lexicon.subtask(state),
        })
    return out


def _flagged_ticks(flags):
    return {ch: [int(t) for t in np.flatnonzero(flags[ch])] for ch in metrics.ALL_CHANNELS}


def _out_of_window_count(flags, window):
    flagged = metrics.detect(flags)
    if window is None:
        return int(flagged.size)
    t0, t1 = window
    hi = t1 + metrics.DEFAULT_LATENCY_TOL
    return int(np.sum((flagged < t0) | (flagged > hi)))


def _query_records(queries, window):
    t0 = t1 = hi = None
    if window is not None:
        t0, t1 = window
        hi = t1 + metrics.DEFAULT_LATENCY_TOL
    out = []
    for q in queries:
        tp = window is not None and t0 <= q.tick <= hi
        out.append({
            "tick": q.tick, "segment_index": q.segment_index, "channel": q.channel, "kind": q.kind,
            "severity": q.severity, "ratio": q.ratio, "text": q.text, "true_positive": tp,
        })
    return out


def _score_and_narrate(hsmm_params, recipe_params, vocab, verb_ids, noun_ids, d_max):
    trace, log_probs, recipe_log_trans = surprise.compute_trace(hsmm_params, recipe_params, verb_ids, noun_ids, d_max)
    flags = surprise.flag(trace, log_probs, recipe_log_trans)
    pi_all = surprise.compute_pi_all(log_probs, verb_ids, noun_ids, d_max)
    queries = narrate.narrate(trace, flags, vocab, hsmm_params, verb_ids, noun_ids, log_probs, recipe_log_trans, pi_all)
    return trace, flags, queries


def sweep(user, user_trials, corpus_params, recipe_params, vocab, lexicon_cache, d_max):
    best = {}
    healthy_by_holdout = []

    for holdout in user_trials:
        calib_trials = [t for t in user_trials if t["trial_id"] != holdout["trial_id"]]
        hsmm_params_u, _ = rollout.calibrate_to_user(corpus_params, calib_trials, d_max)
        lexicon = narrate.Lexicon(vocab, hsmm_params_u)
        traj = generate.trajectory_from_real(hsmm_params_u, holdout["verb_ids"], holdout["noun_ids"], d_max)

        trace_h, flags_h, queries_h = _score_and_narrate(
            hsmm_params_u, recipe_params, vocab, traj["verb_ids"], traj["noun_ids"], d_max
        )
        healthy_by_holdout.append({
            "trial_id": holdout["trial_id"], "hsmm_params": hsmm_params_u, "lexicon": lexicon,
            "traj": traj, "trace": trace_h, "flags": flags_h, "queries": queries_h,
        })

        for scenario in SCENARIOS:
            for select, seed in _combos():
                rng = np.random.default_rng(seed)
                try:
                    if scenario == "stall":
                        injected = rollout.inject_stall(traj, rng, select=select)
                    else:
                        injected = error_injection.inject(scenario, traj, rng, hsmm_params_u, select=select)
                except ValueError:
                    continue  # trajectory too short for this injection's segment requirements

                trace_i, flags_i, queries_i = _score_and_narrate(
                    hsmm_params_u, recipe_params, vocab, injected["verb_ids"], injected["noun_ids"], d_max
                )
                detected, latency, _, _ = metrics.score_trial(flags_i, injected["window"])
                if not detected:
                    continue
                in_window_q = [q for q in _query_records(queries_i, injected["window"]) if q["true_positive"]]
                if not in_window_q:
                    continue

                n_out = _out_of_window_count(flags_i, injected["window"])
                candidate = {
                    "error_type": scenario, "trial_id": holdout["trial_id"], "select": select, "seed": seed,
                    "window": injected["window"], "T": int(len(injected["verb_ids"])),
                    "runs": _runs(injected["verb_ids"], injected["noun_ids"], lexicon),
                    "segments": _segments(narrate.segments_from_z(trace_i.z_star), lexicon),
                    "flagged_channels": _flagged_ticks(flags_i),
                    "queries": _query_records(queries_i, injected["window"]),
                    "latency": latency, "n_out_of_window": n_out,
                    # Pre-injection stream, same holdout, same calibrated hsmm_params_u -- lets
                    # the renderer show what the user actually did above what got fed to the
                    # detector, since insertions/deletions mean tick indices diverge after the
                    # edit point and the two can't just be overlaid on the same row.
                    "healthy_T": int(len(traj["verb_ids"])),
                    "healthy_runs": _runs(traj["verb_ids"], traj["noun_ids"], lexicon),
                }
                current = best.get(scenario)
                if current is None or n_out < current["n_out_of_window"]:
                    best[scenario] = candidate

        print(f"  holdout {holdout['trial_id']}: {len(queries_h)} healthy queries, "
              f"{sum(1 for s in SCENARIOS if s in best)}/{len(SCENARIOS)} scenarios covered so far")

    query_counts = [len(h["queries"]) for h in healthy_by_holdout]
    median_count = sorted(query_counts)[len(query_counts) // 2]
    healthy_pick = min(healthy_by_holdout, key=lambda h: abs(len(h["queries"]) - median_count))
    healthy_runs = _runs(
        healthy_pick["traj"]["verb_ids"], healthy_pick["traj"]["noun_ids"], healthy_pick["lexicon"]
    )
    healthy_record = {
        "error_type": "healthy", "trial_id": healthy_pick["trial_id"], "select": None, "seed": None,
        "window": None, "T": int(len(healthy_pick["traj"]["verb_ids"])),
        "runs": healthy_runs,
        "segments": _segments(narrate.segments_from_z(healthy_pick["trace"].z_star), healthy_pick["lexicon"]),
        "flagged_channels": _flagged_ticks(healthy_pick["flags"]),
        "queries": _query_records(healthy_pick["queries"], None),
        "latency": None, "n_out_of_window": _out_of_window_count(healthy_pick["flags"], None),
        # No injection happened, so the "unaltered" row is identical to the observations row --
        # the renderer still gets the field so it doesn't need a healthy-scenario special case.
        "healthy_T": int(len(healthy_pick["traj"]["verb_ids"])),
        "healthy_runs": healthy_runs,
    }
    best["healthy"] = healthy_record
    return best, [len(h["queries"]) for h in healthy_by_holdout]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset/processed/breakfast")
    parser.add_argument("--user", default=None)
    parser.add_argument("--out", default="dataset/processed/breakfast/flow/anomaly.json")
    args = parser.parse_args()

    with open(f"{args.dataset_dir}/sequences.json") as f:
        all_sequences = json.load(f)
    with open(f"{args.dataset_dir}/vocab.json") as f:
        vocab = json.load(f)
    corpus_params = params.load_params(f"{args.dataset_dir}/hsmm_params.npz")
    recipe_params = recipe_hmm.load_params(f"{args.dataset_dir}/recipe_params.npz")

    if args.user is None:
        from collections import Counter
        counts = Counter(s["trial_id"].split("_", 1)[0] for s in all_sequences)
        args.user = max(counts, key=counts.get)

    user_trials, _ = rollout.load_user_trials(f"{args.dataset_dir}/sequences.json", args.user)
    print(f"user: {args.user}  trials: {len(user_trials)}")

    best, healthy_query_counts = sweep(args.user, user_trials, corpus_params, recipe_params, vocab, {}, D_MAX)

    print("\nselected examples:")
    for scenario in (*SCENARIOS, "healthy"):
        rec = best.get(scenario)
        if rec is None:
            print(f"  {scenario:14s} NO CLEAN DETECTION FOUND across the sweep")
            continue
        tag = f"select={rec['select']} seed={rec['seed']}" if rec["select"] else "representative"
        print(f"  {scenario:14s} {rec['trial_id']:16s} {tag:24s} "
              f"n_queries={len(rec['queries'])} n_out_of_window={rec['n_out_of_window']}")
    print(f"\nhealthy-baseline query counts across all {len(healthy_query_counts)} holdouts: {healthy_query_counts}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"user": args.user, "scenarios": best}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
