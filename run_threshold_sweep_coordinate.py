"""Per-channel alpha, coordinate-descent-style, measured against the UNIONED detector.

run_threshold_sweep_by_channel's summary (render_threshold_sweep_by_channel.py) scores each
surprise channel ALONE -- useful for ranking channels by standalone PR-AUC, but it can't say
what a per-channel alpha would do to the real detector, which unions all seven. A channel's
marginal value depends on how much it overlaps with what the other six already catch, which an
isolated sweep cannot see.

This holds six channels at surprise.DEFAULT_ALPHA and sweeps the seventh's own alpha, re-scoring
the FULL UNION (all seven channels, one varied) at each point. One coordinate-descent pass --
each channel swept independently against the all-default baseline, not a joint 7-dimensional
search, which would be intractable and almost certainly overfit at this sample size. Traces are
computed ONCE per source group (the expensive JAX part); the sweep is cheap re-flagging, built
directly from cook_ad.anomaly.quantile's per-channel threshold functions rather than
surprise.flag_joint's single shared alpha.

    python run_threshold_sweep_coordinate.py --split-file dataset/processed/breakfast/split.json \
        --split-part train --joint-params dataset/processed/breakfast/joint_params_train.npz
"""
import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import narrate, quantile, surprise
from cook_ad.data.config import load_config
from cook_ad.data import split as split_mod
from cook_ad.eval import batch
from cook_ad.hsmm import joint_params
from cook_ad.llm import textify
from cook_ad.synthetic import error_injection, generate

CHANNELS = surprise.CHANNELS
DEFAULT_ALPHA = surprise.DEFAULT_ALPHA

ALPHAS = sorted(
    {round(a, 12) for a in
     [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
      5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]},
    reverse=True,
)


def _mixed_tables(joint_log_probs, r_hat, log_trans_marginal, alphas):
    """Mirrors quantile.threshold_tables_joint, but each of the 5 quantile-table channels is
    built at ITS OWN alpha (from `alphas`) instead of one shared value."""
    log_trans_r = joint_log_probs.log_trans[r_hat]
    return quantile.ThresholdTables(
        emit=quantile.joint_quantile_threshold(joint_log_probs.log_emit_v, joint_log_probs.log_emit_n,
                                               alphas["s_emit"]),
        verb=quantile.categorical_quantile_threshold(joint_log_probs.log_emit_v, alphas["s_verb"]),
        noun=quantile.categorical_quantile_threshold(joint_log_probs.log_emit_n, alphas["s_noun"]),
        transition=quantile.transition_quantile_threshold(log_trans_r, alphas["s_transition"]),
        recipe=quantile.excess_quantile_threshold(log_trans_r, log_trans_marginal, alphas["s_recipe_transition"]),
    )


def flag_joint_mixed(trace, joint_log_probs, r_hat, log_trans_marginal, alphas):
    """surprise.flag_joint, but `alphas` is a dict over all 7 CHANNELS instead of one shared
    scalar -- each channel's threshold table/duration cutoff built at its own alpha."""
    tables = _mixed_tables(joint_log_probs, r_hat, log_trans_marginal, alphas)
    thresholds = {
        "s_temporal": -float(np.log(alphas["s_temporal"])),
        "s_dur_two": -float(np.log(alphas["s_dur_two"])),
    }
    # The `alpha` positional arg is inert here: _duration_thresholds only falls back to it when
    # `thresholds` is empty, and we always supply both duration keys explicitly above.
    flags = surprise._base_flags(trace, tables, DEFAULT_ALPHA, thresholds)
    from_state_valid = trace.from_state != -1
    from_state_safe = np.where(from_state_valid, trace.from_state, 0)
    flags["s_recipe_transition"] = from_state_valid & (
        trace.s_recipe_transition > tables.recipe[from_state_safe]
    )
    return flags


def _union_mask(flags):
    mask = None
    for ch in CHANNELS:
        mask = flags[ch].copy() if mask is None else (mask | flags[ch])
    return mask


def _prepare_trial(traj, degraded, lexicon, is_degraded):
    v_ids = degraded["verb_ids"] if is_degraded else traj["verb_ids"]
    n_ticks = len(v_ids)
    steps = textify.steps_from_ids(v_ids, degraded["noun_ids"] if is_degraded else traj["noun_ids"], lexicon)

    pos_ticks = np.zeros(n_ticks, dtype=bool)
    pos_steps = np.zeros(len(steps), dtype=bool)
    if is_degraded:
        gt_steps = textify.gt_steps_for_window(steps, degraded["window"])
        debris = textify.injection_touched_steps(
            steps, degraded["tick_map"], degraded["edited_ticks"], gt_steps
        )
        positive_step_idx = set(gt_steps) | debris
        for si in positive_step_idx:
            s = steps[si]
            pos_ticks[s.tick_start : s.tick_end] = True
            pos_steps[si] = True

    return {"n_ticks": n_ticks, "steps": steps, "pos_ticks": pos_ticks, "pos_steps": pos_steps}


def _accumulate(counts, mask, static, gt_trial_positive):
    pos_ticks = static["pos_ticks"]
    c = counts["tick"]
    c[0] += int((pos_ticks & mask).sum())
    c[3] += int((pos_ticks & ~mask).sum())
    c[2] += int((~pos_ticks & mask).sum())
    c[1] += int((~pos_ticks & ~mask).sum())

    pos_steps = static["pos_steps"]
    steps = static["steps"]
    step_pred = np.array([bool(mask[s.tick_start : s.tick_end].any()) for s in steps])
    c = counts["step"]
    c[0] += int((pos_steps & step_pred).sum())
    c[3] += int((pos_steps & ~step_pred).sum())
    c[2] += int((~pos_steps & step_pred).sum())
    c[1] += int((~pos_steps & ~step_pred).sum())

    any_flag = bool(mask.any())
    c = counts["trial"]
    if gt_trial_positive:
        c[0 if any_flag else 3] += 1
    else:
        c[2 if any_flag else 1] += 1

    c = counts["trial_loc"]
    if gt_trial_positive:
        hit = bool((pos_ticks & mask).any())
        stray = bool((~pos_ticks & mask).any())
        c[0 if hit else 3] += 1
        if stray:
            c[2] += 1
    else:
        c[2 if any_flag else 1] += 1


def _acc(c):
    tp, tn, fp, fn = c
    tot = tp + tn + fp + fn
    return {
        "accuracy": (tp + tn) / tot if tot else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def score_union(groups, joint_log_probs, log_trans_marginal, alphas):
    counts = {level: [0, 0, 0, 0] for level in ("tick", "step", "trial", "trial_loc")}
    for group_name, trials in groups.items():
        gt_trial_positive = group_name != "healthy"
        for trace, rh, static in trials:
            flags = flag_joint_mixed(trace, joint_log_probs, rh, log_trans_marginal, alphas)
            mask = _union_mask(flags)
            _accumulate(counts, mask, static, gt_trial_positive)
    return {level: _acc(counts[level]) for level in ("tick", "step", "trial", "trial_loc")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train.npz")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--max-real", type=int, default=402)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", default="train", choices=["train", "test"])
    ap.add_argument("--out", default="dataset/processed/breakfast/threshold_sweep_coordinate.json")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lexicon = narrate.Lexicon(vocab, marg)

    seqs = json.load(open(args.sequences))
    if args.split_file:
        split = split_mod.load_split(args.split_file)
        seqs = split_mod.filter_sequences(seqs, split, args.split_part)
    seqs = seqs[: args.max_real]

    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=8)
    usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    print(f"{len(usable)} usable real trials ({args.split_part} split)", flush=True)

    rng = np.random.default_rng(args.seed)
    degraded_by_type = {
        et: [error_injection.inject(et, t, rng, marg) for t in usable]
        for et in error_injection.ERROR_TYPES
    }

    # ---- trace computation: ONCE per group (the expensive JAX part) --------------------------
    groups = {}
    print("computing traces: healthy", flush=True)
    traces, joint_log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(
        jp, usable, d_max, chunk_size=args.chunk_size
    )
    statics = [_prepare_trial(t, None, lexicon, is_degraded=False) for t in usable]
    groups["healthy"] = list(zip(traces, [int(x) for x in r_hat], statics))

    for et in error_injection.ERROR_TYPES:
        print(f"computing traces: {et}", flush=True)
        deg_trials = degraded_by_type[et]
        traces, joint_log_probs, r_hat, log_trans_marginal = batch.compute_traces_joint(
            jp, deg_trials, d_max, chunk_size=args.chunk_size
        )
        statics = [_prepare_trial(t, d, lexicon, is_degraded=True) for t, d in zip(usable, deg_trials)]
        groups[et] = list(zip(traces, [int(x) for x in r_hat], statics))
    print("traces done; sweeping alpha per channel against the union (cheap re-flagging only)", flush=True)

    # ---- baseline: every channel at DEFAULT_ALPHA --------------------------------------------
    default_alphas = {ch: DEFAULT_ALPHA for ch in CHANNELS}
    baseline = score_union(groups, joint_log_probs, log_trans_marginal, default_alphas)
    print(f"baseline (all channels at alpha={DEFAULT_ALPHA:g}): "
          f"trial_loc precision={baseline['trial_loc']['precision']:.3f} "
          f"recall={baseline['trial_loc']['recall']:.3f} "
          f"healthy_fpr={baseline['trial']['fpr']:.3f}", flush=True)

    # ---- coordinate sweep: one channel varied, other six held at default ---------------------
    results = {"baseline": baseline, "per_channel": {}}
    for varied in CHANNELS:
        print(f"\nsweeping {varied} (others fixed at alpha={DEFAULT_ALPHA:g})", flush=True)
        rows = []
        for a in ALPHAS:
            alphas = {**default_alphas, varied: a}
            scored = score_union(groups, joint_log_probs, log_trans_marginal, alphas)
            rows.append({"alpha": a, **scored})
            tl = scored["trial_loc"]
            print(f"  alpha={a:.2e}  trial_loc prec={tl['precision']:.3f} rec={tl['recall']:.3f} "
                  f"healthy_fpr={scored['trial']['fpr']:.3f}", flush=True)
        results["per_channel"][varied] = rows

    with open(args.out, "w") as f:
        json.dump({"config": vars(args), **results}, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
