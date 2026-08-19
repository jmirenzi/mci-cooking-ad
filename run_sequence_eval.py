"""Score `anomaly/sequence.py` as a detector arm, against the tick-level HSMM on the same pool.

The sequence detector reads the Viterbi SEGMENT sequence rather than the tick stream, and runs
three local edit tests that each name their own error type (docs/anomaly.md 6). It exists to close
the one gap the tick channels cannot: `s_transition` fires identically for omission, transposition
and repetition, so the tick arm's transposition diagonal is exactly 0.000 no matter how well it
detects. The question this script answers is whether naming works, not whether detection does --
run_counterfactual.py already established that tick-level transposition detection is real.

Two stages:

  1. CALIBRATE on healthy trials. The swap-gain and duration-ratio statistics are collected over
     every junction/segment of every healthy trial and turned into (1-alpha) empirical quantiles
     via quantile.sequence_thresholds -- the same null-distribution discipline the per-tick
     channels use, with an empirical null in place of a fitted one. The omission test needs no
     table: it inherits narrate.missing_step's fixed nat gate.

  2. SCORE three arms on the identical degraded pool, through element_metrics.evaluate_steps:
     tick-level, sequence-standalone, and tick+relabel. The last is the shipping configuration --
     element_metrics.relabel_with_sequence lets the swap test correct the TYPE of an alarm the
     tick channels already raised, without letting it raise any alarm of its own, so precision,
     recall and healthy FPR are identical to the tick arm by construction and only type_confusion
     can move. The standalone arm is kept because it is the evidence for that containment: scored
     independently the sequence tests lower F1 and raise false alarms (docs/anomaly.md 6).

Everything is recipe-conditioned per trial: transitions come from r_hat's own row set and the
Lexicon from joint_params.select_recipe, matching narrate_joint. Lexicons are built once per
recipe rather than per trial.
"""
import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import narrate, quantile, sequence, surprise
from cook_ad.data.config import load_config
from cook_ad.eval import batch, element_metrics
from cook_ad.hsmm import joint_params
from cook_ad.llm import textify
from cook_ad.synthetic import error_injection, generate


def _lexicon_cache(vocab, jp):
    """Lexicon per recipe (expected_duration is recipe-conditioned), built lazily."""
    cache = {}

    def get(r):
        if r not in cache:
            cache[r] = narrate.Lexicon(vocab, joint_params.select_recipe(jp, r))
        return cache[r]

    return get


def _stats_for_trial(trace, log_trans, lexicon):
    """(swap gains at every junction, duration ratios at every segment) for one trial."""
    segments = narrate.segments_from_z(trace.z_star)
    states = [s for s, _, _ in segments]
    gains = [sequence.transposition_gain(log_trans, states, j) for j in range(len(states) - 1)]
    ratios = [sequence.repetition_ratio(end - start, state, lexicon)
              for state, start, end in segments]
    return gains, ratios


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params.npz")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--max-real", type=int, default=447)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=surprise.DEFAULT_ALPHA,
                    help="tail level for the two empirical sequence thresholds")
    ap.add_argument("--min-bridge-gain", type=float, default=narrate.DEFAULT_MIN_BRIDGE_GAIN)
    ap.add_argument("--out", default="dataset/processed/breakfast/sequence_report.json")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    inject_lex = narrate.Lexicon(vocab, marg)   # recipe-agnostic, for textify/injection
    lex_for = _lexicon_cache(vocab, jp)

    seqs = json.load(open(args.sequences))[: args.max_real]
    traj = [generate.trajectory_from_real_joint(jp, s["verb_ids"], s["noun_ids"], d_max)
            for s in seqs]
    usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    print(f"{len(usable)} usable real trials, alpha={args.alpha:.1e}", flush=True)

    # ---- stage 1: calibrate on healthy trials -------------------------------------------------
    print("computing traces: healthy", flush=True)
    h_traces, log_probs, h_rhat, ltm = batch.compute_traces_joint(
        jp, usable, d_max, chunk_size=args.chunk_size
    )
    h_rhat = [int(x) for x in h_rhat]

    all_gains, all_ratios = [], []
    for tr, rh in zip(h_traces, h_rhat):
        g, r = _stats_for_trial(tr, np.asarray(log_probs.log_trans[rh]), lex_for(rh))
        all_gains += g
        all_ratios += r
    thresholds = quantile.sequence_thresholds(all_gains, all_ratios, args.alpha)
    print(f"calibrated on {len(all_gains)} junctions / {len(all_ratios)} segments: "
          f"transposition > {thresholds.transposition:.3f} nats, "
          f"repetition > {thresholds.repetition:.3f}x expected duration", flush=True)
    # The tail level is only as well-estimated as the sample supports: at alpha it sits at roughly
    # the (alpha * n)th largest value, so report that count rather than implying an exact quantile.
    print(f"  (that is ~the {max(1, int(args.alpha * len(all_gains)))}th largest of "
          f"{len(all_gains)} healthy junction gains)", flush=True)

    def seq_verdicts(trace, rh, steps):
        segments = narrate.segments_from_z(trace.z_star)
        v = sequence.score_segments(segments, np.asarray(log_probs.log_trans[rh]), lex_for(rh),
                                    thresholds, min_bridge_gain=args.min_bridge_gain)
        return element_metrics.from_sequence_verdicts(v, segments, steps)

    healthy_seq = [
        seq_verdicts(tr, rh, textify.steps_from_trajectory(t, inject_lex))
        for tr, rh, t in zip(h_traces, h_rhat, usable)
    ]
    healthy_tick = [
        element_metrics.step_verdicts_from_flags(
            surprise.flag_joint(tr, log_probs, rh, ltm, alpha=surprise.DEFAULT_ALPHA),
            textify.steps_from_trajectory(t, inject_lex), tr, lex_for(rh),
        )
        for tr, rh, t in zip(h_traces, h_rhat, usable)
    ]
    healthy_both = [element_metrics.relabel_with_sequence(a, b)
                    for a, b in zip(healthy_tick, healthy_seq)]

    # ---- stage 2: score both arms on the same degraded pool -----------------------------------
    rng = np.random.default_rng(args.seed)
    deg_seq, deg_tick, deg_both, artifacts = {}, {}, {}, {}
    for et in error_injection.ERROR_TYPES:
        print(f"computing traces: {et}", flush=True)
        degraded = [error_injection.inject(et, t, rng, marg) for t in usable]
        d_traces, log_probs_d, d_rhat, ltm_d = batch.compute_traces_joint(
            jp, degraded, d_max, chunk_size=args.chunk_size
        )
        srows, trows, brows, arows = [], [], [], []
        for i, deg in enumerate(degraded):
            steps = textify.steps_from_ids(deg["verb_ids"], deg["noun_ids"], inject_lex)
            gt = textify.gt_steps_for_window(steps, deg["window"])
            debris = textify.injection_touched_steps(
                steps, deg["tick_map"], deg["edited_ticks"], gt
            )
            rh = int(d_rhat[i])
            sv = seq_verdicts(d_traces[i], rh, steps)
            tv = element_metrics.step_verdicts_from_flags(
                surprise.flag_joint(d_traces[i], log_probs_d, rh, ltm_d,
                                    alpha=surprise.DEFAULT_ALPHA),
                steps, d_traces[i], lex_for(rh),
            )
            srows.append((sv, gt, None))
            trows.append((tv, gt, None))
            brows.append((element_metrics.relabel_with_sequence(tv, sv), gt, None))
            arows.append(debris)
        deg_seq[et], deg_tick[et], deg_both[et], artifacts[et] = srows, trows, brows, arows

    rep_seq = element_metrics.evaluate_steps(healthy_seq, deg_seq, artifact_steps=artifacts)
    rep_tick = element_metrics.evaluate_steps(healthy_tick, deg_tick, artifact_steps=artifacts)
    rep_both = element_metrics.evaluate_steps(healthy_both, deg_both, artifact_steps=artifacts)

    # ---- report --------------------------------------------------------------------------------
    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    print(f"\n=== TRIAL-LOCATED ({len(usable)} real trials) ===\n")
    print(f"{'arm':<22} {'prec':>6} {'recall':>7} {'F1':>6} {'stray':>7} {'healthyFPR':>11}")
    print("-" * 64)
    for lab, rep in (("tick-level HSMM", rep_tick), ("sequence detector", rep_seq),
                     ("tick + RELABEL", rep_both)):
        t = rep["trial_located"]
        print(f"{lab:<22} {t['precision']:>6.3f} {t['recall']:>7.3f} "
              f"{f1(t['precision'], t['recall']):>6.3f} {t['stray_rate']:>7.3f} "
              f"{t['healthy_fpr']:>11.3f}")

    print(f"\n=== per-type trial-located RECALL ===\n")
    print(f"{'error type':<15} {'tick-level':>12} {'sequence':>10} {'relabel':>9}")
    print("-" * 48)
    for et in error_injection.ERROR_TYPES:
        print(f"{et:<15} {rep_tick['trial_located']['per_type'][et]['recall']:>12.3f} "
              f"{rep_seq['trial_located']['per_type'][et]['recall']:>10.3f} "
              f"{rep_both['trial_located']['per_type'][et]['recall']:>9.3f}")

    print(f"\n=== TYPE-NAMING (confusion diagonal) -- the point of this detector ===\n")
    print(f"{'error type':<15} {'tick-level':>12} {'sequence':>10} {'relabel':>9}")
    print("-" * 48)
    for et in error_injection.ERROR_TYPES:
        print(f"{et:<15} {rep_tick['type_confusion'][et].get(et, 0.0):>12.3f} "
              f"{rep_seq['type_confusion'][et].get(et, 0.0):>10.3f} "
              f"{rep_both['type_confusion'][et].get(et, 0.0):>9.3f}")

    with open(args.out, "w") as f:
        json.dump({
            "config": vars(args),
            "thresholds": {"transposition": thresholds.transposition,
                           "repetition": thresholds.repetition,
                           "n_junctions": len(all_gains), "n_segments": len(all_ratios)},
            "sequence": rep_seq, "tick_level": rep_tick, "relabel": rep_both,
        }, f, indent=2, default=str)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
