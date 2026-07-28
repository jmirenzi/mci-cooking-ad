import argparse
import json

import jax
import numpy as np

from cook_ad.anomaly import surprise
from cook_ad.data.config import load_config
from cook_ad.eval import batch, metrics, plotting
from cook_ad.hsmm import joint_params, params
from cook_ad.recipe import recipe_hmm
from cook_ad.synthetic import error_injection, generate

jax.config.update("jax_enable_x64", True)


def _flags_for_all(sequences, hsmm_params, recipe_params, d_max):
    traces = batch.compute_traces(hsmm_params, recipe_params, sequences, d_max)
    return [surprise.flag(t) for t in traces]


def _usable(trajectories):
    return [t for t in trajectories if len(t["segments"]) >= error_injection.MIN_SEGMENTS]


def evaluate_source(trajectories, hsmm_params, recipe_params, d_max, rng, tag, n_nouns):
    """Run the full 5-error evaluation for one healthy source (synthetic or real)."""
    trajectories = _usable(trajectories)
    print(f"\n[{tag}] {len(trajectories)} usable trajectories (>= {error_injection.MIN_SEGMENTS} segments)",
          flush=True)

    healthy_flags = _flags_for_all(trajectories, hsmm_params, recipe_params, d_max)

    degraded_by_type = {}
    degraded_traj_pool = []
    for error_type in error_injection.ERROR_TYPES:
        degraded = [error_injection.inject(error_type, t, rng, hsmm_params) for t in trajectories]
        flags = _flags_for_all(degraded, hsmm_params, recipe_params, d_max)
        degraded_by_type[error_type] = list(zip(flags, (d["window"] for d in degraded)))
        degraded_traj_pool.extend(degraded)
        print(f"  [{tag}] {error_type}: evaluated {len(degraded)} degraded trials", flush=True)

    report = metrics.evaluate(healthy_flags, degraded_by_type)
    report["kl_sanity"] = metrics.kl_sanity(trajectories, degraded_traj_pool, n_nouns)
    return report


def _flags_for_all_joint(sequences, joint_hsmm_params, d_max):
    traces = batch.compute_traces_joint(joint_hsmm_params, sequences, d_max)
    return [surprise.flag(t) for t in traces]


def evaluate_source_joint(trajectories, joint_hsmm_params, marginal_hsmm_params, d_max, rng, tag, n_nouns):
    """Joint-model analogue of evaluate_source. error_injection is unchanged and recipe-
    agnostic by construction (it only reads emissions), so it's handed the pi-weighted
    marginal collapse of the joint params (joint_params.collapse_to_marginal) rather than a
    per-trial recipe-conditioned model -- injecting an error doesn't need to know the trial's
    recipe, only what's typical/atypical for that subtask in general.
    """
    trajectories = _usable(trajectories)
    print(f"\n[{tag}] {len(trajectories)} usable trajectories (>= {error_injection.MIN_SEGMENTS} segments)",
          flush=True)

    healthy_flags = _flags_for_all_joint(trajectories, joint_hsmm_params, d_max)

    degraded_by_type = {}
    degraded_traj_pool = []
    for error_type in error_injection.ERROR_TYPES:
        degraded = [error_injection.inject(error_type, t, rng, marginal_hsmm_params) for t in trajectories]
        flags = _flags_for_all_joint(degraded, joint_hsmm_params, d_max)
        degraded_by_type[error_type] = list(zip(flags, (d["window"] for d in degraded)))
        degraded_traj_pool.extend(degraded)
        print(f"  [{tag}] {error_type}: evaluated {len(degraded)} degraded trials", flush=True)

    report = metrics.evaluate(healthy_flags, degraded_by_type)
    report["kl_sanity"] = metrics.kl_sanity(trajectories, degraded_traj_pool, n_nouns)
    return report


def _print_report(report, tag):
    print(f"\n===== {tag} =====")
    print(f"healthy false-positive rate: {report['healthy']['false_positive_rate']:.3f} "
          f"({report['healthy']['false_positive_trials']}/{report['healthy']['n']} control trials flagged)")
    print(f"KL(healthy||degraded) sanity (nonzero expected): {report['kl_sanity']:.4f}\n")
    print(f"{'error type':>14}  {'n':>4}  {'recall':>7}  {'precision':>9}  {'latency':>8}  {'top channel':>16}")
    for error_type, m in report["per_type"].items():
        attr = report["attribution"][error_type]
        top_ch = max(attr, key=attr.get) if any(attr.values()) else "-"
        lat = "n/a" if np.isnan(m["mean_latency"]) else f"{m['mean_latency']:.1f}"
        print(f"{error_type:>14}  {m['n']:>4}  {m['recall']:>7.3f}  {m['precision']:>9.3f}  "
              f"{lat:>8}  {top_ch:>16}")


def _run_cascade(args, config, d_max, n_nouns):
    hsmm_params = params.load_params(args.params)
    recipe_params = recipe_hmm.load_params(args.recipe_params)
    rng = np.random.default_rng(args.seed)

    print(f"\n[cascade] checkpoint: {args.params}")

    synthetic = generate.generate_healthy(hsmm_params, args.n, rng, args.max_ticks, d_max)
    syn_report = evaluate_source(synthetic, hsmm_params, recipe_params, d_max, rng, "cascade/synthetic", n_nouns)
    _print_report(syn_report, "cascade/synthetic")
    plotting.save_figures(syn_report, args.figures_dir, "cascade_synthetic")

    with open(args.sequences) as f:
        sequences = json.load(f)
    real = [
        generate.trajectory_from_real(hsmm_params, s["verb_ids"], s["noun_ids"], d_max)
        for s in sequences[: args.max_real]
    ]
    real_report = evaluate_source(real, hsmm_params, recipe_params, d_max, rng, "cascade/real", n_nouns)
    _print_report(real_report, "cascade/real")
    plotting.save_figures(real_report, args.figures_dir, "cascade_real")


def _run_joint(args, config, d_max, n_nouns):
    joint_hsmm_params = joint_params.load_params(args.joint_params)
    marginal_hsmm_params = joint_params.collapse_to_marginal(joint_hsmm_params)
    rng = np.random.default_rng(args.seed)

    print(f"\n[joint] checkpoint: {args.joint_params}")

    synthetic = generate.generate_healthy_joint(joint_hsmm_params, args.n, rng, args.max_ticks, d_max)
    syn_report = evaluate_source_joint(
        synthetic, joint_hsmm_params, marginal_hsmm_params, d_max, rng, "joint/synthetic", n_nouns
    )
    _print_report(syn_report, "joint/synthetic")
    plotting.save_figures(syn_report, args.figures_dir, "joint_synthetic")

    with open(args.sequences) as f:
        sequences = json.load(f)
    real = [
        generate.trajectory_from_real_joint(joint_hsmm_params, s["verb_ids"], s["noun_ids"], d_max)
        for s in sequences[: args.max_real]
    ]
    real_report = evaluate_source_joint(
        real, joint_hsmm_params, marginal_hsmm_params, d_max, rng, "joint/real", n_nouns
    )
    _print_report(real_report, "joint/real")
    plotting.save_figures(real_report, args.figures_dir, "joint_real")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breakfast_mini.yaml")
    parser.add_argument("--params", default="dataset/processed/breakfast_mini/hsmm_params.npz")
    parser.add_argument("--recipe-params", default="dataset/processed/breakfast_mini/recipe_params.npz")
    parser.add_argument(
        "--joint-params", default=None,
        help="path to a JointHSMMParams .npz (from run_joint.py); if given, also runs the "
             "joint recipe-conditioned evaluation path",
    )
    parser.add_argument(
        "--cascade", action="store_true",
        help="force the cascade path to run even when --joint-params is given (for a direct "
             "A/B in one invocation); with no --joint-params the cascade path always runs",
    )
    parser.add_argument("--sequences", default="dataset/processed/breakfast_mini/sequences.json")
    parser.add_argument("--vocab", default="dataset/processed/breakfast_mini/vocab.json")
    parser.add_argument("--figures-dir", default="dataset/processed/breakfast_mini/figures")
    parser.add_argument("--n", type=int, default=60, help="synthetic healthy trials to generate")
    parser.add_argument("--max-ticks", type=int, default=100, help="length of each synthetic trajectory")
    parser.add_argument("--max-real", type=int, default=80, help="cap real trials for runtime")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    d_max = config["duration"]["d_max_ticks"]
    n_nouns = config["vocab"]["nouns"]

    print("Phase 6 structured MCI evaluation")
    print("all numbers are conditional on the supplied injection ground truth and checkpoint(s).")

    run_cascade = args.joint_params is None or args.cascade
    run_joint = args.joint_params is not None

    if run_cascade:
        _run_cascade(args, config, d_max, n_nouns)
    if run_joint:
        _run_joint(args, config, d_max, n_nouns)

    print(f"\nfigures written to {args.figures_dir}/")


if __name__ == "__main__":
    main()
