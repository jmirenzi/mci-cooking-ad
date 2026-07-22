import argparse
import json

import jax
import numpy as np

from cook_ad.anomaly import surprise
from cook_ad.data.config import load_config
from cook_ad.eval import batch, metrics, plotting
from cook_ad.hsmm import params
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breakfast_mini.yaml")
    parser.add_argument("--params", default="dataset/processed/breakfast_mini/hsmm_params.npz")
    parser.add_argument("--recipe-params", default="dataset/processed/breakfast_mini/recipe_params.npz")
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

    hsmm_params = params.load_params(args.params)
    recipe_params = recipe_hmm.load_params(args.recipe_params)
    rng = np.random.default_rng(args.seed)

    print("Phase 6 structured MCI evaluation")
    print(f"checkpoint: {args.params}")
    print("all numbers are conditional on the supplied injection ground truth and this checkpoint.")

    # Synthetic healthy source (exact ground truth).
    synthetic = generate.generate_healthy(hsmm_params, args.n, rng, args.max_ticks, d_max)
    syn_report = evaluate_source(synthetic, hsmm_params, recipe_params, d_max, rng, "synthetic", n_nouns)
    _print_report(syn_report, "synthetic")
    plotting.save_figures(syn_report, args.figures_dir, "synthetic")

    # Real held-out Breakfast source (Viterbi-segmented ground truth).
    with open(args.sequences) as f:
        sequences = json.load(f)
    real = [
        generate.trajectory_from_real(hsmm_params, s["verb_ids"], s["noun_ids"], d_max)
        for s in sequences[: args.max_real]
    ]
    real_report = evaluate_source(real, hsmm_params, recipe_params, d_max, rng, "real", n_nouns)
    _print_report(real_report, "real")
    plotting.save_figures(real_report, args.figures_dir, "real")

    print(f"\nfigures written to {args.figures_dir}/")


if __name__ == "__main__":
    main()
