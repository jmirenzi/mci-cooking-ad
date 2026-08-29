"""Can the detector RANK a near substitution below a far one? The paired test docs/anomaly.md's
embeddings argument actually rests on.

run_detect_eval.py's trial_loc scorecard cannot answer this: it asks "was the injection
flagged", and the baseline flags near substitutions about as reliably as far ones -- `water` for
`milk` still lands on a (state, noun) cell with no training evidence, so it is not a quieter
anomaly, merely an equally loud one. What the embedding track is meant to fix is that the two
are indistinguishable in SEVERITY.

Paired so nothing else can move: for each trial one interior segment is chosen once, and two
degraded copies differ in exactly one respect -- the noun that replaces it.

    near  nearest embedding neighbour of the segment's noun   (milk -> water)
    far   a token with no training evidence for that state    (milk -> bowl)

Reports peak s_noun in the injection window for each, plus the rank accuracy -- the fraction of
pairs with s_noun(far) > s_noun(near), where 0.5 is chance, i.e. no gradation at all. Every arm
scores the SAME degraded streams (built from --traj-params), so the contrast is exact.
"""
import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.eval import batch
from cook_ad.hsmm import joint_params, kernel as kernel_mod
from cook_ad.synthetic import error_injection, generate


def paired_injections(traj, rng, marg, neighbours):
    """One segment, two replacement nouns. Returns (near, far) degraded trials or None if the
    trial has no interior segment whose noun admits both."""
    bounds = error_injection._seg_bounds(traj["segments"])
    interior = list(range(1, len(bounds) - 1))
    if not interior:
        return None
    i = int(rng.choice(interior))
    start, end, state, _ = bounds[i]
    current = int(traj["noun_ids"][start])

    near_id = int(np.asarray(neighbours["noun"])[current])
    row = marg.noun_counts[state]
    far_id = int(rng.choice(error_injection._unseen_candidates(row, current)))
    if near_id == current or far_id == current:
        return None

    out = []
    for new_id in (near_id, far_id):
        noun_ids = np.array(traj["noun_ids"])
        noun_ids[start:end] = new_id
        out.append(error_injection._result(
            np.array(traj["verb_ids"]), noun_ids, start, end - 1, "substitution",
            np.arange(len(noun_ids)), edited_ticks=np.arange(start, end),
        ))
    return out[0], out[1], (start, end)


def peak_s_noun(traces, windows):
    return np.array([float(np.max(t.s_noun[a:b])) for t, (a, b) in zip(traces, windows)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train_lexhard.npz")
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", choices=["train", "test"], default="test")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--embeddings", default="dataset/processed/breakfast/embeddings.npz")
    ap.add_argument("--lam", type=float, nargs="*", default=[0.0, 0.15, 0.30])
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--uniform", action="store_true", help="ablation: uniform backoff, not semantic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8)
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    seqs = json.load(open(args.sequences))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(args.split_file), args.split_part)

    base_jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(base_jp)
    _, emb_n = kernel_mod.load_embeddings(args.embeddings)
    neighbours = {"noun": kernel_mod.nearest_neighbours(emb_n)}

    # Injections are placed with the BASELINE model and reused by every arm -- one common set
    # of degraded streams, so nothing but the scoring emission differs between arms.
    traj = generate.trajectories_from_real_joint(base_jp, seqs, d_max, chunk_size=args.chunk_size)
    usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    rng = np.random.default_rng(args.seed)
    pairs = [p for p in (paired_injections(t, rng, marg, neighbours) for t in usable) if p]
    near, far, windows = [p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs]
    print(f"{len(pairs)} paired near/far substitutions from {len(usable)} usable trials "
          f"({args.split_part} split)\n")

    print(f"{'arm':>22} {'s_noun near':>12} {'s_noun far':>11} {'gap':>8} {'rank acc':>9}")
    print("-" * 66)
    for lam in args.lam:
        jp = base_jp
        label = "baseline (no kernel)"
        if lam > 0:
            s_v, s_n = kernel_mod.kernels_from_embeddings(
                args.embeddings, args.tau, lam, lam_verb=0.0, uniform=args.uniform)
            jp = base_jp._replace(kernel_v=s_v, kernel_n=s_n)
            label = f"{'uniform' if args.uniform else 'kernel'} lam={lam}"
        sn = peak_s_noun(batch.compute_traces_joint(jp, near, d_max, chunk_size=args.chunk_size)[0], windows)
        sf = peak_s_noun(batch.compute_traces_joint(jp, far, d_max, chunk_size=args.chunk_size)[0], windows)
        acc = float(np.mean(sf > sn))
        print(f"{label:>22} {sn.mean():12.2f} {sf.mean():11.2f} {sf.mean()-sn.mean():8.2f} {acc:9.3f}")


if __name__ == "__main__":
    main()
