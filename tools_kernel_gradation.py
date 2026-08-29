"""Does the emission actually GRADE a substitution, and what does grading cost on healthy data?

docs/anomaly.md's case for embeddings rests on one claim: a fixed-vocabulary categorical gives
`water for milk` and `knife for milk` nearly the same surprise, so the detector cannot say
"wrong object, but nearly right". This measures that per state and then prices it -- the same
smoothing that separates near from far also lowers surprise on the FAR substitution, which is
what the existing benchmark scores (smooth_params.py's objection).

Reports, per state k with modal noun m:
  healthy  -log P(m | k)                     -- the cost paid on every correct observation
  near     -log P(nearest_neighbour(m) | k)  -- the near substitution
  far      -log P(argmin_count(k) | k)       -- what error_injection's 'random'/'hardest' picks
  gap      far - near                        -- the gradation. ~0 nats == no gradation at all.
"""
import argparse
import json

import numpy as np

from cook_ad.hsmm import joint_params, kernel as kernel_mod


def rows(jp, nn):
    # emissions are shared across recipes, so the joint tables carry the whole story
    log_emit_n = np.asarray(joint_params.to_log_probs_joint(jp, 2).log_emit_n)
    counts = np.asarray(jp.noun_counts)
    out = []
    for k in range(log_emit_n.shape[0]):
        m = int(np.argmax(counts[k]))
        far = int(np.argsort(counts[k])[0])
        if far == m:
            far = int(np.argsort(counts[k])[1])
        out.append((k, m, int(nn[m]), far,
                    -log_emit_n[k, m], -log_emit_n[k, int(nn[m])], -log_emit_n[k, far]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train.npz")
    ap.add_argument("--embeddings", default="dataset/processed/breakfast/embeddings.npz")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--lam", type=float, nargs="*", default=[0.0, 0.15, 0.30])
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=12, help="states listed, by occupancy")
    args = ap.parse_args()

    vocab = json.load(open(args.vocab))
    nw = [w for w, _ in sorted(vocab["nouns"].items(), key=lambda kv: kv[1])]
    _, emb_n = kernel_mod.load_embeddings(args.embeddings)
    nn = kernel_mod.nearest_neighbours(emb_n)
    jp = joint_params.load_params(args.joint_params)
    occupancy = np.asarray(jp.noun_counts).sum(axis=1)
    ranked = np.argsort(-occupancy)[: args.top]

    for lam in args.lam:
        s_n = kernel_mod.similarity_kernel(emb_n, args.tau, lam)
        r = rows(jp._replace(kernel_n=s_n) if lam > 0 else jp, nn)
        name = "BASELINE (no kernel)" if lam == 0 else f"KERNEL lam={lam} tau={args.tau}"
        print(f"\n=== {name} ===")
        print(f"{'state':>5} {'modal noun':>14} {'near sub':>14} {'far sub':>14} "
              f"{'healthy':>8} {'near':>8} {'far':>8} {'gap':>7}")
        for k in ranked:
            _, m, near, far, h, sn, sf = r[k]
            print(f"{k:5d} {nw[m]:>14} {nw[near]:>14} {nw[far]:>14} "
                  f"{h:8.2f} {sn:8.2f} {sf:8.2f} {sf - sn:7.2f}")
        allr = np.array([(h, sn, sf) for _, _, _, _, h, sn, sf in r])
        w = occupancy / occupancy.sum()
        print(f"      occupancy-weighted mean:{'':>30} "
              f"{allr[:, 0] @ w:8.2f} {allr[:, 1] @ w:8.2f} {allr[:, 2] @ w:8.2f} "
              f"{(allr[:, 2] - allr[:, 1]) @ w:7.2f}")


if __name__ == "__main__":
    main()
