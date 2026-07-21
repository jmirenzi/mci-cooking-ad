import argparse
import json
import time

import jax

from cook_ad.data.config import load_config
from cook_ad.hsmm import em, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breakfast.yaml")
    parser.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    parser.add_argument("--out", default="dataset/processed/breakfast/hsmm_params.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-restarts", type=int, default=None, help="override configs/breakfast.yaml em.restarts")
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.sequences) as f:
        sequences = json.load(f)

    n_restarts = args.n_restarts if args.n_restarts is not None else config["em"]["restarts"]

    key = jax.random.PRNGKey(args.seed)
    start = time.time()
    best_params, best_loglik, history = em.run_em(
        key,
        sequences,
        k_subtask=config["k_subtask"],
        d_max=config["duration"]["d_max_ticks"],
        vocab_verbs=config["vocab"]["verbs"],
        vocab_nouns=config["vocab"]["nouns"],
        n_restarts=n_restarts,
        max_iters=args.max_iters,
        tol=args.tol,
        annealing=config["em"]["annealing"],
        chunk_size=config["em"]["chunk_size"],
        progress=True,
    )
    elapsed = time.time() - start

    print(f"sequences: {len(sequences)}")
    print(f"restarts: {n_restarts}, max_iters: {args.max_iters}")
    print(f"best log-likelihood: {best_loglik}")
    print(f"iterations per restart: {[len(h) for h in history]}")
    print(f"elapsed: {elapsed:.1f}s")

    params.save_params(best_params, args.out)
    print(f"saved fitted params to {args.out}")


if __name__ == "__main__":
    main()
