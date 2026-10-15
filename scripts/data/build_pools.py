"""Write the canonical evaluation pools (docs/data.md §5). Run once; every detector is scored on
these files.

    ./py scripts/data/build_pools.py            # breakfast train + test, epic-tent test splits

Breakfast: dataset/processed/breakfast/pools/{train,test}.pkl -- every usable trial healthy plus
one injected copy per error type, placed by joint_params.npz, seed 0. The test pool is checked
against the pool the cached LLM responses were collected on (--llm-golden), when that is present.

EPIC-Tent: dataset/processed/epic_tent/pools/<split>_test.pkl for split.json and split_prego.json
-- the real sessions and their annotation records, frozen; nothing is sampled.
"""
import argparse
import json
import os
import pickle

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.eval import pools  # noqa: E402
from cook_ad.hsmm import params  # noqa: E402

B = "dataset/processed/breakfast"


class _Unpickler(pickle.Unpickler):   # golden stream files name the pre-restructure Flag class
    def find_class(self, module, name):
        return super().find_class("cook_ad.eval.metric_g" if module.endswith("span_metrics")
                                  else module, name)


def same_as_llm_pool(entries, path):
    """True if the pool's ids and spans equal the LLM pool's, trial for trial, bitwise."""
    llm = _Unpickler(open(path, "rb")).load()["pools"]["real"]
    if len(llm) != len(entries):
        return False
    for e, (h, inj) in zip(entries, llm):
        for a, b in [(e["healthy"], h)] + [(e["injected"][t], inj[t]) for t in pools.ERROR_TYPES]:
            if not (np.array_equal(a["verb_ids"], b["verb_ids"])
                    and np.array_equal(a["noun_ids"], b["noun_ids"])
                    and [tuple(x) for x in a.get("spans", [])] == [tuple(x) for x in b.get("spans", [])]):
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm-golden", default="runs/golden/streams_pred_random.pkl")
    args = ap.parse_args()
    d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
    placement = params.load(pools.PLACEMENT["breakfast"])
    for part in ("train", "test"):
        seqs = pools.split_sequences(f"{B}/sequences.json", f"{B}/split.json", part)
        entries = pools.build_pool(seqs, placement, d_max, seed=args.seed)
        meta = {"corpus": "breakfast", "part": part, "seed": args.seed,
                "placement": pools.PLACEMENT["breakfast"], "n_sequences": len(seqs),
                "n_usable": len(entries), "types": list(pools.ERROR_TYPES)}
        if part == "test" and os.path.exists(args.llm_golden):
            meta["equals_llm_pool"] = same_as_llm_pool(entries, args.llm_golden)
            print(f"test pool == LLM pool ({args.llm_golden}): {meta['equals_llm_pool']}")
        pools.save_pool(pools.pool_path("breakfast", part), entries, meta)
        print(f"{part}: {len(entries)} usable of {len(seqs)} -> {pools.pool_path('breakfast', part)}")
    E = "dataset/processed/epic_tent"
    for split in ("split.json", "split_prego.json"):
        path, n = pools.freeze_epic_tent_pool(f"{E}/sequences.json", f"{E}/anomalies.json",
                                              f"{E}/{split}", "test")
        print(f"epic_tent {split} test: {n} sessions -> {path}")
    print(json.dumps({"placement": pools.PLACEMENT["breakfast"]}))


if __name__ == "__main__":
    main()
