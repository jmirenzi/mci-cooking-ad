"""Stage 5 of training (optional): the per-recipe visit-count statistics of the s_count channel
(hsmm/visit_counts.py; docs/detectors.md §5.7), from the final model's own Viterbi decode of the
training split. Written as JSON next to the model.

    ./py scripts/train/fit_visit_counts.py --in runs/joint_final.npz --out runs/joint_final.counts.json
"""
import argparse
import json

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data import split as split_mod  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.hsmm import params  # noqa: E402
from cook_ad.hsmm.train import lexical_init  # noqa: E402
from cook_ad.hsmm.visit_counts import VisitCounts  # noqa: E402
from cook_ad.synthetic import generate  # noqa: E402


def idle_states(jp, vocab, sil_verb="stall"):
    """States whose most likely verb is the idle class: not counted (they are gaps, not steps)."""
    if sil_verb not in vocab["verbs"]:
        return []
    return np.flatnonzero(np.argmax(np.asarray(jp.verb_counts), axis=-1) == vocab["verbs"][sil_verb]).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", default="train")
    ap.add_argument("--labels", default="dataset/processed/breakfast/labels.json")
    ap.add_argument("--vocab", default=None, help="vocab.json, to leave idle states uncounted")
    ap.add_argument("--recipe-labels", action="store_true",
                    help="decode each trial under its labelled dish instead of the inferred one")
    args = ap.parse_args()

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    seqs = json.load(open(args.sequences))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(args.split_file), args.split_part)
    jp = params.load(args.inp)
    r_given = None
    if args.recipe_labels:
        r_given, _ = lexical_init.recipe_label_assignment(seqs, json.load(open(args.labels)))
    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=32, r_hat=r_given)
    ignore = idle_states(jp, json.load(open(args.vocab))) if args.vocab else []
    vc = VisitCounts.fit([(t["recipe_id"], [k for k, _ in t["segments"]]) for t in traj], ignore)
    vc.save(args.out)
    print(f"visit counts from {len(traj)} sessions, {len(vc.n)} recipes, {len(ignore)} idle states "
          f"ignored -> {args.out}")


if __name__ == "__main__":
    main()
