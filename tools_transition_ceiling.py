"""Is the transition channel's miss rate a MODEL limit or a DATA limit?

For every omission / transposition injection this asks a question with no thresholds in it:
the junction the injection creates -- state u followed by state v where the healthy trial had
u -> w -> v -- how often is u -> v something the training split ALREADY contains, either
anywhere or within this trial's own recipe cluster?

A junction the data supports is not an anomaly the model can be blamed for missing: Breakfast
participants genuinely vary the order of steps, so some fraction of "wrong order" injections
produce orderings other people really used. That fraction is the ceiling. Everything below it
is model headroom.
"""
import argparse
import json
from collections import defaultdict

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.hsmm import joint_params
from cook_ad.synthetic import error_injection, generate

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train.npz")
ap.add_argument("--split-part", default="train")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
root = "dataset/processed/breakfast"
seqs = json.load(open(f"{root}/sequences.json"))
seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{root}/split.json"), a.split_part)
jp = joint_params.load_params(a.joint_params)
marg = joint_params.collapse_to_marginal(jp)
traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=32)
usable = [t for t in traj if len(t["segments"]) >= error_injection.MIN_SEGMENTS]

# bigram inventory over the healthy decode: globally, and per recipe cluster
global_bigrams, recipe_bigrams = defaultdict(int), defaultdict(int)
for t in usable:
    st = [s for s, _ in t["segments"]]
    for u, v in zip(st[:-1], st[1:]):
        global_bigrams[(u, v)] += 1
        recipe_bigrams[(t["recipe_id"], u, v)] += 1
print(f"{len(usable)} trials, {sum(global_bigrams.values())} transitions, "
      f"{len(global_bigrams)} distinct global bigrams, {len(recipe_bigrams)} distinct (recipe,bigram)")

rng = np.random.default_rng(a.seed)
for et in ("omission", "transposition"):
    seen_global = seen_recipe = novel = total = 0
    for t in usable:
        st = [s for s, _ in t["segments"]]
        r = t["recipe_id"]
        # replay the injector's own segment choice under the same rng draw sequence
        deg = error_injection.inject(et, t, rng, marg)
        # recover the created junctions by comparing the healthy and degraded state sequences
        # via tick_map: the degraded stream's own run structure over ORIGINAL segment ids
        seg_of_tick = np.concatenate([np.full(d, s) for s, d in t["segments"]])
        mapped = seg_of_tick[deg["tick_map"]]
        runs = [mapped[0]]
        for x in mapped[1:]:
            if x != runs[-1]:
                runs.append(x)
        healthy_pairs = set(zip(st[:-1], st[1:]))
        for u, v in zip(runs[:-1], runs[1:]):
            if (u, v) in healthy_pairs:
                continue  # a junction the healthy trial already had; not created by the injection
            total += 1
            if (r, u, v) in recipe_bigrams:
                seen_recipe += 1
            elif (u, v) in global_bigrams:
                seen_global += 1
            else:
                novel += 1
    print(f"\n{et}: {total} injection-created junctions over {len(usable)} trials")
    print(f"  already in this trial's own recipe cluster: {seen_recipe/total:.3f}")
    print(f"  elsewhere in the corpus but not this recipe: {seen_global/total:.3f}")
    print(f"  never seen anywhere (model CAN flag these):  {novel/total:.3f}")
