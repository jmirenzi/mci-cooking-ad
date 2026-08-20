"""Diagnostics on a fitted joint model: state organisation, transition sparsity, segmentation
quality vs the ground-truth action labels. Read-only; writes nothing."""
import json, sys, argparse
import numpy as np, jax.numpy as jnp, jax
jax.config.update("jax_enable_x64", True)
from cook_ad.hsmm import joint_params, joint_em, em
from cook_ad.recipe import segmentize
from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train.npz")
ap.add_argument("--config", default="configs/breakfast.yaml")
ap.add_argument("--part", default="train")
ap.add_argument("--max", type=int, default=402)
a = ap.parse_args()

d_max = load_config(a.config)["duration"]["d_max_ticks"]
root = "dataset/processed/breakfast"
vocab = json.load(open(f"{root}/vocab.json"))
iv = {v: k for k, v in vocab["verbs"].items()}
inn = {v: k for k, v in vocab["nouns"].items()}
seqs = json.load(open(f"{root}/sequences.json"))
labels = json.load(open(f"{root}/labels.json"))
lab_by_id = {e["trial_id"]: e for e in labels}
split = split_mod.load_split(f"{root}/split.json")
seqs = split_mod.filter_sequences(seqs, split, a.part)[: a.max]

jp = joint_params.load_params(a.joint_params)
lp = joint_params.to_log_probs_joint(jp, d_max)
K = jp.verb_counts.shape[0]; KR = jp.pi_counts.shape[0]
print(f"K={K} K_R={KR} n_trials={len(seqs)}")

v_ids, n_ids, mask = em.pad_batch(seqs)
r_hat, rho, tll = joint_em.infer_recipe(jp, v_ids, n_ids, mask, d_max, chunk_size=8)
r_hat = np.asarray(r_hat)
segres = segmentize.segment_all_conditioned(lp, jnp.asarray(r_hat), v_ids, n_ids, mask, d_max)

# ---- 1. state usage / emission purity
occ = np.zeros(K)
pair_by_state = {}
for i, sr in enumerate(segres):
    z = np.asarray(sr["subtask_per_tick"])
    vv = np.asarray(seqs[i]["verb_ids"]); nn = np.asarray(seqs[i]["noun_ids"])
    for t in range(len(z)):
        occ[z[t]] += 1
        pair_by_state.setdefault(int(z[t]), {}).setdefault((int(vv[t]), int(nn[t])), 0)
        pair_by_state[int(z[t])][(int(vv[t]), int(nn[t]))] += 1
used = int((occ > 0).sum())
print(f"effective K (states used by Viterbi): {used}/{K}")
purity = []
for k, d in pair_by_state.items():
    tot = sum(d.values()); best = max(d.values())
    purity.append(best / tot)
purity = np.array(purity)
w = np.array([sum(pair_by_state[k].values()) for k in pair_by_state])
print(f"state emission purity (max (v,n) share): mean={purity.mean():.3f} occ-weighted={np.average(purity,weights=w):.3f} min={purity.min():.3f}")

# how many distinct (v,n) pairs exist in the data, and how many states map to each
allpairs = {}
for s in seqs:
    for v, n in zip(s["verb_ids"], s["noun_ids"]):
        allpairs[(v, n)] = allpairs.get((v, n), 0) + 1
print(f"distinct (v,n) pairs in data: {len(allpairs)}")
# state -> dominant pair
dom = {k: max(d.items(), key=lambda x: x[1])[0] for k, d in pair_by_state.items()}
from collections import Counter
c = Counter(dom.values())
print(f"distinct dominant pairs covered by states: {len(c)}; pairs with >1 state: {sum(1 for x in c.values() if x>1)}")
print("  most-split pairs:", [(f'{iv[p[0]]}_{inn[p[1]]}', n) for p, n in c.most_common(6)])

# ---- 2. segmentation vs ground truth action runs
gt_segs, dec_segs, agree = [], [], []
import itertools
for i, sr in enumerate(segres):
    tid = seqs[i]["trial_id"]
    vv = seqs[i]["verb_ids"]; nn = seqs[i]["noun_ids"]
    # ground-truth runs = maximal runs of constant (v,n)
    runs = [(k, len(list(g))) for k, g in itertools.groupby(zip(vv, nn))]
    gt_segs.append(len(runs))
    dec_segs.append(len(sr["segments"]))
print(f"segments per trial: ground-truth (v,n) runs mean={np.mean(gt_segs):.1f}  decoded mean={np.mean(dec_segs):.1f}")
print(f"  ratio decoded/gt: {np.mean(np.array(dec_segs)/np.array(gt_segs)):.3f}")

# boundary F1 between decoded segmentation and (v,n)-run segmentation
tp = fp = fn = 0
for i, sr in enumerate(segres):
    vv = seqs[i]["verb_ids"]; nn = seqs[i]["noun_ids"]
    runs = [(k, len(list(g))) for k, g in itertools.groupby(zip(vv, nn))]
    gtb, p = set(), 0
    for _, d in runs[:-1]:
        p += d; gtb.add(p)
    db, p = set(), 0
    for _, d in sr["segments"][:-1]:
        p += d; db.add(p)
    tp += len(gtb & db); fp += len(db - gtb); fn += len(gtb - db)
prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
print(f"boundary precision={prec:.3f} recall={rec:.3f} f1={2*prec*rec/max(1e-9,prec+rec):.3f}")

# ---- 3. transition matrix organisation
lt = np.asarray(lp.log_trans)  # (KR,K,K)
A = np.exp(lt)
ent = -np.sum(np.where(A > 0, A * lt, 0.0), axis=-1)  # (KR,K)
piw = np.asarray(jp.pi_counts); piw = piw / piw.sum()
occn = occ / max(1, occ.sum())
print(f"transition row entropy (nats): mean over used states = {np.mean([ent[:,k].mean() for k in range(K) if occ[k]>0]):.3f} (max possible log(K-1)={np.log(K-1):.3f})")
# effective support: exp(entropy)
eff = np.exp(ent)
print(f"effective out-degree exp(H): mean over used states = {np.mean([eff[:,k].mean() for k in range(K) if occ[k]>0]):.2f}")
print(f"recipe usage: {np.round(np.bincount(r_hat, minlength=KR)/len(r_hat),3)}")
