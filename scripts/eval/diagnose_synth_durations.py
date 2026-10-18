# ruff: noqa: E402  (imports follow the sys.path setup and the variant switch)
"""(docs/drift_synth.md §9.5) Are the HSMM's duration models too smooth? Per (recipe, state) cell of the N = 250 model:
fitted negative binomial vs the empirical durations of the same cell in the model's own Viterbi
decode of its training sessions."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "scripts/eval")
import numpy as np

import eval_synth_prior as E
E.set_variant("cal")
from cook_ad.xla_env import disable_gpu_autotuning
disable_gpu_autotuning()
import jax
jax.config.update("jax_enable_x64", True)
from cook_ad.data.config import load_config
from cook_ad.hsmm import params as hparams
from cook_ad.synthetic import generate

PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0   # ./py scripts/eval/diagnose_synth_durations.py 0
pid = f"p{PID:02d}"
d = E.ROOT / "persons" / pid
labels = json.loads((d / "null" / "labels.json").read_text())
seqs = E._in_order(json.loads((d / "null" / "sequences.json").read_text()), labels)
vocab = json.loads((d / "null" / "vocab.json").read_text())
iv = {i: v for v, i in vocab["verbs"].items()}
inn = {i: n for n, i in vocab["nouns"].items()}
d_max = load_config(E.CONFIG)["duration"]["d_max_ticks"]
p = hparams.load(str(d / "model_s250" / "model.npz"))
tabs = hparams.to_log_tables(p, d_max)
pmf = np.exp(np.asarray(tabs.log_pmf))           # (K_R, K, D): P(D = j + 1)
cdf = np.cumsum(pmf, axis=-1)
traj = generate.trajectories_from_real_joint(p, seqs[:250], d_max, chunk_size=8)

cells = {}
for t in traj:
    r = t["recipe_id"]
    pos = 0
    for k, dd in t["segments"][:-1]:
        tok = (iv[int(t["verb_ids"][pos])], inn[int(t["noun_ids"][pos])])
        cells.setdefault((r, k), {"tok": tok, "d": []})["d"].append(int(dd))
        pos += dd

rows = []
for (r, k), c in cells.items():
    x = np.array(c["d"])
    if len(x) < 20:
        continue
    D = np.arange(1, d_max + 1)
    m_mean = float((pmf[r, k] * D).sum())
    m_sd = float(np.sqrt((pmf[r, k] * (D - m_mean) ** 2).sum()))
    med = float(np.median(x))
    cut = max(1, int(round(0.125 * med)))         # abandonment keeps 5-20% (mid 12.5%)
    q01 = int(np.quantile(x, 0.01))
    rows.append(dict(tok=" ".join(c["tok"]), n=len(x), med=med, emp_sd=float(x.std()), mod_mean=m_mean,
                     mod_sd=m_sd, sd_ratio=m_sd / max(x.std(), 1e-9),
                     mod_cdf_at_emp_q01=float(cdf[r, k, q01 - 1]),
                     cut=cut, emp_cdf_cut=float(np.mean(x <= cut)), mod_cdf_cut=float(cdf[r, k, cut - 1]),
                     idle=c["tok"] == ("stall", "idle")))

def summ(sel, label):
    a = lambda k: np.array([r[k] for r in sel])  # noqa: E731
    print(f"{label:22s} cells {len(sel):3d} | median run {np.median(a('med')):5.1f} ticks | "
          f"model sd / empirical sd: median {np.median(a('sd_ratio')):.2f} (p90 {np.quantile(a('sd_ratio'), .9):.2f}) | "
          f"model P(D <= empirical 1% point): median {np.median(a('mod_cdf_at_emp_q01')):.3f} | "
          f"P(D <= 12.5% of median): empirical {np.median(a('emp_cdf_cut')):.3f}, model {np.median(a('mod_cdf_cut')):.3f}; "
          f"cells where the model gives that cut <= 0.001: {np.mean(a('mod_cdf_cut') <= 1e-3):.2f}")

print(pid)
summ(rows, "all cells")
summ([r for r in rows if r["idle"]], "idle gap")
summ([r for r in rows if not r["idle"] and r["tok"].split()[0] in ("take", "put", "wash")], "take/put/wash")
summ([r for r in rows if not r["idle"] and r["tok"].split()[0] not in ("take", "put", "wash")], "recipe action")
long = sorted([r for r in rows if r["med"] >= 8 and not r["idle"]], key=lambda r: -r["sd_ratio"])[:8]
print("widest recipe-action cells (median >= 8 ticks):")
for r in long:
    print(f"  {r['tok']:22s} n={r['n']:4d} median {r['med']:5.1f} emp sd {r['emp_sd']:6.1f} model sd {r['mod_sd']:6.1f} "
          f"| cut {r['cut']}: emp {r['emp_cdf_cut']:.3f} model {r['mod_cdf_cut']:.4f}")
