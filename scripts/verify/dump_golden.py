"""Golden outputs for the restructure (docs/restructure.md §9.1). TEMPORARY: deleted in Phase G.

Runs the detectors through the library calls the eval scripts make, on the same pools in the same
RNG order, and pickles every per-trial result the equivalence check (compare_golden.py) needs:

    <out-dir>/bf_<model>.pkl   Breakfast test pool (run_live_eval.py's streams), per model config
    <out-dir>/et_<variant>.pkl EPIC-Tent default / prego / prego_trimmed (run_epic_tent_eval.py)

Per trial and arm: the (T, 7) bool flag matrix (channel order CHANNELS), plus r_hat, the
whole-trial log-likelihood (infer_recipe's trial_ll and the live filter's sum log P(o_t|o_<t)),
z_star (encodes the Viterbi segments exactly: the HSMM has no self-transitions), and float
diagnostics (hindsight channel values, live p-values and event posteriors) for locating a
mismatch. LLM arms: EPIC-Tent predictive flags are replayed from the cache (--max-requests 0);
Breakfast LLM streams come from run_llm_eval.py --dump-streams (see run_tables.sh).

    ./py scripts/verify/dump_golden.py --out-dir runs/golden [--only bf_lexhard,et_default] [--skip-llm]
"""
import argparse
import json
import math
import os
import pickle
import time

from cook_ad.xla_env import disable_gpu_autotuning

disable_gpu_autotuning()

import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from cook_ad.core import AnomalyDetector, OperatingPoint, Trial  # noqa: E402
from cook_ad.data import split as split_mod  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.data.parse_epic_tent import CANONICAL_ORDER, ERROR_GROUPS  # noqa: E402
from cook_ad.detectors import llm_predictive as llm_pred  # noqa: E402
from cook_ad.detectors.hsmm_hindsight import CHANNELS, HindsightHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_lag import LagHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_live import LiveHSMMDetector  # noqa: E402
from cook_ad.detectors.narrate import Lexicon  # noqa: E402
from cook_ad.detectors.thresholds import BadSets  # noqa: E402
from cook_ad.eval import metric_e, metric_g, pools  # noqa: E402
from cook_ad.hsmm import batch as hbatch  # noqa: E402
from cook_ad.hsmm import params as hparams  # noqa: E402
from cook_ad.hsmm.model import HSMM  # noqa: E402
from cook_ad.hsmm.train.lexical_init import recipe_label_assignment  # noqa: E402
from cook_ad.llm import client as llm_client  # noqa: E402
from cook_ad.llm import prompts  # noqa: E402
from cook_ad.synthetic import generate  # noqa: E402

DOMAIN, INTERLEAVED = "assembly", (("read", "instruction"),)   # as scripts/eval/eval_epic_tent.py
INF_LAG = "inf"
DEFAULT_ALPHA = 5e-3
# run_detect_eval.ALPHAS, the grid run_live_eval builds its live tables on. Copied, not imported:
# the binned event mass is a cumsum over this grid, so the grid itself is part of the golden.
BF_ALPHA_GRID = [0.5, 0.35, 0.2, 0.15, 0.1, 0.07, 0.05, 0.035, 0.02, 0.015, 0.01, 0.007, 0.005,
                 0.003, 0.002, 1e-3, 5e-4, 1e-4, 1e-5, 1e-7, 1e-10]
BF_HINDSIGHT_ALPHAS = (0.01, 0.05)
PER_CHANNEL = {"s_temporal": 1e-5}          # on top of DEFAULT_ALPHA = 0.005
BF_LIVE_OP = (0.02, 0.5)                    # (alpha, kappa)
LAGS = [0, 5, INF_LAG]
ET_KAPPA = 0.5
EVENT_KEYS = ("stall", "early", "trans", "recipe")

BF = "dataset/processed/breakfast"
ET_DIR = "dataset/processed/epic_tent"
BF_CONFIGS = {
    "bf_lexhard": dict(params=f"{BF}/joint_params_train_lexhard.npz", recipe_labels=False),
    "bf_dish_given": dict(params=f"{BF}/joint_params_train_dish.npz", recipe_labels=True),
    "bf_dish_inferred": dict(params=f"{BF}/joint_params_train_dish.npz", recipe_labels=False),
}
ET_CONFIGS = {
    "et_default": dict(split=f"{ET_DIR}/split.json", params=f"{ET_DIR}/joint_params_clean.npz",
                       trim=False, llm=True),
    "et_prego": dict(split=f"{ET_DIR}/split_prego.json",
                     params=f"{ET_DIR}/joint_params_clean_prego.npz", trim=False, llm=False),
    "et_prego_trimmed": dict(split=f"{ET_DIR}/split_prego.json",
                             params=f"{ET_DIR}/joint_params_clean_prego.npz", trim=True, llm=True),
}


def stack(flag_dict):
    return np.stack([np.asarray(flag_dict[c], dtype=bool) for c in CHANNELS], axis=1)


# --------------------------------------------------------------------------------------------
# The rows, through the detector API
_EVENT_KEY = {"s_temporal": "stall", "s_dur_two": "early", "s_transition": "trans",
              "s_recipe_transition": "recipe"}


class NewEngine:
    """hindsight / trial_ll / live_one through the detector classes."""

    def __init__(self, path, d_max, grid):
        self.hsmm = HSMM.load(path, d_max)
        self.bad = BadSets(self.hsmm, grid)

    @staticmethod
    def trials(seqs, known_r):
        return [Trial(str(i), np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]),
                      None if known_r is None else int(known_r[i])) for i, s in enumerate(seqs)]

    def hindsight(self, seqs, chunk_size, known_r, alphas, per_channel):
        det = HindsightHSMMDetector(self.hsmm, self.bad, OperatingPoint(0.005), chunk_size)
        pc = per_channel_alphas(per_channel)
        out = []
        for tr in det.traces(self.trials(seqs, known_r)):
            arms = {f"hindsight@{a:g}": stack(det.flags(tr, OperatingPoint(a))) for a in alphas}
            arms["hindsight@per_channel"] = stack(det.flags(tr, OperatingPoint(pc)))
            out.append({"arms": arms, "r_hat": tr.r_hat, "z_star": tr.z_star,
                        "values": {c: tr.surprise[c] for c in CHANNELS},
                        "pi_at_zstar": tr.pi_at_zstar})
        return out

    def trial_ll(self, seqs, chunk_size):
        verb, noun, mask = hbatch.pad([s["verb_ids"] for s in seqs], [s["noun_ids"] for s in seqs])
        r, rho, ll = hbatch.infer_recipe(self.hsmm.tables, verb, noun, mask, self.hsmm.d_max,
                                         chunk_size)
        return np.asarray(r), np.asarray(rho), np.asarray(ll)

    def live_one(self, s, recipe, per_channel_alpha, kappa):
        op = OperatingPoint(dict(per_channel_alpha), kappa)
        trial = Trial("t", np.asarray(s["verb_ids"]), np.asarray(s["noun_ids"]), recipe)
        arms, events, tr = {}, {}, None
        for lag in LAGS:
            det = (LiveHSMMDetector(self.hsmm, self.bad, op) if lag == 0 else
                   LagHSMMDetector(self.hsmm, self.bad, op, math.inf if lag == INF_LAG else lag))
            tr = det.trace(trial)
            arms[f"live_L{lag}"] = stack(det.flags(tr))
            events[lag] = {_EVENT_KEY[c]: tr.events[c] for c in _EVENT_KEY}
        diag = {"p_emit": tr.p["s_emit"], "p_verb": tr.p["s_verb"], "p_noun": tr.p["s_noun"],
                "log_norm": tr.log_norm, "events": events, "alphas": np.asarray(tr.alphas)}
        return arms, float(np.sum(tr.log_norm)), diag


def engine(path, d_max, grid):
    return NewEngine(path, d_max, grid)


def per_channel_alphas(overrides, alpha=DEFAULT_ALPHA):
    out = {c: alpha for c in CHANNELS}
    out.update(overrides)
    return out


# --------------------------------------------------------------------------------------------
# Breakfast


def breakfast_pool(params, seqs, d_max, chunk_size, recipe_labels, labels_path):
    """The per-type pool, placement by the scored model. -> ({group: [trajectory]}, known_r)."""
    streams, keep = pools.injected_per_type(params, seqs, d_max, seed=0, chunk_size=chunk_size)
    known_r = None
    if recipe_labels:
        known_r, _ = recipe_label_assignment([seqs[i] for i in keep], json.load(open(labels_path)))
    return streams, known_r


def dump_breakfast(name, cfg, out_dir, chunk_size=8):
    d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
    jp = hparams.load(cfg["params"])
    seqs = json.load(open(f"{BF}/sequences.json"))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{BF}/split.json"), "test")
    streams, known_r = breakfast_pool(jp, seqs, d_max, chunk_size, cfg["recipe_labels"],
                                      f"{BF}/labels.json")
    eng = engine(cfg["params"], d_max, BF_ALPHA_GRID)
    alpha, kappa = BF_LIVE_OP
    live_alpha = {c: alpha for c in CHANNELS}
    groups = {}
    for g, items in streams.items():
        t0 = time.time()
        hs = eng.hindsight(items, chunk_size, known_r, BF_HINDSIGHT_ALPHAS, PER_CHANNEL)
        r_inf, rho, ll = eng.trial_ll(items, chunk_size)
        rows = []
        for i, (s, h) in enumerate(zip(items, hs)):
            rec = None if known_r is None else int(known_r[i])
            arms, live_ll, diag = eng.live_one(s, rec, live_alpha, kappa)
            h["arms"].update(arms)
            rows.append({"verb_ids": np.asarray(s["verb_ids"]), "noun_ids": np.asarray(s["noun_ids"]),
                         "spans": [tuple(x) for x in s.get("spans", [])], **h,
                         "r_infer": int(r_inf[i]), "rho": rho[i], "trial_ll": float(ll[i]),
                         "live_ll": live_ll, "live_diag": diag})
        groups[g] = rows
        print(f"  [{name}] {g}: {len(rows)} trials ({time.time() - t0:.0f}s)", flush=True)
    payload = {"meta": {**cfg, "known_r": None if known_r is None else np.asarray(known_r),
                        "hindsight_alphas": BF_HINDSIGHT_ALPHAS, "per_channel": PER_CHANNEL,
                        "live_op": BF_LIVE_OP, "lags": LAGS, "alpha_grid": BF_ALPHA_GRID,
                        "channels": CHANNELS},
               "groups": groups}
    with open(os.path.join(out_dir, f"{name}.pkl"), "wb") as fh:
        pickle.dump(payload, fh)


# --------------------------------------------------------------------------------------------
# EPIC-Tent


def et_llm_flags(trajectories, lexicon, vocab, labels, sil, trimmed):
    """Both predictive variants, replayed from the cache through LLMPredictiveDetector's step
    loop (reset, step, finish). -> ({arm: [[(tick, strength, channel)]]}, client stats)."""
    out, stats = {}, {}
    for variant in prompts.VARIANTS:
        client = llm_client.ChatClient(
            model="gemma3:27b", base_url="http://localhost:11434/v1",
            cache_dir=f"{ET_DIR}/llm_cache", max_requests=0,
            temperature=llm_pred.TEMPERATURE, max_tokens=llm_pred.MAX_TOKENS)
        det = llm_pred.LLMPredictiveDetector(
            client, vocab, lexicon, variant=variant, labels=labels, domain=DOMAIN,
            sil_verb=sil[0], sil_noun=sil[1], canonical_order=CANONICAL_ORDER,
            interleaved=INTERLEAVED)
        det.fit([])
        flags = []
        for t in trajectories:
            d = AnomalyDetector.detect_trial(det, Trial("t", np.asarray(t["verb_ids"]),
                                                        np.asarray(t["noun_ids"])))
            flags.append(sorted(metric_g.Flag(a.tick, a.score, a.channel) for a in d.alarms))
        if trimmed:
            flags = [metric_e.drop_truncated_under_run(f, t, lexicon)
                     for f, t in zip(flags, trajectories)]
        arm = f"llm-predictive-{variant}"
        out[arm] = [[(int(f.tick), float(f.strength), str(f.channel)) for f in fl] for fl in flags]
        stats[arm] = client.stats()
    return out, stats


def et_rows(eng, trajectories, chunk_size, live_alpha):
    hs = eng.hindsight(trajectories, chunk_size, None, (), PER_CHANNEL)
    r_inf, rho, ll = eng.trial_ll(trajectories, chunk_size)
    rows = []
    for i, (t, h) in enumerate(zip(trajectories, hs)):
        arms, live_ll, diag = eng.live_one(t, None, live_alpha, ET_KAPPA)
        h["arms"].update(arms)
        rows.append({"verb_ids": np.asarray(t["verb_ids"]), "noun_ids": np.asarray(t["noun_ids"]),
                     **h, "r_infer": int(r_inf[i]), "rho": rho[i], "trial_ll": float(ll[i]),
                     "live_ll": live_ll, "live_diag": diag})
    return rows


def dump_epic_tent(name, cfg, out_dir, skip_llm, chunk_size=8):
    config = load_config("configs/epic_tent.yaml")
    d_max = config["duration"]["d_max_ticks"]
    sil = (config["ambient_gaps"]["sil_verb"], config["ambient_gaps"]["sil_noun"])
    vocab = json.load(open(f"{ET_DIR}/vocab.json"))
    labels = json.load(open(f"{ET_DIR}/labels.json"))
    seqs = json.load(open(f"{ET_DIR}/sequences.json"))
    truth_by_id = {t["trial_id"]: t for t in json.load(open(f"{ET_DIR}/anomalies.json"))}
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(cfg["split"]), "test")[:29]
    truth = [truth_by_id[s["trial_id"]] for s in seqs]
    jp = hparams.load(cfg["params"])
    lexicon = Lexicon(vocab, jp, sil_verb=sil[0], sil_noun=sil[1])
    pc = per_channel_alphas(PER_CHANNEL)
    eng = engine(cfg["params"], d_max, sorted({DEFAULT_ALPHA, *PER_CHANNEL.values()}))

    sets = {}
    if cfg["trim"]:
        for gname in ERROR_GROUPS:
            trimmed = pools.trim_at_first_mistake(seqs, truth, ERROR_GROUPS[gname])
            sets[gname] = generate.trajectories_from_real_joint(
                jp, [s for s, _ in trimmed], d_max, chunk_size=chunk_size)
    else:
        sets["all"] = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=chunk_size)

    groups, llm, llm_stats = {}, {}, {}
    for key, trajectories in sets.items():
        t0 = time.time()
        groups[key] = et_rows(eng, trajectories, chunk_size, pc)
        if cfg["llm"] and not skip_llm:
            llm[key], llm_stats[key] = et_llm_flags(trajectories, lexicon, vocab, labels, sil,
                                                    cfg["trim"])
        print(f"  [{name}] {key}: {len(trajectories)} trials ({time.time() - t0:.0f}s)", flush=True)
    payload = {"meta": {**cfg, "per_channel": pc, "kappa": ET_KAPPA, "lags": LAGS,
                        "channels": CHANNELS, "trial_ids": [s["trial_id"] for s in seqs]},
               "groups": groups, "llm": llm, "llm_stats": llm_stats}
    with open(os.path.join(out_dir, f"{name}.pkl"), "wb") as fh:
        pickle.dump(payload, fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="runs/golden")
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--impl", choices=["new"], default="new",
                    help="kept for old command lines; the old library is gone (tag pre-restructure)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    for name, cfg in {**BF_CONFIGS, **ET_CONFIGS}.items():
        if only and name not in only:
            continue
        t0 = time.time()
        print(f"[{name}]", flush=True)
        if name.startswith("bf_"):
            dump_breakfast(name, cfg, args.out_dir)
        else:
            dump_epic_tent(name, cfg, args.out_dir, args.skip_llm)
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
