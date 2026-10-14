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
import os
import pickle
import sys
import time

sys.path.insert(0, os.getcwd())  # run_epic_tent_eval / run_llm_eval live at the repo root

from cook_ad.xla_env import disable_gpu_autotuning  # noqa: E402

disable_gpu_autotuning()

import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

import run_epic_tent_eval as ET  # noqa: E402
import run_llm_eval as R  # noqa: E402
from cook_ad.anomaly import live, narrate, quantile, surprise  # noqa: E402
from cook_ad.data import split as split_mod  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.eval import batch  # noqa: E402
from cook_ad.hsmm import joint_em, joint_params  # noqa: E402
from cook_ad.llm import client as llm_client  # noqa: E402
from cook_ad.llm import predict, prompts  # noqa: E402
from cook_ad.recipe import lexical_init  # noqa: E402
from cook_ad.synthetic import error_injection, generate  # noqa: E402

CHANNELS = surprise.CHANNELS
# run_detect_eval.ALPHAS, the grid run_live_eval builds its live tables on. Copied, not imported:
# the binned event mass is a cumsum over this grid, so the grid itself is part of the golden.
BF_ALPHA_GRID = [0.5, 0.35, 0.2, 0.15, 0.1, 0.07, 0.05, 0.035, 0.02, 0.015, 0.01, 0.007, 0.005,
                 0.003, 0.002, 1e-3, 5e-4, 1e-4, 1e-5, 1e-7, 1e-10]
BF_HINDSIGHT_ALPHAS = (0.01, 0.05)
PER_CHANNEL = {"s_temporal": 1e-5}          # on top of DEFAULT_ALPHA = 0.005
BF_LIVE_OP = (0.02, 0.5)                    # (alpha, kappa)
LAGS = [0, 5, live.INF_LAG]
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


def trial_ll(jp, seqs, d_max, chunk_size):
    verb, noun, mask = batch._pad(seqs, max(len(s["verb_ids"]) for s in seqs))
    r_hat, rho, ll = joint_em.infer_recipe(jp, verb, noun, mask, d_max, chunk_size=chunk_size)
    return np.asarray(r_hat), np.asarray(rho), np.asarray(ll)


def hindsight(jp, seqs, d_max, chunk_size, known_r, alphas, per_channel):
    """-> per-trial dicts: flags per arm, r_hat, z_star, channel values."""
    traces, lp, r_hat, ltm = batch.compute_traces_joint(jp, seqs, d_max, chunk_size=chunk_size,
                                                        r_hat=known_r)
    cache = quantile.JointThresholdCache(lp, ltm)
    pc = surprise.per_channel_alphas(per_channel)
    out = []
    for tr, r in zip(traces, np.asarray(r_hat)):
        arms = {f"hindsight@{a:g}": stack(surprise.flag_joint_cached(cache, tr, lp, int(r), ltm,
                                                                     alpha=a)) for a in alphas}
        arms["hindsight@per_channel"] = stack(
            surprise.flag_joint_per_channel(tr, lp, int(r), ltm, pc))
        out.append({
            "arms": arms, "r_hat": int(r), "z_star": np.asarray(tr.z_star),
            "values": {c: np.asarray(getattr(tr, c), dtype=np.float64) for c in CHANNELS},
            "pi_at_zstar": np.asarray(tr.pi_at_zstar),
        })
    return out


def live_one(tb, s, recipe, per_channel_alpha, kappa):
    """per_channel_alpha: {channel: alpha}. -> (arms {live_L<lag>: (T,7)}, diagnostics)."""
    lt = live.compute_live_trace(tb, s["verb_ids"], s["noun_ids"], lags=LAGS, recipe=recipe)
    arms, events = {}, {}
    for lag in LAGS:
        fl = {c: live.flags(lt, tb.alphas, a, kappa, lag=lag)[c] for c, a in per_channel_alpha.items()}
        arms[f"live_L{lag}"] = stack(fl)
        events[lag] = {k: np.asarray(lt.events[lag][k]) for k in EVENT_KEYS}
    diag = {"p_emit": lt.p_emit, "p_verb": lt.p_verb, "p_noun": lt.p_noun,
            "log_norm": lt.log_norm, "events": events, "alphas": np.asarray(tb.alphas)}
    return arms, float(np.sum(lt.log_norm)), diag


# --------------------------------------------------------------------------------------------
# Breakfast


def breakfast_pool(jp, seqs, d_max, chunk_size, seed, recipe_labels, labels_path):
    """run_live_eval.main's stream loop, verbatim in rng order. -> ({group: [seq]}, known_r)."""
    marg = joint_params.collapse_to_marginal(jp)
    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=chunk_size)
    keep = [i for i, t in enumerate(traj) if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    usable = [traj[i] for i in keep]
    rng = np.random.default_rng(seed)
    streams = {"healthy": usable}
    known_r = None
    if recipe_labels:
        known_r, _ = lexical_init.recipe_label_assignment([seqs[i] for i in keep],
                                                          json.load(open(labels_path)))
    for et in error_injection.ERROR_TYPES:
        streams[et] = [error_injection.inject(et, t, rng, marg, select="random") for t in usable]
    return streams, known_r


def dump_breakfast(name, cfg, out_dir, chunk_size=8):
    d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
    jp = joint_params.load_params(cfg["params"])
    seqs = json.load(open(f"{BF}/sequences.json"))
    seqs = split_mod.filter_sequences(seqs, split_mod.load_split(f"{BF}/split.json"), "test")
    streams, known_r = breakfast_pool(jp, seqs, d_max, chunk_size, 0, cfg["recipe_labels"],
                                      f"{BF}/labels.json")
    tb = live.build_tables(jp, d_max, BF_ALPHA_GRID)
    alpha, kappa = BF_LIVE_OP
    live_alpha = {c: alpha for c in CHANNELS}
    groups = {}
    for g, items in streams.items():
        t0 = time.time()
        hs = hindsight(jp, items, d_max, chunk_size, known_r, BF_HINDSIGHT_ALPHAS, PER_CHANNEL)
        r_inf, rho, ll = trial_ll(jp, items, d_max, chunk_size)
        rows = []
        for i, (s, h) in enumerate(zip(items, hs)):
            rec = None if known_r is None else int(known_r[i])
            arms, live_ll, diag = live_one(tb, s, rec, live_alpha, kappa)
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
    """Both predictive variants, replayed from the cache. -> ({arm: [[(tick, strength, channel)]]}, stats)."""
    out, stats = {}, {}
    for variant in prompts.VARIANTS:
        system_prompt = prompts.build_variant(
            variant, vocab, labels=labels, protocol=predict.PROTOCOL, exemplars=[],
            domain=ET.DOMAIN, sil_verb=sil[0], sil_noun=sil[1],
            canonical_order=ET.CANONICAL_ORDER, interleaved=ET.INTERLEAVED)
        client = llm_client.ChatClient(
            model="gemma3:27b", base_url="http://localhost:11434/v1",
            cache_dir=f"{ET_DIR}/llm_cache", max_requests=0,
            temperature=R.PREDICTIVE_TEMPERATURE, max_tokens=R.PREDICTIVE_MAX_TOKENS)
        arm = f"llm-predictive-{variant}"
        flags, _ = ET.predictive_flags_per_trial(trajectories, lexicon, client, system_prompt,
                                                 vocab, arm)
        if trimmed:
            flags = [ET.drop_truncated_under_run(f, t, lexicon) for f, t in zip(flags, trajectories)]
        out[arm] = [[(int(f.tick), float(f.strength), str(f.channel)) for f in fl] for fl in flags]
        stats[arm] = client.stats()
    return out, stats


def et_rows(jp, tb, trajectories, d_max, chunk_size, live_alpha):
    hs = hindsight(jp, trajectories, d_max, chunk_size, None, (), PER_CHANNEL)
    r_inf, rho, ll = trial_ll(jp, trajectories, d_max, chunk_size)
    rows = []
    for i, (t, h) in enumerate(zip(trajectories, hs)):
        arms, live_ll, diag = live_one(tb, t, None, live_alpha, ET_KAPPA)
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
    jp = joint_params.load_params(cfg["params"])
    lexicon = narrate.Lexicon(vocab, joint_params.collapse_to_marginal(jp), sil_verb=sil[0],
                              sil_noun=sil[1])
    pc = surprise.per_channel_alphas(PER_CHANNEL)
    tb = live.build_tables(jp, d_max, sorted({surprise.DEFAULT_ALPHA, *PER_CHANNEL.values()}))

    sets = {}
    if cfg["trim"]:
        for gname in ET.ERROR_GROUPS:
            trimmed = ET.trim_at_first_mistake(seqs, truth, ET.ERROR_GROUPS[gname])
            sets[gname] = generate.trajectories_from_real_joint(
                jp, [s for s, _ in trimmed], d_max, chunk_size=chunk_size)
    else:
        sets["all"] = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=chunk_size)

    groups, llm, llm_stats = {}, {}, {}
    for key, trajectories in sets.items():
        t0 = time.time()
        groups[key] = et_rows(jp, tb, trajectories, d_max, chunk_size, pc)
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
