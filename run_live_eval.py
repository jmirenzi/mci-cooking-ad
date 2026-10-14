"""Hindsight vs live: the shipped detector against docs/live.md's causal channels, on the SAME
degraded streams, at run_detect_eval.py's `trial_loc` metric.

Arms:
  hindsight   -- surprise.flag_joint on compute_traces_joint (whole-trial r_hat and z_star); swept
                 over alpha exactly as run_detect_eval.py does, so its numbers reproduce that script.
  live_L{lag} -- anomaly/live.py: emission channels as timing-split forecast p-values (no lag),
                 hidden-event channels as posteriors given o_{0:t+lag}, swept over alpha x kappa.
  live_mix_L0 -- as live_L0 but the emission channels use the plain forecast mixture (the
                 version docs/live.md §4.1 first proposed), kept to show what the split buys.

The injection loop (rng seed, group order, trial order) is run_detect_eval.py's verbatim, so
`hindsight` here equals its scorecard for the same --joint-params/--seed.

--oracle-recipe gives BOTH arms the recipe as known: each trial's recipe is pinned to the one its
HEALTHY stream decodes to (tools_oracle_recipe.py's oracle), for the healthy trial and every
degraded copy of it. The model's recipes are latent clusters, not Breakfast's dish labels, so
that is the nearest thing to a ground-truth recipe this model has. --lags accepts "inf" (the
smoother: the whole trial, through the same posterior machinery).

Every arm is scored twice: as run_detect_eval.py scores it (positive range = injection-touched
steps), and again as `{arm}+g1` with the positive range extended ONE tick past each of its runs.
The grace exists because an "ended too early" anomaly is only knowable on the next step's first
tick: a causal detector flags exactly there, one tick outside the range, and is charged a stray for
a correct detection. Measured on 97 test abandonments: the live early-exit channel's recall is 0.00
honest vs 0.43 one tick earlier, at zero healthy false alarms. The hindsight s_dur_two sits on the
ended step's last tick -- a tick before it is knowable -- which is how it lands inside the range.
Both arms get the same grace, so +g1 is the like-for-like comparison.

Metric G (span_metrics.METRIC_G, the headline -- chosen 2026-10-14) is computed for every arm at
every operating point from the union of its channels' flags, against the injector's own span
ground truth (degraded["spans"]): recall per type, precision, alarm onsets per 10 min on healthy
trials, healthy-trial FPR, and run FPR (alarms merged one per run of constant (verb, noun)). The A numbers above are kept, labelled, for continuity.

Per trial and channel only (hit, stray) is kept for every operating point; a trial's raw verdict
is the OR over channels of those, which is exact because "any flag inside the positive range" of
a union is the OR of each channel's own.

    ./py run_live_eval.py --joint-params dataset/processed/breakfast/joint_params_train_lexhard.npz \
        --split-part test --tag lexhard
"""
import argparse
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.anomaly import live, narrate, quantile, surprise
from cook_ad.data import split as split_mod
from cook_ad.data.config import load_config
from cook_ad.eval import batch, span_metrics
from cook_ad.llm import textify
from cook_ad.hsmm import joint_params
from cook_ad.synthetic import error_injection, generate
from run_detect_eval import ALPHAS, positive_ticks

CHANNELS = surprise.CHANNELS
KAPPAS = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
EMISSION = ("s_emit", "s_verb", "s_noun")
EVENT_TO_CHANNEL = {"stall": "s_temporal", "early": "s_dur_two", "trans": "s_transition",
                    "recipe": "s_recipe_transition"}


GRACES = (0, 1)


def with_grace(pos, g):
    """Extend every run of True in `pos` by g ticks to the right (clipped at the trial end)."""
    out = pos.copy()
    for k in range(1, g + 1):
        out[k:] |= pos[:-k]
    return out


def _hit_stray(flag, pos):
    """flag: (T, ...) bool -> (hit, stray) each (...) bool."""
    p = pos.reshape((-1,) + (1,) * (flag.ndim - 1))
    return np.any(flag & p, axis=0), np.any(flag & ~p, axis=0)


def score_from_hits(groups):
    """groups: {name: (hit (n, ...), stray (n, ...))} with the operating-point axes trailing.
    Returns arrays over the operating-point axes, exactly run_detect_eval.score's arithmetic."""
    tp = fn = fp = tn = 0
    per_type = {}
    for g, (hit, stray) in groups.items():
        if g == "healthy":
            fp = fp + stray.sum(0)
            tn = tn + (~stray).sum(0)
            per_type[g] = {"stray": stray.mean(0)}
        else:
            tp = tp + hit.sum(0)
            fn = fn + (~hit).sum(0)
            fp = fp + stray.sum(0)
            per_type[g] = {"recall": hit.mean(0), "stray": stray.mean(0)}
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "healthy_fpr": per_type["healthy"]["stray"], "per_type": per_type}


def _tolist(x):
    if isinstance(x, dict):
        return {k: _tolist(v) for k, v in x.items()}
    return np.asarray(x).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-params", default="dataset/processed/breakfast/joint_params_train_lexhard.npz")
    ap.add_argument("--config", default="configs/breakfast.yaml")
    ap.add_argument("--split-file", default="dataset/processed/breakfast/split.json")
    ap.add_argument("--split-part", choices=["train", "test"], default="test")
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--max", type=int, default=10_000)
    ap.add_argument("--lags", default="0,2,5,10")
    ap.add_argument("--oracle-recipe", action="store_true",
                    help="pin every trial's recipe to its healthy decode's, in both arms")
    ap.add_argument("--labels", default="dataset/processed/breakfast/labels.json")
    ap.add_argument("--recipe-labels", action="store_true",
                    help="give BOTH arms each trial's true dish as its recipe. Only meaningful "
                         "for a model fitted with --recipe-labels (recipe id = sorted dish index)")
    ap.add_argument("--tag", default="live")
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()
    lags = [x if x == live.INF_LAG else int(x) for x in args.lags.split(",")]

    d_max = load_config(args.config)["duration"]["d_max_ticks"]
    vocab = json.load(open(args.vocab))
    jp = joint_params.load_params(args.joint_params)
    marg = joint_params.collapse_to_marginal(jp)
    lexicon = narrate.Lexicon(vocab, marg)

    seqs = json.load(open(args.sequences))
    if args.split_file:
        seqs = split_mod.filter_sequences(seqs, split_mod.load_split(args.split_file), args.split_part)
    seqs = seqs[: args.max]

    # ---- streams: run_detect_eval.py's loop, verbatim in rng order ----
    traj = generate.trajectories_from_real_joint(jp, seqs, d_max, chunk_size=args.chunk_size)
    keep = [i for i, t in enumerate(traj) if len(t["segments"]) >= error_injection.MIN_SEGMENTS]
    usable = [traj[i] for i in keep]
    print(f"[{args.tag}/{args.split_part}] {len(usable)}/{len(seqs)} usable trials", flush=True)
    rng = np.random.default_rng(args.seed)
    streams = {"healthy": [(u, np.zeros(len(u["verb_ids"]), dtype=bool)) for u in usable]}
    known_r = np.array([int(u["recipe_id"]) for u in usable]) if args.oracle_recipe else None
    if args.recipe_labels:
        if args.oracle_recipe:
            ap.error("--recipe-labels and --oracle-recipe are two different known recipes; pick one")
        from cook_ad.recipe import lexical_init
        known_r, names = lexical_init.recipe_label_assignment(
            [seqs[i] for i in keep], json.load(open(args.labels)))
        if len(names) != jp.init_counts.shape[0]:
            ap.error(f"{len(names)} dishes but the model has K_R={jp.init_counts.shape[0]}")
    for et in error_injection.ERROR_TYPES:
        deg = [error_injection.inject(et, t, rng, marg, select="random") for t in usable]
        streams[et] = [(d, positive_ticks(d, lexicon)) for d in deg]

    alphas_desc = list(ALPHAS)
    tb = live.build_tables(jp, d_max, alphas_desc)
    a_idx = {a: int(np.flatnonzero(np.isclose(tb.alphas, a))[0]) for a in alphas_desc}
    kap = np.asarray(KAPPAS)

    # hits[arm][channel][group] = (hit, stray) arrays, trial axis first
    base_arms = ["hindsight"] + [f"live_L{lag}" for lag in lags] + ["live_mix_L0"]
    hits = {}
    for gr in GRACES:
        for a in base_arms:
            hits[a if gr == 0 else f"{a}+g{gr}"] = {c: {} for c in CHANNELS}
    switch_diag = {lag: {} for lag in lags}
    gacc = {a: {} for a in base_arms}   # metric G sufficient statistics, per arm and group

    t0 = time.time()
    alpha_arr = np.asarray(alphas_desc)
    grid = (len(alphas_desc), len(kap))
    for g, items in streams.items():
        seqs_g = [s for s, _ in items]
        pos_g = [p for _, p in items]
        acc = {name: {c: ([], []) for c in CHANNELS} for name in hits}
        spans_g, correct_g, runs_g = [], [], []
        for s_ in seqs_g:
            runs_g.append(span_metrics.run_starts(s_["verb_ids"], s_["noun_ids"]))
            sp = [] if g == "healthy" else [tuple(x) for x in s_["spans"]]
            steps = textify.steps_from_ids(s_["verb_ids"], s_["noun_ids"], lexicon)
            correct_g.append([(st.tick_start, st.tick_end - 1) for st in steps
                              if not any(st.tick_start <= b and st.tick_end - 1 >= a for a, b in sp)])
            spans_g.append(sp)

        def record(arm, per_channel, pos, spans=None, correct=None, runs_now=None):
            """per_channel: {channel: (T, ...) bool}; scored once per grace (metric A), and once
            under metric G from the union over channels."""
            union = np.zeros(next(iter(per_channel.values())).shape, dtype=bool)
            for f in per_channel.values():
                union = union | f
            shape = union.shape[1:]
            r = span_metrics.score_onsets_dense(union.reshape(union.shape[0], -1),
                                                [((a, b), g) for a, b in spans], correct,
                                                run_starts=runs_now)
            st = gacc[arm].setdefault(g, {"hits": 0, "n_spans": 0, "fp": 0, "onsets": 0,
                                          "any_flag": 0, "steps_flagged": 0, "n_correct": 0,
                                          "ticks": 0, "n_trials": 0})
            st["hits"] = st["hits"] + r["hits"].sum(axis=0).reshape(shape)
            st["n_spans"] += len(spans)
            st["fp"] = st["fp"] + r["fp"].reshape(shape)
            st["onsets"] = st["onsets"] + r["onsets"].reshape(shape)
            st["any_flag"] = st["any_flag"] + r["any_flag"].reshape(shape)
            st["steps_flagged"] = st["steps_flagged"] + r["steps_flagged"].reshape(shape)
            st["n_correct"] += r["n_correct_steps"]
            st["ticks"] += union.shape[0]
            st["n_trials"] += 1
            for gr in GRACES:
                p = with_grace(pos, gr)
                name = arm if gr == 0 else f"{arm}+g{gr}"
                for c, f in per_channel.items():
                    hh, ss = _hit_stray(f, p)
                    acc[name][c][0].append(hh)
                    acc[name][c][1].append(ss)

        # ---- hindsight ----
        traces, lp, r_hat, ltm = batch.compute_traces_joint(jp, seqs_g, d_max, chunk_size=args.chunk_size,
                                                            r_hat=known_r)
        cache = quantile.JointThresholdCache(lp, ltm)
        for ti, (tr, r, pos) in enumerate(zip(traces, np.asarray(r_hat), pos_g)):
            per_alpha = [surprise.flag_joint_cached(cache, tr, lp, int(r), ltm, alpha=a) for a in alphas_desc]
            record("hindsight", {c: np.stack([np.asarray(fa[c]) for fa in per_alpha], axis=1)
                                 for c in CHANNELS}, pos, spans_g[ti], correct_g[ti], runs_g[ti])

        # ---- live ----
        sw = {lag: [] for lag in lags}
        for ti, (s, pos) in enumerate(zip(seqs_g, pos_g)):
            rec = None if known_r is None else int(known_r[ti])
            lt = live.compute_live_trace(tb, s["verb_ids"], s["noun_ids"], lags=lags, recipe=rec)
            # emission flags are lag-free (T, A); broadcast over kappa so every live arm is (A, Kp)
            em = {c: np.broadcast_to((p[:, None] <= alpha_arr)[:, :, None], p.shape + grid)
                  for c, p in zip(EMISSION, (lt.p_emit, lt.p_verb, lt.p_noun))}
            em_mix = {c: np.broadcast_to((p[:, None] <= alpha_arr)[:, :, None], p.shape + grid)
                      for c, p in zip(EMISSION, lt.p_mix)}
            for lag in lags:
                ev_flags = {}
                for ev, c in EVENT_TO_CHANNEL.items():
                    prob = lt.events[lag][ev][:, [a_idx[a] for a in alphas_desc]]   # (T, A)
                    ev_flags[c] = prob[:, :, None] > kap[None, None, :]             # (T, A, Kp)
                record(f"live_L{lag}", {**em, **ev_flags}, pos, spans_g[ti], correct_g[ti], runs_g[ti])
                if lag == lags[0]:
                    record("live_mix_L0", {**em_mix, **ev_flags}, pos, spans_g[ti], correct_g[ti],
                           runs_g[ti])
                sw[lag].append(float(lt.switch[lag].sum()))

        for name, per_c in acc.items():
            for c, (hh, ss) in per_c.items():
                hits[name][c][g] = (np.stack(hh), np.stack(ss))
        for lag in lags:
            switch_diag[lag][g] = float(np.mean(sw[lag]))
        print(f"  {g}: {len(items)} trials  ({time.time() - t0:.0f}s)", flush=True)

    # ---- score ----
    out = {"config": vars(args), "alphas": alphas_desc, "kappas": KAPPAS, "lags": lags,
           "n_usable": len(usable), "mean_switch_mass_per_trial": switch_diag, "arms": {}}
    for arm, per_ch in hits.items():
        raw_groups = {}
        for g in streams:
            hit = np.any(np.stack([per_ch[c][g][0] for c in CHANNELS]), axis=0)
            stray = np.any(np.stack([per_ch[c][g][1] for c in CHANNELS]), axis=0)
            raw_groups[g] = (hit, stray)
        arm_out = {"raw": score_from_hits(raw_groups)}
        for c in CHANNELS:
            arm_out[c] = score_from_hits(per_ch[c])
        out["arms"][arm] = _tolist(arm_out)

        r = arm_out["raw"]
        best = np.unravel_index(np.argmax(r["accuracy"]), np.shape(r["accuracy"]))
        op = f"alpha={alphas_desc[best[0]]:.0e}" + (f" kappa={KAPPAS[best[1]]}" if len(best) > 1 else "")
        print(f"{arm:>10} best in-split acc={r['accuracy'][best]:.3f} prec={r['precision'][best]:.3f} "
              f"rec={r['recall'][best]:.3f} healthyFP={r['healthy_fpr'][best]:.3f}  @ {op}", flush=True)

    # ---- metric G ----
    out["metric_g"] = {}
    for arm, per_group in gacc.items():
        tp = sum(st["hits"] for gg, st in per_group.items() if gg != "healthy")
        n_spans = sum(st["n_spans"] for gg, st in per_group.items() if gg != "healthy")
        fp = sum(st["fp"] for gg, st in per_group.items() if gg != "healthy")
        h = per_group["healthy"]
        steps_flagged = sum(st["steps_flagged"] for st in per_group.values())
        n_correct = sum(st["n_correct"] for st in per_group.values())
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            rec = tp / max(n_spans, 1)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        out["metric_g"][arm] = _tolist({
            "precision": prec, "recall": rec, "f1": f1,
            "alarms_per_10min": 600.0 * h["onsets"] / h["ticks"],
            "healthy_trial_fpr": h["any_flag"] / h["n_trials"],
            "run_fpr": steps_flagged / max(n_correct, 1),
            "per_type_recall": {gg: st["hits"] / max(st["n_spans"], 1)
                                for gg, st in per_group.items() if gg != "healthy"},
        })
        best = np.unravel_index(int(np.argmax(f1)), np.shape(f1))
        print(f"[G] {arm:>12} best in-split F1={f1[best]:.3f} prec={prec[best]:.3f} rec={rec[best]:.3f} "
              f"alarms/10min={600.0 * h['onsets'][best] / h['ticks']:.2f} "
              f"runFPR={(steps_flagged / max(n_correct, 1))[best]:.3f}", flush=True)
    out["metric_g_definition"] = dict(span_metrics.METRIC_G)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"live_{args.tag}_{args.split_part}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"written to {path}")


if __name__ == "__main__":
    main()
