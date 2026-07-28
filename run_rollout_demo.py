import argparse
import json
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.anomaly import narrate, surprise
from cook_ad.data.config import load_config
from cook_ad.eval import metrics
from cook_ad.hsmm import params
from cook_ad.lifecycle import duration_drift
from cook_ad.recipe import recipe_hmm, segmentize
from cook_ad.synthetic import error_injection, generate

jax.config.update("jax_enable_x64", True)

STALL_EXTRA_MULTIPLE = 3
STALL_EXTRA_MIN = 20
SCENARIO_CHOICES = ("all", "healthy", *error_injection.ERROR_TYPES, "stall", "drift")


def load_user_trials(sequences_path, user):
    with open(sequences_path) as f:
        all_sequences = json.load(f)
    user_trials = [s for s in all_sequences if s["trial_id"].split("_", 1)[0] == user]
    return user_trials, all_sequences


def segment_trial(hsmm_params, verb_ids, noun_ids, d_max):
    mask = jnp.ones((1, len(verb_ids)), dtype=bool)
    return segmentize.segment_all(
        hsmm_params, jnp.asarray(verb_ids)[None, :], jnp.asarray(noun_ids)[None, :], mask, d_max
    )[0]


def calibrate_to_user(corpus_params, trials, d_max, weight=1.0):
    """One hard-EM step: Viterbi-segment each of the user's trials under the corpus checkpoint,
    accumulate those hard counts onto the corpus posterior. Durations are NOT touched here, the
    same structural gap duration_drift.py works around. This is a calibration accumulation, not
    a per-user fit -- see the caveat main() prints."""
    init_counts = corpus_params.init_counts
    trans_counts = corpus_params.trans_counts
    verb_counts = corpus_params.verb_counts
    noun_counts = corpus_params.noun_counts

    segments_by_trial = []
    for trial in trials:
        verb_ids = np.asarray(trial["verb_ids"])
        noun_ids = np.asarray(trial["noun_ids"])
        seg_result = segment_trial(corpus_params, verb_ids, noun_ids, d_max)
        segments = seg_result["segments"]
        z_star = seg_result["subtask_per_tick"]
        segments_by_trial.append(segments)

        init_counts = init_counts.at[segments[0][0]].add(weight)
        for (a, _), (c, _) in zip(segments[:-1], segments[1:]):
            trans_counts = trans_counts.at[a, c].add(weight)
        verb_counts = verb_counts.at[jnp.asarray(z_star), jnp.asarray(verb_ids)].add(weight)
        noun_counts = noun_counts.at[jnp.asarray(z_star), jnp.asarray(noun_ids)].add(weight)

    user_params = corpus_params._replace(
        init_counts=init_counts, trans_counts=trans_counts,
        verb_counts=verb_counts, noun_counts=noun_counts,
    )
    return user_params, segments_by_trial


def inject_stall(traj, rng, extra_ticks=None, select="random"):
    """Stretch one segment by repeating its final (verb, noun) tokens well past its observed
    duration. Not in synthetic.error_injection because the five canonical error types don't
    include a pure stall; matches error_injection's dict shape so it slots into the same
    downstream code path (score_trial, narrate)."""
    verb_ids = np.array(traj["verb_ids"])
    noun_ids = np.array(traj["noun_ids"])
    bounds = error_injection._seg_bounds(traj["segments"])

    i = error_injection._pick_segment(rng, len(bounds), lo=0, hi=len(bounds), select=select)
    start, end, state, d = bounds[i]

    if extra_ticks is None:
        extra_ticks = max(STALL_EXTRA_MULTIPLE * d, STALL_EXTRA_MIN)

    verb_token = verb_ids[end - 1]
    noun_token = noun_ids[end - 1]
    new_verb_ids = np.concatenate([verb_ids[:end], np.full(extra_ticks, verb_token), verb_ids[end:]])
    new_noun_ids = np.concatenate([noun_ids[:end], np.full(extra_ticks, noun_token), noun_ids[end:]])

    return {
        "verb_ids": new_verb_ids.astype(np.int64),
        "noun_ids": new_noun_ids.astype(np.int64),
        "window": (end, end + extra_ticks - 1),
        "error_type": "stall",
    }


def apply_scenario(scenario, traj, rng, hsmm_params):
    if scenario == "healthy":
        return {
            "verb_ids": np.asarray(traj["verb_ids"], dtype=np.int64),
            "noun_ids": np.asarray(traj["noun_ids"], dtype=np.int64),
            "window": None,
            "error_type": "healthy",
        }
    if scenario == "stall":
        return inject_stall(traj, rng)
    return error_injection.inject(scenario, traj, rng, hsmm_params)


def run_scenario(scenario, traj, rng, hsmm_params, recipe_params, vocab, d_max, verbose=False):
    injected = apply_scenario(scenario, traj, rng, hsmm_params)
    trace, log_probs, recipe_log_trans = surprise.compute_trace(
        hsmm_params, recipe_params, injected["verb_ids"], injected["noun_ids"], d_max
    )
    flags = surprise.flag(trace, log_probs, recipe_log_trans)
    pi_all = surprise.compute_pi_all(log_probs, injected["verb_ids"], injected["noun_ids"], d_max)

    print(f"\n=== scenario: {scenario} ===")
    window = injected["window"]
    print(f"  ground truth window: {window}")

    for ch in metrics.ALL_CHANNELS:
        n_flagged = int(np.sum(flags[ch]))
        if window is not None:
            t0, t1 = window
            hi = t1 + metrics.DEFAULT_LATENCY_TOL
            in_window = int(np.sum(flags[ch][t0 : hi + 1]))
            out_window = n_flagged - in_window
            print(f"    {ch:22s} flagged={n_flagged:4d}  in_window={in_window:4d}  out_of_window={out_window:4d}")
        else:
            print(f"    {ch:22s} flagged={n_flagged:4d}  (healthy trial -- any flag here is a false alarm)")

    queries = narrate.narrate(
        trace, flags, vocab, hsmm_params, injected["verb_ids"], injected["noun_ids"],
        log_probs, recipe_log_trans, pi_all,
    )
    print(f"  {len(queries)} narrated queries:")
    for q in queries:
        if window is not None:
            t0, t1 = window
            hi = t1 + metrics.DEFAULT_LATENCY_TOL
            tag = "TRUE POSITIVE" if t0 <= q.tick <= hi else "FALSE ALARM"
        else:
            tag = "FALSE ALARM"
        print(f"    [{tag}] tick={q.tick:4d} {q.channel:20s} ({q.severity}) {q.text}")
        if verbose:
            print(f"        kind={q.kind}  ratio={q.ratio:.2f}  event={q.event}")

    return trace, flags, queries, injected


def run_drift_demo(user_trials, hsmm_params, vocab, d_max, k_subtask, slow_factor):
    """Constructs a synthetic 'recent window' by scaling one subtask's durations. This is a
    mechanism demonstration, not a measured effect: there is no longitudinal per-user Breakfast
    data, so 'last week' is a copy of the user's own sessions with one state's durations
    scaled."""
    lexicon = narrate.Lexicon(vocab, hsmm_params)

    frozen_segments = []
    for trial in user_trials:
        seg_result = segment_trial(hsmm_params, trial["verb_ids"], trial["noun_ids"], d_max)
        frozen_segments.append(seg_result["segments"])

    all_states = [state for segs in frozen_segments for state, _ in segs]
    if not all_states:
        print("\n=== drift demo ===\n  no segments available for this user")
        return
    busiest_state = max(set(all_states), key=all_states.count)

    recent_segments = [
        [(state, int(round(d * slow_factor)) if state == busiest_state else d) for state, d in segs]
        for segs in frozen_segments
    ]

    rows = duration_drift.duration_drift(recent_segments, frozen_segments, k_subtask, d_max)
    print(f"\n=== drift demo (subtask {busiest_state} = '{lexicon.subtask(busiest_state)}' "
          f"scaled {slow_factor}x) ===")
    for r in rows:
        marker = "REPORTABLE" if r["reportable"] else ""
        print(
            f"  state={r['state']:3d}  n_recent={r['n_recent']:3d}  n_frozen={r['n_frozen']:3d}  "
            f"mean_recent={r['mean_recent']:6.2f}  mean_frozen={r['mean_frozen']:6.2f}  "
            f"delta={r['delta_mean']:+6.2f}  kl={r['kl']:.3f}  p={r['p_value']:.4f}  {marker}"
        )
    lines = duration_drift.narrate_drift(rows, lexicon)
    if lines:
        print("  narrated:")
        for line in lines:
            print(f"    {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breakfast.yaml")
    parser.add_argument("--params", default="dataset/processed/breakfast/hsmm_params.npz")
    parser.add_argument("--recipe-params", default="dataset/processed/breakfast/recipe_params.npz")
    parser.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    parser.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    parser.add_argument("--user", default=None, help="participant id (e.g. P03); defaults to the participant with the most trials")
    parser.add_argument("--holdout", default=None, help="trial_id held out as the rollout source; defaults to the user's last trial")
    parser.add_argument("--calibrate", dest="calibrate", action="store_true", default=True)
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    parser.add_argument("--synthetic", action="store_true", help="sample the rollout from the model instead of using a real held-out trial")
    parser.add_argument("--scenario", default="all", choices=SCENARIO_CHOICES)
    parser.add_argument("--slow-factor", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    d_max = config["duration"]["d_max_ticks"]
    k_subtask = config["k_subtask"]

    corpus_params = params.load_params(args.params)
    recipe_params = recipe_hmm.load_params(args.recipe_params)
    with open(args.vocab) as f:
        vocab = json.load(f)

    with open(args.sequences) as f:
        all_sequences = json.load(f)

    if args.user is None:
        counts = Counter(s["trial_id"].split("_", 1)[0] for s in all_sequences)
        args.user = max(counts, key=counts.get)

    user_trials, _ = load_user_trials(args.sequences, args.user)
    if len(user_trials) < 2:
        raise SystemExit(f"user {args.user!r} has too few trials ({len(user_trials)}) for a holdout rollout")

    holdout_id = args.holdout or user_trials[-1]["trial_id"]
    holdout = next((t for t in user_trials if t["trial_id"] == holdout_id), None)
    if holdout is None:
        raise SystemExit(f"holdout trial {holdout_id!r} not found among {args.user}'s trials")
    calib_trials = [t for t in user_trials if t["trial_id"] != holdout_id]

    print(f"user: {args.user}  trials: {len(user_trials)}  holdout: {holdout_id}  calibration trials: {len(calib_trials)}")

    if args.calibrate and calib_trials:
        print(
            f"NOTE: calibration is one hard-EM count-accumulation step on top of the corpus "
            f"posterior, not a per-user fit -- fitting a K={k_subtask} HSMM on "
            f"{len(calib_trials)} sequences would be memorization, not estimation. This is NOT "
            f"per-user EM."
        )
        hsmm_params, _ = calibrate_to_user(corpus_params, calib_trials, d_max)
    else:
        print("NOTE: --no-calibrate -- using the raw corpus checkpoint, no per-user accumulation.")
        hsmm_params = corpus_params

    rng = np.random.default_rng(args.seed)

    if args.synthetic:
        print(
            "NOTE: --synthetic rolls out from the same model that scores it, which flatters "
            "detection relative to a real held-out trial."
        )
        traj = generate.generate_healthy(hsmm_params, 1, rng, max_ticks=len(holdout["verb_ids"]), d_max=d_max)[0]
    else:
        traj = generate.trajectory_from_real(hsmm_params, holdout["verb_ids"], holdout["noun_ids"], d_max)

    print(
        "NOTE: s_temporal here indexes segments from Viterbi over the whole trial -- a correct "
        "offline retrodiction of when a live system would have spoken, not a claim about a "
        "real-time system."
    )

    if args.scenario in ("drift", "all"):
        run_drift_demo(user_trials, hsmm_params, vocab, d_max, k_subtask, args.slow_factor)
        print(
            "NOTE: the drift demo above is a mechanism demonstration, not a measured effect -- "
            "there is no longitudinal per-user Breakfast data; 'last week' is constructed by "
            "scaling one subtask's durations in a copy of this user's own sessions."
        )
    if args.scenario == "drift":
        return

    scenario_list = [*error_injection.ERROR_TYPES, "stall", "healthy"] if args.scenario == "all" else [args.scenario]

    healthy_query_count = None
    for scenario in scenario_list:
        try:
            _, _, queries, _ = run_scenario(scenario, traj, rng, hsmm_params, recipe_params, vocab, d_max, verbose=args.verbose)
        except ValueError as exc:
            print(f"\n=== scenario: {scenario} ===\n  skipped: {exc}")
            continue
        if scenario == "healthy":
            healthy_query_count = len(queries)

    if healthy_query_count is not None:
        print(
            f"\nNOTE: the healthy scenario's {healthy_query_count} narrated queries are this "
            f"user's clean-baseline false-alarm count for this trial -- run --scenario healthy "
            f"across multiple --user values before presenting; it is likely more load-bearing "
            f"than any single narrated query."
        )


if __name__ == "__main__":
    main()
