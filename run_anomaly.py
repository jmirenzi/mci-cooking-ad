import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.anomaly import surprise
from cook_ad.data.config import load_config
from cook_ad.hsmm import params
from cook_ad.recipe import recipe_hmm, segmentize

jax.config.update("jax_enable_x64", True)


def _load_vocab(vocab_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    id_to_verb = {i: v for v, i in vocab["verbs"].items()}
    id_to_noun = {i: n for n, i in vocab["nouns"].items()}
    return id_to_verb, id_to_noun


def inject_noun(verb_ids, noun_ids, hsmm_params, d_max, tick=None):
    """Swap one tick's noun for the least-likely noun given the believed subtask at that
    tick -- a clean, targeted item-substitution perturbation, isolated from the verb channel."""
    mask = jnp.ones((1, len(verb_ids)), dtype=bool)
    seg_result = segmentize.segment_all(
        hsmm_params, jnp.asarray(verb_ids)[None, :], jnp.asarray(noun_ids)[None, :], mask, d_max
    )[0]
    z_star = seg_result["subtask_per_tick"]

    if tick is None:
        tick = len(verb_ids) // 2
    state = int(z_star[tick])

    log_probs = params.to_log_probs(hsmm_params, d_max)
    log_emit_n = np.asarray(log_probs.log_emit_n)
    new_noun = int(np.argmin(log_emit_n[state]))

    new_noun_ids = list(noun_ids)
    old_noun = new_noun_ids[tick]
    new_noun_ids[tick] = new_noun
    return new_noun_ids, tick, old_noun, new_noun, state


def inject_stall(verb_ids, noun_ids, hsmm_params, d_max, seg_idx=None, extra_ticks=None):
    """Stretch one segment by repeating its dominant (verb, noun) tokens well past its
    expected duration -- isolates the temporal channel (emissions stay unchanged). Defaults
    to the segment whose state has the sharpest fitted hazard (highest dur_p): the hazard
    asymptotes at roughly dur_p as elapsed duration grows (see anomaly/temporal.py), so a
    low-dur_p state gives a real but modest rise -- picking a high-dur_p state makes the
    stall demonstration's rise clearly visible without cherry-picking the surprise itself.
    """
    mask = jnp.ones((1, len(verb_ids)), dtype=bool)
    seg_result = segmentize.segment_all(
        hsmm_params, jnp.asarray(verb_ids)[None, :], jnp.asarray(noun_ids)[None, :], mask, d_max
    )[0]
    segments = seg_result["segments"]

    if seg_idx is None:
        dur_p = np.asarray(hsmm_params.dur_p)
        seg_idx = int(np.argmax([dur_p[state] for state, _ in segments]))

    pos = sum(d for _, d in segments[:seg_idx])
    state, d = segments[seg_idx]

    if extra_ticks is None:
        mean_dur = float(hsmm_params.dur_r[state]) * (1.0 - float(hsmm_params.dur_p[state])) / float(hsmm_params.dur_p[state])
        extra_ticks = max(int(3 * mean_dur), 20)

    insert_at = pos + d
    verb_token = verb_ids[insert_at - 1]
    noun_token = noun_ids[insert_at - 1]

    new_verb_ids = list(verb_ids[:insert_at]) + [verb_token] * extra_ticks + list(verb_ids[insert_at:])
    new_noun_ids = list(noun_ids[:insert_at]) + [noun_token] * extra_ticks + list(noun_ids[insert_at:])
    return new_verb_ids, new_noun_ids, pos, state, extra_ticks


def _print_channel_summary(name, values, flags):
    values = np.asarray(values)
    n_flagged = int(np.sum(flags))
    print(f"  {name:22s} mean={values.mean():7.3f}  max={values.max():7.3f}  flagged_ticks={n_flagged}")


def _report(trace, flags, tag):
    print(f"\n--- {tag} ---")
    for name in surprise.DEFAULT_THRESHOLDS:
        _print_channel_summary(name, getattr(trace, name), flags[name])
    attribution_counts = {label: int(np.sum(trace.attribution == label)) for label in ("item", "action", "none")}
    print(f"  attribution counts: {attribution_counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breakfast_mini.yaml")
    parser.add_argument("--params", default="dataset/processed/breakfast_mini/hsmm_params.npz")
    parser.add_argument("--recipe-params", default="dataset/processed/breakfast_mini/recipe_params.npz")
    parser.add_argument("--sequences", default="dataset/processed/breakfast_mini/sequences.json")
    parser.add_argument("--vocab", default="dataset/processed/breakfast_mini/vocab.json")
    parser.add_argument("--trial", default=None, help="trial_id to analyze; defaults to the first trial")
    parser.add_argument("--inject-noun", action="store_true")
    parser.add_argument("--inject-stall", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-out", default="anomaly_trace.png")
    args = parser.parse_args()

    config = load_config(args.config)
    d_max = config["duration"]["d_max_ticks"]

    hsmm_params = params.load_params(args.params)
    recipe_params = recipe_hmm.load_params(args.recipe_params)
    id_to_verb, id_to_noun = _load_vocab(args.vocab)

    with open(args.sequences) as f:
        sequences = json.load(f)
    entry = sequences[0] if args.trial is None else next(s for s in sequences if s["trial_id"] == args.trial)
    verb_ids, noun_ids = entry["verb_ids"], entry["noun_ids"]
    print(f"trial: {entry['trial_id']}  ticks: {len(verb_ids)}")

    trace = surprise.compute_trace(hsmm_params, recipe_params, verb_ids, noun_ids, d_max)
    flags = surprise.flag(trace)
    _report(trace, flags, "healthy")

    if args.inject_noun:
        new_noun_ids, tick, old_noun, new_noun, state = inject_noun(verb_ids, noun_ids, hsmm_params, d_max)
        trace_n = surprise.compute_trace(hsmm_params, recipe_params, verb_ids, new_noun_ids, d_max)
        flags_n = surprise.flag(trace_n)
        _report(trace_n, flags_n, "noun-injected")
        print(
            f"  injected at tick {tick} (believed subtask {state}): "
            f"noun '{id_to_noun[old_noun]}' -> '{id_to_noun[new_noun]}'"
        )
        print(f"  s_noun={trace_n.s_noun[tick]:.2f}  s_verb={trace_n.s_verb[tick]:.2f}  "
              f"attribution={trace_n.attribution[tick]}  "
              f"expected_noun='{id_to_noun[trace_n.expected_noun[tick]]}'")

    if args.inject_stall:
        new_verb_ids, new_noun_ids, start, state, extra = inject_stall(verb_ids, noun_ids, hsmm_params, d_max)
        trace_s = surprise.compute_trace(hsmm_params, recipe_params, new_verb_ids, new_noun_ids, d_max)
        flags_s = surprise.flag(trace_s)
        _report(trace_s, flags_s, "stall-injected")
        stretch_end = start + extra + 5
        print(f"  stretched segment at tick {start} (state {state}) by {extra} extra ticks")
        print(f"  s_temporal at stretch start={trace_s.s_temporal[start]:.3f}  "
              f"at stretch end={trace_s.s_temporal[min(stretch_end, len(trace_s.s_temporal) - 1)]:.3f}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(surprise.DEFAULT_THRESHOLDS), 1, figsize=(10, 10), sharex=True)
        for ax, name in zip(axes, surprise.DEFAULT_THRESHOLDS):
            ax.plot(getattr(trace, name))
            ax.set_ylabel(name)
        axes[-1].set_xlabel("tick")
        fig.tight_layout()
        fig.savefig(args.plot_out)
        print(f"\nsaved trace plot to {args.plot_out}")


if __name__ == "__main__":
    main()
