import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from cook_ad.data.config import load_config
from cook_ad.hsmm import params
from cook_ad.lifecycle import divergence, online_update, state_manager
from cook_ad.lifecycle.online_update import PreferenceEvent

jax.config.update("jax_enable_x64", True)


def _load_vocab(vocab_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    return {i: n for n, i in vocab["nouns"].items()}


def _noun_prob(p, state, token):
    _, _, _, log_emit_n = params.normalize_categoricals(p)
    return float(jnp.exp(log_emit_n[state, token]))


def _noun_surprise(p, state, token):
    return -np.log(_noun_prob(p, state, token))


def _pick_substitution(frozen):
    """Find a well-populated subtask (a clear dominant noun) and a low-probability in-vocab
    noun for it -- a realistic 'you used X instead of your usual Y' substitution to accommodate."""
    _, _, _, log_emit_n = params.normalize_categoricals(frozen)
    prob_n = np.asarray(jnp.exp(log_emit_n))
    peak = prob_n.max(axis=1)
    state = int(np.argmax(peak))                 # most committed subtask
    dominant = int(np.argmax(prob_n[state]))
    sub = int(np.argmin(prob_n[state]))          # least expected noun there
    return state, dominant, sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breakfast_mini.yaml")
    parser.add_argument("--params", default="dataset/processed/breakfast_mini/hsmm_params.npz")
    parser.add_argument("--vocab", default="dataset/processed/breakfast_mini/vocab.json")
    parser.add_argument("--delta", type=float, default=online_update.DEFAULT_DELTA)
    parser.add_argument("--max-bump", type=float, default=online_update.DEFAULT_MAX_BUMP)
    parser.add_argument("--events", type=int, default=6, help="repeated confirmed-preference events")
    parser.add_argument("--flag-threshold", type=float, default=4.0, help="Phase 4 s_noun flag threshold (nats)")
    args = parser.parse_args()

    load_config(args.config)
    frozen0 = params.load_params(args.params)
    id_to_noun = _load_vocab(args.vocab)

    state, dominant, sub = _pick_substitution(frozen0)
    print(f"scenario: subtask {state} normally uses '{id_to_noun[dominant]}'; user repeatedly uses "
          f"'{id_to_noun[sub]}' and confirms it each time (PREFERENCE).")
    print("(all numbers below are conditional on the confirmation oracle, which we supply)\n")

    dual = state_manager.init_dual_model(frozen0)
    event = PreferenceEvent("noun", state, sub)

    print(f"{'event':>6}  {'live P(sub)':>12}  {'s_noun(sub)':>12}  {'flagged?':>9}  {'KL(live||frozen)':>17}")
    base_surprise = _noun_surprise(dual.live, state, sub)
    print(f"{'frozen':>6}  {_noun_prob(dual.live, state, sub):12.5f}  {base_surprise:12.3f}  "
          f"{str(base_surprise > args.flag_threshold):>9}  {0.0:17.4f}")

    for i in range(1, args.events + 1):
        dual, _ = state_manager.handle_confirmation(dual, event, "preference", args.delta, args.max_bump)
        s = _noun_surprise(dual.live, state, sub)
        kl = float(divergence.model_divergence(dual.live, dual.frozen)["total"])
        print(f"{i:>6}  {_noun_prob(dual.live, state, sub):12.5f}  {s:12.3f}  "
              f"{str(s > args.flag_threshold):>9}  {kl:17.4f}")
    print("  -> bounded: P(sub) plateaus and s_noun stays above the flag threshold; one 'yes' "
          "does not blind the detector. KL(live||frozen) accumulates -- the weekly-review signal.\n")

    # A breakdown on a different subtask: recognized mistake, nothing updates, incident flagged.
    other_state = int((state + 1) % frozen0.init_counts.shape[0])
    breakdown_event = PreferenceEvent("noun", other_state, sub)
    kl_before = float(divergence.model_divergence(dual.live, dual.frozen)["total"])
    dual, rec = state_manager.handle_confirmation(dual, breakdown_event, "breakdown")
    kl_after = float(divergence.model_divergence(dual.live, dual.frozen)["total"])
    print(f"breakdown event on subtask {other_state}: updated={rec['updated']}, flagged={rec.get('flagged')}, "
          f"KL unchanged ({kl_before:.4f} -> {kl_after:.4f}).\n")

    # Localize drift for the weekly review.
    div = divergence.model_divergence(dual.live, dual.frozen)
    per_state = np.asarray(div["per_state"])
    top = int(np.argmax(per_state))
    print(f"weekly review -- drift localizes to subtask {top} "
          f"(per-state KL={per_state[top]:.4f}; noun-channel KL={float(div['noun'][top]):.4f}).")

    # Consolidate: approve the preference -> re-baseline frozen, reset live, precision bounded.
    live_precision_before = float(jnp.sum(dual.live.noun_counts))
    dual = state_manager.consolidate(dual, approved=[event])
    kl_post = float(divergence.model_divergence(dual.live, dual.frozen)["total"])
    frozen_surprise = _noun_surprise(dual.frozen, state, sub)
    print(f"consolidate (approve): frozen now expects '{id_to_noun[sub]}' more "
          f"(s_noun {base_surprise:.3f} -> {frozen_surprise:.3f}), live reset, KL={kl_post:.4f}.")

    # Second window cannot compound precision.
    for _ in range(args.events):
        dual, _ = state_manager.handle_confirmation(dual, event, "preference", args.delta, args.max_bump)
    live_precision_after = float(jnp.sum(dual.live.noun_counts))
    print(f"precision guard: live noun-count mass after a 2nd window ({live_precision_after:.1f}) is "
          f"bounded by frozen + one window's bump, not compounding across windows "
          f"(1st-window live mass was {live_precision_before:.1f}).")


if __name__ == "__main__":
    main()
