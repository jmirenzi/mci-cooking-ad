"""Run-length histogram over the RLE observation steps: the span layer's precondition check.

eval/span_metrics.py merges flags separated by no more than MERGE_TOL ticks into a single alarm,
so that an OSCILLATING observation stream is not charged four false positives where a clean region
is charged one. That default (5) is only defensible if oscillation is a real property of the
stream, so this measures it directly: the fraction of run-length-encoded steps that are 1 or 2
ticks long.

Breakfast observations come from annotated action segments with a median run near 11 ticks
(llm/textify.py's module docstring), so a spike at 1-2 ticks would be surprising. If there is no
spike, the oscillation concern is an EPIC-Kitchens problem rather than a current one and the merge
tolerance can be left at its default without further tuning. If there IS a spike, that fraction
bounds how much of the false-alarm floor and of the LLM query budget it accounts for -- the
predictive arm issues one request per step boundary, so short runs are paid for twice.

Reads the observation stream straight from sequences.json: no model, no fit, no GPU.
"""
import argparse
import collections
import json

import numpy as np

from cook_ad.data import split as split_mod


def run_lengths(verb_ids, noun_ids):
    """Lengths of the maximal runs of constant (verb, noun) -- llm/textify.steps_from_ids' RLE,
    without needing a lexicon to name the tokens."""
    pairs = np.stack([np.asarray(verb_ids), np.asarray(noun_ids)])
    if pairs.shape[1] == 0:
        return []
    breaks = np.flatnonzero(np.any(pairs[:, 1:] != pairs[:, :-1], axis=0)) + 1
    edges = np.concatenate([[0], breaks, [pairs.shape[1]]])
    return np.diff(edges).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", default="dataset/processed/breakfast/sequences.json")
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--split-part", default=None)
    ap.add_argument("--max", type=int, default=None, help="cap on trials, for a quick look")
    ap.add_argument("--short", type=int, default=2,
                    help="a run this long or shorter counts as oscillation (default 2)")
    args = ap.parse_args()

    with open(args.sequences) as f:
        sequences = json.load(f)
    if args.split_file:
        if args.split_part is None:
            ap.error("--split-part is required when --split-file is given")
        sequences = split_mod.filter_sequences(
            sequences, split_mod.load_split(args.split_file), args.split_part)
    if args.max:
        sequences = sequences[: args.max]

    per_trial, lengths = [], []
    for s in sequences:
        rl = run_lengths(s["verb_ids"], s["noun_ids"])
        per_trial.append(len(rl))
        lengths.extend(rl)

    lengths = np.asarray(lengths)
    n = len(lengths)
    if n == 0:
        raise SystemExit("no runs found")
    short = int(np.count_nonzero(lengths <= args.short))

    part = f"{args.split_part} split" if args.split_part else "all sequences"
    print(f"{len(sequences)} trials ({part}); {n} RLE steps, "
          f"median {np.median(per_trial):.0f} steps/trial")
    print(f"run length: median {np.median(lengths):.0f} ticks  mean {lengths.mean():.1f}  "
          f"p10 {np.percentile(lengths, 10):.0f}  p90 {np.percentile(lengths, 90):.0f}  "
          f"max {lengths.max()}")
    print(f"\nfraction of runs <= {args.short} ticks: {short / n:.4f} ({short}/{n})")

    print(f"\n{'ticks':>6}  {'runs':>6}  {'share':>7}  histogram")
    counts = collections.Counter(lengths.tolist())
    top = min(20, int(lengths.max()))
    for k in range(1, top + 1):
        c = counts.get(k, 0)
        print(f"{k:>6}  {c:>6}  {c / n:>7.4f}  {'#' * int(round(60 * c / n))}")
    rest = int(np.count_nonzero(lengths > top))
    if rest:
        print(f"{'>' + str(top):>6}  {rest:>6}  {rest / n:>7.4f}  {'#' * int(round(60 * rest / n))}")

    print()
    if short / n <= 0.10:
        print(f"No 1-{args.short} tick spike ({short / n:.1%} of runs). Oscillation is not a "
              f"property of this corpus, so span_metrics.MERGE_TOL needs no tuning here; revisit "
              f"on EPIC-Kitchens, whose 0.5s ticks halve every run length.")
    else:
        print(f"SPIKE: {short / n:.1%} of runs are <= {args.short} ticks. That fraction bounds "
              f"how much of the false-alarm floor and of the predictive arm's query budget is "
              f"observation oscillation. Tune span_metrics.MERGE_TOL against it.")


if __name__ == "__main__":
    main()
