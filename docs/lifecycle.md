# Next: drift detection

**Not implemented.** This doc says what the lifecycle layer will do. The previous implementation
(`lifecycle/`, deleted in the restructure) is recoverable at the git tag
`pre-restructure`. The new one will be built on `class HSMM` ([`model.md`](model.md)).

**The problem.** A detector that adapts to everything it sees stops detecting: if the user swaps
almond milk for cow's milk, confirms it once, and the model absorbs it fully, later substitutions
go silent, including the ones that are not preferences. A detector that never adapts nags forever.

**Two models.** A *frozen* reference that moves only at an explicit consolidation, and a *live*
copy that absorbs confirmed preferences. Detection reads the live copy; drift is the difference
between the two.

**Preference updates.** A narrated question ([`detectors.md`](detectors.md) §6) gets an answer:

- *"that was on purpose"*: a bounded count bump to the live model's emission or transition row,
  capped relative to the frozen row, so no single confirmation can make the token typical;
- *"that was a mistake"*: update nothing. A recognised error is evidence the model was right.

Consolidation resets frozen to live; it is a human-in-the-loop decision, not an increment.

**Duration drift.** Durations are NB point estimates, so they cannot take count bumps. They get a
report instead: per subtask, compare a recent window's duration histogram against the frozen fit
(drop each trial's censored last segment, fit NB with the training estimator, one-sided
Mann–Whitney on the raw durations). Report only when there are ≥ 5 segments on each side, the mean
shifts by ≥ 2 ticks, and p < 0.05. KL alone is not a decision rule at a handful of sessions a week.
Rendered through the same `Lexicon`: "You've been slower at pour milk this week — about 34 ticks
now vs your usual 21."

**Open.** Localising categorical drift by per-state KL(live ‖ frozen); giving durations a real
live/frozen split instead of a report; whether the corpus supports personalisation at all
(`scripts/analysis/participant_variability.py` tests order and duration multimodality per
participant).
