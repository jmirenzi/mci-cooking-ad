# Next steps

Two pieces of work are planned on top of the current detectors. Neither is implemented. The
numbers quoted here were measured on code that has since been deleted (history in
[`logbook.md`](logbook.md)). They say what each option is likely to buy and cost.

---

## 1. Open vocabulary

**The problem.** Emissions are categoricals over a fixed vocabulary, so the model has two blind
spots:

- **No severity grading.** `water` for `milk` and `knife` for `milk` are just two different columns.
  On the fitted models it is worse than that: 97% of emission cells sit on FLOOR
  ([`model.md`](model.md) §1), so a near and a far substitution score identically (gap 0.00 nats).
  The detector can say "that is wrong" but not "wrong object, but nearly right", and the second is
  the more useful question for a user with MCI.
- **No unseen tokens.** A noun absent at fit time has no column and no probability. That rules
  out moving to a new kitchen, a new recognizer vocabulary, or a corpus like EPIC-KITCHENS
  (305 nouns).

Every option below gets its semantics from a fixed text embedding of each token. The encoder was
all-MiniLM-L6-v2, run in ~40 lines of numpy, with SIL tokens pinned to an orthogonal direction.
The verb embedding space failed a nearest-neighbour sanity check, so all options act on nouns
only. Scores below are metric A accuracy (`trial_loc`, Breakfast test split), the metric of the
time. Any revival should re-measure under metric G.

**Options measured so far:**

| Option | Idea | Severity ranking (near vs far, chance 0.5) | Unseen nouns | Detection cost |
|---|---|---|---|---|
| **A. Similarity kernel** | latent intended token: P(n \| k) = Σ_m B_km S_mn, with S fixed from embedding cosines (λ = leak mass, τ = shape). EM stays exactly monotone; S = I is the baseline bit for bit | 0.90 (uniform-leak ablation: chance) | with a rectangular S, yes | −0.073 accuracy |
| **B. Log-bilinear emission** | P(n \| k) ∝ exp(μ_k · e(n) / τ) over a swappable candidate set; μ fitted by generalised EM | 0.88 | yes | −0.025 |
| **C. Centred centroid** (B with zero optimiser steps) | μ_k = unit(Σ_n (p̂_kn − 1/N) e_n), closed form | 0.85, but on a 0.19-nat gap | best of all: held-out-noun MRR 0.238 vs B's 0.181; transferred to EPIC nouns unseen, 43% top-5 of 320 (27× chance) | −0.055; healthy FPR 0.68 vs 0.43 |
| **D. Floor-only reshaping** | keep every observed cell; re-shape only the FLOOR mass by embedding similarity | per-state rows ordered on 99% of pairs | no | none (accuracy 0.502 vs 0.502) |

Option B/C was the work in the `log-bilinear-emission` worktree (2026-09). Its verdict: the
log-bilinear *parameterisation* is right, because it makes the candidate set swappable. *Fitting*
μ is refuted by its own control: on Breakfast's 36 nouns, μ·E spans every categorical, so all
fitting can add is the L2 prior, and C is that prior's closed-form solution. The code was not
kept.

**Two things any option must deal with:**

1. **Moving mass off observed cells costs detection.** A, B and C all buy their ordering that
   way, and that mass is what the anomaly score is made of: a smoother emission makes a swapped
   noun less surprising (C: substitution recall 1.00 → 0.82). D moves none and is free.
2. **The reader decides whether severity is visible.** `s_noun` scores against the π̃ mixture
   ([`detectors.md`](detectors.md) §5.1), so it asks "does any plausible state emit this noun".
   D's 99%-correct rows moved reported severity rank accuracy only 0.588 → 0.598. Scoring at the
   decoded state instead, −log P(n | z*), is the cheap win (C: 0.845 → 0.918). It is an extra
   readout for narration, not a replacement for `s_noun`.

**A reasonable order:** D plus the conditioned readout first, since it gives severity at zero
detection cost. Then C's candidate-set swap for unseen nouns, kept off the detection path (used to
phrase "I expected X" and to grade severity, not to raise alarms). Always compare emission
variants at a shared decode from the same state space as well as each one's own: an arm can hide
tokens it cannot model by re-segmenting.

---

## 2. Drift detection

**Purpose.** Detect long-term, personal change in how someone cooks, and explain it in terms of
model quantities: "you use butter more often", "stirring has slowed over the last quarter",
"omissions are becoming more frequent". The anomaly detectors answer *is this trial unusual for
this person*; drift detection answers *has this person's normal changed, and how*.

**Terms.**

| Term | Meaning | Comes from |
|---|---|---|
| anomaly | anything out of the ordinary in a trial | the HSMM detectors' channels |
| breakdown | an anomaly that is a failure due to MCI (e.g. stalling) | user, caregiver or an LLM interpreter |
| preference shift | an anomaly that is a harmless change (e.g. a new pan) | user, caregiver or an LLM interpreter |
| drift | a change in behaviour that persists over time | the drift detector |

A one-off strange week and a lasting change look the same in any single comparison. The
difference is **persistence**, which the timescales below handle.

**Models.** A detector that absorbs everything it sees stops detecting, and one that never adapts
nags forever. So:

- During week w the anomaly detectors run against a **frozen** model F_{w−1}, which does not move
  within the week.
- A **live** model is updated with every trial for real-time personalisation. It is never a drift
  test input.
- At the week's end, F_w = F_{w−1} updated with the week's trials, filtered by span: breakdown spans
  excluded, preference shifts included, unlabelled anomalous spans excluded by default.
- Frozen snapshots are **kept** at each timescale's reference start, so drift too small to flag in
  any one week cannot be absorbed into the reference week after week.

**Two signals**, computed by one pipeline:

1. *Normal-behaviour drift*: filtered trials in the window vs the reference. How the person does
   things when nothing went wrong.
2. *Anomaly-rate drift*: per-channel alarm counts per trial on **unfiltered** trials. Decline that
   shows up as more frequent anomalies, which signal 1 is blind to by design. A rising omission
   rate is probably the most clinically meaningful drift statistic.

**Timescales.**

| Timescale | Window | Reference | Cadence | α |
|---|---|---|---|---|
| weekly | last 7 days | previous 4 weeks | weekly | 0.01 |
| monthly | last 4 weeks | the 12 weeks before | every 4 weeks | 0.01 |
| yearly | last quarter | the same quarter a year earlier (seasonality) | quarterly | 0.05 |

Weekly, at 7–14 trials, can catch only large localised changes; longer windows see smaller ones.
At weekly α = 0.01 the expected false alarms are ~0.5 per year, and that rate should be reported.

**Method: a stratified permutation test under one common model.**

- Decode every trial of both periods under the *same* model, F_ref (the snapshot at the reference
  start), using **expected counts** from forward-backward rather than Viterbi paths, so decoding
  uncertainty carries through. Each trial becomes count vectors in one shared state space, and
  permuting periods is re-summing vectors: no refits, no label switching.
- **Blocks**, one per distribution: noun P(N | Z), verb P(V | Z), duration P(D | Z, R) binned by
  reference quantiles, transition P(Z' | Z, R), anomaly rate per channel, and optionally a fit
  residual (how badly F_ref explains the trial), which catches behaviour forced into existing
  states.
- **Statistic**: per block, the two-sample G statistic
  D_b = n_win KL(p_win ‖ p_pool) + n_ref KL(p_ref ‖ p_pool); D = Σ_b D_b. Blocks below a minimum
  count in either period are dropped, not counted as zero.
- **Null**: shuffle window/reference labels *within each recipe*, so a change in recipe mix ("more
  butter because more butter recipes") is not drift. Recipes seen in only one period are excluded
  and reported.
- **Decision**: a global permutation p-value gates; then max-T (Westfall–Young) over standardised
  blocks says *which* blocks changed, with family-wise error control. The same statistic decides
  and explains.

A likelihood + CUSUM detector (mean per-trial log-likelihood under F_ref, null by
posterior-predictive simulation) is kept as a baseline. It may have more power on a *distributed*
shift (a little slower in every state). If permutation power there is within ~10–15 points, use
permutation alone; otherwise CUSUM alarms and permutation localises.

**Evaluation.** The synthetic route is being built: [`drift_synth.md`](drift_synth.md).
No public dataset has real long-term within-task kitchen drift. Plan: real
longitudinal data if obtainable (Oatmeal107 / CHEF-VL; the CMU older-adult meal-preparation
study), else drift injected into real trial sequences at controlled onset, rate and size (abrupt,
gradual, incremental, recurring). Test cases: null calibration; one state's duration +20%; one
noun's probability 0.1 → 0.35; recipe mix only (no alarm); +5% duration everywhere; a new recipe
(excluded, no alarm); labelled breakdowns (behaviour blocks unchanged, anomaly-rate block fires).
Compare against generic change-point detectors (PELT, BOCPD) on the same per-trial surprise
stream. Localisation accuracy (is the top flagged block the injected one?) is the contribution
generic detectors cannot match. Closest prior work: Masciadri et al. 2017 (HMM + likelihood
ratio between periods).

**Where it attaches to the current code.**

- Expected counts under a fixed model already exist: `HSMM.filter` + `HSMM.smooth(fr, math.inf)`
  per trial, or training's `hsmm/train/messages.combine_sufficient_stats`.
- The recipe stratum is the trial's labelled dish or r̂ (`HSMM.infer_recipe`).
- The anomaly-rate block reads `Detection.alarms` per channel (`s_temporal` is the plan's stall
  channel, `s_dur_two` its abandonment channel).
- Narration's `Lexicon` should render drift reports, so a subtask has one name everywhere.

**Open decisions.**

- How to filter the frozen update: drop the whole anomalous trial, mask only the span, or
  down-weight by P(anomalous). Dropping whole trials truncates the duration tail, tightening F
  and inflating false alarms. Log all three and compare.
- The default for unlabelled anomalies, which depends on caregiver or interpreter attention.
- Permutation alone vs CUSUM + permutation.
- Whether the residual block is needed.
- Minimum block count, smoothing and permutation count, set by null calibration (starting points:
  5 per period, 0.5, 2000).

Out of scope for now: sequential/anytime-valid tests, the LLM interpreter's prompts, and
sensor or recognizer health checks. The last of these is required before deployment, because a
moved camera or a recognizer update looks exactly like emission drift.
