# Logbook

Dated entries, newest first, ≤ 5 lines each: what was tried, the number, the verdict, and the
script or commit that measured it. Every number states its metric letter where it is a detection
score ([`eval.md`](eval.md)). Scripts named here before 2026-10-14 were deleted in the restructure
and are recoverable at the git tag `pre-restructure`.

## 2026-10-19 — metric H (any flag hits, one miss per run), delay 2

Breakfast dish-given test (97/group), metric H delay 2: hindsight 0.655, live 0.656, LLM 0.35-0.40,
null (tick) 0.301. EPIC-Tent procedural default split (9 sessions, 51 spans): hindsight 0.326, live
0.337, LLM 0.175, null 0.169. Live alarms/10 min 9.3 vs 2.6 under G: stretches expose flicker.
`eval/metric_h.py`, `COOK_METRIC=H`; delay sweep 0/2/5/10 on EPIC-Tent only.

## 2026-10-20 — nonparametric duration hazard tested on synthetic, Breakfast, EPIC-Tent (metric G)

Kaplan-Meier hazard from training decodes, swapped for the NB tables. Synthetic scrubbed, 250
sessions: 0.834 -> 0.871 (abandonment 0.13 -> 0.57); 10-30 sessions -0.02 to -0.035. Breakfast
hindsight 0.659 -> 0.682 (train-picked), abandonment at alpha 0.005 0.35 -> 0.71. EPIC-Tent
unchanged. next_steps.md §3.

## 2026-10-19 — idle scrubbing beats every add-on (metric G)

Same persons refit on idle-free streams: the plain HSMM scores 0.451 / 0.644 / 0.834 at 10 / 30 /
250 sessions, with healthy FPR 0.03 at 250 (idle kept: 0.218 / 0.274 / 0.409; best add-ons 0.827 at
FPR 0.11). s_count adds nothing on scrubbed streams. Abandonment (0.13) is what's left. §9.8.

## 2026-10-19 — idle scrubbed before the model; overnight rerun started

`data/scrub.py`: idle ticks dropped before training and scoring ("A, pause, A" merges; Jack);
`scrub_kept.json` keeps original ticks. `eval_synth_prior.py --variant cal_scrub`,
`overnight_scrub.sh`: same 4 persons refit, step-level pools, plain HSMM and + count channel.

## 2026-10-18 — skip-idle population prior swept to W = 100 (metric G)

W = 30 / 100: 0.371 / 0.371 (10 sessions), 0.576 / 0.578 (30), 0.794 / 0.827 (250; healthy FPR
0.11). Cold start levels off by W = 30. W was compared on test, not picked on validation yet.

## 2026-10-18 — population prior for the skip-idle channel (metric G)

W population pseudo-counts per row (prior arm): W = 10 gives 0.354 / 0.561 / 0.767 F1 at 10 / 30 / 250
sessions (healthy FPR 0.80 / 0.36 / 0.20), against 0.292 / 0.444 / 0.727 without. On the re-scored
Breakfast-injector pool the channels also help: 0.687 -> 0.744 at 250. `--struct-prior`, §9.7.

## 2026-10-18 — step-level injectors; idle-skipping, count and empirical-duration channels (metric G)

Step-level pools (`inject.py`): the HSMM alone scores 0.409 at 250 sessions (transposition 0.08,
repetition 0.00): idle gaps hide first-order context. + skip-idle + count: 0.706; + empirical short
duration: 0.727 (omission 0.88, transposition 0.84, repetition 0.86, abandonment 0.68), healthy
FPR 0.26. Cold start (10 / 30 sessions) 0.29 / 0.44. drift_synth.md §9.7.

## 2026-10-18 — re-scored on oracle-confirmed anomalies (metric G)

Same models, non-anomalous injections dropped from val and test: F1 at 250 sessions 0.393 -> 0.687
(prior), healthy FPR 0.05. Transposition recall 0.67; omission 0.19, abandonment 0.01, repetition
0.05 stay low. `eval_synth_prior.py --pool real`, drift_synth.md §9.6.

## 2026-10-18 — diagnosis: two thirds of structural injections are not anomalies

Oracle against 3,400 extra healthy sessions per person: only 34% of abandonment/omission/
transposition/repetition injections are outside healthy behaviour (gaps and take/put/wash are hit
uniformly; repetitions merge in place). HSMM hits 9–69% of the real ones; omissions need counting
(74% count-only). NB durations: right width, fat left tail. `diagnose_synth_injections.py`, §9.5.

## 2026-10-18 — calibrated prior experiment on the revised generator (metric G)

Mean F1 person only → prior (split α) at 10 / 30 / 250 sessions: 0.161 → 0.180, 0.210 → 0.228,
0.362 → 0.393 (healthy FPR 0.35 → 0.12 at 250). Old generator: 0.117 → 0.147, 0.185 → 0.301,
0.353 → 0.373. Injected repetition recall still 0.04–0.08. docs/drift_synth.md §9.4.

## 2026-10-18 — overnight rerun on the revised generator started

`overnight_v2.sh`: all 37 conditions, sanity plots, synthetic AD eval (noise on/off), calibrated prior
experiment. fine_motor pools now target θ 0.6 / 0.75 / 0.9 (0.6 / 1.1 / 1.7 repeats per session);
+0.2 / 0.4 from a healthy θ0 would have stayed below the hinge. 18-recipe outputs archived.
Then made opt-in (`--capability-pools`): for now the AD eval scores injected errors only (Jack).
The AD eval was then stopped mid-training and its output deleted: it uses the uncalibrated generator,
and only the calibrated prior experiment finished overnight.

## 2026-10-18 — take/put/wash no longer stretch with the CC4D step scale

Logistics inherited their step's scale (a `take salt` next to a 14× step lasted 36 s). Now they keep
EPIC time (median 1.7 s) and the scales are refit on the step's own actions: global median 4.51,
session medians within 6% of CC4D on all 8 recipes. `calibrate_cc4d.py --write-scales`.

## 2026-10-18 — rollout fixes: no doubled actions, no fetched dishes, narrower durations

3-person rollouts (`rollout_examples.py`) had 153 back-to-back identical actions in 480 sessions
(put-backs after the recipe's own put, same-ingredient additions, aliased sauces), `take pancake` in
every pancake session, and 13× spreads on one person's `take banana`. Now 0 doubled actions in 1200
sessions, 0 fetched dishes, durations log-sd × 0.5 (mean kept).

## 2026-10-17 — generator revision: 8 recipes, no healthy repeats, two-sided flavour knobs

18-recipe sessions were 40 runs (Breakfast 6), 32 pairs per recipe (8), all unique; cut to 8 simple
recipes. Healthy fine_motor repeated 12.5% of its tokens (52% merged into one run): now hinged at
θ 0.5, healthy θ0 U(0, 0.4), 0 repeats in 634 null sessions. sour/sweet/salty/fatty two-sided with
LLM affinity (awaiting review). Not yet regenerated or re-evaluated.

## 2026-10-17 — CC4D-calibrated prior rerun (metric G)

Mean F1, person only → prior λ = 1 (split α), at 10 / 30 / 250 sessions: 0.117 → 0.147, 0.185 → 0.301,
0.353 → 0.373; healthy FPR at 250 is 0.23 → 0.10. Every person gains at 30 and 250, p00 included; p00
and p03 tie or lose at 10 (loose α_trans). Fading = fixed again. `overnight_cal.sh`, 4 h 47 min.

## 2026-10-17 — training size counted in sessions; calibrated rerun started

`eval_synth_prior.py` splits by session index, not day: train on the first N sessions, validate on
250–324, test on 325–399 (same 150 held-out sessions per person). The CC4D-calibrated overnight
run (`overnight_cal.sh`, N = 10 / 30 / 250 sessions) was launched on this split.

## 2026-10-16 — split transition α and a fading prior (metric G)

Split α turns the fixed prior's 100–200-day loss into a gain or tie (mean F1 0.382 / 0.394 against
person-only 0.360 / 0.399). A fading prior λ₀/(1+n/n₀) is within 0.02 of fixed everywhere. p00
(100–200 days) and p03 (10–30) still lose with any prior. The CC4D-calibrated overnight rerun was
started, then stopped by Jack: early problems first.

## 2026-10-16 — global population prior + per-person fine-tune (metric G)

60 simulated people → one count prior. Mean over 4 persons, F1 person-only → +prior: 10 days
0.152 → 0.200, 30 days 0.241 → 0.301, 100 days 0.360 → 0.362, 200 days 0.398 → 0.377. λ (1–100)
barely matters. All 4 losing cells are validation picks jumping to a loose α. CC4D: real orders
are as varied as uniform, real step time is 4–5× ours (`docs/drift_synth.md` §9.2–9.3).

## 2026-10-16 — lowering the synthetic healthy alarm floor (metric G)

Hindsight F1 / healthy-trial FPR: baseline 0.202 / 0.90; 600 training days 0.238 / 0.51 (learns
the noise, structural recall falls); habit 2 0.214 / 0.86; noise × 0.5 0.263 / 0.78; all three
0.290 / 0.56; noise-free 0.355 / 0.45. The residual floor is unseen transitions (70–80% of
flags). `s_recipe_transition` duplicates `s_transition` here (`docs/drift_synth.md` §9.1).

## 2026-10-16 — HSMM anomaly detector on synthetic data (metric G)

One person (seed 0, p06), fitted on 265 normal sessions. Injected errors: F1 0.355 noise-free,
0.202 with recognizer noise (healthy-trial FPR 0.45 → 0.90; unseen transitions set the floor).
slow and fine_motor at up to +0.55 θ: no per-trial signal (Δ alarms vs twin ≈ 0). Found and fixed:
metric G's alarms/10 min assumed 1 s ticks (`GridScore(tick_seconds)`).

## 2026-10-15 — synthetic drift data: all conditions + sanity plots (M4)

31 conditions × 3 seeds × 8 persons generated (about 22k sessions, 70 s). At |Δθ| 0.45 every knob
moves its own feature group and leaves the others at MC noise. The weakest is mise_en_place
(bigrams 3.9×). Dose-response is monotone. Sessions run 2.8 min against CC4D's 15.7, because
waits are unmodelled (`docs/drift_synth.md` §8, `scripts/drift_synth/sanity_plots.py`).

## 2026-10-15 — synthetic drift data: step expansions (M2)

gemma3, one pass per step: zero variants, padded pairs ("add pan"). Two passes plus validation:
257/257 steps, 26 calorie omits, 0 substitutes. Asked per step it put sour into almost every step;
one call per recipe gives 21 sites. `null` generates: median session 2.9 min against CC4D's ~15,
because EPIC waits are seconds. Waiting on Jack's spot-check.

## 2026-10-15 — synthetic drift data: phases, tags, seeded generator (M1 + M3)

The 18 recipes were approved. gemma3 phases needed two prompt rules: adding to a hot pan is cook,
and a recipe with no heat has no cook steps. Its tags proposal over-tags (knife and spoon as
ingredient, whisk as heat, 34 fine-motor verbs), so it is left as proposed for review. Stages 4–10
pass the spec's 9 tests on a toy world (`tests/test_drift_synth_generator.py`). EPIC durations
have no long waits, so sessions will run shorter than CC4D recordings unless expansions use `wait`.

## 2026-10-15 — synthetic drift data: pre-checks

Branch `synthetic-drift`. CC4D has no public fine-grained actions, so durations and exemplars come
from EPIC-100. All 24 task graphs validate. 18 of 24, including every stovetop dish, leave
`mise_en_place` nothing to reorder. Proposed 18 recipes, every knob covered by ≥ 6
(`scripts/drift_synth/precheck.py`, `docs/drift_synth.md` §5). Waiting on Jack's approval.

## 2026-10-15 — next steps written down

`docs/next_steps.md` replaces `lifecycle.md`: the open-vocabulary options with their measured
costs, and the drift-detection plan (frozen references, recipe-stratified permutation G-test under
one common model, max-T localisation). The uncommitted log-bilinear worktree was discarded; its
results are in the 2026-09-03 entries and next_steps.md §1.

## 2026-10-15 — post-restructure cleanup

Deleted 3.0 GB of untracked outputs that nothing reads (restructure goldens, outputs of deleted
scripts, cascade and superseded checkpoints, old figures and reports) and two stale worktrees. The
config keys left over from the cascade warm start (`warm_start`, `cascade_*`, `restarts`, ...) are
gone; `configs/*.yaml` now hold only keys the code reads.

## 2026-10-15 — the restructure (branch `restructure`, Phases A–G)

The repo was rebuilt around one `AnomalyDetector` contract under one rule: behaviour does not
change. Phase A tagged `pre-restructure` and dumped golden per-trial flags, tables and LLM streams
from the old code. B added `core.py`, `class HSMM` (`hsmm/model.py` on pure-JAX `hsmm/kernels.py`)
and `class BadSets`. C built the Live, Lag and Hindsight detectors and `narrate`. D built the two LLM
detectors. E built `eval/{metric_g,metric_e,pools,harness}.py`, `hsmm/train/`, the `scripts/` tree
and the demo, and deleted the cascade, `anomaly/`, `recipe/`, `lifecycle/`, metrics
A/B/D/F/G-gap, `s_pair`, `sequence.py`, the similarity kernel, `noun_tilt` and 43 root scripts. F
froze canonical pools. G rewrote the docs. **Verified**: every per-trial golden reproduced bitwise
through the new API; every EPIC-Tent table and the Breakfast headline table reproduced to 3
decimals; all 158,234 cached LLM requests replayed with 0 misses; cost 0.93–1.00× the old code.
Deviations from the spec: `HSMM.smooth(fr, L)` takes a `FilterResult`; `narrate` takes the detector,
not the HSMM; Lag raises emission alarms at t; `hsmm/batch.py` was added for bitwise batching;
`llm/textify`'s ground-truth helpers were kept; the EPIC-Tent injected control is now scored under
G (its G-gap numbers moved) and the `@merge_tol=0` row is gone; `eval_breakfast.py` also scores the
LLM verdict arms. Two unifications were tried and not adopted: π̃ from the filter flipped 1/582
hindsight flags; r̂ from the filter agreed 582/582 but cost several times more. Breakfast
train-split grids moved ≤ 0.004, inside the old code's own run-to-run spread: exactly tied Viterbi
paths were broken nondeterministically by the old code (the new code is deterministic).

## 2026-10-15 — canonical pools (restructure Phase F)

Every detector is now scored on fixed pools (`scripts/data/build_pools.py`, placement by
`joint_params.npz`, seed 0). The Breakfast test pool IS the LLM pool, bitwise, so the headline
table did not move to 3 decimals (one live κ pick 0.1 → 0.7, same numbers). Per-model test-grid
best F1 moved ≤ 0.007 (G). EPIC-Tent unchanged. Picks: `configs/operating_points.yaml`.

## 2026-10-15 — live detection costs no F1

Metric G, train-picked, Breakfast test: live vs hindsight F1 within ±0.01 on all three models
(dish given 0.662 vs 0.659). EPIC-Tent PREGO split, procedural: live 0.426 vs hindsight 0.402 at
2.54 vs 3.06 alarms/10 min; the LLM predictive arm 0.220–0.226 at ~43 (not rerun in Phase F). Lag
L = 0…∞ identical. Dish given vs inferred ≤ 0.011 F1 (dish accuracy 94.1%). `run_live_eval.py`.

## 2026-10-15 — metric G merges one alarm per run

The 5-tick gap merge let flag-every-tick read 0.000 F1 at 4.8 alarms/10 min while alarming
nonstop. G now merges one alarm per run of constant (verb, noun); flag-every-tick = flag-every-run
(Breakfast F1 0.348). The old rule is G-gap (retired). Say "run", not "step".

## 2026-10-15 — LLM replays need the original base URL

The cache key includes `base_url`. Breakfast's responses were collected through
`http://localhost:11435/v1`; replaying with the default port missed on the first request and looked
like a prompt change. An hour lost. Replay with that URL and `--max-requests 0`.

## 2026-10-14 — live emission test split by timing

The plain forecast p-value double-counted step timing: on 39 healthy Breakfast test trials at α =
0.005, all 28 live `s_emit` flags sat on step changes after steps that ran a median 0.52 of their
expected length. Taking the larger p-value of "stayed" and "switched" cut trials with any emission
flag from 15 to 6. `anomaly/live.py:split_emission_pvalues`.

## 2026-10-14 — causal flags need a one-tick grace under metric A

The live early-exit channel fires on the next run's first tick; metric A's positive range ends one
tick before that. Live abandonment recall was 0.00 honest vs 0.43 with flags moved back one tick
(healthy FP 0.00). The hindsight `s_dur_two` sits one tick early, which is part of its 0.70. Metric G's
deadline removes the artifact.

## 2026-10-14 — metric G adopted as the headline

Too many unlabelled scorers. D let any flag find a span, so flag-every-tick scored D precision 0.85
/ F1 0.92 on EPIC-Tent, above every detector. A caps FPs at one per trial and has a 0-tick
deadline. Decision: G (alarm onsets only) everywhere, E only against PREGO, every table labelled.

## 2026-10-14 — the restructure agreed; 43 scripts deleted

The owner found the repo unreadable. Spec agreed (`AnomalyDetector`, `HSMM`, Live/Lag/Hindsight,
LLM detectors; delete dead code; golden-output verification). The deleted scripts' findings are in
the dated entries below: threshold sweeps, counterfactual, sequence detector, laundering, oracle
recipe, transition ceiling, duration power, run lengths, alarm load, the embedding track.

## 2026-10-12 — the HSMM arm reads the future

Every HSMM flag read the whole-trial r̂ and Viterbi path, and in the joint model even π̃ was
computed under r̂; the LLM arm was strictly causal. So every HSMM-vs-LLM number compared hindsight
with live. Led to the live design (`docs/live.md`, 2026-10-14) and the 2026-10-15 measurement.

## 2026-10-12 — EPIC-Tent errors inside deleted actions

First-come-wins overlap resolution can swallow the action an error was committed in (p20 ties the
top inside a 119 s `insert support`). 15 records moved to `unobservable_records`; still excised from
training, so every checkpoint stayed byte-identical. `parse_epic_tent.host_swallowed`.

## 2026-09-29 — EPIC-Tent: real errors, scored per error group

Metric D (retired), 9 test sessions, procedural: HSMM F1 0.383 at 2.8 alarms/10 min vs the
flag-every-boundary null 0.262; the predictive LLM (0.233) was dominated by that null; giving it the
instruction order bought nothing. Temporal errors were caught by the transition channel (4 of 5),
not the duration channels. The emission channels fired 0 times. `run_epic_tent_eval.py`.

## 2026-09-29 — error-excised training on EPIC-Tent

Every training session contains annotated errors. Fitting on error-free fragments instead (metric D,
all six types): precision 0.806 → 0.647, recall 0.197 → 0.260, F1 0.316 → 0.371; order recall 0.300 →
0.425. Contamination makes the model permissive. Adopted: `joint_params_clean.npz`.

## 2026-09-29 — injected vs annotated errors on the same sessions

The HSMM on EPIC-Tent with *injected* errors: precision 0.093, recall 0.370, F1 0.148 (G-gap, now
rescored under G by `eval_epic_tent.py`). Injected errors are found more often and far less cleanly
than annotated ones. The control's "healthy" sessions still carry real errors.

## 2026-09-29 — PREGO's protocol reproduced (metric E)

PREGO's split recovered exactly (15 test ids = their prediction file). Trimmed, step level: HSMM
3/15 at 8% of correct steps flagged; predictive LLM 15/15 at 95%; flag-every-step 0.089–0.108
precision. PREGO's published 0.107/0.867 implies ~71% of correct steps flagged. TI-PREGO's balanced
precision scores flag-every-step at 0.500 / F1_b 0.667, above all its Epic-tent-O rows.

## 2026-09-29 — the one-recipe channel needs two fixes

At K_R = 1 `s_recipe_transition` fired on a flooring artifact (up to ~6 nats); it was the earliest
channel on 17 of 25 caught spans. A zero statistic alone looked complete for a week, until the
excised fit's sub-zero thresholds fired 6 ticks; +inf thresholds were also needed. Breakfast's 112
flat rows are safe while the comparison stays strict.

## 2026-09-22 — EPIC-Tent ingested

29 sessions, 1 s ticks, 12 actions factor into 12 (verb, noun) pairs, so subtask ARI 1.000; K_R = 1,
and joint EM converges in 9 iterations (the limit cycle is a recipe-layer effect). 367 mapped error
records; PREGO's subset is exactly their 163. Overlaps drop 127 of 1,252 segments, 79 of them
instruction reads. `parse_epic_tent.py`.

## 2026-09-22 — EPIC-Tent order errors are not bigram-visible

Errors are annotated against the printed instructions: the transition into an out-of-order segment
has median corpus count 23 against a 22 baseline. A rank-jump heuristic scores F1 0.22 at a 0.14
base rate. Yet `s_transition` was the earliest channel on 11 of 12 caught order errors: it detects a
locally unusual pair, not out-of-orderness.

## 2026-09-22 — few-shot breaks on long sessions

Three EPIC-Tent exemplars render as 191 lines behind a 34-line grammar; gemma3:27b abandoned the
grammar on every request (0 parsed flags vs 57 at 0-shot). EPIC-Tent runs 0-shot; when exemplars
are requested they come from error-free fragments.

## 2026-09-03 — open vocabulary, phase 2: fitted log-bilinear refuted

Vocabulary holdout, shared decode, 4 seeds (mean log P / MRR): categorical −29.8 / 0.052, kernel
−8.1 / 0.082, log-bilinear −7.2 / 0.181, closed-form centred centroid −3.5 / 0.238. Fitting μ is beaten
by its own zero-step control. Every open-vocab arm costs metric-A accuracy (0.502 → 0.447–0.477).
Not shipped; deleted in the restructure.

## 2026-09-03 — severity lives in the reader

`s_noun` is a mixture over π̃, so fixing the emission rows' near/far ordering (99% correct) moved
reported severity rank accuracy only 0.588 → 0.598. Scoring at the decoded state recovers part
(centroid arm 0.845 → 0.918). Also: always report an emission variant at a shared decode as well as
its own, or an arm can hide tokens by re-segmenting (kernel arm: worse than chance when shared).

## 2026-09-02 — the predictive LLM arm (arm 3)

PREGO-style: the LLM predicts duration and next run, code compares. Smoke test on 4 healthy trials:
symbol disagreement 0.68 per boundary, over-run 0.62, under-run 0.17; gemma3 under-predicts
durations. The under-run channel is not in PREGO and is what lets it see abandonment.

## 2026-09-02 — score spans by existence, not by tick

Tick-summing recall measures flag density, and span widths differ by type (median 13.5–15 ticks;
abandonment 1). Span scoring counts a span found or not. Its run-merging precondition held on
Breakfast: median run 11 ticks, 9.4% of runs ≤ 2 ticks, no spike (`tools_run_lengths.py`).

## 2026-09-02 — span metric D's saturation exploit

Breakfast, 97 test trials: flag-every-tick scored D F1 0.706 against the HSMM's 0.530, because TPs
counted raw flags and FPs merged ones. The predictive arm was dominated by flag-every-boundary on
precision, recall and F1. Never quote a span F1 without its null and an alarm rate; led to G.

## 2026-08-31 — the LLM at tick resolution

`--unit tick`: one request per tick (146,724 for the test split, both variants). Lines carry elapsed
time; the "ended" note lands on the next run's first tick. Ground truth became points, not ranges:
at the tick unit a transposition window covered 29.9% of a trial vs 1.8% by its junction points.
`run_tick_test.sh`, base URL 11435.

## 2026-08-29 — noun tilt and the λ anchoring dial

A per-recipe noun reweighting gave the recipe latent a direct channel into content (at K = 128 on
EPIC only 2.1% of transition cells survive the MAP mode; Breakfast K = 64: 31.3%). A λ-scheduled
recipe prior anchored assignments. No shipped model uses tilt; it was deleted in the restructure
(λ remains in `joint_em`).

## 2026-08-28 — per-channel α

`s_temporal` at α = 1e-5 with the rest at 5e-3 removed 37% of false-positive ticks for 0.006 recall
(metric A, picked on train, confirmed on test by `run_threshold_test_tuned.py`). Kept as the
per-channel default; EPIC-Tent still runs at it.

## 2026-08-23 — EPIC-KITCHENS ingest and the XLA autotune trap

0.5 s ticks (at 1 s, 25% of narrations vanish), 2–15 minute visits, first-come-wins overlaps.
Recipe clustering needs both nouns and IDF: dish ARI −0.009 → 0.538. A "hanging" EPIC-scale E-step
was XLA autotuning: compile did not finish in 4 min with it on, 3.0 s off (`xla_env`).

## 2026-08-22 — open vocabulary, phase 1: a similarity kernel

Near-vs-far substitution rank accuracy: baseline 0.53 (chance), semantic kernel λ = 0.15: 0.90,
uniform ablation: chance. Metric-A accuracy 0.552 → 0.480. The planned decisive test was wrong:
baseline near-substitution recall is already 0.98. Not shipped.

## 2026-08-22 — emission rows are deltas

97% of (state, noun) cells hold count ≤ 1, and the MAP mode max(c − 1, FLOOR) floors every one:
the background mass meant to keep off-pair surprise finite is erased. Near and far substitutions
score identically (gap 0.00 nats). This is also why lag later bought nothing.

## 2026-08-22 — injector redesign

Substitution edits a whole segment's verb or noun with an unseen token; abandonment keeps 5–20%;
every injector is interior-only. Before: 28% of substitutions hit the first/last idle segment, 21%
of abandonments the trailing idle, 3.2% of abandonments were no-ops. HSMM recall on the LLM pool fell
0.79 → 0.47 (metric B), almost all from abandonment (0.87 → 0.34): harder, not broken.

## 2026-08-21 — narration is silent more often than flagging

On the 95-trial figure pool, `narrate_joint` asked nothing on 69% of repetition trials, 62% of
omission, 55% of abandonment, 42% of transposition, 0% of substitution. Figures that required the
HSMM to have spoken were therefore selected for its successes.

## 2026-08-20 — moving the detector from the training side

Lexical warm start + hard EM + transition smoothing + duration refit: test accuracy 0.468 → 0.508
(metric A, four injection seeds; always-flag scores accuracy 0.455, F1 0.625, so quote accuracy).
`--init-prior-scale 0` fits a better detector (0.556 vs 0.518 train) at a worse objective.

## 2026-08-20 — the training objective is not the selection criterion

The lexical warm start begins at −12,948, above where cascade-started joint EM converges (−14,245);
soft EM from it lowers subtask ARI 0.999 → 0.940; the highest-likelihood fit (−11,863, ARI 0.935) was
beaten on detection by fits a thousand nats lower. `--init-prior-scale 1` ends higher (−12,594 vs
−12,931) yet detects worse; the fits differ in transition-row entropy (0.470 vs 0.518).

## 2026-08-20 — select regularisation out of sample

Backoff τ chosen on train looked best at 2; on a nested dev fold (322 / 80) τ = 2 was worst and 30
best. Symptom: flagged healthy steps 0.8% train vs 4.3% test. With τ = 30, step precision 0.691 vs
the cascade's 0.626 at matched recall 0.511 (metric B). `make_dev_split.py`.

## 2026-08-20 — where detections are lost

Viterbi re-explains ~70% of omissions that create a never-seen junction: the cascade fit split 10 of
48 (verb, noun) pairs across 2–4 states, a legal detour (30% → 92% of junctions survive under the
lexical fit). The MAP recipe flips on 21% of
omissions and 59% of transpositions; pinning r̂ recovers that recall (`tools_oracle_recipe.py`).
~49% of injected junctions are never seen in training. Repetition is duration-limited (28% at α = 0.05).

## 2026-08-20 — trial_loc hides nagging

Metric A charges at most one FP per trial. Counting contiguous alarms (`tools_alarm_load.py`):
1.1–1.4 alarms per charged FP for the HSMM, 3.2 for the LLM, so A flattered the chattier arm. Under A,
accuracy = (5R + 1 − h)/(6 + 5s), so recall is worth ~5× the healthy FPR. This led to G.

## 2026-08-19 — choosing α

Threshold sweep (metric A): α = 5e-3 gave recall 0.453, precision 0.728, 14.1% of healthy trials
alarmed; α = 0.05 bought 16 points of recall for ~3× the false alarms. `s_emit` had fired on ~99% of
healthy trials until the 1e-6 tie tolerance on the dilution bound. `run_threshold_sweep.py`.

## 2026-08-19 — the trial's last segment is not scored for duration

The censored final segment fired "idle only took 8 ticks — you usually spend about 186" at the end of
healthy trials: 62% of the residual healthy false-positive rate at tight α (0.100 → 0.038, metric A).
Trimming the trailing idle (same day) fixed the cause: the always-last idle state had a fitted mean
of 734 ticks against a median run of 5.

## 2026-08-19 — counterfactual pairing (metric F)

Each injected trial scored against its own un-injected twin: matched null 0.017–0.055, ~5× below a
uniform-placement estimate. Lift (419 trials, α = 5e-3): substitution +0.955, abandonment +0.368,
omission +0.263, transposition +0.322, repetition +0.169, all McNemar χ² ≥ 69. Detection on every
type is real, at low sensitivity.

## 2026-08-19 — a segment-sequence detector

Swap/bridge/duration tests over the Viterbi segments named transpositions (0.000 → 0.380) but
detected worse (recall 0.186 vs 0.377), and unioned with the tick channels lowered F1 0.558 → 0.549
(metric A). As a type relabeller only, it lifted transposition naming to 0.199 for −0.032 on
omission. Deleted in the restructure (`anomaly/sequence.py`).

## 2026-08-18 — the LLM baseline

gemma3:27b served locally, 95 trials, conversational, metric A in-sample: HSMM F1 0.503, LLM 0.483 /
0.496 (no / with recipes), healthy-trial FPR 0.211 vs 0.663 / 0.832. Conversational beat incremental
(precision 0.32 vs 0.19). Step-level LLM precision was 1.2–1.5× chance (metric B), with parse
failures on 0.25% of 19,253 requests: a capability result, not a formatting one.

## 2026-08-18 — synthetic healthy trials are not neutral ground

Ancestral samples from the HSMM flatter it and penalise the LLM: LLM healthy FPR 0.767 on synthetic
trials vs 0.418 on real ones (metric B), because samples contain pairings and orders no cook would
produce. Report real trials only; the synthetic arm was dropped.

## 2026-08-18 — serving the LLM locally

Raising `OLLAMA_NUM_PARALLEL` without bounding `OLLAMA_CONTEXT_LENGTH` put two thirds of gemma3:27b
on the CPU silently; bounding it made the run 4× faster (850 s → 212 s). qwen3's thinking cannot be
disabled through the OpenAI endpoint (~250 tokens per one-line verdict).

## 2026-08-18 — which joint checkpoint

Effective K_R (recipes actually used) was the only axis that separated checkpoints: 5 → 14 raised
healthy FPR 0.050 → 0.100 and lowered step precision 0.951 → 0.893 at flat recall (metric B, 40
trials). `joint_params.npz` (effective K_R 5) was kept; it now places the canonical pools.

## 2026-08-11 — the recipe layer

Tripling recipe ARI (0.31 → 0.86) moved detection by nothing (mean recall 0.823 → 0.821); the joint
model still beat the cascade (healthy FPR 0.104 → 0.065). A random partition beat the true one by
175 nats of training likelihood. Bag-of-nouns k-means alone reaches ARI 0.88. EM degrades a given
partition. The recipe mixture is a capacity mechanism, not a classifier.

## 2026-08-11 — joint EM limit cycle

Joint EM settles into a period-3 cycle (−0.111, +0.045, +0.070 per iteration), so `tol` never
fires. Iterations 100 → 320 changed recipe ARI, effective K and subtask ARI by nothing. Raising
`global_damping` to 0.9 sharpened the cycle. Stop at ~100.

## 2026-08-06 — damping the global duration fit

The pooled per-state duration fit, refit from scratch each M-step, swung ρ 471 → 38 in one step for a
near-empty state, perturbing all 16 recipe copies and driving a ~30-iteration objective decline. An
EMA (`global_damping` = 0.7) fixed it.

## 2026-07-28 — calibrated thresholds and coherent narration

Per-state exact α-tail thresholds replaced hand-set nat thresholds (the best achievable threshold is
only 2.5–3.3 nats, so a flat floor would override every state). The dilution correction made the
mixture score comparable with one state's threshold. "I expected X" now conditions on the
unflagged token, after the naive version produced "pour kitchen".

## 2026-07-21 — the HSMM core

Explicit-duration HSMM with negative-binomial durations, right-censoring by ECM imputation, and
banned self-transitions. Newton on the NB shape is seeded by method of moments: seeded from the old
value (ρ = 10 against data from ρ = 4) it diverged to ~38,000.
