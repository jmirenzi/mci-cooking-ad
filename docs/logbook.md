# Logbook

Dated entries, newest first, ≤ 5 lines each: what was tried, the number, the verdict, and the
script or commit that measured it. (Older history is folded in from the previous docs during the
restructure's Phase G.)

## 2026-10-15 — restructure, Phase E verified

New-API-only code reproduces every per-trial golden bitwise, every EPIC-Tent table, every
Breakfast test grid and the Breakfast headline table; train-split grids move by <= 0.004, inside
the old code's own run-to-run spread (Viterbi ties). Cost 0.93-1.00x. `docs/restructure.md`.

## 2026-10-14 — restructure, Phase E: scripts deleted, and what each had found

Forty-three root scripts and the cascade/metric-A/B/D/F/G-gap code were deleted (all recoverable at
tag `pre-restructure`). One line per script that had a finding; the rest were plumbing.

- `run_threshold_sweep.py`: alpha 5e-3 sat on the trial_loc shoulder (recall 0.453, precision 0.728, 14.1% healthy trials alarmed); s_emit fired on ~99% of healthy trials until the 1e-6 tie tolerance.
- `run_threshold_sweep_coordinate.py` / `run_threshold_test_tuned.py`: s_temporal at 1e-5 removed 37% of false-positive ticks for 0.006 recall (train-picked, confirmed on test); kept as the per-channel default.
- `render_threshold_sweep_by_channel.py`: per-channel standalone curves; motivated the coordinate sweep above.
- `run_step_sweep.py` / `tools_alarm_load.py`: trial_loc caps false positives at one per trial; step- and event-level counts disagree about which fit is better (led to metric G).
- `report_final.py`: alpha must be picked on train and reported on test (the protocol eval_breakfast.py keeps).
- `run_counterfactual.py`: matched-null "attributable" detection; tick-level transposition detection was real, not chance.
- `run_sequence_eval.py`: the segment-sequence detector could name transpositions the tick channels cannot, at no recall gain; dropped with `anomaly/sequence.py`.
- `tools_launder.py`: Viterbi re-explains ~70% of omissions; the recipe re-inference flips on ~59% of transpositions.
- `tools_oracle_recipe.py`: pinning r̂ to the healthy decode recovers transposition recall lost to recipe re-inference (40–60% of trials move cluster).
- `tools_transition_ceiling.py`: counted injected junctions the data never contains — the ceiling for s_transition.
- `tools_duration_power.py`: s_dur_two alone cannot carry repetition; s_temporal does.
- `tools_run_lengths.py`: no 1–2-tick run spike on Breakfast (median run ~11 ticks), so gap merging was unnecessary there.
- `tools_state_organisation.py`, `tools_kernel_gradation.py`, `tools_embed_vocab.py`, `tools_severity_ranking.py`: the vocabulary-embedding line — kernel emissions buy severity ranking (chance → 0.90) at −0.073 trial_loc; not shipped.
- `run_live_eval.py` / `tools_live_summary.py` / `tools_breakfast_g_table.py`: causal (live) detection matches hindsight F1 within ±0.01 under metric G; lag buys nothing (folded into `scripts/eval/eval_breakfast.py`).
- `run_llm_eval.py`, `render_llm_*`, `run_detect_eval.py`, `run_evaluation.py`: the LLM-vs-HSMM comparison, now `eval_breakfast.py`; metric B (step verdicts) is gone.
- Cascade/flow/lifecycle/mini (`run_experiment`, `run_recipe`, `run_joint`, `run_anomaly`, `run_rollout_demo`, `run_lifecycle`, `export_*`, `render_flow*`, `render_anomaly_png`, `build_mini_dataset`, `main.py`): superseded by the joint model; drift detection is to be rebuilt on `HSMM`.

## 2026-10-14 — restructure, Phase D

`LLMVerdictDetector` and `LLMPredictiveDetector` (detectors/) replay all 158,234 cached LLM
requests with 0 misses, through both the concurrent fast path and the tick-by-tick step loop;
every golden stream and table reproduces bitwise.

## 2026-10-14 — restructure, Phase C

`LiveHSMMDetector`, `LagHSMMDetector`, `HindsightHSMMDetector` and `narrate` (detectors/)
reproduce every golden bitwise at 1.00× cost. π̃-from-the-filter (unification 1) flips flags on
1/582 Breakfast trials and was rejected; r̂-from-the-filter (2) agrees 582/582 but costs too much.
Details: `docs/restructure.md` "Progress notes".

## 2026-10-14 — restructure, Phases A–B

Tagged `pre-restructure`; golden outputs of the old code (`scripts/verify/dump_golden.py`,
`run_tables.sh`) are bitwise reproducible run to run, and every LLM replay is 100% cache hits.
Phase B moved the HSMM inference kernels into `hsmm/kernels.py`, added `class HSMM`
(`hsmm/model.py`), `class BadSets` (`detectors/thresholds.py`) and `core.py`, and shimmed the old
modules onto them. Equivalence and cost results: see `docs/restructure.md` "Progress notes".
