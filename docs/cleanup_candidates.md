# Cleanup candidates

Untracked paths that nothing in the current code reads, proposed for deletion. **Nothing here has
been deleted.** The owner approves (or strikes) each row; then they can be removed in one go.
Everything listed is gitignored, so deleting it loses only the file, not history. Code that made
these files is recoverable at the tag `pre-restructure`.

**Never deletable, and deliberately not listed:** `dataset/processed/*/llm_cache/` (days of GPU
time; Breakfast's is 1.1 GB of the 1.2 GB in that directory); the shipped models (Breakfast
`joint_params.npz`, `joint_params_train_lexhard.npz`, `joint_params_train_dish.npz`; EPIC-Tent
`joint_params_clean.npz`, `joint_params_clean_prego.npz`, with their `.meta.json`); every
`sequences*.json`, `labels*.json`, `vocab.json` and `split*.json`; the canonical pools
(`dataset/processed/*/pools/`); the raw annotations (`dataset/breakfast_actions/`,
`dataset/epic_tent/`).

Sizes from `du -sh` on 2026-10-15.

## `runs/` (2.9 GB)

| Path | Size | Reason |
|---|---|---|
| `runs/golden/`, `golden_A2/`, `golden_B/` … `golden_E/`, `golden_preexisting/`, `golden_*.log`, `golden_D_result.txt` | ~2.8 GB | restructure verification dumps; the verifier (`scripts/verify/`) is deleted. Keep until the branch is merged if a re-check might be wanted |
| `runs/cmp_pre/`, `cmp_disk/`, `cmpB_partial/`, `cmpC/`, `cmpC2/`, `cmpD/` | ~3 MB | restructure table comparisons, same reason |
| `runs/train_new/`, `train_new_*.log` | ~4 MB | Phase E training-reproduction check |
| `runs/smoke_eval_breakfast.json` | small | a smoke run of the new eval script |
| `runs/live_*.json` (18 files), `runs/live_logs/`, `runs/live_oracle_test.json` | ~90 MB | outputs of the deleted `run_live_eval.py`; superseded by `scripts/eval/eval_breakfast.py` (results are in the logbook, 2026-10-15) |
| `runs/breakfast_g_table.json` | small | output of the deleted `tools_breakfast_g_table.py`; same table is now printed by `eval_breakfast.py` |
| `runs/gate_seed*_*.npz`, `full_seed0_*.npz`, `floor_tau*.npz`, `holdout_*.json`, `transfer_*.json`, `detect_arm_*_test.json`, `detect_floor*_test.json`, `detect_repro_base_test.json` | ~30 MB | open-vocabulary emission experiments (kernel, bilinear, centroid); that code is deleted, and `params.load` refuses the extra keys the non-baseline fits carry |
| `runs/epic_base.*`, `epic_both.*`, `epic_lam1.*`, `epic_tilt.*` (`.npz`, `.meta.json`, `.log`) | ~3 MB | EPIC-KITCHENS noun-tilt / λ experiments; tilt is deleted |

Keep `runs/phase_F/` and `runs/phase_F_tables.log`: they are the run that produced README's tables.

## `dataset/processed/breakfast/` (~35 MB of the 1.2 GB; the rest is `llm_cache/`)

| Path | Size | Reason |
|---|---|---|
| `figures/` and 13 `figures_*/` (`conv100`, `conv100_fixed`, `current_cascade`, `current_real`, `fixed`, `newhsmm`, `test_train`, `tick`, `tick_compare`, `tick_narrate`, `tick_smoke`, `tick_smoke2`, `trialloc`) | 21 MB | rendered by deleted `render_*` scripts; the current figure is `scripts/figures/trial_figure.py` |
| `flow/` | 0.1 MB | `export_flow.py` output (deleted) |
| `hsmm_params.npz`, `hsmm_params_train.npz`, `recipe_params.npz`, `recipe_params_train.npz` | 0.2 MB | the deleted cascade model; not loadable by `hsmm/params.load`. `configs/breakfast.yaml` still names two of them under `joint_em` (stale keys, read by nothing) |
| `joint_clean.npz`, `joint_d09.npz`, `joint_kmeans10.npz`, `joint_noun10.npz`, `joint_noun16.npz` (+ `.meta.json`) | 2.8 MB | earlier joint checkpoints compared in the 2026-08-11 / 08-18 logbook entries; not used by any script |
| `joint_params_train.npz` (+ `.meta.json`) | 0.6 MB | the pre-lexhard train-split fit (`run_detect_eval.py`'s default); superseded by `joint_params_train_lexhard.npz` |
| `joint_dish_soft.npz` (+ `.meta.json`), `joint_dish_hard.npz`, `joint_dish_smooth.npz` | 1.1 MB | intermediate stages of an earlier dish fit; `train_hsmm.sh` now writes its own `joint_params_train_dish_*` intermediates |
| `embeddings.npz` | 0.2 MB | vocabulary embeddings for the deleted similarity kernel |
| `counterfactual_report.json`, `sequence_report.json`, `sequence_relabel_report.json`, `sequence_relabel2_report.json` | small | outputs of deleted runners (metric F, `sequence.py`) |
| `threshold_sweep*.json`, `threshold_sweep_tick.log`, `threshold_test_tuned.json`, `tick_alpha_test.log` | 1 MB | metric-A threshold sweeps (deleted scripts; findings in the logbook) |
| `llm_report*.json`, `llm_full_report*.json`, `llm_conv*_report.json`, `llm_trialloc_report.json`, `llm_tick_smoke*.json`, `llm_tick_test*.json`, `llm_tick_test.log` | 0.4 MB | `run_llm_eval.py` reports (metrics A/B); the LLM arms are now scored by `eval_breakfast.py` from the cache |
| `rerun_logs/`, `render_tick_narrate.log` | small | logs of deleted scripts |
| `reports/` | 0.9 MB | `scripts/analysis/participant_variability.py` output. That script is kept, so this is regenerable; delete only if not wanted |

## Other corpora

| Path | Size | Reason |
|---|---|---|
| `dataset/processed/union/` | 1.3 MB | only `embeddings.npz` for the deleted kernel |
| `dataset/processed/epic_tent/joint_params.npz` (+ `.meta.json`) | small | the whole-session (contaminated) fit; every script uses the error-excised `joint_params_clean*.npz` |
| `dataset/processed/epic_tent/epic_tent_report*.json`, `report_hsmm_clean.json` | 0.3 MB | pre-restructure reports, several under retired metrics (D, G-gap). `eval_epic_tent.py` rewrites `epic_tent_report.json` by default |
| `dataset/processed/epic_tent/figures/` | 1.6 MB | narration figures from deleted render scripts |

## Worktrees

Not files in this checkout, but disk the owner may want back (`git worktree remove <path>`):

| Worktree | Reason |
|---|---|
| `../mci-cooking-ad-old` (detached at `pre-restructure`) | old-code copy used to time the restructure; not needed after merge |
| `.claude/worktrees/log-bilinear-emission-e76024`, `.claude/worktrees/log-bilinear-emission-epic-7c4138` | open-vocabulary experiment branches (2026-09); their results are in the logbook and the approach was not shipped |
