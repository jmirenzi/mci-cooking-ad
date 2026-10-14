#!/usr/bin/env bash
# Golden TABLES (docs/restructure.md §9.1): the real eval scripts, outputs into $1. TEMPORARY:
# deleted in Phase G. Per-trial goldens come from dump_golden.py; this is the printed-table half.
#
#     scripts/verify/run_tables.sh runs/golden [--skip-llm]
#
# LLM arms replay from the cache only (--max-requests 0): any miss aborts, which means a prompt
# byte changed. EPIC-Tent's cache was filled through http://localhost:11434/v1, Breakfast's
# through :11435 -- the base URL is part of the cache key, so these must not change.
set -eu
cd "$(dirname "$0")/../.."
OUT=${1:?usage: run_tables.sh OUT_DIR [--skip-llm]}
SKIP_LLM=${2:-}
mkdir -p "$OUT"
D=dataset/processed/epic_tent
B=dataset/processed/breakfast
log() { echo "[$(date +%T)] $*"; }

log "breakfast live/hindsight tables"
./py run_live_eval.py --joint-params $B/joint_params_train_lexhard.npz --lags 0,5,inf \
    --tag golden_lexhard --out-dir "$OUT" > "$OUT/live_lexhard.log" 2>&1
./py run_live_eval.py --joint-params $B/joint_params_train_dish.npz --recipe-labels --lags 0,5,inf \
    --tag golden_dishknown --out-dir "$OUT" > "$OUT/live_dishknown.log" 2>&1
./py run_live_eval.py --joint-params $B/joint_params_train_dish.npz --lags 0,5,inf \
    --tag golden_dish --out-dir "$OUT" > "$OUT/live_dish.log" 2>&1

log "epic-tent tables (no LLM)"
./py run_epic_tent_eval.py --live --live-lags 0,5,inf --skip-llm \
    --out "$OUT/et_live.json" > "$OUT/et_live.log" 2>&1
./py run_epic_tent_eval.py --live --live-lags 0,5,inf --skip-llm \
    --split-file $D/split_prego.json --joint-params $D/joint_params_clean_prego.npz \
    --out "$OUT/et_live_prego.json" > "$OUT/et_live_prego.log" 2>&1
./py run_epic_tent_eval.py --live --live-lags 0,5,inf --skip-llm --trim \
    --split-file $D/split_prego.json --joint-params $D/joint_params_clean_prego.npz \
    --out "$OUT/et_live_prego_trimmed.json" > "$OUT/et_live_prego_trimmed.log" 2>&1

[ "$SKIP_LLM" = --skip-llm ] && { log "done (LLM skipped)"; exit 0; }

log "epic-tent LLM replays"
./py run_epic_tent_eval.py --live --live-lags 0 --variant both --skip-preflight \
    --base-url http://localhost:11434/v1 --max-requests 0 \
    --out "$OUT/et_G_llm.json" > "$OUT/et_G_llm.log" 2>&1
./py run_epic_tent_eval.py --live --live-lags 0 --variant both --skip-preflight --trim \
    --split-file $D/split_prego.json --joint-params $D/joint_params_clean_prego.npz \
    --base-url http://localhost:11434/v1 --max-requests 0 \
    --out "$OUT/et_E_llm_prego_trimmed.json" > "$OUT/et_E_llm_prego_trimmed.log" 2>&1

log "breakfast LLM replays (streams)"
COMMON=(--config configs/breakfast.yaml --unit tick --source real
        --split-file $B/split.json --split-part test --max-real 150 --alpha 0.005
        --model gemma3:27b --base-url http://localhost:11435/v1 --max-requests 0
        --figures-dir "$OUT/figs")
./py run_llm_eval.py "${COMMON[@]}" --protocol incremental --variant both --skip-hsmm \
    --out "$OUT/llm_verdict_report.json" --dump-streams "$OUT/streams_verdict.pkl" \
    > "$OUT/streams_verdict.log" 2>&1
./py run_llm_eval.py "${COMMON[@]}" --protocol predictive --variant both --shot-select random \
    --out "$OUT/llm_pred_random.json" --dump-streams "$OUT/streams_pred_random.pkl" \
    > "$OUT/streams_pred_random.log" 2>&1
./py run_llm_eval.py "${COMMON[@]}" --protocol predictive --variant both --shot-select autocot \
    --shot-cot --skip-hsmm --out "$OUT/llm_pred_cot.json" --dump-streams "$OUT/streams_pred_cot.pkl" \
    > "$OUT/streams_pred_cot.log" 2>&1
log "done"
