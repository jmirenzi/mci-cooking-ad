#!/usr/bin/env bash
# Golden-comparison TABLES from the new eval scripts, into $1. TEMPORARY: deleted in Phase G.
# Compare against runs/golden with scripts/verify/compare_tables.py.
#
#     scripts/verify/run_tables.sh runs/golden_E [--skip-llm]
#
# LLM arms replay from the cache only (--max-requests 0): any miss aborts, which means a prompt
# byte changed. EPIC-Tent's cache was filled through http://localhost:11434/v1, Breakfast's
# through :11435 -- the base URL is part of the cache key.
set -eu
cd "$(dirname "$0")/../.."
OUT=${1:?usage: run_tables.sh OUT_DIR [--skip-llm]}
SKIP_LLM=${2:-}
mkdir -p "$OUT"
D=dataset/processed/epic_tent
E=scripts/eval/eval_epic_tent.py
log() { echo "[$(date +%T)] $*"; }

log "epic-tent tables (no LLM)"
./py $E --live --live-lags 0,5,inf --skip-llm --out "$OUT/et_live.json" > "$OUT/et_live.log" 2>&1
./py $E --live --live-lags 0,5,inf --skip-llm --split-file $D/split_prego.json \
    --joint-params $D/joint_params_clean_prego.npz --out "$OUT/et_live_prego.json" \
    > "$OUT/et_live_prego.log" 2>&1
./py $E --live --live-lags 0,5,inf --skip-llm --trim --split-file $D/split_prego.json \
    --joint-params $D/joint_params_clean_prego.npz --out "$OUT/et_live_prego_trimmed.json" \
    > "$OUT/et_live_prego_trimmed.log" 2>&1

if [ "$SKIP_LLM" != --skip-llm ]; then
    log "epic-tent LLM replays"
    ./py $E --live --live-lags 0 --variant both --skip-preflight \
        --base-url http://localhost:11434/v1 --max-requests 0 \
        --out "$OUT/et_G_llm.json" > "$OUT/et_G_llm.log" 2>&1
    ./py $E --live --live-lags 0 --variant both --skip-preflight --trim \
        --split-file $D/split_prego.json --joint-params $D/joint_params_clean_prego.npz \
        --base-url http://localhost:11434/v1 --max-requests 0 \
        --out "$OUT/et_E_llm_prego_trimmed.json" > "$OUT/et_E_llm_prego_trimmed.log" 2>&1
    DET=hsmm-hindsight,hsmm-live,hsmm-lag,llm-predictive,llm-verdict,nulls
else
    DET=hsmm-hindsight,hsmm-live,hsmm-lag,nulls
fi

log "breakfast (train picks, test grids, LLM-pool table)"
./py scripts/eval/eval_breakfast.py --detectors $DET --max-requests 0 \
    --out "$OUT/eval_breakfast.json" --operating-points "$OUT/operating_points.yaml" \
    > "$OUT/eval_breakfast.log" 2>&1
log "done"
