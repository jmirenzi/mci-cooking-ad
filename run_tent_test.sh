#!/usr/bin/env bash
# EPIC-Tent end to end: parse -> split -> fit the clean checkpoint -> score every arm
# under each error group (procedural, procedural+temporal).
#
# The three things run_epic_tent_eval.py cannot do for itself, all about the ollama server:
#   * make sure the model is GPU-resident. ollama fixes layer placement at LOAD time, so a model
#     loaded while JAX held the card sits mostly on CPU with no error and no warning. The Python
#     runner detects that and warns; only this script can unload and reload to fix it.
#   * re-pin keep_alive. /v1/chat/completions carries no keep_alive field, so ollama applies its
#     5-minute default and silently overrides whatever was set at load time.
#   * wait for another run to finish rather than contending for the card.
#
# Everything else -- the context preflight, the controls, the report -- is in the runner.
#
#   ./run_tent_test.sh                     # full pipeline, both prompt variants
#   STAGE=score ./run_tent_test.sh         # skip parse and fit
#   MODEL=qwen3:32b ./run_tent_test.sh
set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
D=dataset/processed/epic_tent
MODEL=${MODEL:-gemma3:27b}
BASE_URL=${BASE_URL:-http://localhost:11434/v1}
HOST=${BASE_URL%/v1}
STAGE=${STAGE:-all}
VARIANT=${VARIANT:-both}
N_SHOT=${N_SHOT:-0}      # 0 deliberately -- see run_epic_tent_eval.py's docstring
LOG=$D/epic_tent_test.log

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$D"

# ---- 0. do not contend for the GPU ---------------------------------------------------------
# pgrep -f would match this script's own command line, so match the interpreter's exe instead.
while :; do
    busy=0
    for pid in $(pgrep -f 'run_epic_tent_eval\.py|run_llm_eval\.py' 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        readlink -f "/proc/$pid/exe" 2>/dev/null | grep -q python && busy=1
    done
    [ "$busy" -eq 0 ] && break
    log "another eval is running; waiting 60s"
    sleep 60
done

# ---- 1. data -------------------------------------------------------------------------------
if [ "$STAGE" = all ]; then
    if [ ! -f dataset/epic_tent/Synchronised_action_label.txt ]; then
        log "fetching annotations (~16 MB, plus uncertainty for the PREGO split)"
        $PY tools_fetch_epic_tent.py --out dataset/epic_tent --uncertainty >>"$LOG" 2>&1 \
            || { log "ABORT: fetch failed"; exit 1; }
    fi
    log "parsing"
    $PY -m cook_ad.data.parse_epic_tent --config configs/epic_tent.yaml >>"$LOG" 2>&1 \
        || { log "ABORT: parse failed"; exit 1; }
    $PY split_dataset.py --sequences $D/sequences.json --labels $D/labels.json \
        --out $D/split.json --test-frac 0.3 --seed 0 >>"$LOG" 2>&1
    $PY tools_epic_tent_split.py >>"$LOG" 2>&1
    tail -4 "$LOG"
fi

# ---- 2. the clean checkpoint ---------------------------------------------------------------
# The HSMM is fitted on the ERROR-EXCISED fragments only. Every training session contains
# annotated errors, and a model fitted on whole sessions learns them as normal (measured: it
# alarms less on both counts). split_clean.json inherits session membership from split.json.
if [ "$STAGE" = all ] && [ ! -f "$D/joint_params_clean.npz" ]; then
    log "fitting joint_params_clean"
    $PY run_joint_lexical.py --config configs/epic_tent.yaml \
        --sequences "$D/sequences_clean.json" --labels "$D/labels_clean.json" \
        --split-file "$D/split_clean.json" --split-part train \
        --out "$D/joint_params_clean.npz" >>"$LOG" 2>&1 || { log "ABORT: fit failed"; exit 1; }
    grep -E "joint EM:|subtask ARI" "$LOG" | tail -2
fi

# ---- 3. the HSMM arm, scored per error group --------------------------------------------------
log "HSMM arm (clean fit), procedural and procedural+temporal groups"
$PY run_epic_tent_eval.py --skip-llm --out "$D/report_hsmm.json" 2>&1 | grep -vE '%\|' \
    | tee -a "$LOG" | grep -E "^GROUP|hsmm-joint  "

# ---- 4. GPU placement, then arm 3 ----------------------------------------------------------
placement() {
    curl -s -m 20 "$HOST/api/ps" | $PY -c "
import json,sys
for m in json.load(sys.stdin).get('models',[]):
    if m['name']=='$MODEL' and m.get('size'):
        print(m['size_vram']/m['size']); break
else:
    print(0.0)"
}
for attempt in 1 2; do
    curl -s -m 600 "$HOST/api/generate" \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"24h\"}" \
        -o /dev/null
    frac=$(placement)
    log "GPU placement attempt $attempt: $frac of $MODEL in VRAM"
    awk -v f="$frac" 'BEGIN{exit !(f>0.99)}' && break
    log "reloading: layer placement is fixed at load time and cannot be corrected in place"
    curl -s -m 60 "$HOST/api/generate" \
        -d "{\"model\":\"$MODEL\",\"keep_alive\":0}" -o /dev/null
    sleep 5
    [ "$attempt" = 2 ] && { log "ABORT: $MODEL will not load onto the GPU"; exit 1; }
done

log "all arms ($VARIANT, ${N_SHOT}-shot), both error groups"
$PY run_epic_tent_eval.py --variant "$VARIANT" \
    --n-shot "$N_SHOT" --model "$MODEL" --base-url "$BASE_URL" \
    --out "$D/epic_tent_report.json" 2>&1 | grep -vE '%\|' | tee -a "$LOG" \
    | grep -E "^GROUP|^  (hsmm|llm|control)"

# Re-pin: the /v1 requests above reset keep_alive to ollama's 5-minute default.
curl -s -m 60 "$HOST/api/generate" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"24h\"}" -o /dev/null
log "done -- reports in $D, full log in $LOG"
