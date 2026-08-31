#!/usr/bin/env bash
# Full test-split evaluation at the TICK unit, one LLM prompt variant per invocation.
#
#     VARIANT=with-recipes ./run_tick_test.sh     # the arm that reads labels.json
#     VARIANT=no-recipes   ./run_tick_test.sh     # the like-for-like comparison vs the HSMM
#
# Built to survive losing whatever started it. Launch detached:
#
#     setsid nohup env VARIANT=no-recipes ./run_tick_test.sh > /dev/null 2>&1 < /dev/null &
#
# and it keeps running with no controlling terminal. Re-running it after a crash, a kill, or a
# reboot is always safe and always the right move: every LLM response is cached on
# sha256(base_url + model + messages + temperature), so a restart replays the completed work from
# disk at no cost and resumes at the first request that never landed. Nothing here is destructive
# and no stage overwrites a finished stage's results.
#
# The two arms run as SEPARATE invocations into the same --out, which run_llm_eval._write_report
# merges (guarded on POOL_DEFINING_ARGS, so it refuses to merge arms that were scored on
# different pools). That is not just tidiness: the HSMM arm wants JAX on the GPU and the LLM arm
# wants ollama holding 22 GB of weights on the same card, and running them in one process means
# both are resident at once. Sequencing them keeps each one alone with the card.
set -u

cd "$(dirname "$0")" || exit 1

# Pin imports to THIS checkout's src, the same guard ./py applies to every other runner. The
# venv's editable install resolves to whichever checkout last synced it, so when several git
# worktrees share one venv a bare `import cook_ad` can land on a copy with no tick unit, and the
# run dies on the first elements_from_trajectory(..., unit=) call. Cheap to set, so it is set
# unconditionally rather than only when that is true.
#
# If the editable pointer really is wrong, re-sync from the checkout you want -- but WITH THE
# EXTRAS: a bare `uv sync` uninstalls 46 packages here, including jax-cuda12-plugin, which drops
# JAX to CPU without saying so.
#
#     uv sync --extra dev --extra gpu
#
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

VARIANT="${VARIANT:-with-recipes}"
case "$VARIANT" in
    with-recipes|no-recipes) ;;
    *) echo "VARIANT must be with-recipes or no-recipes, got '$VARIANT'" >&2; exit 2 ;;
esac

PY=.venv/bin/python
BASE_URL=http://localhost:11435/v1
MODEL=gemma3:27b
OUT=dataset/processed/breakfast/llm_tick_test.json
FIGS=dataset/processed/breakfast/figures_tick
LOG=dataset/processed/breakfast/llm_tick_test.log
MAX_LLM_ATTEMPTS=5

COMMON=(--config configs/breakfast.yaml
        --unit tick
        --source real
        --split-file dataset/processed/breakfast/split.json --split-part test
        --max-real 150
        --alpha 0.005
        --model "$MODEL"
        --base-url "$BASE_URL"
        --out "$OUT"
        --figures-dir "$FIGS")

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ------------------------------------------------------------------------------------------
# 0. don't contend with another run for the server's 8 slots
# ------------------------------------------------------------------------------------------
# Counts only real interpreter processes. A plain `pgrep -f run_llm_eval.py` also matches every
# SHELL whose command line happens to mention the script -- a monitoring one-liner, the wrapper
# that launched this -- and the first version of this loop deadlocked against exactly that,
# waiting forever on two greps. The same self-match makes `pkill -f run_tick_test.sh` kill the
# shell that runs it; stop this script by PID.
running_evals() {
    local n=0 pid exe
    for pid in $(pgrep -f 'run_llm_eval\.py' 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null) || continue
        case "$exe" in *python*) n=$((n + 1)) ;; esac
    done
    echo "$n"
}

while [ "$(running_evals)" != "0" ]; do
    log "waiting: another run_llm_eval is still using the server's slots"
    sleep 60
done

log "=== tick-unit test-split evaluation, $VARIANT LLM arm ==="
log "out=$OUT  model=$MODEL  base_url=$BASE_URL  variant=$VARIANT"

# ------------------------------------------------------------------------------------------
# 1. HSMM arm (JAX). Runs first and exits, so the card is free before the LLM arm starts.
# ------------------------------------------------------------------------------------------
if $PY - "$OUT" <<'EOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))["reports"]
except Exception:
    sys.exit(1)
# Any HSMM arm counts, including one whose key carries per-channel alphas
# (real/hsmm-joint@s_temporal=1e-05). An exact-suffix match would miss those and recompute the
# arm on every invocation.
sys.exit(0 if any("/hsmm-" in k and not v.get("incomplete")
                  for k, v in r.items()) else 1)
EOF
then
    log "stage 1 (HSMM): already present in $OUT, skipping"
else
    log "stage 1 (HSMM): starting"
    $PY run_llm_eval.py "${COMMON[@]}" --skip-llm >> "$LOG" 2>&1
    rc=$?
    log "stage 1 (HSMM): exit $rc"
    [ $rc -ne 0 ] && { log "ABORT: HSMM arm failed; LLM arm not started"; exit 1; }
fi

# ------------------------------------------------------------------------------------------
# 2. Check the model is loaded and WHOLLY on the GPU before spending ~10 hours.
#
# ollama fixes layer placement from free VRAM at load time and keeps it. If the model was loaded
# while JAX held the card it sits at ~18/63 layers on GPU and every request runs at CPU speed --
# with no error, no warning, and a ~4x slowdown. Checked here, and repaired by unloading and
# reloading now that stage 1 has exited and the card is free.
# ------------------------------------------------------------------------------------------
gpu_fraction() {
    curl -s "${BASE_URL%/v1}/api/ps" | $PY -c "
import json, sys
m = [x for x in json.load(sys.stdin).get('models', []) if x['name'] == '$MODEL']
print(0.0 if not m else m[0]['size_vram'] / m[0]['size'])
" 2>/dev/null || echo 0.0
}

for attempt in 1 2; do
    frac=$(gpu_fraction)
    log "model on GPU: $(echo "$frac" | cut -c1-5) of its weight"
    if [ "$(echo "$frac > 0.99" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        break
    fi
    [ $attempt -eq 2 ] && { log "ABORT: model still not wholly on GPU; a run now would be CPU-bound"; exit 1; }
    log "reloading the model with the card free"
    curl -s "${BASE_URL%/v1}/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":0}" > /dev/null
    sleep 5
    curl -s "${BASE_URL%/v1}/api/generate" \
         -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"keep_alive\":\"24h\",\"stream\":false}" > /dev/null
done

# ------------------------------------------------------------------------------------------
# 2b. Preflight the LONGEST prompt against the server and read back the token count it actually
#     charged. run_llm_eval prints an estimate; this asks. Silent front-truncation drops the
#     system block -- the response grammar, the vocabulary, the anomaly definitions -- and still
#     returns a plausible reply, so it shows up as a parse-failure cliff on long trials and as
#     nothing else. Measured: the longest test trial (466 ticks, with-recipes) is 7611 tokens
#     against a served 8192, which fits but is not comfortable. Refuse to start if it does not.
# ------------------------------------------------------------------------------------------
log "preflight: measuring the longest request against the server"
$PY - "$BASE_URL" "$MODEL" "$VARIANT" <<'EOF' >> "$LOG" 2>&1
import json, sys, urllib.request
from cook_ad.llm import prompts, textify, detect     # deliberately no JAX import here
base_url, model, variant = sys.argv[1], sys.argv[2], sys.argv[3]
vocab = json.load(open('dataset/processed/breakfast/vocab.json'))
labels = json.load(open('dataset/processed/breakfast/labels.json'))

class Lex:
    def __init__(self, v):
        self.v = {i: s for s, i in v["verbs"].items()}
        self.n = {i: s for s, i in v["nouns"].items()}
    def verb(self, i): return self.v[int(i)]
    def noun(self, i): return self.n[int(i)]

seqs = {t['trial_id']: t for t in json.load(open('dataset/processed/breakfast/sequences.json'))}
test = json.load(open('dataset/processed/breakfast/split.json'))['test_trial_ids']
longest = max((seqs[i] for i in test if i in seqs), key=lambda t: len(t['verb_ids']))
ticks = textify.ticks_from_ids(longest['verb_ids'], longest['noun_ids'], Lex(vocab))
system = prompts.build_variant(variant, vocab, labels, 'incremental', 'tick')
messages = detect.incremental_messages(system, ticks, 'tick')[-1]

body = json.dumps({"model": model, "messages": messages,
                   "temperature": 0.0, "max_tokens": 1}).encode()
req = urllib.request.Request(base_url + "/chat/completions", data=body,
                             headers={"Authorization": "Bearer local",
                                      "Content-Type": "application/json"})
used = json.load(urllib.request.urlopen(req, timeout=900))["usage"]["prompt_tokens"]

ps = json.load(urllib.request.urlopen(base_url.rsplit('/v1', 1)[0] + "/api/ps", timeout=60))
served = next((m["context_length"] for m in ps.get("models", []) if m["name"] == model), 0)
print(f"preflight [{variant}]: longest request {used} tokens over {len(ticks)} ticks; "
      f"served context {served}; headroom {served - used}")
if served - used < 256:
    print("preflight FAILED: raise OLLAMA_CONTEXT_LENGTH; prompts would be truncated from the "
          "front, silently dropping the system prompt")
    sys.exit(1)
EOF
rc=$?
tail -2 "$LOG"
[ $rc -ne 0 ] && { log "ABORT: preflight failed"; exit 1; }

# Re-pin the model. The preflight goes through /v1/chat/completions, which carries no keep_alive
# field, so ollama applies its 5-minute default and silently overrides the 24h set at load time --
# which is exactly how the model came to be unloaded between two runs here. Requests during the
# sweep keep it warm on their own; this covers the gap before the first one lands.
curl -s "${BASE_URL%/v1}/api/generate" \
     -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"keep_alive\":\"24h\",\"stream\":false}" > /dev/null
log "model re-pinned at keep_alive=24h"

# ------------------------------------------------------------------------------------------
# 3. LLM arm, with-recipes. Retried on failure because the cache makes a retry nearly free: a
#    run that died at request 40,000 replays those 40,000 from disk in minutes and continues.
# ------------------------------------------------------------------------------------------
if $PY - "$OUT" "$VARIANT" <<'EOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))["reports"]
except Exception:
    sys.exit(1)
sys.exit(0 if any(k.endswith("/llm-" + sys.argv[2]) and not v.get("incomplete")
                  for k, v in r.items()) else 1)
EOF
then
    log "stage 2 (LLM $VARIANT): already complete in $OUT, nothing to do"
else
    for attempt in $(seq 1 $MAX_LLM_ATTEMPTS); do
        log "stage 2 (LLM $VARIANT): attempt $attempt/$MAX_LLM_ATTEMPTS"
        $PY run_llm_eval.py "${COMMON[@]}" --skip-hsmm --variant "$VARIANT" >> "$LOG" 2>&1
        rc=$?
        log "stage 2 (LLM $VARIANT): exit $rc"
        [ $rc -eq 0 ] && break
        [ $attempt -eq $MAX_LLM_ATTEMPTS ] && { log "ABORT: LLM arm failed $MAX_LLM_ATTEMPTS times"; exit 1; }
        log "retrying in 60s (completed requests replay from cache)"
        sleep 60
    done
fi

log "=== done. report: $OUT  figures: $FIGS/ ==="
