#!/usr/bin/env bash
# Breakfast evaluation with live LLM requests (scripts/eval/eval_breakfast.py), wrapped in the
# three things the Python runner cannot do for itself, all about the ollama server:
#   * make sure the model is wholly GPU-resident. ollama fixes layer placement at LOAD time; a
#     model loaded while JAX held the card runs ~4x slower with no error. Checked, and repaired
#     by unloading and reloading.
#   * preflight the longest request against the served context: ollama truncates an over-long
#     prompt from the FRONT, silently dropping the system prompt.
#   * re-pin keep_alive (/v1 requests reset it to ollama's 5-minute default).
# Retried on failure: every response is cached on sha256(base_url + model + messages +
# temperature), so a restart replays the finished work from disk and continues.
#
#     setsid nohup scripts/eval/llm_breakfast.sh > /dev/null 2>&1 < /dev/null &
#
# Stop it by PID: `pkill -f llm_breakfast.sh` also matches the shell that runs it.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=.venv/bin/python
BASE_URL=${BASE_URL:-http://localhost:11435/v1}   # the URL the Breakfast cache was filled through
MODEL=${MODEL:-gemma3:27b}
OUT=${OUT:-runs/eval_breakfast.json}
LOG=${LOG:-runs/eval_breakfast.log}
MAX_ATTEMPTS=5
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 0. do not contend with another eval for the server's slots (match the interpreter, not shells)
busy() {
    local pid
    for pid in $(pgrep -f 'eval_breakfast\.py|eval_epic_tent\.py' 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        readlink -f "/proc/$pid/exe" 2>/dev/null | grep -q python && return 0
    done
    return 1
}
while busy; do log "waiting: another eval is running"; sleep 60; done

# 1. the model wholly on the GPU
gpu_fraction() {
    curl -s "${BASE_URL%/v1}/api/ps" | $PY -c "
import json, sys
m = [x for x in json.load(sys.stdin).get('models', []) if x['name'] == '$MODEL']
print(0.0 if not m else m[0]['size_vram'] / m[0]['size'])" 2>/dev/null || echo 0.0
}
for attempt in 1 2; do
    frac=$(gpu_fraction)
    log "model on GPU: $(echo "$frac" | cut -c1-5) of its weight"
    awk -v f="$frac" 'BEGIN{exit !(f>0.99)}' && break
    [ $attempt -eq 2 ] && { log "ABORT: model still not wholly on GPU"; exit 1; }
    log "reloading the model with the card free"
    curl -s "${BASE_URL%/v1}/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":0}" > /dev/null
    sleep 5
    curl -s "${BASE_URL%/v1}/api/generate" \
         -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"keep_alive\":\"24h\",\"stream\":false}" > /dev/null
done

# 2. preflight: the longest tick-unit verdict request (the longest any arm sends)
$PY - "$BASE_URL" "$MODEL" <<'PYEOF' >> "$LOG" 2>&1 || { log "ABORT: preflight failed"; exit 1; }
import json, sys, urllib.request
from cook_ad.data import split as split_mod
from cook_ad.llm import detect, prompts, textify
base_url, model = sys.argv[1:3]
B = "dataset/processed/breakfast"
vocab, labels = json.load(open(f"{B}/vocab.json")), json.load(open(f"{B}/labels.json"))
class Lex:
    def __init__(self, v):
        self.v = {i: s for s, i in v["verbs"].items()}; self.n = {i: s for s, i in v["nouns"].items()}
    def verb(self, i): return self.v[int(i)]
    def noun(self, i): return self.n[int(i)]
test = split_mod.filter_sequences(json.load(open(f"{B}/sequences.json")),
                                  split_mod.load_split(f"{B}/split.json"), "test")[:150]
longest = max(test, key=lambda t: len(t["verb_ids"]))
ticks = textify.ticks_from_ids(longest["verb_ids"], longest["noun_ids"], Lex(vocab))
system = prompts.build_variant("with-recipes", vocab, labels, "incremental", "tick")
messages = detect.incremental_messages(system, ticks, "tick")[-1]
body = json.dumps({"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 1}).encode()
req = urllib.request.Request(base_url + "/chat/completions", data=body,
                             headers={"Authorization": "Bearer local", "Content-Type": "application/json"})
used = json.load(urllib.request.urlopen(req, timeout=900))["usage"]["prompt_tokens"]
ps = json.load(urllib.request.urlopen(base_url.rsplit("/v1", 1)[0] + "/api/ps", timeout=60))
served = next((m["context_length"] for m in ps.get("models", []) if m["name"] == model), 0)
print(f"preflight: longest request {used} tokens over {len(ticks)} ticks; served context {served}")
sys.exit(0 if served - used >= 256 else 1)
PYEOF
curl -s "${BASE_URL%/v1}/api/generate" \
     -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"keep_alive\":\"24h\",\"stream\":false}" > /dev/null

# 3. the evaluation, retried
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    log "eval_breakfast: attempt $attempt/$MAX_ATTEMPTS"
    $PY scripts/eval/eval_breakfast.py --base-url "$BASE_URL" --model "$MODEL" --out "$OUT" \
        --operating-points configs/operating_points.yaml "$@" >> "$LOG" 2>&1 && { log "done: $OUT"; exit 0; }
    log "failed; retrying in 60s (completed requests replay from cache)"
    sleep 60
done
log "ABORT: eval failed $MAX_ATTEMPTS times"; exit 1
