#!/usr/bin/env bash
# The session-time generator (docs/drift_synth.md §9.9): sessions only, recognizer noise off, no
# gaps, CC4D step totals. Rebuilds the prior experiment from scratch in drift_synth_prior_sess/:
# population prior, 4 persons x 3 training sizes (stage 4b included), then the step-level pools
# scored by Hindsight under metric G and metric H (delay 2), and by Live under G.
# Logs in runs/drift_synth/overnight_sess/.
#
#   setsid nohup scripts/drift_synth/overnight_sess.sh > runs/drift_synth/overnight_sess/run.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/../.."
L=runs/drift_synth/overnight_sess
mkdir -p "$L"
E="./py scripts/eval/eval_synth_prior.py --variant sess"
stamp() { echo "[$(date +%F' '%T)] $*"; }

stamp "population";  $E --stage population > "$L/population.log" 2>&1
stamp "persons";     $E --stage persons > "$L/persons.log" 2>&1
stamp "train"
pids=()
for p in 0 1 2 3; do $E --stage train --person $p > "$L/train_p$p.log" 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
stamp "eval"
pids=()
for p in "0 1" "2 3"; do
  args=$(printf -- "--person %s " $p)
  $E --pool gen --np-stage --stage eval-split $args > "$L/eval_G_${p// /}.log" 2>&1 & pids+=($!)
  COOK_METRIC=H COOK_DELAY=2 $E --pool gen --np-stage --stage eval-split $args > "$L/eval_H2_${p// /}.log" 2>&1 & pids+=($!)
  $E --pool gen --np-stage --stage eval-live $args > "$L/live_G_${p// /}.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
$E --pool gen --np-stage --stage summary-split > "$L/summary_G.md" 2>&1
COOK_METRIC=H COOK_DELAY=2 $E --pool gen --np-stage --stage summary-split > "$L/summary_H2.md" 2>&1
stamp "done"
