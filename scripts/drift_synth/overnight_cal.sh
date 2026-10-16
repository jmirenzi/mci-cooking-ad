#!/usr/bin/env bash
# The CC4D-calibrated rerun of the global-prior experiment (docs/drift_synth.md §9.2-9.3):
# population -> 4 test persons -> 12 person fits (2 workers) -> split-alpha eval (2 workers) ->
# summary + plot. ~8 h on one GPU. Logs in runs/drift_synth/overnight_cal/.
#
#   nohup scripts/drift_synth/overnight_cal.sh > runs/drift_synth/overnight_cal/run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
L=runs/drift_synth/overnight_cal
mkdir -p "$L"
E="./py scripts/eval/eval_synth_prior.py --variant cal"
stamp() { echo "[$(date +%F' '%T)] $*"; }

stamp "population";  $E --stage population > "$L/population.log" 2>&1
stamp "persons";     $E --stage persons > "$L/persons.log" 2>&1
stamp "train";       $E --stage train --person 0 --person 1 > "$L/train_a.log" 2>&1 & a=$!
                     $E --stage train --person 2 --person 3 > "$L/train_b.log" 2>&1 & b=$!
wait "$a"; wait "$b"   # each: a failed worker stops the run here
stamp "eval-split";  $E --stage eval-split --person 0 --person 1 > "$L/eval_a.log" 2>&1 & a=$!
                     $E --stage eval-split --person 2 --person 3 > "$L/eval_b.log" 2>&1 & b=$!
wait "$a"; wait "$b"
$E --stage summary-split > "$L/summary.md" 2>&1
$E --stage plot-split >> "$L/summary.md" 2>&1
stamp "done"
