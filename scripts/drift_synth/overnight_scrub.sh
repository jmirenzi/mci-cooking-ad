#!/usr/bin/env bash
# Idle scrubbed before the model (data/scrub.py, docs/drift_synth.md §9.8): the calibrated prior
# experiment rebuilt on scrubbed streams -- population prior, 4 persons x 3 training sizes refit,
# then the step-level pools scored by the plain HSMM and with the count channel, in parallel.
# Logs in runs/drift_synth/overnight_scrub/.
#
#   setsid nohup scripts/drift_synth/overnight_scrub.sh > runs/drift_synth/overnight_scrub/run.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/../.."
L=runs/drift_synth/overnight_scrub
mkdir -p "$L"
E="./py scripts/eval/eval_synth_prior.py --variant cal_scrub"
stamp() { echo "[$(date +%F' '%T)] $*"; }

stamp "population";  $E --stage population > "$L/population.log" 2>&1
stamp "persons";     $E --stage persons > "$L/persons.log" 2>&1
stamp "train";       $E --stage train --person 0 --person 1 > "$L/train_a.log" 2>&1 & a=$!
                     $E --stage train --person 2 --person 3 > "$L/train_b.log" 2>&1 & b=$!
wait "$a"; wait "$b"
stamp "eval (plain, + count)"
$E --pool gen --stage eval-split --person 0 --person 1 > "$L/plain_a.log" 2>&1 & a=$!
$E --pool gen --stage eval-split --person 2 --person 3 > "$L/plain_b.log" 2>&1 & b=$!
$E --pool gen --count-channel --stage eval-split --person 0 --person 1 > "$L/count_a.log" 2>&1 & c=$!
$E --pool gen --count-channel --stage eval-split --person 2 --person 3 > "$L/count_b.log" 2>&1 & e=$!
wait "$a"; wait "$b"; wait "$c"; wait "$e"
$E --pool gen --stage summary-split > "$L/summary.md" 2>&1
$E --pool gen --count-channel --stage summary-split >> "$L/summary.md" 2>&1
stamp "done"
