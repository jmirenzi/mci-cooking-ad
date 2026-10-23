#!/usr/bin/env bash
# The synthetic anomaly-detection experiment end to end (docs/drift_synth.md §7): population
# prior, 4 test persons x 3 training sizes (training stages 1-4b), then the step-level pools scored
# by Hindsight and Live under metric H (delay 2), and by Hindsight under metric G. About 4 h on one GPU.
# Logs in runs/drift_synth/prior_experiment/, results in runs/drift_synth/prior/.
#
#   setsid nohup scripts/drift_synth/prior_experiment.sh > runs/drift_synth/prior_experiment/run.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/../.."
L=runs/drift_synth/prior_experiment
mkdir -p "$L"
E="./py scripts/eval/eval_synth_prior.py"
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
  $E --stage eval $args > "$L/eval_H_${p// /}.log" 2>&1 & pids+=($!)
  COOK_METRIC=G $E --stage eval $args > "$L/eval_G_${p// /}.log" 2>&1 & pids+=($!)
  $E --stage eval-live $args > "$L/live_H_${p// /}.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
$E --stage summary > "$L/summary_H.md" 2>&1
COOK_METRIC=G $E --stage summary > "$L/summary_G.md" 2>&1
stamp "done"
