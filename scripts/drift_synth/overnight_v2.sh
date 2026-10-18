#!/usr/bin/env bash
# The 2026-10-18 generator (8 recipes, no healthy repeats, two-sided flavour knobs, narrower
# durations, unscaled logistics), end to end: every drift condition + sanity plots, then in
# parallel the synthetic AD eval (noise on, then off) and the CC4D-calibrated global-prior
# experiment (overnight_cal.sh). Logs in runs/drift_synth/overnight_v2/.
#
#   setsid nohup scripts/drift_synth/overnight_v2.sh > runs/drift_synth/overnight_v2/run.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/../.."
L=runs/drift_synth/overnight_v2
mkdir -p "$L" runs/drift_synth/overnight_cal
stamp() { echo "[$(date +%F' '%T)] $*"; }

stamp "generate";  ./py scripts/drift_synth/generate.py > "$L/generate.log" 2>&1
stamp "sanity";    ./py scripts/drift_synth/sanity_plots.py > "$L/sanity.log" 2>&1
stamp "ad eval + prior experiment"
( set -e
  for n in on off; do
    for st in data train eval; do
      echo "[$(date +%T)] noise $n: $st"
      ./py scripts/eval/eval_synth_ad.py --stage "$st" --noise "$n"
    done
  done ) > "$L/ad.log" 2>&1 & ad=$!
scripts/drift_synth/overnight_cal.sh > runs/drift_synth/overnight_cal/run.log 2>&1 & pr=$!
wait "$ad"; stamp "ad eval done"
wait "$pr"; stamp "prior experiment done"
