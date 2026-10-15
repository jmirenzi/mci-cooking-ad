#!/usr/bin/env bash
# Fit an HSMM end to end (docs/training.md). Presets:
#
#   PRESET=lexhard     Breakfast, recipes = latent clusters (joint_params_train_lexhard.npz):
#                      fit_lexical -> fit_hard_em -> smooth_params -> refit_durations
#   PRESET=dish        the same 4 stages with recipe = labelled dish (joint_params_train_dish.npz)
#   PRESET=epic_tent   EPIC-Tent: fit_lexical on the error-excised clean split (joint_params_clean.npz)
#   PRESET=epic_tent_prego   the same on PREGO's split (joint_params_clean_prego.npz)
#
#     PRESET=dish OUT=runs/dish scripts/train/train_hsmm.sh [--max-iters N] [--iters N]
#
# Extra flags: --max-iters (soft EM, stage 1) and --iters (hard EM, stage 2) override the
# preset's iteration counts, for quick checks. OUT is the output directory (default: the preset's
# dataset directory); the final model is $OUT/<preset model name>.npz.
set -euo pipefail
cd "$(dirname "$0")/../.."
PRESET=${PRESET:?set PRESET=lexhard|dish|epic_tent}
MAX_ITERS=60
HARD_ITERS=5
while [ $# -gt 0 ]; do
    case "$1" in
        --max-iters) MAX_ITERS=$2; shift 2 ;;
        --iters) HARD_ITERS=$2; shift 2 ;;
        *) echo "unknown flag $1" >&2; exit 2 ;;
    esac
done
T=scripts/train

case "$PRESET" in
    lexhard|dish)
        OUT=${OUT:-dataset/processed/breakfast}
        mkdir -p "$OUT"
        if [ "$PRESET" = dish ]; then RL=--recipe-labels; NAME=joint_params_train_dish;
        else RL=; NAME=joint_params_train_lexhard; fi
        ./py $T/fit_lexical.py $RL --init-prior-scale 0.0 --anchor 50 --max-iters "$MAX_ITERS" \
            --out "$OUT/${NAME}_soft.npz"
        ./py $T/fit_hard_em.py $RL --init-from "$OUT/${NAME}_soft.npz" --keep-init-emissions \
            --iters "$HARD_ITERS" --out "$OUT/${NAME}_hard.npz"
        ./py $T/smooth_params.py --in "$OUT/${NAME}_hard.npz" --out "$OUT/${NAME}_smooth.npz" \
            --strength 0.7 --backoff-tau 30
        ./py $T/refit_durations.py $RL --in "$OUT/${NAME}_smooth.npz" --out "$OUT/$NAME.npz" \
            --kappa 0.001
        ;;
    epic_tent|epic_tent_prego)
        D=dataset/processed/epic_tent
        OUT=${OUT:-$D}
        mkdir -p "$OUT"
        if [ "$PRESET" = epic_tent ]; then SPLIT=split_clean; NAME=joint_params_clean;
        else SPLIT=split_clean_prego; NAME=joint_params_clean_prego; fi
        ./py $T/fit_lexical.py --config configs/epic_tent.yaml \
            --sequences "$D/sequences_clean.json" --labels "$D/labels_clean.json" \
            --split-file "$D/$SPLIT.json" --split-part train --max-iters "$MAX_ITERS" \
            --out "$OUT/$NAME.npz"
        ;;
    *) echo "PRESET must be lexhard, dish, epic_tent or epic_tent_prego" >&2; exit 2 ;;
esac
echo "DONE ($PRESET) -> $OUT"
