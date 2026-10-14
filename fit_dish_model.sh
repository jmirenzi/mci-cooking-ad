#!/usr/bin/env bash
# The shipped Breakfast detector's fitting pipeline (joint_params_train_lexhard.npz's meta.json),
# with recipe = labelled dish instead of a latent cluster: K_R = 10, every training trial's
# recipe pinned to its label in every stage. Hyperparameters are the lexhard ones, unchanged --
# they were selected for the clustered model and have NOT been re-selected for this one.
#
#     setsid nohup ./fit_dish_model.sh > runs/live_logs/fit_dish.log 2>&1 < /dev/null &
set -euo pipefail
root=dataset/processed/breakfast
./py run_joint_lexical.py --recipe-labels --init-prior-scale 0.0 --anchor 50 --max-iters 60 \
    --out "$root/joint_dish_soft.npz"
./py run_hard_em.py --recipe-labels --init-from "$root/joint_dish_soft.npz" --keep-init-emissions \
    --iters 5 --out "$root/joint_dish_hard.npz"
./py smooth_params.py --in "$root/joint_dish_hard.npz" --out "$root/joint_dish_smooth.npz" \
    --strength 0.7 --backoff-tau 30
./py refit_durations.py --recipe-labels --in "$root/joint_dish_smooth.npz" \
    --out "$root/joint_params_train_dish.npz" --kappa 0.001
echo "DONE $root/joint_params_train_dish.npz"
