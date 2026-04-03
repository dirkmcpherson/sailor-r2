#!/bin/bash
# Full 3-phase SAILOR-R2 training: gymhil/pickcube, left-hand demos, image-only
#
# Phase 1: Diffusion policy pretraining (behavior cloning on 20 left-hand demos)
# Phase 2: World model warmstart (R2-Dreamer RSSM + Barlow Twins, no decoder)
# Phase 3: Iterative MPPI residual refinement
#
# MPPI hyperparams scaled 10x smaller for gymhil's small action magnitudes.

set -euo pipefail
cd "$(dirname "$0")/.."

SAILOR_ROOT="${SAILOR_ROOT:-$(realpath ../SAILOR)}"
export SAILOR_ROOT
SCRATCH="scratch_dir/left"
DATADIR="${SAILOR_ROOT}/datasets/gymhil_datasets"

echo "=== SAILOR-R2: gymhil/pickcube (left, all 3 phases) ==="
echo "SAILOR_ROOT=${SAILOR_ROOT}"
echo "Demos: ${DATADIR}/pickcube/trajectory_hil_left.h5"
echo "Scratch: ${SCRATCH}"
echo ""

PYTHONUNBUFFERED=1 python train.py \
    --configs cfg_dp_mppi gymhil gymhil_image \
    --task gymhil__pickcube \
    --seed 0 \
    --use_wandb False \
    --visualize_eval True \
    --eval_num_runs 10 \
    --num_envs 5 \
    --num_exp_trajs 20 \
    --num_exp_val_trajs 5 \
    --h5_filename trajectory_hil_left.h5 \
    --datadir "${DATADIR}" \
    --scratch_dir "${SCRATCH}" \
    --set dp.train_steps 8000 \
    --set dp.eval_freq 2000 \
    "$@" \
    2>&1 | tee experiments/left_gymhil_training.log
