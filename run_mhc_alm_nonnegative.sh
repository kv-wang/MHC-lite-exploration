#!/usr/bin/env bash
#
# Train 4-stream mHC with augmented-Lagrangian H_res constraints:
#   row sums = 1, column sums = 1, and entries >= 0.
# Runs small, medium, then large model configurations.
#
# Usage:
#   ./run_mhc_alm_nonnegative.sh
#   N_GPUS=1 MAX_ITERS=100 ./run_mhc_alm_nonnegative.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_alm_nonnegative.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-ablation_num_streams}"

echo ""
echo "================================================================"
echo " Running 4-stream mHC ALM-nonnegative training for small/medium/large"
echo " train_config:  $TRAIN_CONFIG"
echo " method_config: $METHOD_CONFIG"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " n_gpus:        $N_GPUS"
echo " wandb_prefix:  $WANDB_PROJECT_PREFIX"
echo "================================================================"

run_model() {
  local model_name="$1"
  local model_config="$2"
  local wandb_run_name="mhc-${model_name}-mhc-alm-nonnegative-4streams-${MAX_ITERS}iter"
  local wandb_project="${WANDB_PROJECT_PREFIX}_${model_name}"
  local out_prefix_method="mhc-alm-nonnegative-4streams-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running ${model_name} mHC ALM-nonnegative"
  echo " model_config:      $model_config"
  echo " wandb_project:     $wandb_project"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$model_config"
    "$METHOD_CONFIG"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --wandb_log=True
    --wandb_project="$wandb_project"
    --wandb_run_name="$wandb_run_name"
    --out_prefix_method="$out_prefix_method"
    --wandb_log_layer_stats=False
    --wandb_log_layer_cosine=False
  )

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}"
  else
    python train.py "${common_args[@]}"
  fi
}

#run_model "small" "config/small_model.py"
#run_model "medium" "config/medium_model.py"
run_model "large" "config/large_model.py"

echo ""
echo "================================================================"
echo " 4-stream mHC ALM-nonnegative small/medium/large training completed"
echo "================================================================"
