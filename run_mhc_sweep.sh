#!/usr/bin/env bash
#
# Train ordinary mHC (Sinkhorn) for n=8 then n=16.
# Each stream count runs small, medium, then large.
#
# wandb projects:
#   final_experiments_small / final_experiments_medium / final_experiments_large
#
# Usage:
#   ./run_mhc_sweep.sh
#   N_GPUS=1 MAX_ITERS=100 ./run_mhc_sweep.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-1}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-final_experiments}"

echo ""
echo "================================================================"
echo " Running ordinary mHC sweep: n=8 then n=16, small/medium/large"
echo " train_config:  $TRAIN_CONFIG"
echo " method_config: $METHOD_CONFIG"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " n_gpus:        $N_GPUS"
echo " wandb_prefix:  $WANDB_PROJECT_PREFIX"
echo "================================================================"

run_one() {
  local n_streams="$1"
  local model_name="$2"
  local model_config="$3"
  local wandb_project="${WANDB_PROJECT_PREFIX}_${model_name}"
  local wandb_run_name="mhc-${model_name}-mhc-sinkhorn-${n_streams}streams-${MAX_ITERS}iter"
  local out_prefix_method="mhc-sinkhorn-${n_streams}streams-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running ${model_name} ordinary mHC with ${n_streams} streams"
  echo " train_config:      $TRAIN_CONFIG"
  echo " model_config:      $model_config"
  echo " method_config:     $METHOD_CONFIG"
  echo " n_streams:         $n_streams"
  echo " max_iters:         $MAX_ITERS"
  echo " eval_iters:        $EVAL_ITERS"
  echo " n_gpus:            $N_GPUS"
  echo " wandb_project:     $wandb_project"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$model_config"
    "$METHOD_CONFIG"
    --hyper_conn_n="$n_streams"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --wandb_log=True
    --wandb_project="$wandb_project"
    --wandb_run_name="$wandb_run_name"
    --out_prefix_method="$out_prefix_method"
    --wandb_log_layer_stats=False
    --wandb_log_layer_cosine=False
    --wandb_log_layer_grad_norm=False
    --wandb_log_layer_activation_norm=False
    --wandb_log_layer_activation_grad_norm=False
    --mhc_log_constraint_errors=False
  )

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}"
  else
    python train.py "${common_args[@]}"
  fi
}

for n_streams in 8 16; do
  run_one "$n_streams" "small" "config/small_model.py"
  #run_one "$n_streams" "medium" "config/medium_model.py"
  #run_one "$n_streams" "large" "config/large_model.py"
done

echo ""
echo "================================================================"
echo " ordinary mHC sweep completed (n=8, n=16 x small/medium/large)"
echo "================================================================"
