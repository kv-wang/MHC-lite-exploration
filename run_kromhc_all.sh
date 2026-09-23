#!/usr/bin/env bash
# Run KromHC training sequentially for the small, medium, and large models.
# KromHC follows the official implementation: n=4, repeat expansion, and sum
# reduction. The residual matrix is an exact Kronecker product of 2x2 doubly
# stochastic factors.
#
# Usage:
#   ./run_kromhc_all.sh
#   CUDA_DEVICES=0,1 N_GPUS=2 MAX_ITERS=100 ./run_kromhc_all.sh
#   N_GPUS=0 MAX_ITERS=10 ./run_kromhc_all.sh  # CPU/debug invocation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
N_GPUS="${N_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_fineweb_edu.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_kromhc.py}"
N_STREAMS="${N_STREAMS:-4}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-sum}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
COMPILE="${COMPILE:-False}"
WANDB_LOG="${WANDB_LOG:-True}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-True}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-kromhc_fineweb_edu}"

EXTRA_ARGS=("$@")

run_one() {
  local model_name="$1"
  local model_config="config/${model_name}_model.py"
  local run_name="kromhc-${model_name}-${N_STREAMS}streams-${MAX_ITERS}iter"
  local out_prefix="kromhc-${model_name}-${N_STREAMS}streams-${MAX_ITERS}iter"
  local wandb_project="${WANDB_PROJECT_PREFIX}_${model_name}"

  echo
  echo "================================================================"
  echo " Running KromHC: ${model_name}"
  echo " model_config:         $model_config"
  echo " n_streams:            $N_STREAMS"
  echo " reduce_mode:          $REDUCE_STREAM_MODE"
  echo " cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
  echo " n_gpus:               $N_GPUS"
  echo " max_iters:            $MAX_ITERS"
  echo " eval_iters:           $EVAL_ITERS"
  echo " checkpoint_interval:  $CHECKPOINT_INTERVAL"
  echo " wandb_project:        $wandb_project"
  echo "================================================================"

  local train_args=(
    "$TRAIN_CONFIG"
    "$model_config"
    "$METHOD_CONFIG"
    --hyper_conn_n="$N_STREAMS"
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
    --hyper_conn_expand_stream_mode=repeat
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --checkpoint_interval="$CHECKPOINT_INTERVAL"
    --compile="$COMPILE"
    --wandb_log="$WANDB_LOG"
    --wandb_project="$wandb_project"
    --wandb_run_name="$run_name"
    --out_prefix_method="$out_prefix"
    --save_checkpoints="$SAVE_CHECKPOINTS"
    --wandb_log_layer_stats=False
    --wandb_log_layer_cosine=False
    --wandb_log_layer_grad_norm=False
    --wandb_log_h_matrix_grad_norm=False
    --wandb_log_layer_activation_norm=False
    --wandb_log_layer_activation_grad_norm=False
    --mhc_log_constraint_errors=False
    "${EXTRA_ARGS[@]}"
  )

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${train_args[@]}"
  else
    python train.py "${train_args[@]}"
  fi
}

for model_name in small medium large; do
  run_one "$model_name"
done

echo
echo "================================================================"
echo " KromHC small/medium/large training completed"
echo "================================================================"
