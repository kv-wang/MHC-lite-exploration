#!/usr/bin/env bash
# Run FineWeb-edu small-model identity, identity+tanh-offdiag, and Sinkhorn mHC.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
N_GPUS="${N_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_fineweb_edu.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
N_STREAMS="${N_STREAMS:-4}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-fineweb_small_baselines}"
WANDB_LOG="${WANDB_LOG:-True}"
COMPILE="${COMPILE:-True}"

EXTRA_ARGS=("$@")

run_one() {
  local method="$1"
  local method_config="$2"
  shift 2
  local method_args=("$@")
  local run_name="fineweb-small-${method}-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"
  local out_prefix="fineweb-small-${method}-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"

  echo
  echo "================================================================"
  echo " Running FineWeb small: ${method}"
  echo " cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
  echo " n_gpus:              $N_GPUS"
  echo " n_streams:           $N_STREAMS"
  echo " reduce_mode:         $REDUCE_STREAM_MODE"
  echo " max_iters:           $MAX_ITERS"
  echo " method_config:       $method_config"
  echo "================================================================"

  local train_args=(
    "$TRAIN_CONFIG"
    "$MODEL_CONFIG"
    "$method_config"
    --hyper_conn_n="$N_STREAMS"
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --checkpoint_interval="$CHECKPOINT_INTERVAL"
    --compile="$COMPILE"
    --wandb_log="$WANDB_LOG"
    --wandb_project="$WANDB_PROJECT"
    --wandb_run_name="$run_name"
    --out_prefix_method="$out_prefix"
    --wandb_log_layer_stats=False
    --wandb_log_layer_cosine=False
    --wandb_log_layer_grad_norm=False
    --wandb_log_layer_activation_norm=False
    --wandb_log_layer_activation_grad_norm=False
    --mhc_log_constraint_errors=False
    "${method_args[@]}"
    "${EXTRA_ARGS[@]}"
  )

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${train_args[@]}"
  else
    python train.py "${train_args[@]}"
  fi
}

run_one identity config/with_mhc.py --mhc_identity_h_res=True
run_one identity_tanh_offdiag config/with_mhc_identity_tanh_offdiag.py
run_one mhc_sinkhorn config/with_mhc.py --mhc_h_res_mode=sinkhorn

echo
echo "All FineWeb small n=${N_STREAMS}, reduce=${REDUCE_STREAM_MODE} runs completed."
