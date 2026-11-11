#!/usr/bin/env bash
#
# Compare the two direct trainable-diagonal config flags for small n=4 ITO-HC.
# Both variants initialize H_res diagonal entries to 1 and train them directly.
#
# Usage:
#   ./run_mhc_identity_tanh_offdiag_diag_ablation_small_4.sh
#   N_GPUS=1 MAX_ITERS=100 EVAL_ITERS=20 ./run_mhc_identity_tanh_offdiag_diag_ablation_small_4.sh

set -euo pipefail

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

N_GPUS="${N_GPUS:-1}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_identity_tanh_offdiag.py}"
N_STREAMS=4
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-identity_tanh_offdiag_diag_ablation_small}"
EXTRA_ARGS=("$@")

run_variant() {
  local variant="$1"
  local diag_trainable="$2"
  local diag_unbounded="$3"
  local run_name="mhc-small-ito-hc-${variant}-${N_STREAMS}streams-${MAX_ITERS}iter"
  local out_prefix="mhc-ito-hc-${variant}-${N_STREAMS}streams-${MAX_ITERS}iter"

  echo "Running ${variant}: n=${N_STREAMS}, diag_trainable=${diag_trainable}, diag_unbounded=${diag_unbounded}"
  local common_args=(
    "$TRAIN_CONFIG"
    "$MODEL_CONFIG"
    "$METHOD_CONFIG"
    --hyper_conn_n="$N_STREAMS"
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
    --mhc_h_res_diag_trainable="$diag_trainable"
    --mhc_h_res_diag_unbounded="$diag_unbounded"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --checkpoint_interval="$CHECKPOINT_INTERVAL"
    --wandb_log=True
    --wandb_project="$WANDB_PROJECT"
    --wandb_run_name="$run_name"
    --out_prefix_method="$out_prefix"
    --wandb_log_layer_stats=False
    --wandb_log_layer_cosine=False
    --wandb_log_layer_grad_norm=False
    --wandb_log_layer_activation_norm=False
    --wandb_log_layer_activation_grad_norm=False
    --mhc_log_constraint_errors=False
  )

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
  else
    python train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
  fi
}

run_variant diag_trainable True False
run_variant diag_unbounded False True

echo "ITO-HC diagonal ablation completed."
