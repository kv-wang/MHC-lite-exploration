#!/usr/bin/env bash
#
# Run all available methods on the small model preset.
#
# Usage:
#   ./run_small.sh
#   N_GPUS=4 ./run_small.sh
#   N_GPUS=0 ./run_small.sh
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
WANDB_PROJECT="${WANDB_PROJECT:-mhc-lite}"
NUM_STREAMS="${NUM_STREAMS:-}"
EXPAND_STREAM_MODE="${EXPAND_STREAM_MODE:-}"

run_one() {
  local name="$1"
  shift
  echo ""
  echo "================================================================"
  echo " Running: $name"
  echo "================================================================"

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py \
      "$TRAIN_CONFIG" "$MODEL_CONFIG" \
      --wandb_project="$WANDB_PROJECT" \
      --wandb_log_layer_stats=False \
      --wandb_log_layer_cosine=False \
      --wandb_log_layer_grad_norm=False \
      --wandb_log_layer_activation_norm=False \
      --wandb_log_layer_activation_grad_norm=False \
      --mhc_log_constraint_errors=False \
      ${NUM_STREAMS:+--num_streams="$NUM_STREAMS"} \
      ${EXPAND_STREAM_MODE:+--expand_stream_mode="$EXPAND_STREAM_MODE"} \
      "$@"
  else
    python train.py \
      "$TRAIN_CONFIG" "$MODEL_CONFIG" \
      --wandb_project="$WANDB_PROJECT" \
      --wandb_log_layer_stats=False \
      --wandb_log_layer_cosine=False \
      --wandb_log_layer_grad_norm=False \
      --wandb_log_layer_activation_norm=False \
      --wandb_log_layer_activation_grad_norm=False \
      --mhc_log_constraint_errors=False \
      ${NUM_STREAMS:+--num_streams="$NUM_STREAMS"} \
      ${EXPAND_STREAM_MODE:+--expand_stream_mode="$EXPAND_STREAM_MODE"} \
      "$@"
  fi
}


# 13) mHC-lite block depth
run_one "mHC-lite-block-depth" \
  config/with_mhc_lite_block_depth.py

echo ""
echo "================================================================"
echo " All runs completed!"
echo "================================================================"
