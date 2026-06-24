#!/usr/bin/env bash
#
# Run all methods: Residual, HC, mHC, mHC-lite, AttnRes
# Each mHC/mHC-lite variant uses default sigmoid gate + full H_res.
# To test other combos, pass --mhc_gate_fn=softmax or --mhc_identity_h_res=True.
#
# Usage:
#   ./run_all_methods.sh                    # 8 GPUs (default)
#   N_GPUS=4 ./run_all_methods.sh           # 4 GPUs
#   N_GPUS=0 ./run_all_methods.sh           # single GPU

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"


N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/large_model.py}"
WANDB_PROJECT="${WANDB_PROJECT:-mhc-lite-large_owt}"
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

# # 1) Residual (default)
# run_one "Residual"
#
# # 2) HC
#run_one "HC" \
  #config/with_hc.py
#
# # 3) mHC (sigmoid, default)
#run_one "mHC-sigmoid" \
#  config/with_mhc.py
#
# # 4) mHC-lite (sigmoid, default)
#run_one "mHC-lite-sigmoid" \
  #config/with_mhc_lite.py
#
# # 5) Attention Residuals
# run_one "AttnRes" \
#   config/with_attn_res.py

# # 6) mHC (softmax, default)
# run_one "mHC-softmax" \
#   config/with_mhc.py \
#   --mhc_gate_fn=softmax
#
# # 7) mHC-lite (softmax, default)
# run_one "mHC-lite-softmax" \
#   config/with_mhc_lite.py \
#   --mhc_gate_fn=softmax
#
# # 8) mHC (softmax, identity H_res)
# run_one "mHC-softmax-idH" \
#   config/with_mhc.py \
#   --mhc_gate_fn=softmax \
#   --mhc_identity_h_res=True
#
# 9) Attention Residuals
#run_one "AttnRes" \
  #config/with_attn_res.py

#run_one "mHC-lite-selective" \
  #config/with_mhc_lite_selective.py

#run_one "mHC-lite-block-attn" \
  #config/with_mhc_lite_block_attn.py \
  #--mhc_identity_h_res=True

#run_one "mHC-lite-block-attn" \
  #config/with_mhc_lite_block_attn.py

#run_one "mHC-lite-depth-attn" \
  #config/with_mhc_lite_depth_attn.py

run_one "mHC-lite-block-depth" \
  config/with_mhc_lite_block_depth.py

echo ""
echo "================================================================"
echo " All runs completed!"
echo "================================================================"
