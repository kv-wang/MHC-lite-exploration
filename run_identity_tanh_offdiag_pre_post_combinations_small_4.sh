#!/usr/bin/env bash
#
# Sequentially train three small-model, 4-stream identity-tanh-offdiag runs:
#   1) static H_pre, static H_post
#   2) dynamic H_pre, static H_post
#   3) static H_pre, dynamic H_post
#
# H_res follows config/with_mhc_identity_tanh_offdiag.py and is fixed at its
# identity+tanh-offdiag parameterization (fixed gamma by default).
#
# Usage:
#   ./run_identity_tanh_offdiag_pre_post_combinations_small_4.sh
#   MAX_ITERS=100 EVAL_ITERS=20 ./run_identity_tanh_offdiag_pre_post_combinations_small_4.sh
#   CUDA_DEVICES=2,3 N_GPUS=2 ./run_identity_tanh_offdiag_pre_post_combinations_small_4.sh
#

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
N_GPUS="${N_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_identity_tanh_offdiag.py}"
N_STREAMS="${N_STREAMS:-4}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-identity_tanh_offdiag_pre_post_small}"
WANDB_LOG_H_MATRIX_GRAD_NORM="${WANDB_LOG_H_MATRIX_GRAD_NORM:-False}"

EXTRA_ARGS=("$@")

run_one() {
  local combination="$1"
  local disable_pre="$2"
  local disable_post="$3"
  local run_name="mhc-small-identity-tanh-offdiag-${combination}-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"
  local out_prefix="mhc-identity-tanh-offdiag-${combination}-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running small mHC identity-tanh-offdiag: ${combination}"
  echo " cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
  echo " n_gpus:              $N_GPUS"
  echo " n_streams:           $N_STREAMS"
  echo " reduce_mode:         $REDUCE_STREAM_MODE"
  echo " disable_h_pre:       $disable_pre"
  echo " disable_h_post:      $disable_post"
  echo " max_iters:           $MAX_ITERS"
  echo " eval_iters:          $EVAL_ITERS"
  echo " checkpoint_interval: $CHECKPOINT_INTERVAL"
  echo " wandb_project:       $WANDB_PROJECT"
  echo " wandb_run_name:      $run_name"
  echo " out_prefix_method:   $out_prefix"
  echo "================================================================"

  torchrun --standalone --nproc_per_node="$N_GPUS" train.py \
    "$TRAIN_CONFIG" \
    "$MODEL_CONFIG" \
    "$METHOD_CONFIG" \
    --hyper_conn_n="$N_STREAMS" \
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE" \
    --mhc_disable_dynamic_h_pre="$disable_pre" \
    --mhc_disable_dynamic_h_post="$disable_post" \
    --max_iters="$MAX_ITERS" \
    --eval_iters="$EVAL_ITERS" \
    --checkpoint_interval="$CHECKPOINT_INTERVAL" \
    --wandb_log=True \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_name="$run_name" \
    --out_prefix_method="$out_prefix" \
    --wandb_log_layer_stats=False \
    --wandb_log_layer_cosine=False \
    --wandb_log_layer_grad_norm=False \
    --wandb_log_h_matrix_grad_norm="$WANDB_LOG_H_MATRIX_GRAD_NORM" \
    --wandb_log_layer_activation_norm=False \
    --wandb_log_layer_activation_grad_norm=False \
    --mhc_log_constraint_errors=False \
    --mhc_log_h_res_diag_mass=False \
    "${EXTRA_ARGS[@]}"
}

run_one static-static True True
run_one dynamic-static False True
run_one static-dynamic True False

echo ""
echo "================================================================"
echo " small identity-tanh-offdiag n=${N_STREAMS} H_pre/H_post combinations completed"
echo "================================================================"
