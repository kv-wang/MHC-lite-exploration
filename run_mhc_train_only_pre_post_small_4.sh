#!/usr/bin/env bash
#
# Train small 4-stream mHC with H_res fixed to identity and only H_pre/H_post trainable among H matrices.
#
# Usage:
#   ./run_mhc_train_only_pre_post_small_4.sh
#   N_GPUS=4 MAX_ITERS=100 ./run_mhc_train_only_pre_post_small_4.sh
#   CHECKPOINT_INTERVAL=100 WANDB_PROJECT=final_experiments_small ./run_mhc_train_only_pre_post_small_4.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
N_STREAMS="${N_STREAMS:-4}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
WANDB_PROJECT="${WANDB_PROJECT:-final_experiments_small}"
WANDB_LOG_H_MATRIX_GRAD_NORM="${WANDB_LOG_H_MATRIX_GRAD_NORM:-False}"

WANDB_RUN_NAME="mhc-small-train-only-pre-post-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"
OUT_PREFIX_METHOD="mhc-train-only-pre-post-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"

echo ""
echo "================================================================"
echo " Running small mHC train-only-H_pre-H_post"
echo " train_config:      $TRAIN_CONFIG"
echo " model_config:      $MODEL_CONFIG"
echo " method_config:     $METHOD_CONFIG"
echo " n_streams:         $N_STREAMS"
echo " reduce_mode:       $REDUCE_STREAM_MODE"
echo " max_iters:         $MAX_ITERS"
echo " eval_iters:        $EVAL_ITERS"
echo " ckpt_interval:     $CHECKPOINT_INTERVAL"
echo " n_gpus:            $N_GPUS"
echo " wandb_project:     $WANDB_PROJECT"
echo " wandb_run_name:    $WANDB_RUN_NAME"
echo " out_prefix_method: $OUT_PREFIX_METHOD"
echo " h_grad_norms:      $WANDB_LOG_H_MATRIX_GRAD_NORM"
echo "================================================================"

common_args=(
  "$TRAIN_CONFIG"
  "$MODEL_CONFIG"
  "$METHOD_CONFIG"
  --hyper_conn_n="$N_STREAMS"
  --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
  --mhc_train_only_pre_post=True
  --max_iters="$MAX_ITERS"
  --eval_iters="$EVAL_ITERS"
  --checkpoint_interval="$CHECKPOINT_INTERVAL"
  --wandb_log=True
  --wandb_project="$WANDB_PROJECT"
  --wandb_run_name="$WANDB_RUN_NAME"
  --out_prefix_method="$OUT_PREFIX_METHOD"
  --wandb_log_layer_stats=False
  --wandb_log_layer_cosine=False
  --wandb_log_layer_grad_norm=False
  --wandb_log_h_matrix_grad_norm="$WANDB_LOG_H_MATRIX_GRAD_NORM"
  --wandb_log_layer_activation_norm=False
  --wandb_log_layer_activation_grad_norm=False
  --mhc_log_constraint_errors=False
  --mhc_log_h_res_diag_mass=False
)

EXTRA_ARGS=("$@")

if [[ "$N_GPUS" -gt 0 ]]; then
  torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
else
  python train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
fi

echo ""
echo "================================================================"
echo " small mHC train-only-H_pre-H_post training completed"
echo "================================================================"
