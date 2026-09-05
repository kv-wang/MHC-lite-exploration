#!/usr/bin/env bash
#
# Train small 4-stream mHC with reverse-KL S-prox-ALM H_res projection.
#
# Usage:
#   ./run_mhc_admm_reverse_kl_sprox_alm_small_4.sh
#   N_GPUS=4 MAX_ITERS=100 ./run_mhc_admm_reverse_kl_sprox_alm_small_4.sh
#   CHECKPOINT_INTERVAL=100 WANDB_PROJECT=final_experiments_small ./run_mhc_admm_reverse_kl_sprox_alm_small_4.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_admm_reverse_kl_sprox_alm.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
N_STREAMS="${N_STREAMS:-4}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
WANDB_PROJECT="${WANDB_PROJECT:-final_experiments_small}"
WANDB_LOG_H_MATRIX_GRAD_NORM="${WANDB_LOG_H_MATRIX_GRAD_NORM:-False}"
MHC_LOG_CONSTRAINT_ERRORS="${MHC_LOG_CONSTRAINT_ERRORS:-True}"
MHC_CONSTRAINT_LOG_INTERVAL="${MHC_CONSTRAINT_LOG_INTERVAL:-100}"

WANDB_RUN_NAME="mhc-small-mhc-admm-reverse-kl-sprox-alm-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"
OUT_PREFIX_METHOD="mhc-admm-reverse-kl-sprox-alm-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"

echo ""
echo "================================================================"
echo " Running small mHC reverse-KL S-prox-ALM"
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
echo " constraint_log:    $MHC_LOG_CONSTRAINT_ERRORS"
echo " constraint_itv:    $MHC_CONSTRAINT_LOG_INTERVAL"
echo "================================================================"

common_args=(
  "$TRAIN_CONFIG"
  "$MODEL_CONFIG"
  "$METHOD_CONFIG"
  --hyper_conn_n="$N_STREAMS"
  --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
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
  --mhc_log_constraint_errors="$MHC_LOG_CONSTRAINT_ERRORS"
  --mhc_constraint_log_interval="$MHC_CONSTRAINT_LOG_INTERVAL"
)

EXTRA_ARGS=("$@")

if [[ "$N_GPUS" -gt 0 ]]; then
  torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
else
  python train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
fi

echo ""
echo "================================================================"
echo " small mHC reverse-KL S-prox-ALM training completed"
echo "================================================================"
