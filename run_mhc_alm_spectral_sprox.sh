#!/usr/bin/env bash
#
# Train mHC with S-prox-ALM spectral H_res constraints.
#
# Usage:
#   ./run_mhc_alm_spectral_sprox.sh
#   MODEL_NAME=medium N_STREAMS=8 ./run_mhc_alm_spectral_sprox.sh
#   MODEL_NAME=large N_STREAMS=32 REDUCE_STREAM_MODE=4mean N_GPUS=4 ./run_mhc_alm_spectral_sprox.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-1}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_alm_spectral_sprox.py}"
MODEL_NAME="${MODEL_NAME:-small}"
N_STREAMS="${N_STREAMS:-32}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-ablation_num_streams}"

case "$MODEL_NAME" in
  small)
    MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
    ;;
  medium)
    MODEL_CONFIG="${MODEL_CONFIG:-config/medium_model.py}"
    ;;
  large)
    MODEL_CONFIG="${MODEL_CONFIG:-config/large_model.py}"
    ;;
  *)
    echo "Invalid MODEL_NAME: $MODEL_NAME. Expected small, medium, or large." >&2
    exit 1
    ;;
esac

WANDB_PROJECT="${WANDB_PROJECT:-${WANDB_PROJECT_PREFIX}_${MODEL_NAME}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mhc-${MODEL_NAME}-mhc-alm-spectral-sprox-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter}"
OUT_PREFIX_METHOD="${OUT_PREFIX_METHOD:-mhc-alm-spectral-sprox-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter}"

echo ""
echo "================================================================"
echo " Running mHC ALM-spectral-S-prox training"
echo " train_config:      $TRAIN_CONFIG"
echo " model_name:        $MODEL_NAME"
echo " model_config:      $MODEL_CONFIG"
echo " method_config:     $METHOD_CONFIG"
echo " n_streams:         $N_STREAMS"
echo " reduce_mode:       $REDUCE_STREAM_MODE"
echo " max_iters:         $MAX_ITERS"
echo " eval_iters:        $EVAL_ITERS"
echo " n_gpus:            $N_GPUS"
echo " wandb_project:     $WANDB_PROJECT"
echo " wandb_run_name:    $WANDB_RUN_NAME"
echo " out_prefix_method: $OUT_PREFIX_METHOD"
echo "================================================================"

common_args=(
  "$TRAIN_CONFIG"
  "$MODEL_CONFIG"
  "$METHOD_CONFIG"
  --hyper_conn_n="$N_STREAMS"
  --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
  --max_iters="$MAX_ITERS"
  --eval_iters="$EVAL_ITERS"
  --wandb_log=True
  --wandb_project="$WANDB_PROJECT"
  --wandb_run_name="$WANDB_RUN_NAME"
  --out_prefix_method="$OUT_PREFIX_METHOD"
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

echo ""
echo "================================================================"
echo " mHC ALM-spectral-S-prox training completed"
echo "================================================================"
