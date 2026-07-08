#!/usr/bin/env bash
#
# Train small mHC ALM-nonnegative-cap with configurable residual streams.
#
# Usage:
#   ./run_alm_nonnegative_cap.sh
#   N_GPUS=1 MAX_ITERS=100 ./run_alm_nonnegative_cap.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_alm_nonnegative_cap.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-ablation_num_streams_small}"
STREAMS_LIST="${STREAMS_LIST:-32 4}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
MHC_H_RES_CAP="${MHC_H_RES_CAP:-1.5}"
MHC_ADMM_PROX_WEIGHT="${MHC_ADMM_PROX_WEIGHT:-1.0}"
MHC_ADMM_STEP_SCALE="${MHC_ADMM_STEP_SCALE:-0.0001}"
WANDB_LOG_H_MATRIX_GRAD_NORM="${WANDB_LOG_H_MATRIX_GRAD_NORM:-True}"

echo ""
echo "================================================================"
echo " Running small mHC ALM-nonnegative-cap training for streams: $STREAMS_LIST"
echo " train_config:  $TRAIN_CONFIG"
echo " model_config:  $MODEL_CONFIG"
echo " method_config: $METHOD_CONFIG"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " n_gpus:        $N_GPUS"
echo " wandb_project: $WANDB_PROJECT"
echo " reduce_mode:   $REDUCE_STREAM_MODE"
echo " h_grad_norms:  $WANDB_LOG_H_MATRIX_GRAD_NORM"
echo " h_res_cap:     $MHC_H_RES_CAP"
echo " prox_weight:   $MHC_ADMM_PROX_WEIGHT"
echo " step_scale:    $MHC_ADMM_STEP_SCALE"
echo "================================================================"

cap_tag() {
  printf "%s" "$1" | sed 's/\./p/g'
}

run_streams() {
  local n_streams="$1"
  local cap_slug
  cap_slug="$(cap_tag "$MHC_H_RES_CAP")"
  local wandb_run_name="mhc-small-mhc-alm-nonnegative-cap-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-cap${cap_slug}-${MAX_ITERS}iter"
  local out_prefix_method="mhc-alm-nonnegative-cap-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-cap${cap_slug}-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running small mHC ALM-nonnegative-cap with ${n_streams} streams"
  echo " wandb_project:     $WANDB_PROJECT"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo " h_res_cap:         $MHC_H_RES_CAP"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$MODEL_CONFIG"
    "$METHOD_CONFIG"
    --hyper_conn_n="$n_streams"
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
    --mhc_h_res_cap="$MHC_H_RES_CAP"
    --mhc_admm_prox_weight="$MHC_ADMM_PROX_WEIGHT"
    --mhc_admm_step_scale="$MHC_ADMM_STEP_SCALE"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --wandb_log=True
    --wandb_project="$WANDB_PROJECT"
    --wandb_run_name="$wandb_run_name"
    --out_prefix_method="$out_prefix_method"
    --wandb_log_layer_stats=False
    --wandb_log_layer_cosine=False
    --wandb_log_layer_grad_norm=False
    --wandb_log_h_matrix_grad_norm="$WANDB_LOG_H_MATRIX_GRAD_NORM"
    --wandb_log_layer_activation_norm=False
    --wandb_log_layer_activation_grad_norm=False
    --mhc_log_constraint_errors=True
  )

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}"
  else
    python train.py "${common_args[@]}"
  fi
}

for n_streams in $STREAMS_LIST; do
  run_streams "$n_streams"
done

echo ""
echo "================================================================"
echo " small mHC ALM-nonnegative-cap stream training completed"
echo "================================================================"
