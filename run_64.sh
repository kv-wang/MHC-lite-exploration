#!/usr/bin/env bash
#
# Train n=64 mHC runs sequentially:
#   1) small identity+tanh-offdiag H_res
#   2) small identity H_res
#   3) large identity+tanh-offdiag H_res
#   4) large identity H_res
#
# Usage:
#   ./run_64.sh
#   N_GPUS=4 MAX_ITERS=100 ./run_64.sh
#   MHC_H_RES_OFFDIAG_INIT_SCALE=0.05 ./run_64.sh
#   MHC_H_RES_OFFDIAG_TRAINABLE=True ./run_64.sh
#   MHC_LOG_H_RES_GAMMA=True MHC_H_RES_GAMMA_LOG_INTERVAL=500 ./run_64.sh
#   WANDB_LOG_H_MATRIX_GRAD_NORM=True H_RES_GRAD_DUMP_INTERVAL=500 ./run_64.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
N_STREAMS="${N_STREAMS:-64}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-final_experiments}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
MHC_H_RES_OFFDIAG_INIT_SCALE="${MHC_H_RES_OFFDIAG_INIT_SCALE:-0.05}"
MHC_H_RES_OFFDIAG_TRAINABLE="${MHC_H_RES_OFFDIAG_TRAINABLE:-True}"
WANDB_LOG_H_MATRIX_GRAD_NORM="${WANDB_LOG_H_MATRIX_GRAD_NORM:-False}"
H_RES_GRAD_DUMP_INTERVAL="${H_RES_GRAD_DUMP_INTERVAL:-0}"
H_RES_GRAD_DUMP_DIR="${H_RES_GRAD_DUMP_DIR:-}"
MHC_LOG_H_RES_GAMMA="${MHC_LOG_H_RES_GAMMA:-False}"
MHC_H_RES_GAMMA_LOG_INTERVAL="${MHC_H_RES_GAMMA_LOG_INTERVAL:-500}"

echo ""
echo "================================================================"
echo " Running small then large mHC n=64 identity-tanh-offdiag and identity"
echo " train_config:  $TRAIN_CONFIG"
echo " n_streams:     $N_STREAMS"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " ckpt_interval: $CHECKPOINT_INTERVAL"
echo " n_gpus:        $N_GPUS"
echo " wandb_prefix:  $WANDB_PROJECT_PREFIX"
echo " reduce_mode:   $REDUCE_STREAM_MODE"
echo " h_grad_norms:  $WANDB_LOG_H_MATRIX_GRAD_NORM"
echo " offdiag_scale: $MHC_H_RES_OFFDIAG_INIT_SCALE"
echo " gamma_train:   $MHC_H_RES_OFFDIAG_TRAINABLE"
echo " gamma_log:     $MHC_LOG_H_RES_GAMMA"
echo " gamma_log_itv: $MHC_H_RES_GAMMA_LOG_INTERVAL"
echo " grad_dump_itv: $H_RES_GRAD_DUMP_INTERVAL"
echo " grad_dump_dir: ${H_RES_GRAD_DUMP_DIR:-<out_dir>/h_res_gradients}"
echo "================================================================"

scale_tag() {
  printf "%s" "$1" | sed 's/\./p/g'
}

run_method() {
  local model_name="$1"
  local method="$2"
  local model_config
  local wandb_project
  local method_config
  local wandb_run_name
  local out_prefix_method
  local method_desc
  local extra_method_args=()

  case "$model_name" in
    small)
      model_config="config/small_model.py"
      ;;
    medium)
      model_config="config/medium_model.py"
      ;;
    large)
      model_config="config/large_model.py"
      ;;
    *)
      echo "Unknown model: $model_name" >&2
      exit 1
      ;;
  esac

  wandb_project="${WANDB_PROJECT_PREFIX}_${model_name}"

  case "$method" in
    identity_tanh_offdiag)
      local scale_slug
      scale_slug="$(scale_tag "$MHC_H_RES_OFFDIAG_INIT_SCALE")"
      local gamma_mode
      if [[ "$MHC_H_RES_OFFDIAG_TRAINABLE" == "True" ]]; then
        gamma_mode="trainable"
      else
        gamma_mode="fixed"
      fi

      method_config="config/with_mhc_identity_tanh_offdiag.py"
      method_desc="${model_name} mHC identity-tanh-offdiag"
      wandb_run_name="mhc-${model_name}-mhc-identity-tanh-offdiag-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-gamma${scale_slug}-${gamma_mode}-${MAX_ITERS}iter"
      out_prefix_method="mhc-identity-tanh-offdiag-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-gamma${scale_slug}-${gamma_mode}-${MAX_ITERS}iter"
      extra_method_args+=(
        --mhc_h_res_offdiag_init_scale="$MHC_H_RES_OFFDIAG_INIT_SCALE"
        --mhc_h_res_offdiag_trainable="$MHC_H_RES_OFFDIAG_TRAINABLE"
        --mhc_log_h_res_gamma="$MHC_LOG_H_RES_GAMMA"
        --mhc_h_res_gamma_log_interval="$MHC_H_RES_GAMMA_LOG_INTERVAL"
      )
      ;;
    identity)
      method_config="config/with_mhc.py"
      method_desc="${model_name} mHC identity H_res"
      wandb_run_name="mhc-${model_name}-mhc-identity-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"
      out_prefix_method="mhc-identity-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"
      extra_method_args+=(--mhc_identity_h_res=True)
      ;;
    *)
      echo "Unknown method: $method" >&2
      exit 1
      ;;
  esac

  echo ""
  echo "================================================================"
  echo " Running $method_desc with ${N_STREAMS} streams"
  echo " model_config:      $model_config"
  echo " method_config:     $method_config"
  echo " wandb_project:     $wandb_project"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$model_config"
    "$method_config"
    --hyper_conn_n="$N_STREAMS"
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --checkpoint_interval="$CHECKPOINT_INTERVAL"
    --wandb_log=True
    --wandb_project="$wandb_project"
    --wandb_run_name="$wandb_run_name"
    --out_prefix_method="$out_prefix_method"
    --wandb_log_layer_stats=False
    --wandb_log_layer_cosine=False
    --wandb_log_layer_grad_norm=False
    --wandb_log_h_matrix_grad_norm="$WANDB_LOG_H_MATRIX_GRAD_NORM"
    --h_res_grad_dump_interval="$H_RES_GRAD_DUMP_INTERVAL"
    --h_res_grad_dump_dir="$H_RES_GRAD_DUMP_DIR"
    --wandb_log_layer_activation_norm=False
    --wandb_log_layer_activation_grad_norm=False
    --mhc_log_constraint_errors=False
  )

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}" "${extra_method_args[@]}" "${EXTRA_ARGS[@]}"
  else
    python train.py "${common_args[@]}" "${extra_method_args[@]}" "${EXTRA_ARGS[@]}"
  fi
}

EXTRA_ARGS=("$@")

run_method small identity_tanh_offdiag

echo ""
echo "================================================================"
echo " small and large mHC n=64 identity-tanh-offdiag and identity training completed"
echo "================================================================"
