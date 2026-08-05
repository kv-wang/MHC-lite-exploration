#!/usr/bin/env bash
#
# Train large mHC models with identity+tanh-offdiag H_res across stream counts.
#
# Usage:
#   ./run_identity.sh
#   N_GPUS=1 MAX_ITERS=100 STREAMS_LIST="4 8 16" ./run_identity.sh
#   MHC_H_RES_OFFDIAG_INIT_SCALE=0.0001 MHC_H_RES_OFFDIAG_TRAINABLE=True ./run_identity.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-128}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/large_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_identity_tanh_offdiag.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
STREAMS_LIST="${STREAMS_LIST:-4 8 16 32}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-final_experiments}"
MHC_H_RES_OFFDIAG_INIT_SCALE="${MHC_H_RES_OFFDIAG_INIT_SCALE:-0.05}"
MHC_H_RES_OFFDIAG_TRAINABLE="${MHC_H_RES_OFFDIAG_TRAINABLE:-True}"
WANDB_LOG_H_MATRIX_GRAD_NORM="${WANDB_LOG_H_MATRIX_GRAD_NORM:-True}"
H_RES_GRAD_DUMP_INTERVAL="${H_RES_GRAD_DUMP_INTERVAL:-0}"
H_RES_GRAD_DUMP_DIR="${H_RES_GRAD_DUMP_DIR:-}"
MHC_LOG_H_RES_GAMMA="${MHC_LOG_H_RES_GAMMA:-True}"
MHC_H_RES_GAMMA_LOG_INTERVAL="${MHC_H_RES_GAMMA_LOG_INTERVAL:-500}"

echo ""
echo "================================================================"
echo " Running large mHC identity-tanh-offdiag training across stream counts"
echo " train_config:  $TRAIN_CONFIG"
echo " model_config:  $MODEL_CONFIG"
echo " method_config: $METHOD_CONFIG"
echo " streams_list:  $STREAMS_LIST"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " ckpt_interval: $CHECKPOINT_INTERVAL"
echo " n_gpus:        $N_GPUS"
echo " batch_size:    $BATCH_SIZE (micro-batch per GPU)"
echo " grad_accum:    $GRADIENT_ACCUMULATION_STEPS (global; tokens/step = grad_accum * batch_size * block_size)"
echo " reduce_mode:   $REDUCE_STREAM_MODE"
echo " wandb_prefix:  $WANDB_PROJECT_PREFIX"
echo " offdiag_scale: $MHC_H_RES_OFFDIAG_INIT_SCALE"
echo " gamma_train:   $MHC_H_RES_OFFDIAG_TRAINABLE"
echo " gamma_log:     $MHC_LOG_H_RES_GAMMA"
echo " gamma_log_itv: $MHC_H_RES_GAMMA_LOG_INTERVAL"
echo " h_grad_norms:  $WANDB_LOG_H_MATRIX_GRAD_NORM"
echo " grad_dump_itv: $H_RES_GRAD_DUMP_INTERVAL"
echo " grad_dump_dir: ${H_RES_GRAD_DUMP_DIR:-<out_dir>/h_res_gradients}"
echo "================================================================"

scale_tag() {
  printf "%s" "$1" | sed 's/\./p/g'
}

run_model() {
  local n_streams="$1"
  local model_name="large"
  local model_config="$MODEL_CONFIG"
  local wandb_project="${WANDB_PROJECT_PREFIX}_${model_name}"
  local scale_slug
  scale_slug="$(scale_tag "$MHC_H_RES_OFFDIAG_INIT_SCALE")"
  local gamma_mode
  if [[ "$MHC_H_RES_OFFDIAG_TRAINABLE" == "True" ]]; then
    gamma_mode="trainable"
  else
    gamma_mode="fixed"
  fi
  local out_prefix_method="mhc-identity-tanh-offdiag-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-gamma${scale_slug}-${gamma_mode}-${MAX_ITERS}iter"
  local wandb_run_name="mhc-${model_name}-identity-tanh-offdiag-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-gamma${scale_slug}-${gamma_mode}-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running ${model_name} mHC identity-tanh-offdiag with ${n_streams} streams"
  echo " model_config:      $model_config"
  echo " n_streams:         $n_streams"
  echo " wandb_project:     $wandb_project"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo " offdiag_scale:     $MHC_H_RES_OFFDIAG_INIT_SCALE"
  echo " gamma_trainable:   $MHC_H_RES_OFFDIAG_TRAINABLE"
  echo " gamma_log:         $MHC_LOG_H_RES_GAMMA"
  echo " gamma_log_interval: $MHC_H_RES_GAMMA_LOG_INTERVAL"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$model_config"
    "$METHOD_CONFIG"
    --hyper_conn_n="$n_streams"
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
    --mhc_h_res_offdiag_init_scale="$MHC_H_RES_OFFDIAG_INIT_SCALE"
    --mhc_h_res_offdiag_trainable="$MHC_H_RES_OFFDIAG_TRAINABLE"
    --batch_size="$BATCH_SIZE"
    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS"
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
    --mhc_log_h_res_gamma="$MHC_LOG_H_RES_GAMMA"
    --mhc_h_res_gamma_log_interval="$MHC_H_RES_GAMMA_LOG_INTERVAL"
  )

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
  else
    python train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
  fi
}

EXTRA_ARGS=("$@")

for n_streams in $STREAMS_LIST; do
  run_model "$n_streams"
done

echo ""
echo "================================================================"
echo " large mHC identity-tanh-offdiag stream-count training completed"
echo "================================================================"
