#!/usr/bin/env bash
#
# Train large mHC identity+clip-offdiag H_res with fixed gamma, n_streams=4.
#
# Usage:
#   ./run_mhc_identity_clip_offdiag_large_4.sh
#   N_GPUS=4 MAX_ITERS=100 ./run_mhc_identity_clip_offdiag_large_4.sh
#   MHC_H_RES_OFFDIAG_INIT_SCALE=0.05 ./run_mhc_identity_clip_offdiag_large_4.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/large_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_identity_clip_offdiag.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-ablation_num_streams_large}"
STREAMS_LIST="${STREAMS_LIST:-4}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
MHC_H_RES_OFFDIAG_INIT_SCALE="${MHC_H_RES_OFFDIAG_INIT_SCALE:-0.05}"
# Fixed gamma for this script.
MHC_H_RES_OFFDIAG_TRAINABLE="${MHC_H_RES_OFFDIAG_TRAINABLE:-False}"
WANDB_LOG_H_MATRIX_GRAD_NORM="${WANDB_LOG_H_MATRIX_GRAD_NORM:-True}"
H_RES_GRAD_DUMP_INTERVAL="${H_RES_GRAD_DUMP_INTERVAL:-500}"
H_RES_GRAD_DUMP_DIR="${H_RES_GRAD_DUMP_DIR:-}"

echo ""
echo "================================================================"
echo " Running large mHC identity-clip-offdiag (fixed gamma)"
echo " streams:       $STREAMS_LIST"
echo " train_config:  $TRAIN_CONFIG"
echo " model_config:  $MODEL_CONFIG"
echo " method_config: $METHOD_CONFIG"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " ckpt_interval: $CHECKPOINT_INTERVAL"
echo " n_gpus:        $N_GPUS"
echo " wandb_project: $WANDB_PROJECT"
echo " reduce_mode:   $REDUCE_STREAM_MODE"
echo " h_grad_norms:  $WANDB_LOG_H_MATRIX_GRAD_NORM"
echo " offdiag_scale: $MHC_H_RES_OFFDIAG_INIT_SCALE"
echo " gamma_train:   $MHC_H_RES_OFFDIAG_TRAINABLE"
echo " grad_dump_itv: $H_RES_GRAD_DUMP_INTERVAL"
echo " grad_dump_dir: ${H_RES_GRAD_DUMP_DIR:-<out_dir>/h_res_gradients}"
echo "================================================================"

scale_tag() {
  printf "%s" "$1" | sed 's/\./p/g'
}

run_streams() {
  local n_streams="$1"
  local scale_slug
  scale_slug="$(scale_tag "$MHC_H_RES_OFFDIAG_INIT_SCALE")"
  local gamma_mode="fixed"
  local wandb_run_name="mhc-large-mhc-identity-clip-offdiag-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-gamma${scale_slug}-${gamma_mode}-${MAX_ITERS}iter"
  local out_prefix_method="mhc-identity-clip-offdiag-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-gamma${scale_slug}-${gamma_mode}-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running large mHC identity-clip-offdiag with ${n_streams} streams"
  echo " wandb_project:     $WANDB_PROJECT"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo " offdiag_scale:     $MHC_H_RES_OFFDIAG_INIT_SCALE"
  echo " gamma_trainable:   $MHC_H_RES_OFFDIAG_TRAINABLE"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$MODEL_CONFIG"
    "$METHOD_CONFIG"
    --hyper_conn_n="$n_streams"
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
    --mhc_h_res_offdiag_init_scale="$MHC_H_RES_OFFDIAG_INIT_SCALE"
    --mhc_h_res_offdiag_trainable="$MHC_H_RES_OFFDIAG_TRAINABLE"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --checkpoint_interval="$CHECKPOINT_INTERVAL"
    --wandb_log=True
    --wandb_project="$WANDB_PROJECT"
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
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
  else
    python train.py "${common_args[@]}" "${EXTRA_ARGS[@]}"
  fi
}

EXTRA_ARGS=("$@")

for n_streams in $STREAMS_LIST; do
  run_streams "$n_streams"
done

echo ""
echo "================================================================"
echo " large mHC identity-clip-offdiag (fixed gamma, n=4) training completed"
echo "================================================================"
