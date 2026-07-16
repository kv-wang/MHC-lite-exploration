#!/usr/bin/env bash
#
# Train large mHC models with H_res fixed to identity across stream counts.
#
# Usage:
#   ./run_identity.sh
#   N_GPUS=1 MAX_ITERS=100 STREAMS_LIST="4 8 16" ./run_identity.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-256}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
STREAMS_LIST="${STREAMS_LIST:-32}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-ablation_num_streams}"

echo ""
echo "================================================================"
echo " Running large mHC identity-H_res training across stream counts"
echo " train_config:  $TRAIN_CONFIG"
echo " model_config:  $MODEL_CONFIG"
echo " method_config: $METHOD_CONFIG"
echo " streams_list:  $STREAMS_LIST"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " n_gpus:        $N_GPUS"
echo " batch_size:    $BATCH_SIZE (micro-batch per GPU)"
echo " grad_accum:    $GRADIENT_ACCUMULATION_STEPS (global; tokens/step = grad_accum * batch_size * block_size)"
echo " reduce_mode:   $REDUCE_STREAM_MODE"
echo " wandb_prefix:  $WANDB_PROJECT_PREFIX"
echo "================================================================"

run_model() {
  local n_streams="$1"
  local model_name="small"
  local model_config="$MODEL_CONFIG"
  local wandb_project="${WANDB_PROJECT_PREFIX}_${model_name}"
  local out_prefix_method="mhc-identity-h-res-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"
  local wandb_run_name="mhc-${model_name}-identity-h-res-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running ${model_name} mHC identity-H_res with ${n_streams} streams"
  echo " model_config:      $model_config"
  echo " n_streams:         $n_streams"
  echo " wandb_project:     $wandb_project"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$model_config"
    "$METHOD_CONFIG"
    --hyper_conn_n="$n_streams"
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
    --mhc_identity_h_res=True
    --batch_size="$BATCH_SIZE"
    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --wandb_log=True
    --wandb_project="$wandb_project"
    --wandb_run_name="$wandb_run_name"
    --out_prefix_method="$out_prefix_method"
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
}

for n_streams in $STREAMS_LIST; do
  run_model "$n_streams"
done

echo ""
echo "================================================================"
echo " large mHC identity-H_res stream-count training completed"
echo "================================================================"
