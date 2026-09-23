#!/usr/bin/env bash
#
# Train small-model mHC Sinkhorn runs with 32 and 64 residual streams.
# Physical GPUs 0 and 1 are selected by default for two-process DDP training.
#
# Usage:
#   ./run_sinkhorn_all.sh
#   MAX_ITERS=100 EVAL_ITERS=20 ./run_sinkhorn_all.sh
#   CUDA_DEVICES=0,1 N_STREAMS_LIST="32 64" ./run_sinkhorn_all.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
N_GPUS="${N_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc.py}"
N_STREAMS_LIST="${N_STREAMS_LIST:-64}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-False}"
WANDB_PROJECT="${WANDB_PROJECT:-ablation_num_streams_small}"

echo ""
echo "================================================================"
echo " Running large mHC Sinkhorn training"
echo " cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
echo " n_gpus:              $N_GPUS"
echo " train_config:         $TRAIN_CONFIG"
echo " model_config:         $MODEL_CONFIG"
echo " method_config:        $METHOD_CONFIG"
echo " streams_list:         $N_STREAMS_LIST"
echo " reduce_mode:          $REDUCE_STREAM_MODE"
echo " max_iters:            $MAX_ITERS"
echo " eval_iters:           $EVAL_ITERS"
echo " checkpoint_interval:  $CHECKPOINT_INTERVAL"
echo " save_checkpoints:    $SAVE_CHECKPOINTS"
echo " wandb_project:        $WANDB_PROJECT"
echo "================================================================"

run_one() {
  local n_streams="$1"
  local run_name="mhc-small-mhc-sinkhorn-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"
  local out_prefix="mhc-sinkhorn-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running large mHC Sinkhorn with ${n_streams} streams"
  echo " wandb_run_name:   $run_name"
  echo " out_prefix_method: $out_prefix"
  echo "================================================================"

  torchrun --standalone --nproc_per_node="$N_GPUS" train.py \
    "$TRAIN_CONFIG" \
    "$MODEL_CONFIG" \
    "$METHOD_CONFIG" \
    --hyper_conn_n="$n_streams" \
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE" \
    --mhc_h_res_mode=sinkhorn \
    --max_iters="$MAX_ITERS" \
    --eval_iters="$EVAL_ITERS" \
    --checkpoint_interval="$CHECKPOINT_INTERVAL" \
    --save_checkpoints="$SAVE_CHECKPOINTS" \
    --wandb_log=True \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_name="$run_name" \
    --out_prefix_method="$out_prefix" \
    --wandb_log_layer_stats=False \
    --wandb_log_layer_cosine=False \
    --wandb_log_layer_grad_norm=False \
    --wandb_log_layer_activation_norm=False \
    --wandb_log_layer_activation_grad_norm=False \
    --mhc_log_constraint_errors=False \
    "${EXTRA_ARGS[@]}"
}

EXTRA_ARGS=("$@")

for n_streams in $N_STREAMS_LIST; do
  run_one "$n_streams"
done

echo ""
echo "================================================================"
echo " large mHC Sinkhorn n=32,64 training completed"
echo "================================================================"
