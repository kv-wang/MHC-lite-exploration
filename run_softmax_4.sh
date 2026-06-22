#!/usr/bin/env bash
#
# Train small mHC ALM-nonnegative with softmax_4mean stream reduction.
#
# Usage:
#   ./run_softmax_4.sh
#   N_GPUS=1 MAX_ITERS=100 ./run_softmax_4.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_alm_nonnegative.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-ablation_num_streams_small}"
STREAMS_LIST="${STREAMS_LIST:-32 16 8 4}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-softmax_4mean}"

echo ""
echo "================================================================"
echo " Running small mHC ALM-nonnegative softmax_4mean training"
echo " train_config:  $TRAIN_CONFIG"
echo " model_config:  $MODEL_CONFIG"
echo " method_config: $METHOD_CONFIG"
echo " streams_list:  $STREAMS_LIST"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " n_gpus:        $N_GPUS"
echo " wandb_project: $WANDB_PROJECT"
echo " reduce_mode:   $REDUCE_STREAM_MODE"
echo "================================================================"

run_streams() {
  local n_streams="$1"
  local wandb_run_name="mhc-small-mhc-alm-nonnegative-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"
  local out_prefix_method="mhc-alm-nonnegative-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running small mHC ALM-nonnegative with ${n_streams} streams"
  echo " wandb_project:     $WANDB_PROJECT"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$MODEL_CONFIG"
    "$METHOD_CONFIG"
    --hyper_conn_n="$n_streams"
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --wandb_log=True
    --wandb_project="$WANDB_PROJECT"
    --wandb_run_name="$wandb_run_name"
    --out_prefix_method="$out_prefix_method"
    --wandb_log_layer_stats=False
    --wandb_log_layer_cosine=False
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
echo " small mHC ALM-nonnegative softmax_4mean stream training completed"
echo "================================================================"
