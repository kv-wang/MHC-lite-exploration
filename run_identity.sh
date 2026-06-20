#!/usr/bin/env bash
#
# Train small mHC models with H_res fixed to identity across stream counts.
#
# Usage:
#   ./run_identity.sh
#   N_GPUS=1 MAX_ITERS=100 STREAMS_LIST="8 16" ./run_identity.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
STREAMS_LIST="${STREAMS_LIST:-8 16 32}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-ablation_num_streams}"

echo ""
echo "================================================================"
echo " Running small mHC identity-H_res training across stream counts"
echo " train_config:  $TRAIN_CONFIG"
echo " method_config: $METHOD_CONFIG"
echo " streams_list:  $STREAMS_LIST"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " n_gpus:        $N_GPUS"
echo " wandb_prefix:  $WANDB_PROJECT_PREFIX"
echo "================================================================"

run_model() {
  local n_streams="$1"
  local model_name="small"
  local model_config="config/small_model.py"
  local wandb_project="${WANDB_PROJECT_PREFIX}_${model_name}"
  local out_prefix_method="mhc-identity-h-res-${n_streams}streams-${MAX_ITERS}iter"
  local wandb_run_name="mhc-${model_name}-identity-h-res-${n_streams}streams-${MAX_ITERS}iter"

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
    --mhc_identity_h_res=True
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --wandb_log=True
    --wandb_project="$wandb_project"
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
  run_model "$n_streams"
done

echo ""
echo "================================================================"
echo " small mHC identity-H_res stream-count training completed"
echo "================================================================"
