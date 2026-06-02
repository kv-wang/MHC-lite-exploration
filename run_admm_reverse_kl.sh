#!/usr/bin/env bash
#
# Train a 4-stream mHC model from scratch with reverse-KL primal ADMM H_res.
#
# Usage:
#   ./run_admm_reverse_kl.sh
#   N_GPUS=1 MAX_ITERS=100 ./run_admm_reverse_kl.sh

set -e

export NCCL_TIMEOUT=1800

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/medium_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_admm_reverse_kl.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-ablation_num_streams_medium}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mhc-medium-mhc-admm-reverse-kl-4streams-10000iter}"
OUT_PREFIX_METHOD="${OUT_PREFIX_METHOD:-mhc-admm-reverse-kl-4streams-10000iter}"

echo ""
echo "================================================================"
echo " Running 4-stream mHC reverse-KL ADMM training"
echo " train_config:      $TRAIN_CONFIG"
echo " model_config:      $MODEL_CONFIG"
echo " method_config:     $METHOD_CONFIG"
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
  --max_iters="$MAX_ITERS"
  --eval_iters="$EVAL_ITERS"
  --wandb_log=True
  --wandb_project="$WANDB_PROJECT"
  --wandb_run_name="$WANDB_RUN_NAME"
  --out_prefix_method="$OUT_PREFIX_METHOD"
  --wandb_log_layer_stats=False
  --wandb_log_layer_cosine=False
)

if [[ "$N_GPUS" -gt 0 ]]; then
  torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}"
else
  python train.py "${common_args[@]}"
fi

echo ""
echo "================================================================"
echo " 4-stream mHC reverse-KL ADMM training completed"
echo "================================================================"
