#!/usr/bin/env bash
#
# Train 32-stream mHC adapter-cap models initialized from the trained
# 4-stream checkpoint at out-owt-medium-mhc-num-streams-4/ckpt.pt.
# Runs both Sinkhorn-style and ADMM constraint variants.
#
# Usage:
#   ./run_mhc_adapter_cap_1000.sh
#   N_GPUS=0 MAX_ITERS=0 ./run_mhc_adapter_cap_1000.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/medium_model.py}"
SINKHORN_METHOD_CONFIG="${SINKHORN_METHOD_CONFIG:-config/with_mhc_adapter_cap.py}"
ADMM_METHOD_CONFIG="${ADMM_METHOD_CONFIG:-config/with_mhc_adapter_cap_admm.py}"
MAX_ITERS="${MAX_ITERS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-6e-5}"
WANDB_PROJECT="${WANDB_PROJECT:-ablation_num_streams_medium}"
SINKHORN_WANDB_RUN_NAME="${SINKHORN_WANDB_RUN_NAME:-mhc-medium-mhc-adapter-cap-sinkhorn-4to32-5000iter}"
ADMM_WANDB_RUN_NAME="${ADMM_WANDB_RUN_NAME:-mhc-medium-mhc-adapter-cap-admm-4to32-5000iter}"
SINKHORN_OUT_PREFIX_METHOD="${SINKHORN_OUT_PREFIX_METHOD:-mhc-adapter-cap-sinkhorn-4to32-5000iter}"
ADMM_OUT_PREFIX_METHOD="${ADMM_OUT_PREFIX_METHOD:-mhc-adapter-cap-admm-4to32-5000iter}"

echo ""
echo "================================================================"
echo " Running mHC adapter-cap 4-to-32 training variants"
echo " train_config:      $TRAIN_CONFIG"
echo " model_config:      $MODEL_CONFIG"
echo " sinkhorn_config:   $SINKHORN_METHOD_CONFIG"
echo " admm_config:       $ADMM_METHOD_CONFIG"
echo " max_iters:         $MAX_ITERS"
echo " learning_rate:     $LEARNING_RATE"
echo " n_gpus:            $N_GPUS"
echo " wandb_project:     $WANDB_PROJECT"
echo "================================================================"

run_variant() {
  local variant_name="$1"
  local method_config="$2"
  local wandb_run_name="$3"
  local out_prefix_method="$4"

  echo ""
  echo "================================================================"
  echo " Running mHC adapter-cap ${variant_name}"
  echo " method_config:     $method_config"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$MODEL_CONFIG"
    "$method_config"
    --max_iters="$MAX_ITERS"
    --learning_rate="$LEARNING_RATE"
    --min_lr="$LEARNING_RATE"
    --decay_lr=False
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

#run_variant "sinkhorn" "$SINKHORN_METHOD_CONFIG" "$SINKHORN_WANDB_RUN_NAME" "$SINKHORN_OUT_PREFIX_METHOD"
run_variant "admm" "$ADMM_METHOD_CONFIG" "$ADMM_WANDB_RUN_NAME" "$ADMM_OUT_PREFIX_METHOD"

echo ""
echo "================================================================"
echo " mHC adapter-cap training variants completed"
echo "================================================================"
