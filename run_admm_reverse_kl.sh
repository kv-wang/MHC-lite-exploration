#!/usr/bin/env bash
#
# Train 4-stream mHC variants from scratch:
#   1. reverse-KL primal ADMM H_res
#   2. ordinary mHC with Sinkhorn H_res
#
# Usage:
#   ./run_admm_reverse_kl.sh
#   N_GPUS=1 MAX_ITERS=100 ./run_admm_reverse_kl.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/large_model.py}"
ADMM_METHOD_CONFIG="${ADMM_METHOD_CONFIG:-config/with_mhc_admm_reverse_kl.py}"
MHC_METHOD_CONFIG="${MHC_METHOD_CONFIG:-config/with_mhc.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-ablation_num_streams_large}"
ADMM_WANDB_RUN_NAME="${ADMM_WANDB_RUN_NAME:-mhc-large-mhc-admm-reverse-kl-4streams-10000iter}"
MHC_WANDB_RUN_NAME="${MHC_WANDB_RUN_NAME:-mhc-large-mhc-sinkhorn-4streams-10000iter}"
ADMM_OUT_PREFIX_METHOD="${ADMM_OUT_PREFIX_METHOD:-mhc-admm-reverse-kl-4streams-10000iter}"
MHC_OUT_PREFIX_METHOD="${MHC_OUT_PREFIX_METHOD:-mhc-sinkhorn-4streams-10000iter}"

echo ""
echo "================================================================"
echo " Running 4-stream mHC training variants"
echo " train_config:      $TRAIN_CONFIG"
echo " model_config:      $MODEL_CONFIG"
echo " admm_config:       $ADMM_METHOD_CONFIG"
echo " mhc_config:        $MHC_METHOD_CONFIG"
echo " max_iters:         $MAX_ITERS"
echo " eval_iters:        $EVAL_ITERS"
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
  echo " Running 4-stream mHC ${variant_name}"
  echo " method_config:     $method_config"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$MODEL_CONFIG"
    "$method_config"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --wandb_log=True
    --wandb_project="$WANDB_PROJECT"
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

run_variant "reverse-KL ADMM" "$ADMM_METHOD_CONFIG" "$ADMM_WANDB_RUN_NAME" "$ADMM_OUT_PREFIX_METHOD"
#run_variant "ordinary Sinkhorn" "$MHC_METHOD_CONFIG" "$MHC_WANDB_RUN_NAME" "$MHC_OUT_PREFIX_METHOD"

echo ""
echo "================================================================"
echo " 4-stream mHC training variants completed"
echo "================================================================"
