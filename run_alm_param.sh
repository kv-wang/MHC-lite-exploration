#!/usr/bin/env bash
#
# Train small then medium mHC ALM-nonnegative models with the identity-init
# H_res hyperparameters selected from the 500-step sweep.
#
# Usage:
#   ./run_alm_param.sh
#   N_GPUS=1 MAX_ITERS=499 ./run_alm_param.sh  # 500 updates: iter 0..499

set -euo pipefail

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_alm_nonnegative.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
WANDB_LOG="${WANDB_LOG:-True}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-alm_param_identity_init}"

NUM_STREAMS="${NUM_STREAMS:-4}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
H_RES_INIT_DIAG_MASS_FRAC="${H_RES_INIT_DIAG_MASS_FRAC:-1.0}"
MHC_ADMM_STEP_SCALE="${MHC_ADMM_STEP_SCALE:-0.0425}"
MHC_ADMM_RHO="${MHC_ADMM_RHO:-5.0}"
MHC_ADMM_PROX_WEIGHT="${MHC_ADMM_PROX_WEIGHT:-0.2}"
MHC_ADMM_DUAL_STEP="${MHC_ADMM_DUAL_STEP:-1.0}"
MHC_DISABLE_DYNAMIC_H_RES="${MHC_DISABLE_DYNAMIC_H_RES:-True}"
WANDB_LOG_H_MATRIX_GRAD_NORM="${WANDB_LOG_H_MATRIX_GRAD_NORM:-False}"

PARAM_TAG="idinit-${NUM_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-step4p25e-2-rho5-prox0p2-dual1"

echo ""
echo "================================================================"
echo " Running small then medium mHC ALM-nonnegative with sweep params"
echo " train_config:  $TRAIN_CONFIG"
echo " method_config: $METHOD_CONFIG"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " n_gpus:        $N_GPUS"
echo " streams:       $NUM_STREAMS"
echo " reduce_mode:   $REDUCE_STREAM_MODE"
echo " diag_init:     $H_RES_INIT_DIAG_MASS_FRAC"
echo " step_scale:    $MHC_ADMM_STEP_SCALE"
echo " rho:           $MHC_ADMM_RHO"
echo " prox_weight:   $MHC_ADMM_PROX_WEIGHT"
echo " dual_step:     $MHC_ADMM_DUAL_STEP"
echo " wandb_log:     $WANDB_LOG"
echo "================================================================"

run_model() {
  local model_name="$1"
  local model_config="$2"
  local wandb_project="${WANDB_PROJECT_PREFIX}_${model_name}"
  local wandb_run_name="mhc-${model_name}-alm-nonnegative-${PARAM_TAG}-${MAX_ITERS}iter"
  local out_prefix_method="mhc-alm-nonnegative-${PARAM_TAG}-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running ${model_name} mHC ALM-nonnegative"
  echo " model_config:      $model_config"
  echo " wandb_project:     $wandb_project"
  echo " wandb_run_name:    $wandb_run_name"
  echo " out_prefix_method: $out_prefix_method"
  echo "================================================================"

  local common_args=(
    "$TRAIN_CONFIG"
    "$model_config"
    "$METHOD_CONFIG"
    --hyper_conn_n="$NUM_STREAMS"
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
    --mhc_h_res_init_diag_mass_frac="$H_RES_INIT_DIAG_MASS_FRAC"
    --mhc_admm_step_scale="$MHC_ADMM_STEP_SCALE"
    --mhc_admm_rho="$MHC_ADMM_RHO"
    --mhc_admm_prox_weight="$MHC_ADMM_PROX_WEIGHT"
    --mhc_admm_dual_step="$MHC_ADMM_DUAL_STEP"
    --mhc_disable_dynamic_h_res="$MHC_DISABLE_DYNAMIC_H_RES"
    --max_iters="$MAX_ITERS"
    --eval_iters="$EVAL_ITERS"
    --wandb_log="$WANDB_LOG"
    --wandb_project="$wandb_project"
    --wandb_run_name="$wandb_run_name"
    --out_prefix_method="$out_prefix_method"
    --wandb_log_layer_stats=False
    --wandb_log_layer_cosine=False
    --wandb_log_layer_grad_norm=False
    --wandb_log_h_matrix_grad_norm="$WANDB_LOG_H_MATRIX_GRAD_NORM"
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

run_model "small" "config/small_model.py"
run_model "medium" "config/medium_model.py"

echo ""
echo "================================================================"
echo " small and medium mHC ALM-nonnegative parameter runs completed"
echo "================================================================"
