#!/usr/bin/env bash
#
# Train small mHC ALM-nonnegative models with H_res initialized so that
# diagonal mass fraction is 1 / n_streams.
#
# Usage:
#   ./run_mhc_alm_nonnegative_small_diag_mass.sh
#   N_GPUS=1 MAX_ITERS=100 STREAMS_LIST="4 8" ./run_mhc_alm_nonnegative_small_diag_mass.sh

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
STREAMS_LIST="${STREAMS_LIST:-4 8 16 32}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
H_RES_DIAG_MASS_LOG_INTERVAL="${H_RES_DIAG_MASS_LOG_INTERVAL:-1000}"
WANDB_LOG_H_MATRIX_GRAD_NORM="${WANDB_LOG_H_MATRIX_GRAD_NORM:-True}"
MHC_ADMM_PROX_WEIGHT="${MHC_ADMM_PROX_WEIGHT:-0.1}"
MHC_ADMM_STEP_SCALE="${MHC_ADMM_STEP_SCALE:-0.001}"
WANDB_PROJECT="${WANDB_PROJECT:-ablation_num_streams_small}"

echo ""
echo "================================================================"
echo " Running small mHC ALM-nonnegative with diag mass fraction 1/n"
echo " train_config:  $TRAIN_CONFIG"
echo " model_config:  $MODEL_CONFIG"
echo " method_config: $METHOD_CONFIG"
echo " streams_list:  $STREAMS_LIST"
echo " reduce_mode:   $REDUCE_STREAM_MODE"
echo " max_iters:     $MAX_ITERS"
echo " eval_iters:    $EVAL_ITERS"
echo " n_gpus:        $N_GPUS"
echo " diag_log_int:  $H_RES_DIAG_MASS_LOG_INTERVAL"
echo " h_grad_norms:  $WANDB_LOG_H_MATRIX_GRAD_NORM"
echo " prox_weight:   $MHC_ADMM_PROX_WEIGHT"
echo " step_scale:    $MHC_ADMM_STEP_SCALE"
echo " wandb_project: $WANDB_PROJECT"
echo "================================================================"

diag_mass_for_streams() {
  local n_streams="$1"
  awk -v n="$n_streams" 'BEGIN { printf "%.12g", 1.0 / n }'
}

run_streams() {
  local n_streams="$1"
  local diag_mass_frac
  diag_mass_frac="$(diag_mass_for_streams "$n_streams")"
  local diag_tag="diagmass-1over${n_streams}"
  local wandb_run_name="mhc-small-mhc-alm-nonnegative-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-${diag_tag}-${MAX_ITERS}iter"
  local out_prefix_method="mhc-alm-nonnegative-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-${diag_tag}-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running small mHC ALM-nonnegative with ${n_streams} streams"
  echo " diag_mass_frac:    $diag_mass_frac"
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
    --mhc_h_res_init_diag_mass_frac="$diag_mass_frac"
    --mhc_admm_prox_weight="$MHC_ADMM_PROX_WEIGHT"
    --mhc_admm_step_scale="$MHC_ADMM_STEP_SCALE"
    --mhc_log_h_res_diag_mass=True
    --mhc_h_res_diag_mass_log_interval="$H_RES_DIAG_MASS_LOG_INTERVAL"
    --wandb_log_h_matrix_grad_norm="$WANDB_LOG_H_MATRIX_GRAD_NORM"
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

for n_streams in $STREAMS_LIST; do
  run_streams "$n_streams"
done

echo ""
echo "================================================================"
echo " small mHC ALM-nonnegative 1/n diagonal-mass training completed"
echo "================================================================"
