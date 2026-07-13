#!/usr/bin/env bash
#
# Continue the 4-stream small mHC ALM-nonnegative-cap checkpoint with an
# unconstrained static H_res. H_res is updated by AdamW from the LM gradient,
# with no Sinkhorn, ALM/S-prox update, nonnegative projection, or row/column
# constraint.
#
# The checkpoint is at iter 10000. MAX_ITERS is the target total iteration,
# so the default MAX_ITERS=20000 continues training for another 10000 steps.
#
# Usage:
#   ./run_alm_nonnegative_cap_continue.sh
#   N_GPUS=1 MAX_ITERS=11000 LOG_INTERVAL=10 ./run_alm_nonnegative_cap_continue.sh

set -euo pipefail

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/nanogpt/bin/python}"
N_GPUS="${N_GPUS:-1}"

TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/small_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc_alm_nonnegative_cap.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$SCRIPT_DIR/alm_cap_4stream/ckpt.pt}"

MAX_ITERS="${MAX_ITERS:-20000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"

WANDB_PROJECT="${WANDB_PROJECT:-ablation_num_streams_small}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-mhc-small-unconstrained-h-res-4streams-from-alm-cap-fixed-ckpt-lr-to-${MAX_ITERS}iter}"
OUT_PREFIX_METHOD="${OUT_PREFIX_METHOD:-mhc-unconstrained-h-res-4streams-from-alm-cap-fixed-ckpt-lr-to-${MAX_ITERS}iter}"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

echo ""
echo "================================================================"
echo " Continuing small mHC ALM-nonnegative-cap training"
echo " checkpoint:        $CHECKPOINT_PATH"
echo " H_res mode:        unconstrained AdamW"
echo " target max_iters:  $MAX_ITERS"
echo " fixed learning rate: checkpoint final optimizer LR"
echo " eval_iters:        $EVAL_ITERS"
echo " log_interval:      $LOG_INTERVAL"
echo " checkpoint_interval: $CHECKPOINT_INTERVAL"
echo " n_gpus:            $N_GPUS"
echo " wandb_project:     $WANDB_PROJECT"
echo " wandb_run_name:    $WANDB_RUN_NAME"
echo " out_prefix_method: $OUT_PREFIX_METHOD"
echo " H grad norms:      enabled"
echo "================================================================"

common_args=(
  "$TRAIN_CONFIG"
  "$MODEL_CONFIG"
  "$METHOD_CONFIG"
  --init_from=continue_ckpt
  --continue_ckpt_path="$CHECKPOINT_PATH"
  --continue_load_optimizer=True
  --continue_reset_iter=False
  --continue_reset_best_val_loss=False
  --continue_fixed_lr_from_ckpt=True
  --continue_override_mhc_h_res_mode=True
  --mhc_h_res_mode=unconstrained
  --max_iters="$MAX_ITERS"
  --eval_iters="$EVAL_ITERS"
  --log_interval="$LOG_INTERVAL"
  --checkpoint_interval="$CHECKPOINT_INTERVAL"
  --wandb_log=True
  --wandb_project="$WANDB_PROJECT"
  --wandb_run_name="$WANDB_RUN_NAME"
  --out_prefix_method="$OUT_PREFIX_METHOD"
  --wandb_log_h_matrix_grad_norm=True
  --wandb_log_layer_stats=False
  --wandb_log_layer_cosine=False
  --wandb_log_layer_grad_norm=False
  --wandb_log_layer_activation_norm=False
  --wandb_log_layer_activation_grad_norm=False
  --mhc_log_constraint_errors=False
)

if [[ "$N_GPUS" -gt 0 ]]; then
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$N_GPUS" \
    train.py "${common_args[@]}"
else
  "$PYTHON_BIN" train.py "${common_args[@]}"
fi

echo ""
echo "================================================================"
echo " ALM-nonnegative-cap checkpoint continuation completed"
echo "================================================================"
