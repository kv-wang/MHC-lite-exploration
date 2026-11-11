#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY must be set in the environment." >&2
  exit 2
fi

# Match the W&B endpoint used by this repository's existing HC launch scripts.
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
TRAIN_PYTHON="${TRAIN_PYTHON:-/root/miniconda3/envs/nanogpt/bin/python}"
if [[ ! -x "$TRAIN_PYTHON" ]]; then
  TRAIN_PYTHON="python"
fi

MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-mhc-lite}"
N_STREAMS="${N_STREAMS:-4}"

run_model() {
  local model="$1"
  local run_name="hc-${model}-${N_STREAMS}streams-single-gpu"
  local out_prefix="hc-${N_STREAMS}streams-single-gpu"

  echo "[$(date -Is)] Starting ${model}: ${MAX_ITERS} iterations on GPU ${CUDA_VISIBLE_DEVICES}"
  "$TRAIN_PYTHON" train.py \
    config/train_owt.py "config/${model}_model.py" config/with_hc.py \
    --hyper_conn_type=hc \
    --hyper_conn_n="$N_STREAMS" \
    --max_iters="$MAX_ITERS" \
    --lr_decay_iters="$MAX_ITERS" \
    --eval_iters="$EVAL_ITERS" \
    --wandb_log=True \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_name="$run_name" \
    --wandb_group=hc-medium-small-single-gpu \
    --out_prefix_method="$out_prefix" \
    --wandb_log_layer_stats=False \
    --wandb_log_layer_cosine=False \
    --wandb_log_layer_grad_norm=False \
    --wandb_log_h_matrix_grad_norm=False \
    --wandb_log_layer_activation_norm=False \
    --wandb_log_layer_activation_grad_norm=False \
    --mhc_log_constraint_errors=False
  echo "[$(date -Is)] Finished ${model}"
}

run_model medium
run_model small
echo "[$(date -Is)] Both HC training runs completed."
