#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOMOE_DIR="${NANOMOE_DIR:-/root/autodl-tmp/nanoMoE}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data/openwebtext}"

N_GPUS="${N_GPUS:-8}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_gpt2.py}"
WANDB_PROJECT="${WANDB_PROJECT:-nano-moe-gpt2-124m-owt}"

# Use GPT-2 124M backbone dimensions from nanoMoE/config/train_gpt2.py,
# but enable MoE so TEON_moe has expert tensors to operate on.
N_EXP="${N_EXP:-8}"
TOP_K="${TOP_K:-2}"
STRIDE="${STRIDE:-2}"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

run_one() {
  local mode="$1"
  local run_name="$2"
  local out_dir="$3"
  shift 3

  echo ""
  echo "================================================================"
  echo " Running: $mode"
  echo " run_name: $run_name"
  echo " out_dir:  $out_dir"
  echo " data_dir: $DATA_DIR"
  echo "================================================================"

  cd "$NANOMOE_DIR"

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py \
      "$TRAIN_CONFIG" \
      --dataset=openwebtext \
      --data_dir="$DATA_DIR" \
      --n_exp="$N_EXP" \
      --top_k="$TOP_K" \
      --stride="$STRIDE" \
      --use_aux_loss=True \
      --use_router_z_loss=True \
      --use_switch_tfm_init=True \
      --router_use_full_prec=True \
      --wandb_project="$WANDB_PROJECT" \
      --wandb_run_name="$run_name" \
      --out_dir="$out_dir" \
      "$@"
  else
    python train.py \
      "$TRAIN_CONFIG" \
      --dataset=openwebtext \
      --data_dir="$DATA_DIR" \
      --n_exp="$N_EXP" \
      --top_k="$TOP_K" \
      --stride="$STRIDE" \
      --use_aux_loss=True \
      --use_router_z_loss=True \
      --use_switch_tfm_init=True \
      --router_use_full_prec=True \
      --wandb_project="$WANDB_PROJECT" \
      --wandb_run_name="$run_name" \
      --out_dir="$out_dir" \
      "$@"
  fi
}

run_one \
  "Muon" \
  "gpt2-124M-moe-owt-muon" \
  "out-gpt2-124m-moe-owt-muon" \
  --opt=muon

run_one \
  "Muon+TEON" \
  "gpt2-124M-moe-owt-muon-teon" \
  "out-gpt2-124m-moe-owt-muon-teon" \
  --opt=muon \
  --enable_teon=True

run_one \
  "Muon+TEON_moe" \
  "gpt2-124M-moe-owt-muon-teon-moe" \
  "out-gpt2-124m-moe-owt-muon-teon-moe" \
  --opt=muon \
  --enable_teon_moe=True

run_one \
  "Muon+TEON+TEON_moe" \
  "gpt2-124M-moe-owt-muon-teon-teon-moe" \
  "out-gpt2-124m-moe-owt-muon-teon-teon-moe" \
  --opt=muon \
  --enable_teon=True \
  --enable_teon_moe=True

echo ""
echo "================================================================"
echo " All runs completed!"
echo "================================================================"
