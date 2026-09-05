#!/usr/bin/env bash
#
# Train small mHC-Sinkhorn with randomly initialized, frozen H_res logits.
#
# Usage:
#   ./run_sinkhorn_random.sh
#   N_GPUS=2 MAX_ITERS=100 ./run_sinkhorn_random.sh
#   N_STREAMS=4 MHC_H_RES_RANDOM_INIT_LOGIT_SCALE=0.5 ./run_sinkhorn_random.sh

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-2}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
MODEL_CONFIG="${MODEL_CONFIG:-config/large_model.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc.py}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
N_STREAMS="${N_STREAMS:-4}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
WANDB_PROJECT="${WANDB_PROJECT:-final_experiments_large}"
MHC_H_RES_RANDOM_INIT_LOGIT_SCALE="${MHC_H_RES_RANDOM_INIT_LOGIT_SCALE:-1.0}"

scale_tag() {
  printf "%s" "$1" | sed 's/\./p/g'
}

SCALE_SLUG="$(scale_tag "$MHC_H_RES_RANDOM_INIT_LOGIT_SCALE")"
WANDB_RUN_NAME="mhc-large-sinkhorn-random-logits-frozen-hres-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-scale${SCALE_SLUG}-${MAX_ITERS}iter"
OUT_PREFIX_METHOD="mhc-sinkhorn-random-logits-frozen-hres-${N_STREAMS}streams-reduce-${REDUCE_STREAM_MODE}-scale${SCALE_SLUG}-${MAX_ITERS}iter"

echo ""
echo "================================================================"
echo " Running large mHC-Sinkhorn with random frozen H_res logits"
echo " train_config:      $TRAIN_CONFIG"
echo " model_config:      $MODEL_CONFIG"
echo " method_config:     $METHOD_CONFIG"
echo " n_streams:         $N_STREAMS"
echo " reduce_mode:       $REDUCE_STREAM_MODE"
echo " random_logit_scale: $MHC_H_RES_RANDOM_INIT_LOGIT_SCALE"
echo " freeze_h_res:      True"
echo " max_iters:         $MAX_ITERS"
echo " eval_iters:        $EVAL_ITERS"
echo " ckpt_interval:     $CHECKPOINT_INTERVAL"
echo " n_gpus:            $N_GPUS"
echo " wandb_project:     $WANDB_PROJECT"
echo " wandb_run_name:    $WANDB_RUN_NAME"
echo " out_prefix_method: $OUT_PREFIX_METHOD"
echo "================================================================"

common_args=(
  "$TRAIN_CONFIG"
  "$MODEL_CONFIG"
  "$METHOD_CONFIG"
  --hyper_conn_n="$N_STREAMS"
  --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE"
  --mhc_h_res_mode=sinkhorn
  --mhc_h_res_random_init_logits=True
  --mhc_h_res_random_init_logit_scale="$MHC_H_RES_RANDOM_INIT_LOGIT_SCALE"
  --mhc_freeze_h_res=True
  --max_iters="$MAX_ITERS"
  --eval_iters="$EVAL_ITERS"
  --checkpoint_interval="$CHECKPOINT_INTERVAL"
  --wandb_log=True
  --wandb_project="$WANDB_PROJECT"
  --wandb_run_name="$WANDB_RUN_NAME"
  --out_prefix_method="$OUT_PREFIX_METHOD"
  --wandb_log_layer_stats=False
  --wandb_log_layer_cosine=False
  --wandb_log_layer_grad_norm=False
  --wandb_log_h_matrix_grad_norm=False
  --wandb_log_layer_activation_norm=False
  --wandb_log_layer_activation_grad_norm=False
  --mhc_log_constraint_errors=False
)

if [[ "$N_GPUS" -gt 0 ]]; then
  torchrun --standalone --nproc_per_node="$N_GPUS" train.py "${common_args[@]}" "$@"
else
  python train.py "${common_args[@]}" "$@"
fi

echo ""
echo "================================================================"
echo " mHC-Sinkhorn random-logits training completed"
echo "================================================================"
