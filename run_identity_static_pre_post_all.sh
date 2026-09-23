#!/usr/bin/env bash
#
# Train mHC identity runs with model-specific stream counts and no dynamic
# H_pre/H_post parameters. Runs small, medium, then large sequentially on two
# GPUs by default.

set -e

export DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
N_GPUS="${N_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

TRAIN_CONFIG="${TRAIN_CONFIG:-config/train_owt.py}"
METHOD_CONFIG="${METHOD_CONFIG:-config/with_mhc.py}"
N_STREAMS_SMALL="${N_STREAMS_SMALL:-4}"
N_STREAMS_MEDIUM="${N_STREAMS_MEDIUM:-13828}"
N_STREAMS_LARGE="${N_STREAMS_LARGE:-18436}"
REDUCE_STREAM_MODE="${REDUCE_STREAM_MODE:-4mean}"
MAX_ITERS="${MAX_ITERS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-final_experiments}"
WANDB_LOG_H_MATRIX_GRAD_NORM="${WANDB_LOG_H_MATRIX_GRAD_NORM:-False}"

EXTRA_ARGS=("$@")

run_one() {
  local model_name="$1"
  local model_config="config/${model_name}_model.py"
  local n_streams
  case "$model_name" in
    small)  n_streams="$N_STREAMS_SMALL" ;;
    medium) n_streams="$N_STREAMS_MEDIUM" ;;
    large)  n_streams="$N_STREAMS_LARGE" ;;
    *)
      echo "Unsupported model: $model_name" >&2
      return 1
      ;;
  esac
  local wandb_project="${WANDB_PROJECT_PREFIX}_${model_name}"
  local run_name="mhc-${model_name}-identity-static-pre-post-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"
  local out_prefix="mhc-identity-static-pre-post-${n_streams}streams-reduce-${REDUCE_STREAM_MODE}-${MAX_ITERS}iter"

  echo ""
  echo "================================================================"
  echo " Running ${model_name} mHC identity static-H_pre/H_post"
  echo " cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
  echo " n_gpus:              $N_GPUS"
  echo " train_config:        $TRAIN_CONFIG"
  echo " model_config:        $model_config"
  echo " method_config:       $METHOD_CONFIG"
  echo " n_streams:           $n_streams"
  echo " reduce_mode:         $REDUCE_STREAM_MODE"
  echo " max_iters:           $MAX_ITERS"
  echo " eval_iters:          $EVAL_ITERS"
  echo " checkpoint_interval: $CHECKPOINT_INTERVAL"
  echo " wandb_project:       $wandb_project"
  echo " wandb_run_name:      $run_name"
  echo " out_prefix_method:   $out_prefix"
  echo "================================================================"

  torchrun --standalone --nproc_per_node="$N_GPUS" train.py \
    "$TRAIN_CONFIG" \
    "$model_config" \
    "$METHOD_CONFIG" \
    --hyper_conn_n="$n_streams" \
    --hyper_conn_reduce_stream_mode="$REDUCE_STREAM_MODE" \
    --mhc_train_only_pre_post=True \
    --mhc_disable_dynamic_pre_post=True \
    --compile=False \
    --max_iters="$MAX_ITERS" \
    --eval_iters="$EVAL_ITERS" \
    --checkpoint_interval="$CHECKPOINT_INTERVAL" \
    --wandb_log=True \
    --wandb_project="$wandb_project" \
    --wandb_run_name="$run_name" \
    --out_prefix_method="$out_prefix" \
    --wandb_log_layer_stats=False \
    --wandb_log_layer_cosine=False \
    --wandb_log_layer_grad_norm=False \
    --wandb_log_h_matrix_grad_norm="$WANDB_LOG_H_MATRIX_GRAD_NORM" \
    --wandb_log_layer_activation_norm=False \
    --wandb_log_layer_activation_grad_norm=False \
    --mhc_log_constraint_errors=False \
    --mhc_log_h_res_diag_mass=False \
    "${EXTRA_ARGS[@]}"
}

for model_name in small medium large; do
  run_one "$model_name"
done

echo ""
echo "================================================================"
echo " model-specific identity static-H_pre/H_post runs completed"
echo "================================================================"
