#!/usr/bin/env bash
#
# Run mHC stream-count ablations for default mHC, identity H_res,
# and optional ADMM / Cayley H_res on the medium model preset.
#
# Usage:
#   ./stream_ablation.sh
#   N_GPUS=8 ./stream_ablation.sh
#   N_GPUS=0 ./stream_ablation.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export WANDB_API_KEY="${WANDB_API_KEY:-2eaf5d3e15da1d68fbce32137184e1eaba001ff6}"
#export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

N_GPUS="${N_GPUS:-4}"
TRAIN_CONFIG="config/train_owt.py"
MODEL_CONFIG="config/medium_model.py"
MHC_GATE_FN="${MHC_GATE_FN:-sigmoid}"
HYPER_CONN_REDUCE_STREAM_MODE="${HYPER_CONN_REDUCE_STREAM_MODE:-mean}"
HYPER_CONN_EXPAND_STREAM_MODE="${HYPER_CONN_EXPAND_STREAM_MODE:-repeat}"
MHC_ZERO_INIT_PRE_POST_LOGITS="${MHC_ZERO_INIT_PRE_POST_LOGITS:-False}"
WANDB_PROJECT_MHC="${WANDB_PROJECT_MHC:-ablation_num_streams_medium}"
STREAM_COUNTS=(8)

run_one() {
  local num_streams="$1"
  local variant="$2"
  local method_config
  local run_name
  local out_prefix_method
  local wandb_project
  local method_desc
  local gate_tag="$MHC_GATE_FN"
  local reduce_tag="reduce-${HYPER_CONN_REDUCE_STREAM_MODE}"
  local expand_tag="expand-${HYPER_CONN_EXPAND_STREAM_MODE}"
  local init_tag
  local extra_args=()

  if [[ "$MHC_ZERO_INIT_PRE_POST_LOGITS" == "True" ]]; then
    init_tag="zero-init"
  else
    init_tag="main-init"
  fi

  case "$variant" in
    mhc)
      method_config="config/with_mhc.py"
      run_name="mhc-medium-num-streams-${num_streams}-${gate_tag}-${init_tag}-${reduce_tag}-${expand_tag}"
      out_prefix_method="mhc-num-streams-${num_streams}-${gate_tag}-${init_tag}-${reduce_tag}-${expand_tag}"
      wandb_project="$WANDB_PROJECT_MHC"
      method_desc="mHC"
      ;;
    identity)
      method_config="config/with_mhc.py"
      run_name="mhc-medium-num-streams-${num_streams}-idH-${gate_tag}-${init_tag}-${reduce_tag}-${expand_tag}"
      out_prefix_method="mhc-num-streams-${num_streams}-idH-${gate_tag}-${init_tag}-${reduce_tag}-${expand_tag}"
      wandb_project="$WANDB_PROJECT_MHC"
      method_desc="mHC (identity H_res)"
      extra_args+=(--mhc_identity_h_res=True)
      ;;
    admm)
      method_config="config/with_mhc_admm.py"
      run_name="mhc-medium-num-streams-${num_streams}-admm-${gate_tag}-${init_tag}-${reduce_tag}-${expand_tag}"
      out_prefix_method="mhc-num-streams-${num_streams}-admm-${gate_tag}-${init_tag}-${reduce_tag}-${expand_tag}"
      wandb_project="$WANDB_PROJECT_MHC"
      method_desc="mHC (ADMM H_res)"
      extra_args+=(--mhc_h_res_mode=admm)
      ;;
    cayley)
      method_config="config/with_mhc_cayley.py"
      run_name="mhc-medium-num-streams-${num_streams}-cayley-${gate_tag}-${init_tag}-${reduce_tag}-${expand_tag}"
      out_prefix_method="mhc-num-streams-${num_streams}-cayley-${gate_tag}-${init_tag}-${reduce_tag}-${expand_tag}"
      wandb_project="$WANDB_PROJECT_MHC"
      method_desc="mHC (Cayley H_res)"
      extra_args+=(--mhc_h_res_mode=cayley)
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 1
      ;;
  esac

  echo ""
  echo "================================================================"
  echo " Running: mHC medium stream ablation"
  echo " model:           medium"
  echo " method:          $method_desc"
  echo " num_streams:     $num_streams"
  echo " mhc_gate_fn:    $MHC_GATE_FN"
  echo " reduce_stream:   $HYPER_CONN_REDUCE_STREAM_MODE"
  echo " expand_stream:   $HYPER_CONN_EXPAND_STREAM_MODE"
  echo " zero_init_pre:   $MHC_ZERO_INIT_PRE_POST_LOGITS"
  echo " wandb_project:   $wandb_project"
  echo " wandb_run_name:  $run_name"
  echo " out_prefix:      $out_prefix_method"
  echo "================================================================"

  if [[ "$N_GPUS" -gt 0 ]]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" train.py \
      "$TRAIN_CONFIG" "$MODEL_CONFIG" "$method_config" \
      --num_streams="$num_streams" \
      --mhc_gate_fn="$MHC_GATE_FN" \
      --mhc_zero_init_pre_post_logits="$MHC_ZERO_INIT_PRE_POST_LOGITS" \
      --hyper_conn_reduce_stream_mode="$HYPER_CONN_REDUCE_STREAM_MODE" \
      --hyper_conn_expand_stream_mode="$HYPER_CONN_EXPAND_STREAM_MODE" \
      "${extra_args[@]}" \
      --wandb_project="$wandb_project" \
      --wandb_run_name="$run_name" \
      --out_prefix_method="$out_prefix_method" \
      --wandb_log_layer_stats=False \
      --wandb_log_layer_cosine=False \
      --wandb_log_layer_grad_norm=False \
      --wandb_log_layer_activation_norm=False \
      --wandb_log_layer_activation_grad_norm=False \
      --mhc_log_constraint_errors=False
  else
    python train.py \
      "$TRAIN_CONFIG" "$MODEL_CONFIG" "$method_config" \
      --num_streams="$num_streams" \
      --mhc_gate_fn="$MHC_GATE_FN" \
      --mhc_zero_init_pre_post_logits="$MHC_ZERO_INIT_PRE_POST_LOGITS" \
      --hyper_conn_reduce_stream_mode="$HYPER_CONN_REDUCE_STREAM_MODE" \
      --hyper_conn_expand_stream_mode="$HYPER_CONN_EXPAND_STREAM_MODE" \
      "${extra_args[@]}" \
      --wandb_project="$wandb_project" \
      --wandb_run_name="$run_name" \
      --out_prefix_method="$out_prefix_method" \
      --wandb_log_layer_stats=False \
      --wandb_log_layer_cosine=False \
      --wandb_log_layer_grad_norm=False \
      --wandb_log_layer_activation_norm=False \
      --wandb_log_layer_activation_grad_norm=False \
      --mhc_log_constraint_errors=False
  fi
}

for num_streams in "${STREAM_COUNTS[@]}"; do
  #run_one "$num_streams" mhc
  #run_one "$num_streams" identity
  run_one "$num_streams" admm
  #run_one "$num_streams" cayley
done

echo ""
echo "================================================================"
echo " All stream ablation runs completed!"
echo "================================================================"
