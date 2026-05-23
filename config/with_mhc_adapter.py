hyper_conn_type = "mhc"
hyper_conn_n = 32
hyper_conn_reduce_stream_mode = "sum"
hyper_conn_expand_stream_mode = "repeat_base_zero_rest"

init_from = "mhc_adapter"
mhc_adapter_ckpt_path = "out-owt-medium-mhc-num-streams-4/ckpt.pt"
mhc_adapter_base_streams = 4
mhc_adapter_epsilon = 1e-6
mhc_adapter_cross_logit = -20.0
mhc_adapter_new_block_logit = -20.0
mhc_adapter_inactive_logit = -20.0

mhc_h_res_mode = "adapter_epsilon"

wandb_notes = "mhc-adapter-strict-eq-eps1e-6"
out_prefix_method = "mhc-adapter-strict-eq-eps1e-6"
