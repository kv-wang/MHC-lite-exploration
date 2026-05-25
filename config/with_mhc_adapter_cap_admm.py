hyper_conn_type = "mhc"
hyper_conn_n = 32
hyper_conn_reduce_stream_mode = "sum"
hyper_conn_expand_stream_mode = "repeat_base_zero_rest"

init_from = "mhc_adapter"
mhc_adapter_ckpt_path = "out-owt-medium-mhc-num-streams-4/ckpt.pt"
mhc_adapter_base_streams = 4
mhc_adapter_cap = 1.0
mhc_adapter_cross_logit = -40.0
mhc_adapter_new_block_logit = -40.0
mhc_adapter_inactive_logit = -40.0
mhc_adapter_h_res_init_mode = "copy_logits"
mhc_adapter_admm_input_mode = "raw_logits"
mhc_adapter_admm_checkpoint = True

mhc_h_res_mode = "adapter_cap_admm"
mhc_admm_iters = 50
mhc_admm_rho = 0.1

wandb_notes = "mhc-adapter-cap-admm"
out_prefix_method = "mhc-adapter-cap-admm"
