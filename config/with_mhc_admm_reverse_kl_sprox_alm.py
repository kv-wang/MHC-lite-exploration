hyper_conn_type = "mhc"
hyper_conn_n = 4

mhc_h_res_mode = "admm_reverse_kl_sprox_alm"
mhc_admm_iters = 250
mhc_admm_rho = 10.0
mhc_adapter_base_streams = 4
mhc_adapter_cap = 1.0
mhc_adapter_admm_input_floor = 1e-30
mhc_adapter_admm_checkpoint = False

wandb_notes = "mhc-admm-reverse-kl-sprox-alm"
out_prefix_method = "mhc-admm-reverse-kl-sprox-alm"
