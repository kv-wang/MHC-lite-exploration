hyper_conn_type = "mhc"
hyper_conn_n = 4
wandb_notes = "continue 4x4 mHC from checkpoint with fixed checkpoint LR"

init_from = "continue_ckpt"
continue_ckpt_path = "out-owt-medium-mhc-num-streams-4/ckpt.pt"
continue_load_optimizer = True
continue_reset_iter = True
continue_reset_best_val_loss = True
# The exact fixed LR is inferred from the checkpoint optimizer param group.
continue_fixed_lr_from_ckpt = True

out_prefix_method = "mhc-num-streams-4-continue-fixed-lr"
wandb_run_name = "mhc-medium-mhc-num-streams-4-continue-fixed-lr"
