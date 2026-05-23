eval_interval = 100
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = True
wandb_project = 'mhc-lite'
out_prefix_dataset = "owt"

dataset = 'openwebtext'
gradient_accumulation_steps = 128
batch_size = 2
block_size = 1024

dtype = 'bfloat16'
