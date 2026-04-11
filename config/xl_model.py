wandb_group = "xl"
out_prefix_model = "xl"

n_layer = 48
n_head = 25
n_embd = 1600
dropout = 0.0

learning_rate = 3e-4
min_lr = 3e-5
max_iters = 10000
lr_decay_iters = 10000
warmup_iters = 200
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
