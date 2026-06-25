"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from datetime import timedelta
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from hyper_conn.mhc import sinkhorn_knopps
from pprint import pprint
import warnings
import json
import glob

# suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# ----- hyper conn start -----
hyper_conn_type = "none" # none, hc, mhc, mhc_lite, attn_res
hyper_conn_n = 1 # num_streams
hyper_conn_reduce_stream_mode = "sum" # "sum", "mean", "4mean", or "softmax_4mean" for final multi-stream reduction
hyper_conn_expand_stream_mode = "repeat" # "repeat", "split", or "repeat_base_zero_rest" for initial multi-stream expansion
mhc_gate_fn = "sigmoid"    # "softmax" or "sigmoid" for H_pre/H_post (mhc/mhc_lite only)
mhc_zero_init_pre_post_logits = False # True = initialize H_pre/H_post static logits to all zeros (mhc only)
mhc_identity_h_res = False # True = H_res fixed to I, no stream mixing (mhc/mhc_lite only)
mhc_h_res_mode = "sinkhorn" # "sinkhorn", "admm", "admm_reverse_kl", "admm_reverse_kl_sprox_alm", "admm_l2", "alm_signed", "alm_nonnegative", "alm_signed_sprox", "alm_spectral_sprox", "cayley", "adapter_epsilon", "adapter_cap", or "adapter_cap_admm" for H_res (mhc only)
mhc_admm_iters = 20        # ADMM steps for H_res when mhc_h_res_mode uses ADMM
mhc_admm_rho = 1.0         # ADMM penalty for H_res when mhc_h_res_mode uses ADMM
mhc_admm_dual_step = 0.5   # S-prox-ALM dual update step for S-prox H_res modes
mhc_admm_prox_weight = None # S-prox-ALM proximal weight; None uses mhc_admm_rho
mhc_admm_smooth_beta = 0.5 # S-prox-ALM auxiliary smoothing factor
mhc_admm_step_scale = 1.0  # S-prox-ALM primal step multiplier
mhc_log_constraint_errors = False # log projected H_res row/column constraint errors during training
mhc_constraint_log_interval = 100 # iteration interval for H_res constraint error logging
mhc_adapter_ckpt_path = "out-owt-medium-mhc-num-streams-4/ckpt.pt"
mhc_adapter_base_streams = 4
mhc_adapter_epsilon = 0.1
mhc_adapter_cap = 1.0
mhc_adapter_cross_logit = -40.0
mhc_adapter_new_block_logit = -40.0
mhc_adapter_inactive_logit = -40.0
mhc_adapter_h_res_init_mode = "copy_logits" # "copy_logits" or "sinkhorn_projected_log"
mhc_adapter_projected_init_iters = 20
mhc_adapter_projected_init_floor = 1e-30
mhc_adapter_admm_input_mode = "raw_logits" # "raw_logits" or "sinkhorn_base_log"
mhc_adapter_admm_input_floor = 1e-30
mhc_adapter_admm_checkpoint = False
mhc_lite_h_res_mode = "doubly_stochastic" # "doubly_stochastic" or "newton_schulz" for H_res (mhc_lite only)
mhc_lite_ns_steps = 5      # Newton-Schulz steps for H_res when mhc_lite_h_res_mode="newton_schulz"
mhc_lite_method = "base"   # base, selective, depth_attn, block_attn, block_depth
mhc_lite_perm_topk = 0     # selective only; 0 defaults to num_streams
mhc_lite_block_size = 4    # block_attn only; measured in sublayers
# ----- hyper conn end -----

# I/O
seed = 1337
out_prefix_dataset = ""
out_prefix_model = ""
out_prefix_method = "residual"
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # legacy config; checkpoints are saved only once at the final iteration
init_from = 'scratch' # 'scratch', 'resume', 'continue_ckpt', 'mhc_adapter', or 'gpt2*'
continue_ckpt_path = "out-owt-medium-mhc-num-streams-4/ckpt.pt"
continue_load_optimizer = True
continue_reset_iter = True
continue_reset_best_val_loss = True
continue_fixed_lr_from_ckpt = True
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'exp' # 'run' + str(time.time())
wandb_group = "default"
wandb_notes = ""
wandb_log_layer_stats = True
wandb_log_layer_cosine = True
wandb_log_layer_grad_norm = True
wandb_log_layer_activation_norm = True
wandb_log_layer_activation_grad_norm = True
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
# auto-generate descriptive wandb_run_name and out_prefix_method
if wandb_run_name == 'exp':
    if hyper_conn_type == "none":
        wandb_run_name = "residual"
    elif hyper_conn_type == "hc":
        wandb_run_name = "hc"
    elif hyper_conn_type in ("mhc", "mhc_lite"):
        tag = hyper_conn_type.replace("_", "-")
        if hyper_conn_type == "mhc_lite" and mhc_lite_method != "base":
            tag += f"-{mhc_lite_method.replace('_', '-')}"
        tag += f"-{mhc_gate_fn}"
        if hyper_conn_type == "mhc" and mhc_h_res_mode != "sinkhorn":
            tag += f"-{mhc_h_res_mode.replace('_', '-')}"
        if hyper_conn_type == "mhc_lite" and mhc_lite_h_res_mode != "doubly_stochastic":
            tag += f"-{mhc_lite_h_res_mode.replace('_', '-')}"
        if mhc_identity_h_res:
            tag += "-idH"
        if hyper_conn_expand_stream_mode != "repeat":
            tag += f"-expand-{hyper_conn_expand_stream_mode}"
        wandb_run_name = tag
    elif hyper_conn_type == "attn_res":
        wandb_run_name = "attn-res"
    else:
        wandb_run_name = hyper_conn_type

if out_prefix_method == "residual" and hyper_conn_type != "none":
    out_prefix_method = wandb_run_name

config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    ddp_timeout_seconds = int(os.environ.get('DDP_TIMEOUT', '1800'))
    init_process_group(backend=backend, timeout=timedelta(seconds=ddp_timeout_seconds))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(
        project = wandb_project , 
        name    = wandb_run_name,
        group   = wandb_group,
        config  = config , 
        notes   = wandb_notes,
    )

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

out_dir = f"out-{out_prefix_dataset}-{out_prefix_model}-{out_prefix_method}"
if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
if dataset == 'openwebtext':
    data_dir = '/root/autodl-tmp/MHC-backup-20260413-023555/examples/nanogpt/data/openwebtext'
else:
    data_dir = os.path.join('data', dataset)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Batch sampling (simple random contiguous windows)




# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
vocab_size = meta_vocab_size

# model init
model_args = dict(
    n_layer=n_layer, 
    n_head=n_head, 
    n_embd=n_embd, 
    block_size=block_size,
    bias=bias, 
    vocab_size=vocab_size, 
    dropout=dropout,
    hyper_conn_n=hyper_conn_n,
    hyper_conn_type=hyper_conn_type,
    hyper_conn_reduce_stream_mode=hyper_conn_reduce_stream_mode,
    hyper_conn_expand_stream_mode=hyper_conn_expand_stream_mode,
    mhc_gate_fn=mhc_gate_fn,
    mhc_zero_init_pre_post_logits=mhc_zero_init_pre_post_logits,
    mhc_identity_h_res=mhc_identity_h_res,
    mhc_h_res_mode=mhc_h_res_mode,
    mhc_admm_iters=mhc_admm_iters,
    mhc_admm_rho=mhc_admm_rho,
    mhc_admm_dual_step=mhc_admm_dual_step,
    mhc_admm_prox_weight=mhc_admm_prox_weight,
    mhc_admm_smooth_beta=mhc_admm_smooth_beta,
    mhc_admm_step_scale=mhc_admm_step_scale,
    mhc_adapter_base_streams=mhc_adapter_base_streams,
    mhc_adapter_epsilon=mhc_adapter_epsilon,
    mhc_adapter_cap=mhc_adapter_cap,
    mhc_adapter_admm_input_mode=mhc_adapter_admm_input_mode,
    mhc_adapter_admm_input_floor=mhc_adapter_admm_input_floor,
    mhc_adapter_admm_checkpoint=mhc_adapter_admm_checkpoint,
    mhc_lite_h_res_mode=mhc_lite_h_res_mode,
    mhc_lite_ns_steps=mhc_lite_ns_steps,
    mhc_lite_method=mhc_lite_method,
    mhc_lite_perm_topk=mhc_lite_perm_topk,
    mhc_lite_block_size=mhc_lite_block_size,
)

def strip_compile_prefix(state_dict):
    unwanted_prefix = '_orig_mod.'
    fixed = {}
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            k = k[len(unwanted_prefix):]
        fixed[k] = v
    return fixed

def adapt_mhc_4_to_n_state_dict(
    target_state,
    source_state,
    base_streams,
    target_streams,
    cross_logit,
    new_block_logit,
    inactive_logit,
    h_res_init_mode,
    projected_init_iters,
    projected_init_floor,
):
    valid_h_res_init_modes = {"copy_logits", "sinkhorn_projected_log"}
    if h_res_init_mode not in valid_h_res_init_modes:
        raise ValueError(f"Invalid mhc_adapter_h_res_init_mode: {h_res_init_mode}")
    if projected_init_iters < 1:
        raise ValueError("mhc_adapter_projected_init_iters must be >= 1")
    if projected_init_floor <= 0:
        raise ValueError("mhc_adapter_projected_init_floor must be > 0")

    adapted = {k: v.clone() for k, v in target_state.items()}
    copied_equal = 0
    copied_adapted = 0

    for k, v in source_state.items():
        if k in adapted and adapted[k].shape == v.shape:
            adapted[k].copy_(v)
            copied_equal += 1

    for k, old in source_state.items():
        if k not in adapted or adapted[k].shape == old.shape:
            continue

        new = adapted[k]

        if k.endswith(".static_alpha"):
            old_streams = old.shape[0]
            new_streams = new.shape[0]
            old_views = old.shape[1] - old_streams
            new_views = new.shape[1] - new_streams
            if old_streams != base_streams or new_streams != target_streams or old_views != new_views:
                continue

            new[:base_streams, :new_views].copy_(old[:base_streams, :old_views])
            new[base_streams:, :new_views] = inactive_logit
            old_h_res = old[:base_streams, old_views:]
            if h_res_init_mode == "sinkhorn_projected_log":
                projected_h_res = sinkhorn_knopps(
                    old_h_res.to(dtype=new.dtype),
                    iters=projected_init_iters,
                )
                projected_h_res = projected_h_res.clamp_min(projected_init_floor).log()
                new[:base_streams, new_views:new_views + base_streams].copy_(projected_h_res)
            else:
                new[:base_streams, new_views:new_views + base_streams].copy_(old_h_res)
            new[:base_streams, new_views + base_streams:] = cross_logit
            new[base_streams:, new_views:new_views + base_streams] = cross_logit
            new[base_streams:, new_views + base_streams:] = new_block_logit
            copied_adapted += 1

        elif k.endswith(".dynamic_alpha_fn"):
            old_streams = base_streams
            new_streams = target_streams
            old_dim = old.shape[0] // old_streams
            new_dim = new.shape[0] // new_streams
            old_t = old.shape[1] // old_streams
            new_t = new.shape[1] // new_streams
            old_views = old_t - old_streams
            new_views = new_t - new_streams
            if old_dim != new_dim or old_views != new_views:
                continue

            old_rows = old_streams * old_dim
            new.zero_()
            for source_idx in range(base_streams):
                old_col = source_idx * old_t
                new_col = source_idx * new_t
                new[:old_rows, new_col:new_col + new_views].copy_(
                    old[:old_rows, old_col:old_col + old_views]
                )
                new[:old_rows, new_col + new_views:new_col + new_views + base_streams].copy_(
                    old[:old_rows, old_col + old_views:old_col + old_t]
                )
            copied_adapted += 1

        elif k.endswith(".static_beta"):
            if old.shape[0] != base_streams or new.shape[0] != target_streams:
                continue
            new[:base_streams].copy_(old)
            new[base_streams:] = inactive_logit
            copied_adapted += 1

        elif k.endswith(".dynamic_beta_fn"):
            old_dim = old.shape[0] // base_streams
            new_dim = new.shape[0] // target_streams
            if old_dim != new_dim or old.shape[1] != base_streams or new.shape[1] != target_streams:
                continue
            old_rows = base_streams * old_dim
            new.zero_()
            new[:old_rows, :base_streams].copy_(old)
            copied_adapted += 1

        elif k.endswith(".norm.gamma"):
            old_dim = old.shape[0] // base_streams
            new_dim = new.shape[0] // target_streams
            if old_dim != new_dim:
                continue
            old_len = base_streams * old_dim
            new[:old_len].copy_(old)
            copied_adapted += 1

    print(f"mHC adapter copied {copied_equal} same-shape tensors and adapted {copied_adapted} tensors")
    return adapted

def get_checkpoint_lr(checkpoint):
    optimizer_state = checkpoint.get('optimizer')
    if isinstance(optimizer_state, dict):
        param_groups = optimizer_state.get('param_groups', [])
        lrs = [
            float(group['lr'])
            for group in param_groups
            if isinstance(group, dict) and 'lr' in group
        ]
        if lrs:
            return lrs[0]

    ckpt_config = checkpoint.get('config', {})
    ckpt_iter = checkpoint.get('iter_num', 0)
    ckpt_learning_rate = float(ckpt_config.get('learning_rate', learning_rate))
    ckpt_min_lr = float(ckpt_config.get('min_lr', min_lr))
    ckpt_decay_lr = bool(ckpt_config.get('decay_lr', decay_lr))
    ckpt_warmup_iters = int(ckpt_config.get('warmup_iters', warmup_iters))
    ckpt_lr_decay_iters = int(ckpt_config.get('lr_decay_iters', lr_decay_iters))

    if not ckpt_decay_lr:
        return ckpt_learning_rate
    if ckpt_iter < ckpt_warmup_iters:
        return ckpt_learning_rate * (ckpt_iter + 1) / (ckpt_warmup_iters + 1)
    if ckpt_iter > ckpt_lr_decay_iters:
        return ckpt_min_lr
    decay_span = ckpt_lr_decay_iters - ckpt_warmup_iters
    if decay_span <= 0:
        return ckpt_min_lr
    decay_ratio = (ckpt_iter - ckpt_warmup_iters) / decay_span
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return ckpt_min_lr + coeff * (ckpt_learning_rate - ckpt_min_lr)

if master_process:
    print ("="*100)
    for k, v in model_args.items():
        print (f"{k} = {v}")
    print ("="*100)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = strip_compile_prefix(checkpoint['model'])
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from == 'continue_ckpt':
    print(f"Continuing training from checkpoint {continue_ckpt_path}")
    checkpoint = torch.load(continue_ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in model_args:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = strip_compile_prefix(checkpoint['model'])
    model.load_state_dict(state_dict)

    loaded_best_val_loss = checkpoint.get('best_val_loss', 1e9)
    if loaded_best_val_loss is None:
        loaded_best_val_loss = 1e9
    iter_num = 0 if continue_reset_iter else checkpoint.get('iter_num', 0)
    best_val_loss = 1e9 if continue_reset_best_val_loss else loaded_best_val_loss
    if continue_fixed_lr_from_ckpt:
        checkpoint_lr = get_checkpoint_lr(checkpoint)
        learning_rate = checkpoint_lr
        min_lr = checkpoint_lr
        decay_lr = False
        config.update({
            'learning_rate': learning_rate,
            'min_lr': min_lr,
            'decay_lr': decay_lr,
        })
        if master_process:
            print(f"Freezing learning rate at checkpoint lr: {checkpoint_lr:.8g}")
            if wandb_log:
                wandb.config.update({
                    'learning_rate': learning_rate,
                    'min_lr': min_lr,
                    'decay_lr': decay_lr,
                }, allow_val_change=True)
elif init_from == 'mhc_adapter':
    print(f"Initializing mHC adapter from {mhc_adapter_ckpt_path}")
    if hyper_conn_type != "mhc":
        raise ValueError("init_from='mhc_adapter' requires hyper_conn_type='mhc'")
    if hyper_conn_n < mhc_adapter_base_streams:
        raise ValueError("hyper_conn_n must be >= mhc_adapter_base_streams")

    checkpoint = torch.load(mhc_adapter_ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    source_streams = checkpoint_model_args.get('hyper_conn_n')
    if source_streams != mhc_adapter_base_streams:
        raise ValueError(
            f"adapter checkpoint has {source_streams} streams, expected {mhc_adapter_base_streams}"
        )

    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    source_state = strip_compile_prefix(checkpoint['model'])
    adapted_state = adapt_mhc_4_to_n_state_dict(
        model.state_dict(),
        source_state,
        base_streams=mhc_adapter_base_streams,
        target_streams=hyper_conn_n,
        cross_logit=mhc_adapter_cross_logit,
        new_block_logit=mhc_adapter_new_block_logit,
        inactive_logit=mhc_adapter_inactive_logit,
        h_res_init_mode=mhc_adapter_h_res_init_mode,
        projected_init_iters=mhc_adapter_projected_init_iters,
        projected_init_floor=mhc_adapter_projected_init_floor,
    )
    model.load_state_dict(adapted_state)
    iter_num = 0
    best_val_loss = 1e9
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
elif init_from == 'continue_ckpt' and continue_load_optimizer:
    if 'optimizer' not in checkpoint:
        raise ValueError("continue_load_optimizer=True but checkpoint has no optimizer state")
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ---------------------------------------------------------------------------
# Layer-level diagnostics (ported from MHC)
# ---------------------------------------------------------------------------

raw_model = model.module if ddp else model

def save_final_checkpoint(saved_iter_num):
    if not master_process:
        return
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': saved_iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    print(f"saving final checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

def _get_hc_modules(block):
    if hasattr(block, 'hc_attn') and hasattr(block, 'hc_mlp'):
        return block.hc_attn, block.hc_mlp
    return None, None

_has_hc_modules = any(
    _get_hc_modules(block)[0] is not None for block in raw_model.transformer.h
)
_use_h_res_alm_loss_training = (
    hyper_conn_type == "mhc"
    and mhc_h_res_mode == "alm_signed"
    and _has_hc_modules
)
_use_h_res_sprox_training = (
    hyper_conn_type == "mhc"
    and mhc_h_res_mode in {"alm_nonnegative", "alm_signed_sprox", "alm_spectral_sprox"}
    and _has_hc_modules
)

if mhc_log_constraint_errors and mhc_constraint_log_interval < 1:
    raise ValueError("mhc_constraint_log_interval must be >= 1")


def _iter_hc_modules():
    for block_idx, block in enumerate(raw_model.transformer.h):
        hc_attn, hc_mlp = _get_hc_modules(block)
        if hc_attn is None:
            continue
        yield block_idx, "attn", hc_attn
        yield block_idx, "mlp", hc_mlp


def reset_h_res_constraint_errors():
    for _, _, hc in _iter_hc_modules():
        if hasattr(hc, "reset_h_res_constraint_errors"):
            hc.reset_h_res_constraint_errors()


def set_h_res_constraint_error_collection(enabled):
    for _, _, hc in _iter_hc_modules():
        if hasattr(hc, "collect_h_res_constraint_errors"):
            hc.collect_h_res_constraint_errors = enabled


def collect_h_res_constraint_errors():
    rows = []
    for block_idx, component, hc in _iter_hc_modules():
        if not hasattr(hc, "get_h_res_constraint_errors"):
            continue
        stats = hc.get_h_res_constraint_errors()
        if stats is None:
            continue
        layer_index = block_idx * 2 + (0 if component == "attn" else 1)
        rows.append((layer_index, component, stats))
    return rows


def build_h_res_constraint_error_log(rows):
    if not rows:
        return {}

    row_max = max(stats["row_err_max"] for _, _, stats in rows)
    col_max = max(stats["col_err_max"] for _, _, stats in rows)
    nonneg_violation_max = max(stats["nonneg_violation_max"] for _, _, stats in rows)
    spectral_violation_max = max(stats.get("spectral_violation_max", 0.) for _, _, stats in rows)
    row_mean = float(np.mean([stats["row_err_mean"] for _, _, stats in rows]))
    col_mean = float(np.mean([stats["col_err_mean"] for _, _, stats in rows]))
    nonneg_violation_mean = float(np.mean([stats["nonneg_violation_mean"] for _, _, stats in rows]))
    spectral_violation_mean = float(np.mean([stats.get("spectral_violation_mean", 0.) for _, _, stats in rows]))
    count = sum(stats["count"] for _, _, stats in rows)

    log = {
        "train/h_res_constraint/row_err_max": row_max,
        "train/h_res_constraint/col_err_max": col_max,
        "train/h_res_constraint/nonneg_violation_max": nonneg_violation_max,
        "train/h_res_constraint/spectral_violation_max": spectral_violation_max,
        "train/h_res_constraint/row_err_mean": row_mean,
        "train/h_res_constraint/col_err_mean": col_mean,
        "train/h_res_constraint/nonneg_violation_mean": nonneg_violation_mean,
        "train/h_res_constraint/spectral_violation_mean": spectral_violation_mean,
        "train/h_res_constraint/num_samples": count,
    }
    for layer_index, component, stats in rows:
        key = f"layer_{layer_index}_{component}"
        log[f"train/h_res_constraint/row_err_max/{key}"] = stats["row_err_max"]
        log[f"train/h_res_constraint/col_err_max/{key}"] = stats["col_err_max"]
        log[f"train/h_res_constraint/nonneg_violation_max/{key}"] = stats["nonneg_violation_max"]
        log[f"train/h_res_constraint/spectral_violation_max/{key}"] = stats.get("spectral_violation_max", 0.)
    return log


def reset_h_res_alm_forward_losses():
    for _, _, hc in _iter_hc_modules():
        if hasattr(hc, "reset_h_res_alm_forward_loss"):
            hc.reset_h_res_alm_forward_loss()


def collect_h_res_alm_loss():
    total = None
    for _, _, hc in _iter_hc_modules():
        if not hasattr(hc, "get_h_res_alm_loss"):
            continue
        loss = hc.get_h_res_alm_loss()
        if loss is None:
            continue
        total = loss if total is None else total + loss
    return total


def reset_h_res_alm_dual_update_accumulators():
    for _, _, hc in _iter_hc_modules():
        if hasattr(hc, "reset_h_res_alm_dual_update_accumulators"):
            hc.reset_h_res_alm_dual_update_accumulators()


def update_h_res_alm_duals():
    rows = []
    for block_idx, component, hc in _iter_hc_modules():
        if not hasattr(hc, "update_h_res_alm_duals"):
            continue
        stats = hc.update_h_res_alm_duals()
        if stats is None:
            continue
        layer_index = block_idx * 2 + (0 if component == "attn" else 1)
        rows.append((layer_index, component, stats))
    return rows


def prepare_h_res_sprox_steps():
    for _, _, hc in _iter_hc_modules():
        if hasattr(hc, "prepare_h_res_sprox_step"):
            hc.prepare_h_res_sprox_step()


def step_h_res_sprox_alm():
    rows = []
    for block_idx, component, hc in _iter_hc_modules():
        if not hasattr(hc, "step_h_res_sprox_alm"):
            continue
        stats = hc.step_h_res_sprox_alm()
        if stats is None:
            continue
        layer_index = block_idx * 2 + (0 if component == "attn" else 1)
        rows.append((layer_index, component, stats))
    return rows


def build_h_res_alm_update_log(rows, alm_loss):
    if not rows:
        return {}

    row_err_max = max(stats["row_err_max"] for _, _, stats in rows)
    col_err_max = max(stats["col_err_max"] for _, _, stats in rows)
    nonneg_violation_max = max(stats["nonneg_violation_max"] for _, _, stats in rows)
    spectral_violation_max = max(stats.get("spectral_violation_max", 0.) for _, _, stats in rows)
    spectral_norm_max = max(stats.get("spectral_norm", 0.) for _, _, stats in rows)
    row_err_mean = float(np.mean([stats["row_err_mean"] for _, _, stats in rows]))
    col_err_mean = float(np.mean([stats["col_err_mean"] for _, _, stats in rows]))
    nonneg_violation_mean = float(np.mean([stats["nonneg_violation_mean"] for _, _, stats in rows]))
    spectral_violation_mean = float(np.mean([stats.get("spectral_violation_mean", 0.) for _, _, stats in rows]))
    spectral_norm_mean = float(np.mean([stats.get("spectral_norm", 0.) for _, _, stats in rows]))
    row_dual_norm_max = max(stats["row_dual_norm"] for _, _, stats in rows)
    col_dual_norm_max = max(stats["col_dual_norm"] for _, _, stats in rows)
    nonneg_dual_norm_max = max(stats["nonneg_dual_norm"] for _, _, stats in rows)
    row_dual_norm_mean = float(np.mean([stats["row_dual_norm"] for _, _, stats in rows]))
    col_dual_norm_mean = float(np.mean([stats["col_dual_norm"] for _, _, stats in rows]))
    nonneg_dual_norm_mean = float(np.mean([stats["nonneg_dual_norm"] for _, _, stats in rows]))
    lm_grad_norm_max = max(stats.get("lm_grad_norm", 0.) for _, _, stats in rows)
    lm_grad_norm_mean = float(np.mean([stats.get("lm_grad_norm", 0.) for _, _, stats in rows]))
    count = sum(stats["count"] for _, _, stats in rows)

    log = {
        "train/h_res_alm/loss": alm_loss,
        "train/h_res_alm/row_res_max": row_err_max,
        "train/h_res_alm/col_res_max": col_err_max,
        "train/h_res_alm/nonneg_violation_max": nonneg_violation_max,
        "train/h_res_alm/spectral_violation_max": spectral_violation_max,
        "train/h_res_alm/spectral_norm_max": spectral_norm_max,
        "train/h_res_alm/row_res_mean": row_err_mean,
        "train/h_res_alm/col_res_mean": col_err_mean,
        "train/h_res_alm/nonneg_violation_mean": nonneg_violation_mean,
        "train/h_res_alm/spectral_violation_mean": spectral_violation_mean,
        "train/h_res_alm/spectral_norm_mean": spectral_norm_mean,
        "train/h_res_alm/row_dual_norm_max": row_dual_norm_max,
        "train/h_res_alm/col_dual_norm_max": col_dual_norm_max,
        "train/h_res_alm/nonneg_dual_norm_max": nonneg_dual_norm_max,
        "train/h_res_alm/lm_grad_norm_max": lm_grad_norm_max,
        "train/h_res_alm/row_dual_norm_mean": row_dual_norm_mean,
        "train/h_res_alm/col_dual_norm_mean": col_dual_norm_mean,
        "train/h_res_alm/nonneg_dual_norm_mean": nonneg_dual_norm_mean,
        "train/h_res_alm/lm_grad_norm_mean": lm_grad_norm_mean,
        "train/h_res_alm/num_samples": count,
    }
    for layer_index, component, stats in rows:
        key = f"layer_{layer_index}_{component}"
        log[f"train/h_res_alm/row_res_max/{key}"] = stats["row_err_max"]
        log[f"train/h_res_alm/col_res_max/{key}"] = stats["col_err_max"]
        log[f"train/h_res_alm/nonneg_violation_max/{key}"] = stats["nonneg_violation_max"]
        log[f"train/h_res_alm/spectral_violation_max/{key}"] = stats.get("spectral_violation_max", 0.)
        log[f"train/h_res_alm/spectral_norm/{key}"] = stats.get("spectral_norm", 0.)
        log[f"train/h_res_alm/row_dual_norm/{key}"] = stats["row_dual_norm"]
        log[f"train/h_res_alm/col_dual_norm/{key}"] = stats["col_dual_norm"]
        log[f"train/h_res_alm/nonneg_dual_norm/{key}"] = stats["nonneg_dual_norm"]
        log[f"train/h_res_alm/lm_grad_norm/{key}"] = stats.get("lm_grad_norm", 0.)
    return log


def collect_hc_layer_stats():
    layer_count = len(raw_model.transformer.h) * 2
    layer_stats = {}
    for block_idx, block in enumerate(raw_model.transformer.h):
        hc_attn, hc_mlp = _get_hc_modules(block)
        if hc_attn is None:
            continue
        for sub_idx, hc in enumerate((hc_attn, hc_mlp)):
            if not hasattr(hc, "last_stats"):
                continue
            layer_index = block_idx * 2 + sub_idx
            for key, value in hc.last_stats.items():
                layer_stats.setdefault(key, [None] * layer_count)
                layer_stats[key][layer_index] = value.item()
    return layer_stats


def build_layer_table(layer_stats):
    if not layer_stats:
        return None
    keys = sorted(layer_stats.keys())
    layer_count = max(len(v) for v in layer_stats.values())
    table = wandb.Table(columns=["layer"] + keys)
    for i in range(layer_count):
        row_vals = []
        for key in keys:
            values = layer_stats[key]
            val = values[i] if i < len(values) else None
            row_vals.append(val)
        if all(v is None for v in row_vals):
            continue
        table.add_data(i, *row_vals)
    return table


def _activation_norm_hook(module, _, output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        return
    activation_norm = torch.linalg.vector_norm(output.detach().float()).item()
    module._activation_norm_sum = (
        getattr(module, "_activation_norm_sum", 0.0) + activation_norm
    )
    module._activation_norm_count = getattr(module, "_activation_norm_count", 0) + 1
    if output.requires_grad:
        output.register_hook(lambda grad: _activation_grad_norm_hook(module, grad))


def _activation_grad_norm_hook(module, grad):
    if not torch.is_tensor(grad):
        return
    activation_grad_norm = torch.linalg.vector_norm(grad.detach().float()).item()
    module._activation_grad_norm_sum = (
        getattr(module, "_activation_grad_norm_sum", 0.0) + activation_grad_norm
    )
    module._activation_grad_norm_count = (
        getattr(module, "_activation_grad_norm_count", 0) + 1
    )


def reset_layer_activation_norms():
    for block in raw_model.transformer.h:
        hc_attn, hc_mlp = _get_hc_modules(block)
        if hc_attn is None:
            continue
        for module in (hc_attn, hc_mlp):
            module._activation_norm_sum = 0.0
            module._activation_norm_count = 0
            module._activation_grad_norm_sum = 0.0
            module._activation_grad_norm_count = 0


def collect_layer_activation_norms():
    rows = []
    for block_idx, block in enumerate(raw_model.transformer.h):
        hc_attn, hc_mlp = _get_hc_modules(block)
        if hc_attn is None:
            continue
        for sub_idx, (name, module) in enumerate(
            (("attn", hc_attn), ("mlp", hc_mlp))
        ):
            count = getattr(module, "_activation_norm_count", 0)
            if count <= 0:
                continue
            layer_index = block_idx * 2 + sub_idx
            activation_norm = getattr(module, "_activation_norm_sum", 0.0) / count
            rows.append((layer_index, name, activation_norm))
    return rows


def build_layer_scalar_log(prefix, rows):
    if not rows:
        return {}
    return {
        f"{prefix}/layer_{layer_index}_{component}": value
        for layer_index, component, value in rows
    }


def collect_layer_activation_grad_norms():
    rows = []
    for block_idx, block in enumerate(raw_model.transformer.h):
        hc_attn, hc_mlp = _get_hc_modules(block)
        if hc_attn is None:
            continue
        for sub_idx, (name, module) in enumerate(
            (("attn", hc_attn), ("mlp", hc_mlp))
        ):
            count = getattr(module, "_activation_grad_norm_count", 0)
            if count <= 0:
                continue
            layer_index = block_idx * 2 + sub_idx
            activation_grad_norm = (
                getattr(module, "_activation_grad_norm_sum", 0.0) / count
            )
            rows.append((layer_index, name, activation_grad_norm))
    return rows


def _module_grad_norm(module):
    total_sq = None
    for param in module.parameters():
        if param.grad is None:
            continue
        grad_sq = param.grad.detach().float().pow(2).sum()
        total_sq = grad_sq if total_sq is None else total_sq + grad_sq
    if total_sq is None:
        return None
    return torch.sqrt(total_sq).item()


def collect_layer_grad_norms():
    rows = []
    for block_idx, block in enumerate(raw_model.transformer.h):
        hc_attn, hc_mlp = _get_hc_modules(block)
        if hc_attn is None:
            continue
        for sub_idx, (name, module) in enumerate(
            (("attn", hc_attn), ("mlp", hc_mlp))
        ):
            grad_norm = _module_grad_norm(module)
            if grad_norm is None:
                continue
            layer_index = block_idx * 2 + sub_idx
            rows.append((layer_index, name, grad_norm))
    return rows


def forward_with_layer_cosine(x, y):
    sims = []
    prev = [None]
    handles = []

    def hook(_, __, output):
        out = output.detach()
        if prev[0] is not None:
            prev_flat = prev[0].reshape(-1, prev[0].shape[-1])
            out_flat = out.reshape(-1, out.shape[-1])
            sim = F.cosine_similarity(prev_flat, out_flat, dim=-1).mean()
            sims.append(sim)
        prev[0] = out

    for block in raw_model.transformer.h:
        handles.append(block.register_forward_hook(hook))

    with ctx:
        _, loss = model(x, y)

    for handle in handles:
        handle.remove()

    sims = [s.item() for s in sims]
    return loss, sims


if wandb_log and _has_hc_modules and (
    wandb_log_layer_stats
    or wandb_log_layer_activation_norm
    or wandb_log_layer_activation_grad_norm
):
    for block in raw_model.transformer.h:
        hc_attn, hc_mlp = _get_hc_modules(block)
        if hc_attn is None:
            continue
        for hc in (hc_attn, hc_mlp):
            if wandb_log_layer_stats and hasattr(hc, 'collect_stats'):
                hc.collect_stats = True
            if wandb_log_layer_activation_norm or wandb_log_layer_activation_grad_norm:
                hc.register_forward_hook(_activation_norm_hook)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    out = {}
    layer_cosine = None
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            if (
                layer_cosine is None
                and wandb_log
                and wandb_log_layer_cosine
                and split == "train"
                and k == 0
            ):
                loss, layer_cosine = forward_with_layer_cosine(X, Y)
            else:
                with ctx:
                    _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out, layer_cosine


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
train_losses = []
tpss = []
grad_norms = []
while True:

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate
    if iter_num % eval_interval == 0:
        losses, layer_cosine = estimate_loss()
        avg_train_loss = [np.mean(train_losses), np.std(train_losses)] if len(train_losses) > 0 else [0, 0]
        avg_grad_norm_val = [np.mean(grad_norms), np.std(grad_norms)] if len(grad_norms) > 0 else [0, 0]

        if master_process:
            desc = f"[step {iter_num}]" + ", ".join([
                f"train loss: {losses['train']:.4f}",
                f"val loss: {losses['val']:.4f}",
                f"(avg train loss: {avg_train_loss[0]:.4f} ± {avg_train_loss[1]:.4f})",
                f"(avg grad norm: {avg_grad_norm_val[0]:.4f} ± {avg_grad_norm_val[1]:.4f})",
            ])
            print(desc)

            if wandb_log:
                eval_log = {
                    "iter": iter_num,
                    "train/loss_est": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                    "train/avg_loss": avg_train_loss[0],
                    "train/avg_gnorm": avg_grad_norm_val[0],
                }
                wandb.log(eval_log, step=iter_num)
                if wandb_log_layer_cosine and layer_cosine is not None:
                    layer_table = wandb.Table(columns=["layer", "cosine"])
                    for idx, value in enumerate(layer_cosine):
                        layer_table.add_data(idx, value)
                    wandb.log({"hc/layer_cosine": layer_table}, step=iter_num)
                if wandb_log_layer_stats and _has_hc_modules:
                    layer_stats = collect_hc_layer_stats()
                    layer_stats_table = build_layer_table(layer_stats)
                    if layer_stats_table is not None:
                        wandb.log({"hc/layer_stats": layer_stats_table}, step=iter_num)
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
    if iter_num == 0 and eval_only:
        break

    log_h_res_constraints_this_iter = (
        mhc_log_constraint_errors
        and _has_hc_modules
        and iter_num % mhc_constraint_log_interval == 0
    )
    if log_h_res_constraints_this_iter:
        reset_h_res_constraint_errors()
        set_h_res_constraint_error_collection(True)
    if _use_h_res_alm_loss_training:
        reset_h_res_alm_dual_update_accumulators()

    # training step with gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    if wandb_log and (
        wandb_log_layer_activation_norm
        or wandb_log_layer_activation_grad_norm
    ) and _has_hc_modules:
        reset_layer_activation_norms()

    train_time_start = time.time()
    h_res_alm_loss_accum = 0.
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        if _use_h_res_alm_loss_training:
            reset_h_res_alm_forward_losses()
        with ctx:
            logits, loss = model(X, Y)
            if _use_h_res_alm_loss_training:
                h_res_alm_loss = collect_h_res_alm_loss()
                if h_res_alm_loss is not None:
                    loss = loss + h_res_alm_loss
                    h_res_alm_loss_accum += h_res_alm_loss.detach().float().item() / gradient_accumulation_steps
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    h_res_constraint_errors = None
    if log_h_res_constraints_this_iter:
        set_h_res_constraint_error_collection(False)
        h_res_constraint_errors = collect_h_res_constraint_errors()

    # gradient clipping + collect diagnostics
    grad_norm = -1.
    layer_grad_norms = None
    layer_activation_norms = None
    layer_activation_grad_norms = None
    grads_unscaled = False
    if _use_h_res_sprox_training and scaler.is_enabled():
        scaler.unscale_(optimizer)
        grads_unscaled = True
    if _use_h_res_sprox_training:
        prepare_h_res_sprox_steps()
    if grad_clip > 0.:
        if not grads_unscaled:
            scaler.unscale_(optimizer)
            grads_unscaled = True
        if wandb_log and wandb_log_layer_activation_norm and _has_hc_modules:
            layer_activation_norms = collect_layer_activation_norms()
        if wandb_log and wandb_log_layer_activation_grad_norm and _has_hc_modules:
            layer_activation_grad_norms = collect_layer_activation_grad_norms()
        if wandb_log and wandb_log_layer_grad_norm and _has_hc_modules:
            layer_grad_norms = collect_layer_grad_norms()
        ret_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm = float(ret_grad_norm)

    scaler.step(optimizer)
    scaler.update()
    h_res_alm_update_log = None
    if _use_h_res_alm_loss_training:
        h_res_alm_update_log = build_h_res_alm_update_log(
            update_h_res_alm_duals(),
            h_res_alm_loss_accum,
        )
    elif _use_h_res_sprox_training:
        h_res_alm_update_log = build_h_res_alm_update_log(
            step_h_res_sprox_alm(),
            0.,
        )
    train_time_end = time.time()
    d_train_time = train_time_end - train_time_start
    tokens_per_sec = tokens_per_iter / d_train_time

    if h_res_constraint_errors is not None and master_process:
        h_res_constraint_log = build_h_res_constraint_error_log(h_res_constraint_errors)
        if h_res_constraint_log:
            print(
                f"[iter {iter_num}] h_res constraint "
                f"row_err_max: {h_res_constraint_log['train/h_res_constraint/row_err_max']:.6g}, "
                f"col_err_max: {h_res_constraint_log['train/h_res_constraint/col_err_max']:.6g}, "
                f"nonneg_violation_max: {h_res_constraint_log['train/h_res_constraint/nonneg_violation_max']:.6g}, "
                f"spectral_violation_max: {h_res_constraint_log['train/h_res_constraint/spectral_violation_max']:.6g}, "
                f"row_err_mean: {h_res_constraint_log['train/h_res_constraint/row_err_mean']:.6g}, "
                f"col_err_mean: {h_res_constraint_log['train/h_res_constraint/col_err_mean']:.6g}, "
                f"nonneg_violation_mean: {h_res_constraint_log['train/h_res_constraint/nonneg_violation_mean']:.6g}, "
                f"spectral_violation_mean: {h_res_constraint_log['train/h_res_constraint/spectral_violation_mean']:.6g}"
            )
            if wandb_log:
                wandb.log(h_res_constraint_log, step=iter_num)
    if h_res_alm_update_log and master_process and iter_num % log_interval == 0:
        print(
            f"[iter {iter_num}] h_res ALM "
            f"loss: {h_res_alm_update_log['train/h_res_alm/loss']:.6g}, "
            f"row_res_max: {h_res_alm_update_log['train/h_res_alm/row_res_max']:.6g}, "
            f"col_res_max: {h_res_alm_update_log['train/h_res_alm/col_res_max']:.6g}, "
            f"nonneg_violation_max: {h_res_alm_update_log['train/h_res_alm/nonneg_violation_max']:.6g}, "
            f"spectral_norm_max: {h_res_alm_update_log['train/h_res_alm/spectral_norm_max']:.6g}, "
            f"spectral_violation_max: {h_res_alm_update_log['train/h_res_alm/spectral_violation_max']:.6g}, "
            f"row_dual_norm_max: {h_res_alm_update_log['train/h_res_alm/row_dual_norm_max']:.6g}, "
            f"col_dual_norm_max: {h_res_alm_update_log['train/h_res_alm/col_dual_norm_max']:.6g}, "
            f"nonneg_dual_norm_max: {h_res_alm_update_log['train/h_res_alm/nonneg_dual_norm_max']:.6g}"
        )

    tpss.append(tokens_per_sec)
    train_losses.append(float(loss.detach()))
    grad_norms.append(float(grad_norm))
    if len(train_losses) > 200:
        train_losses.pop(0)
    if len(grad_norms) > 200:
        grad_norms.pop(0)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        tokens_seen = iter_num * tokens_per_iter

        desc = f"[iter {iter_num}]" + ", ".join([
            f"loss: {lossf:.4f}",
            f"tokens/sec: {np.mean(tpss):.2f} ± {np.std(tpss):.2f}" if len(tpss) > 0 else "0.00 ± 0.00",
            f"tokens seen: {tokens_seen}",
            f"grad norm: {grad_norm:.4f}",
        ])
        if wandb_log:
            log_dict = {
                "train/loss": lossf,
                "train/tokens_per_sec": np.mean(tpss) if len(tpss) > 0 else 0,
                "train/tokens_seen": tokens_seen,
                "train/grad_norm": grad_norm,
            }
            if device_type == "cuda":
                log_dict["perf/max_mem_allocated_mb"] = (
                    torch.cuda.max_memory_allocated() / 1e6
                )
            if h_res_alm_update_log:
                log_dict.update(h_res_alm_update_log)
            wandb.log(log_dict, step=iter_num)
            if wandb_log_layer_grad_norm and layer_grad_norms is not None:
                lg = build_layer_scalar_log("train/layer_parameter_grad_norm", layer_grad_norms)
                if lg:
                    wandb.log(lg, step=iter_num)
            if wandb_log_layer_activation_norm and layer_activation_norms is not None:
                lg = build_layer_scalar_log("train/layer_activation_norm", layer_activation_norms)
                if lg:
                    wandb.log(lg, step=iter_num)
            if wandb_log_layer_activation_grad_norm and layer_activation_grad_norms is not None:
                lg = build_layer_scalar_log("train/layer_activation_grad_norm", layer_activation_grad_norms)
                if lg:
                    wandb.log(lg, step=iter_num)
            if device_type == "cuda":
                torch.cuda.reset_peak_memory_stats()

        print(desc)
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if not eval_only and iter_num > 0:
    save_final_checkpoint(iter_num - 1)

if ddp:
    destroy_process_group()
