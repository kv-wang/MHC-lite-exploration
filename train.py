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
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
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
hyper_conn_reduce_stream_mode = "sum" # "sum" or "mean" for final multi-stream reduction
mhc_gate_fn = "sigmoid"    # "softmax" or "sigmoid" for H_pre/H_post (mhc/mhc_lite only)
mhc_zero_init_pre_post_logits = False # True = initialize H_pre/H_post static logits to all zeros (mhc only)
mhc_identity_h_res = False # True = H_res fixed to I, no stream mixing (mhc/mhc_lite only)
mhc_h_res_mode = "sinkhorn" # "sinkhorn", "admm", or "cayley" for H_res (mhc only)
mhc_admm_iters = 20        # ADMM steps for H_res when mhc_h_res_mode="admm"
mhc_admm_rho = 1.0         # ADMM penalty for H_res when mhc_h_res_mode="admm"
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
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
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
    init_process_group(backend=backend)
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
    data_dir = '/root/autodl-tmp/nanoMoE-mhc/data/openwebtext'
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
    mhc_gate_fn=mhc_gate_fn,
    mhc_zero_init_pre_post_logits=mhc_zero_init_pre_post_logits,
    mhc_identity_h_res=mhc_identity_h_res,
    mhc_h_res_mode=mhc_h_res_mode,
    mhc_admm_iters=mhc_admm_iters,
    mhc_admm_rho=mhc_admm_rho,
    mhc_lite_h_res_mode=mhc_lite_h_res_mode,
    mhc_lite_ns_steps=mhc_lite_ns_steps,
    mhc_lite_method=mhc_lite_method,
    mhc_lite_perm_topk=mhc_lite_perm_topk,
    mhc_lite_block_size=mhc_lite_block_size,
)
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
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
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

def _get_hc_modules(block):
    if hasattr(block, 'hc_attn') and hasattr(block, 'hc_mlp'):
        return block.hc_attn, block.hc_mlp
    return None, None

_has_hc_modules = any(
    _get_hc_modules(block)[0] is not None for block in raw_model.transformer.h
)


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
    if iter_num % eval_interval == 0 and master_process:
        losses, layer_cosine = estimate_loss()
        avg_train_loss = [np.mean(train_losses), np.std(train_losses)] if len(train_losses) > 0 else [0, 0]
        avg_grad_norm_val = [np.mean(grad_norms), np.std(grad_norms)] if len(grad_norms) > 0 else [0, 0]

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
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # training step with gradient accumulation
    optimizer.zero_grad(set_to_none=True)
    if wandb_log and (
        wandb_log_layer_activation_norm
        or wandb_log_layer_activation_grad_norm
    ) and _has_hc_modules:
        reset_layer_activation_norms()

    train_time_start = time.time()
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    # gradient clipping + collect diagnostics
    grad_norm = -1.
    layer_grad_norms = None
    layer_activation_norms = None
    layer_activation_grad_norms = None
    if grad_clip > 0.:
        scaler.unscale_(optimizer)
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
    train_time_end = time.time()
    d_train_time = train_time_end - train_time_start
    tokens_per_sec = tokens_per_iter / d_train_time

    tpss.append(tokens_per_sec)
    train_losses.append(float(loss))
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

if ddp:
    destroy_process_group()
