from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import GPT, GPTConfig


OPENWEBTEXT_DATA_DIR = Path(
    "/root/autodl-tmp/MHC-backup-20260413-023555/examples/nanogpt/data/openwebtext"
)


MODEL_ARG_KEYS = {
    field
    for field in getattr(GPTConfig, "__dataclass_fields__", {})
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paired-evaluate a checkpoint before and after zeroing all "
            "off-diagonal entries in each mHC static H_res block."
        )
    )
    parser.add_argument("--ckpt-path", required=True, help="Input checkpoint path.")
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=None,
        help="Number of random validation batches. Defaults to checkpoint config eval_iters, or 200.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Evaluation device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("float32", "bfloat16", "float16"),
        help="Autocast dtype. Defaults to checkpoint config dtype.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Torch RNG seed for random val windows. Defaults to checkpoint config seed.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help=(
            "Number of paired random validation-window repeats. Each repeat uses "
            "the same windows for original and zero-offdiag evaluation."
        ),
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1000003,
        help="Seed increment between paired repeats.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Validation micro-batch size. Defaults to checkpoint config batch_size.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Sequence block size. Defaults to checkpoint model_args block_size.",
    )
    parser.add_argument(
        "--save-modified-ckpt",
        action="store_true",
        help="Save the in-memory modified checkpoint to --modified-ckpt-path.",
    )
    parser.add_argument(
        "--modified-ckpt-path",
        default=None,
        help="Output checkpoint path when --save-modified-ckpt is set.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON file path for the summary.",
    )
    return parser.parse_args()


def strip_compile_prefix(state_dict):
    unwanted_prefix = "_orig_mod."
    fixed = {}
    for key, value in state_dict.items():
        if key.startswith(unwanted_prefix):
            key = key[len(unwanted_prefix):]
        fixed[key] = value
    return fixed


def as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    if value is None:
        return None
    return float(value)


def data_dir_for_dataset(dataset: str):
    if dataset == "openwebtext":
        return OPENWEBTEXT_DATA_DIR
    return REPO_ROOT / "data" / dataset


def make_eval_indices(data_dir, block_size, batch_size, eval_iters, seed):
    data_path = data_dir / "val.bin"
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(
        len(data) - block_size,
        (eval_iters, batch_size),
        generator=generator,
    )


def get_batch_from_indices(data, ix, block_size, device):
    x = torch.stack([
        torch.from_numpy((data[i:i + block_size]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64))
        for i in ix
    ])
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def build_model_args(checkpoint, config, data_dir):
    model_args = dict(checkpoint.get("model_args", {}))
    for key in MODEL_ARG_KEYS:
        if key not in model_args and key in config:
            model_args[key] = config[key]

    meta_path = data_dir / "meta.pkl"
    if "vocab_size" not in model_args or model_args["vocab_size"] is None:
        if meta_path.exists():
            with meta_path.open("rb") as f:
                model_args["vocab_size"] = pickle.load(f)["vocab_size"]
        else:
            model_args["vocab_size"] = 50304

    return {key: model_args[key] for key in MODEL_ARG_KEYS if key in model_args}


def zero_h_res_offdiag(state_dict):
    modified = 0
    stats = []
    for key, value in state_dict.items():
        if not key.endswith(".static_alpha"):
            continue
        if value.ndim != 2:
            continue

        rows, cols = value.shape
        num_input_views = cols - rows
        if num_input_views < 0:
            continue

        h_res = value[:, num_input_views:]
        if h_res.shape != (rows, rows):
            continue

        with torch.no_grad():
            work = h_res.detach().float()
            eye = torch.eye(rows, device=work.device, dtype=torch.bool)
            offdiag = work.masked_select(~eye)
            offdiag_abs = offdiag.abs()
            diag_before = work.diag().clone()
            mask = torch.eye(rows, device=h_res.device, dtype=h_res.dtype)
            h_res.mul_(mask)
            work_after = h_res.detach().float()

        stats.append({
            "key": key,
            "rows": rows,
            "offdiag_abs_max_before": float(offdiag_abs.max().item()) if offdiag_abs.numel() else 0.0,
            "offdiag_abs_mean_before": float(offdiag_abs.mean().item()) if offdiag_abs.numel() else 0.0,
            "diag_abs_change_max": float((work_after.diag() - diag_before).abs().max().item()),
        })
        modified += 1

    return modified, stats


def filter_state_dict_for_model(state_dict, model):
    target_state = model.state_dict()
    filtered = {}
    dropped = []
    mismatched = []

    for key, value in state_dict.items():
        if key not in target_state:
            dropped.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            mismatched.append((key, tuple(value.shape), tuple(target_state[key].shape)))
            continue
        filtered[key] = value

    if mismatched:
        details = "\n".join(
            f"{key}: checkpoint {ckpt_shape} vs model {model_shape}"
            for key, ckpt_shape, model_shape in mismatched[:20]
        )
        raise RuntimeError(f"Shape mismatches while loading checkpoint:\n{details}")

    return filtered, dropped


@torch.no_grad()
def estimate_val_loss(model, eval_indices, data_dir, block_size, device, ctx):
    model.eval()
    eval_iters = eval_indices.shape[0]
    data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    losses = torch.empty(eval_iters, dtype=torch.float32)
    for idx in range(eval_iters):
        x, y = get_batch_from_indices(data, eval_indices[idx], block_size, device)
        with ctx:
            _, loss = model(x, y)
        losses[idx] = loss.detach().float().cpu()
    return losses


@torch.no_grad()
def estimate_repeated_val_loss(
    model,
    repeat_seeds,
    eval_iters,
    batch_size,
    data_dir,
    block_size,
    device,
    ctx,
):
    losses = []
    for repeat_seed in repeat_seeds:
        eval_indices = make_eval_indices(
            data_dir=data_dir,
            block_size=block_size,
            batch_size=batch_size,
            eval_iters=eval_iters,
            seed=repeat_seed,
        )
        losses.append(
            estimate_val_loss(
                model=model,
                eval_indices=eval_indices,
                data_dir=data_dir,
                block_size=block_size,
                device=device,
                ctx=ctx,
            )
        )
    return torch.cat(losses, dim=0)


def load_model_for_eval(model_args, state_dict, device):
    model = GPT(GPTConfig(**model_args))
    filtered_state_dict, dropped_keys = filter_state_dict_for_model(state_dict, model)
    load_result = model.load_state_dict(filtered_state_dict, strict=False)
    if load_result.missing_keys:
        raise RuntimeError(f"Missing keys while loading checkpoint: {load_result.missing_keys[:20]}")
    model.to(device)
    return model, dropped_keys


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt_path)
    if args.save_modified_ckpt and args.modified_ckpt_path is None:
        raise ValueError("--modified-ckpt-path is required with --save-modified-ckpt")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available")

    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})
    dataset = config.get("dataset", "openwebtext")
    data_dir = data_dir_for_dataset(dataset)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    eval_iters = args.eval_iters
    if eval_iters is None:
        eval_iters = int(config.get("eval_iters", 200))
    if eval_iters < 1:
        raise ValueError("--eval-iters must be >= 1")
    if args.num_repeats < 1:
        raise ValueError("--num-repeats must be >= 1")

    seed = args.seed
    if seed is None:
        seed = int(config.get("seed", 1337))
    repeat_seeds = [seed + args.seed_stride * i for i in range(args.num_repeats)]
    dtype = args.dtype or config.get("dtype", "float32")
    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError(f"Unsupported dtype: {dtype}")
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = nullcontext() if "cuda" not in device or dtype == "float32" else torch.amp.autocast(
        device_type="cuda",
        dtype=ptdtype,
    )

    model_args = build_model_args(checkpoint, config, data_dir)
    if args.block_size is not None:
        model_args["block_size"] = args.block_size
    block_size = int(model_args["block_size"])
    batch_size = int(args.batch_size if args.batch_size is not None else config.get("batch_size", 2))

    state_dict = strip_compile_prefix(checkpoint["model"])
    original_model, dropped_keys = load_model_for_eval(model_args, state_dict, device)
    original_losses = estimate_repeated_val_loss(
        model=original_model,
        repeat_seeds=repeat_seeds,
        eval_iters=eval_iters,
        batch_size=batch_size,
        data_dir=data_dir,
        block_size=block_size,
        device=device,
        ctx=ctx,
    )
    del original_model
    if "cuda" in device:
        torch.cuda.empty_cache()

    modified_count, h_res_stats = zero_h_res_offdiag(state_dict)
    zero_model, zero_dropped_keys = load_model_for_eval(model_args, state_dict, device)
    if zero_dropped_keys != dropped_keys:
        raise RuntimeError("Original and zero-offdiag model loading dropped different state keys")
    zero_losses = estimate_repeated_val_loss(
        model=zero_model,
        repeat_seeds=repeat_seeds,
        eval_iters=eval_iters,
        batch_size=batch_size,
        data_dir=data_dir,
        block_size=block_size,
        device=device,
        ctx=ctx,
    )
    del zero_model
    if "cuda" in device:
        torch.cuda.empty_cache()

    delta_losses = zero_losses - original_losses
    delta_std = float(delta_losses.std(unbiased=False).item())
    total_eval_batches = int(delta_losses.numel())
    delta_sem = delta_std / (total_eval_batches ** 0.5)
    delta_ci95 = 1.96 * delta_sem

    offdiag_max_values = [row["offdiag_abs_max_before"] for row in h_res_stats]
    offdiag_mean_values = [row["offdiag_abs_mean_before"] for row in h_res_stats]
    summary = {
        "ckpt_path": str(ckpt_path),
        "iter_num": checkpoint.get("iter_num"),
        "best_val_loss": as_float(checkpoint.get("best_val_loss")),
        "original_recomputed_val_loss": float(original_losses.mean().item()),
        "original_recomputed_val_loss_std": float(original_losses.std(unbiased=False).item()),
        "zero_offdiag_val_loss": float(zero_losses.mean().item()),
        "zero_offdiag_val_loss_std": float(zero_losses.std(unbiased=False).item()),
        "delta_zero_minus_original_mean": float(delta_losses.mean().item()),
        "delta_zero_minus_original_std": delta_std,
        "delta_zero_minus_original_sem": delta_sem,
        "delta_zero_minus_original_95ci": delta_ci95,
        "delta_zero_minus_original_min": float(delta_losses.min().item()),
        "delta_zero_minus_original_max": float(delta_losses.max().item()),
        "num_batches_zero_better": int((delta_losses < 0).sum().item()),
        "num_batches_original_better": int((delta_losses > 0).sum().item()),
        "num_batches_equal": int((delta_losses == 0).sum().item()),
        "eval_iters": eval_iters,
        "num_repeats": args.num_repeats,
        "total_eval_batches": total_eval_batches,
        "batch_size": batch_size,
        "block_size": block_size,
        "device": device,
        "dtype": dtype,
        "seed": seed,
        "repeat_seeds": repeat_seeds,
        "num_static_alpha_modified": modified_count,
        "offdiag_abs_max_before_max": max(offdiag_max_values) if offdiag_max_values else 0.0,
        "offdiag_abs_max_before_mean": float(np.mean(offdiag_max_values)) if offdiag_max_values else 0.0,
        "offdiag_abs_mean_before_mean": float(np.mean(offdiag_mean_values)) if offdiag_mean_values else 0.0,
        "dropped_unexpected_state_keys": len(dropped_keys),
        "dropped_unexpected_state_key_examples": dropped_keys[:20],
        "h_res_stats": h_res_stats,
    }

    if args.save_modified_ckpt:
        checkpoint["model"] = state_dict
        output_path = Path(args.modified_ckpt_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_path)
        summary["modified_ckpt_path"] = str(output_path)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(summary, f, indent=2)

    printable = dict(summary)
    printable.pop("h_res_stats", None)
    print(json.dumps(printable, indent=2))
    print("\nPer-H_res offdiag stats:")
    for row in h_res_stats:
        print(
            f"{row['key']}: "
            f"offdiag_abs_max_before={row['offdiag_abs_max_before']:.6g}, "
            f"offdiag_abs_mean_before={row['offdiag_abs_mean_before']:.6g}"
        )


if __name__ == "__main__":
    main()
