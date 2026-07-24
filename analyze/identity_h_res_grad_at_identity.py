from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import GPT, GPTConfig


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    unwanted_prefix = "_orig_mod."
    return {
        key[len(unwanted_prefix):] if key.startswith(unwanted_prefix) else key: value
        for key, value in state_dict.items()
    }


def parse_component(name: str) -> tuple[int, str] | None:
    parts = name.split(".")
    if len(parts) < 4:
        return None
    if parts[0] != "transformer" or parts[1] != "h":
        return None
    if parts[3] not in {"hc_attn", "hc_mlp"}:
        return None
    try:
        layer = int(parts[2])
    except ValueError:
        return None
    component = "attn" if parts[3] == "hc_attn" else "mlp"
    return layer, component


def build_model(checkpoint: dict, device: str) -> GPT:
    checkpoint_model_args = checkpoint["model_args"]
    config_keys = {field.name for field in fields(GPTConfig)}
    model_args = {key: value for key, value in checkpoint_model_args.items() if key in config_keys}
    if "mhc_disable_dynamic_h_res" not in model_args:
        model_args["mhc_disable_dynamic_h_res"] = False
    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(strip_compile_prefix(checkpoint["model"]))
    model.to(device)
    return model


def iter_hc_modules(model: GPT):
    for name, module in model.named_modules():
        parsed = parse_component(name)
        if parsed is None or not hasattr(module, "static_alpha"):
            continue
        yield name, parsed[0], parsed[1], module


def get_gamma(module) -> torch.Tensor:
    log_scale = getattr(module, "h_res_offdiag_log_scale", None)
    if log_scale is None:
        gamma = torch.as_tensor(
            getattr(module, "mhc_h_res_offdiag_init_scale", 0.05),
            device=module.static_alpha.device,
            dtype=module.static_alpha.dtype,
        )
    else:
        gamma = log_scale.to(device=module.static_alpha.device, dtype=module.static_alpha.dtype).exp()
    return gamma


def h_res_slice(module) -> tuple[slice, slice]:
    rows = module.static_alpha.shape[0]
    cols = module.static_alpha.shape[1]
    num_input_views = cols - rows
    if num_input_views < 0:
        raise ValueError(f"invalid static_alpha shape {tuple(module.static_alpha.shape)}")
    return slice(None), slice(num_input_views, None)


def zero_h_res_to_identity(model: GPT):
    with torch.no_grad():
        for _, _, _, module in iter_hc_modules(model):
            row_slice, col_slice = h_res_slice(module)
            module.static_alpha[row_slice, col_slice].zero_()


def offdiag_mask_like(h: torch.Tensor) -> torch.Tensor:
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"expected square matrix, got {tuple(h.shape)}")
    return ~torch.eye(h.shape[0], dtype=torch.bool, device=h.device)


def collect_grad_rows(model: GPT, suffix: str) -> dict[str, dict[str, float]]:
    rows = {}
    for name, layer, component, module in iter_hc_modules(model):
        row_slice, col_slice = h_res_slice(module)
        grad = module.static_alpha.grad
        if grad is None:
            continue
        raw_grad = grad[row_slice, col_slice].detach().float()
        offdiag = offdiag_mask_like(raw_grad)
        gamma = get_gamma(module).detach().float()
        h_eff_grad = raw_grad[offdiag] / gamma.float()
        raw_grad_offdiag = raw_grad[offdiag]
        key = f"{layer:03d}_{component}"
        rows[key] = {
            "layer": layer,
            "component": component,
            f"raw_w_grad_norm_{suffix}": float(raw_grad_offdiag.norm().item()),
            f"h_eff_grad_norm_{suffix}": float(h_eff_grad.norm().item()),
            "gamma": float(gamma.item()),
        }
    return rows


def collect_update_rows(model: GPT) -> dict[str, dict[str, float]]:
    rows = {}
    for name, layer, component, module in iter_hc_modules(model):
        row_slice, col_slice = h_res_slice(module)
        h_logits = module.static_alpha[row_slice, col_slice].detach().float()
        offdiag = offdiag_mask_like(h_logits)
        raw_update = h_logits[offdiag]
        h_eff = module.effective_static_h_res().detach().float()
        eye = torch.eye(h_eff.shape[0], dtype=h_eff.dtype, device=h_eff.device)
        h_eff_update = (h_eff - eye)[offdiag]
        key = f"{layer:03d}_{component}"
        rows[key] = {
            "layer": layer,
            "component": component,
            "adamw_raw_w_update_norm": float(raw_update.norm().item()),
            "adamw_h_eff_update_norm": float(h_eff_update.norm().item()),
            "adamw_h_eff_update_max_abs": float(h_eff_update.abs().max().item()) if h_eff_update.numel() else 0.0,
        }
    return rows


def get_batch(
    data: np.memmap,
    *,
    block_size: int,
    batch_size: int,
    device: str,
    device_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy((data[i:i + block_size]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64))
        for i in ix
    ])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure per-layer identity_tanh_offdiag H_res gradients at the identity point."
    )
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--output-dir", default="analyze/identity_h_res_grad_at_identity_results")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--data-dir", default="", help="Defaults to data/<dataset> from checkpoint config.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="", choices=["", "float32", "bfloat16", "float16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--use-checkpoint-batch", type=str2bool, default=False)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--apply-grad-clip", type=str2bool, default=True)
    parser.add_argument("--load-optimizer-state", type=str2bool, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    config = checkpoint.get("config", {})
    model_args = checkpoint.get("model_args", {})

    dataset = config.get("dataset", "openwebtext")
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data" / dataset
    data_path = data_dir / f"{args.split}.bin"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    device = args.device
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype_name = args.dtype or config.get("dtype", "float32")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda" and dtype_name != "float32"
        else nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and dtype_name == "float16"))

    batch_size = int(config.get("batch_size", 1)) if args.use_checkpoint_batch else args.batch_size
    grad_accum_steps = (
        int(config.get("gradient_accumulation_steps", 1))
        if args.use_checkpoint_batch
        else args.grad_accum_steps
    )
    block_size = args.block_size or int(model_args.get("block_size", config.get("block_size", 1024)))
    seed = args.seed or int(config.get("seed", 1337))
    learning_rate = float(config.get("learning_rate", 0.0))
    weight_decay = float(config.get("weight_decay", 0.0))
    beta1 = float(config.get("beta1", 0.9))
    beta2 = float(config.get("beta2", 0.95))
    grad_clip = float(config.get("grad_clip", 0.0))

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_model(checkpoint, device=device)
    model.train()
    if getattr(model.config, "mhc_h_res_mode", "") not in {"identity_tanh_offdiag", "identity_clip_offdiag"}:
        raise ValueError(f"checkpoint mode is {model.config.mhc_h_res_mode!r}, expected identity_*_offdiag")
    zero_h_res_to_identity(model)

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if args.load_optimizer_state:
        if "optimizer" not in checkpoint:
            raise ValueError("--load-optimizer-state=True but checkpoint has no optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])

    optimizer.zero_grad(set_to_none=True)
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    loss_sum = 0.0
    for _ in range(grad_accum_steps):
        x, y = get_batch(
            data,
            block_size=block_size,
            batch_size=batch_size,
            device=device,
            device_type=device_type,
        )
        with ctx:
            _, loss = model(x, y)
            scaled_loss = loss / grad_accum_steps
        loss_sum += float(loss.detach().float().item()) / grad_accum_steps
        scaler.scale(scaled_loss).backward()

    if scaler.is_enabled():
        scaler.unscale_(optimizer)

    preclip_rows = collect_grad_rows(model, "preclip")
    global_grad_norm_preclip = math.nan
    global_grad_norm_postclip = math.nan
    if args.apply_grad_clip and grad_clip > 0:
        global_grad_norm_preclip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item())
        global_grad_norm_postclip = min(global_grad_norm_preclip, grad_clip)
    postclip_rows = collect_grad_rows(model, "postclip")

    lrs = [float(group["lr"]) for group in optimizer.param_groups]
    lr = lrs[0] if lrs else learning_rate
    scaler.step(optimizer)
    scaler.update()
    update_rows = collect_update_rows(model)

    merged = []
    keys = sorted(set(preclip_rows) | set(postclip_rows) | set(update_rows))
    for key in keys:
        row = {}
        row.update(preclip_rows.get(key, {}))
        row.update(postclip_rows.get(key, {}))
        row.update(update_rows.get(key, {}))
        row["lr"] = lr
        if "h_eff_grad_norm_postclip" in row:
            row["lr_times_h_eff_grad_norm_postclip"] = lr * row["h_eff_grad_norm_postclip"]
        if "raw_w_grad_norm_postclip" in row:
            row["lr_times_raw_w_grad_norm_postclip"] = lr * row["raw_w_grad_norm_postclip"]
        merged.append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.parent.name or ckpt_path.stem
    csv_path = out_dir / f"{stem}_identity_grad_update.csv"
    json_path = out_dir / f"{stem}_identity_grad_update_summary.json"
    write_csv(csv_path, merged)

    summary = {
        "checkpoint": str(ckpt_path),
        "iter_num": checkpoint.get("iter_num"),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "mode": getattr(model.config, "mhc_h_res_mode", None),
        "split": args.split,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "block_size": block_size,
        "tokens": batch_size * grad_accum_steps * block_size,
        "loss": loss_sum,
        "lr": lr,
        "dtype": dtype_name,
        "device": device,
        "load_optimizer_state": args.load_optimizer_state,
        "apply_grad_clip": args.apply_grad_clip,
        "grad_clip": grad_clip,
        "global_grad_norm_preclip": global_grad_norm_preclip,
        "global_grad_norm_postclip": global_grad_norm_postclip,
        "csv_path": str(csv_path),
    }
    with json_path.open("w") as f:
        json.dump({"summary": summary, "rows": merged}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
