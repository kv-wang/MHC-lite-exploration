#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import GPT, GPTConfig


def strip_compile_prefix(state_dict):
    prefix = "_orig_mod."
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a checkpoint, replace selected identity H_res modules with "
            "trainable matrices initialized at identity, and train only those "
            "matrices for a short projected-SGD probe."
        )
    )
    parser.add_argument(
        "--ckpt",
        default="out-owt-large-mhc-identity-h-res-4streams-reduce-4mean-10000iter/ckpt.pt",
    )
    parser.add_argument(
        "--kkt-json",
        default=(
            "analyze/h_res_identity_kkt_results/"
            "out-owt-large-mhc-identity-h-res-4streams-reduce-4mean-10000iter_"
            "analytic_val_8x1_seed1337.json"
        ),
        help="JSON produced by verify_h_res_identity_kkt.py; failed analytic layers are trained.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset override. Defaults to checkpoint config.")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=0, help="Defaults to checkpoint block_size.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--project-each-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection-floor", type=float, default=1e-12)
    parser.add_argument("--sinkhorn-iters", type=int, default=200)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-7)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument(
        "--eval-log-interval",
        type=int,
        default=0,
        help="If > 0, evaluate the same fixed val batches every N training iterations.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def iter_mhc_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "static_h_res") and hasattr(module, "num_residual_streams"):
            yield name, module


def load_failed_module_names(kkt_json_path):
    with open(kkt_json_path, "r") as f:
        data = json.load(f)
    failed_rows = [row for row in data["layers"] if not row.get("analytic_kkt_pass", row.get("kkt_pass", False))]
    failed_names = {row["module_name"] for row in failed_rows}
    row_by_name = {row["module_name"]: row for row in failed_rows}
    return failed_names, row_by_name, data


def set_forced_h_res_params(model, failed_names, device):
    entries = []
    train_entries = []
    for ordinal, (name, module) in enumerate(iter_mhc_modules(model)):
        n = int(module.num_residual_streams) * int(getattr(module, "num_fracs", 1))
        requires_grad = name in failed_names
        param = nn.Parameter(
            torch.eye(n, device=device, dtype=torch.float32),
            requires_grad=requires_grad,
        )
        module._kkt_forced_h_res = param
        row = (ordinal, name, module, param)
        entries.append(row)
        if requires_grad:
            train_entries.append(row)
    if not entries:
        raise RuntimeError("No mHC modules were found.")
    missing = sorted(failed_names - {name for _, name, _, _ in entries})
    if missing:
        raise RuntimeError(f"KKT JSON refers to modules not present in checkpoint model: {missing}")
    return entries, train_entries


def sinkhorn_project(matrix, num_iters, tol=0.0):
    out = matrix
    tiny = torch.finfo(out.dtype).tiny
    for iter_idx in range(num_iters):
        out = out / out.sum(dim=-1, keepdim=True).clamp_min(tiny)
        out = out / out.sum(dim=-2, keepdim=True).clamp_min(tiny)
        if tol > 0.0 and (iter_idx + 1 == num_iters or (iter_idx + 1) % 25 == 0):
            row_err = (out.sum(dim=-1) - 1.0).abs().max()
            col_err = (out.sum(dim=-2) - 1.0).abs().max()
            if max(float(row_err.item()), float(col_err.item())) <= tol:
                break
    return out


def project_trainable_h_res_(train_entries, args):
    with torch.no_grad():
        for _, _, _, param in train_entries:
            projected = sinkhorn_project(
                param.detach().float().clamp_min(args.projection_floor),
                args.sinkhorn_iters,
                args.sinkhorn_tol,
            )
            param.copy_(projected.to(device=param.device, dtype=param.dtype))


def load_memmap(data_dir, split):
    path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=np.uint16, mode="r")


def make_batch(data, batch_size, seq_len, generator, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([
        torch.from_numpy(np.asarray(data[i:i + seq_len], dtype=np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(np.asarray(data[i + 1:i + 1 + seq_len], dtype=np.int64))
        for i in ix
    ])
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def autocast_context(device, amp_dtype):
    if "cuda" not in device or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


@torch.no_grad()
def estimate_loss(model, data, args, seq_len, seed, num_batches):
    if num_batches <= 0:
        return None
    model.eval()
    losses = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ctx = autocast_context(args.device, args.amp_dtype)
    for _ in range(num_batches):
        x, y = make_batch(data, args.batch_size, seq_len, generator, args.device)
        with ctx:
            _, loss = model(x, y)
        losses.append(float(loss.detach().float().item()))
    model.train()
    return float(np.mean(losses))


def matrix_stats(param):
    h = param.detach().float().cpu()
    n = h.shape[0]
    eye = torch.eye(n)
    offdiag_mask = ~torch.eye(n, dtype=torch.bool)
    delta = h - eye
    return {
        "matrix": h.tolist(),
        "delta": delta.tolist(),
        "delta_l2": float(torch.linalg.vector_norm(delta).item()),
        "delta_abs_max": float(delta.abs().max().item()),
        "diag_min": float(h.diag().min().item()),
        "diag_mean": float(h.diag().mean().item()),
        "diag_max": float(h.diag().max().item()),
        "offdiag_sum": float(h[offdiag_mask].sum().item()),
        "offdiag_max": float(h[offdiag_mask].max().item()),
        "offdiag_min": float(h[offdiag_mask].min().item()),
        "row_sum_abs_err": float((h.sum(dim=1) - 1.0).abs().max().item()),
        "col_sum_abs_err": float((h.sum(dim=0) - 1.0).abs().max().item()),
        "min_entry": float(h.min().item()),
    }


def summarize_entries(train_entries, failed_row_by_name):
    rows = []
    for ordinal, name, _, param in train_entries:
        stats = matrix_stats(param)
        kkt_row = failed_row_by_name.get(name, {})
        rows.append({
            "module_ordinal": ordinal,
            "module_name": name,
            "kkt_min_pair_swap_pair": kkt_row.get("min_pair_swap_pair"),
            "kkt_min_pair_swap_derivative": kkt_row.get("min_pair_swap_derivative"),
            "kkt_min_offdiag_slack": kkt_row.get("min_offdiag_kkt_slack"),
            **stats,
        })
    rows.sort(key=lambda row: row["delta_l2"], reverse=True)
    return rows


def main():
    args = parse_args()
    if args.train_iters < 0:
        raise ValueError("--train-iters must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model_args = dict(checkpoint["model_args"])
    config = dict(checkpoint.get("config", {}))
    dataset = args.dataset or config.get("dataset", "openwebtext")
    block_size = int(model_args.get("block_size", config.get("block_size", 1024)))
    seq_len = args.seq_len or block_size
    if seq_len > block_size:
        raise ValueError(f"seq_len={seq_len} exceeds checkpoint block_size={block_size}")
    data_dir = os.path.join(REPO_ROOT, "data", dataset)

    failed_names, failed_row_by_name, kkt_data = load_failed_module_names(args.kkt_json)
    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(strip_compile_prefix(checkpoint["model"]), strict=True)
    model.to(args.device)
    model.train()

    for param in model.parameters():
        param.requires_grad_(False)

    entries, train_entries = set_forced_h_res_params(model, failed_names, args.device)
    train_params = [param for _, _, _, param in train_entries]
    if not train_params:
        raise RuntimeError("No failed KKT layers to train.")

    optimizer_cls = torch.optim.SGD if args.optimizer == "sgd" else torch.optim.Adam
    optimizer = optimizer_cls(train_params, lr=args.lr)
    train_data = load_memmap(data_dir, args.split)
    eval_data = load_memmap(data_dir, "val")
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    eval_seed = args.seed + 31
    ctx = autocast_context(args.device, args.amp_dtype)

    initial_rows = summarize_entries(train_entries, failed_row_by_name)
    initial_eval_loss = estimate_loss(model, eval_data, args, seq_len, eval_seed, args.eval_batches)
    logs = []

    for iter_idx in range(1, args.train_iters + 1):
        x, y = make_batch(train_data, args.batch_size, seq_len, train_generator, args.device)
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip) if args.grad_clip > 0 else None
        optimizer.step()
        if args.project_each_step:
            project_trainable_h_res_(train_entries, args)

        if iter_idx == 1 or iter_idx % args.log_interval == 0 or iter_idx == args.train_iters:
            rows = summarize_entries(train_entries, failed_row_by_name)
            eval_loss = None
            if args.eval_log_interval > 0 and (
                iter_idx == 1 or iter_idx % args.eval_log_interval == 0 or iter_idx == args.train_iters
            ):
                eval_loss = estimate_loss(model, eval_data, args, seq_len, eval_seed, args.eval_batches)
            log_row = {
                "iter": iter_idx,
                "train_loss": float(loss.detach().float().item()),
                "eval_loss": eval_loss,
                "eval_loss_delta_from_initial": None if eval_loss is None or initial_eval_loss is None else eval_loss - initial_eval_loss,
                "grad_norm": None if grad_norm is None else float(grad_norm.detach().float().item()),
                "max_delta_l2": rows[0]["delta_l2"],
                "max_offdiag_sum": max(row["offdiag_sum"] for row in rows),
                "max_offdiag_max": max(row["offdiag_max"] for row in rows),
                "max_row_sum_abs_err": max(row["row_sum_abs_err"] for row in rows),
                "max_col_sum_abs_err": max(row["col_sum_abs_err"] for row in rows),
                "min_entry": min(row["min_entry"] for row in rows),
            }
            logs.append(log_row)
            eval_text = "" if eval_loss is None else f" eval={eval_loss:.6f}"
            print(
                f"iter {iter_idx:4d} loss={log_row['train_loss']:.6f}{eval_text} "
                f"grad={log_row['grad_norm']:.3e} max_delta={log_row['max_delta_l2']:.3e} "
                f"max_offdiag_sum={log_row['max_offdiag_sum']:.3e}"
            )

    final_eval_loss = estimate_loss(model, eval_data, args, seq_len, eval_seed, args.eval_batches)
    final_rows = summarize_entries(train_entries, failed_row_by_name)

    result = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "checkpoint": args.ckpt,
        "checkpoint_iter": int(checkpoint.get("iter_num", -1)),
        "kkt_json": args.kkt_json,
        "kkt_json_loss_at_forced_identity": kkt_data.get("loss_at_forced_identity"),
        "dataset": dataset,
        "split": args.split,
        "device": args.device,
        "seed": args.seed,
        "eval_seed": eval_seed,
        "train_iters": args.train_iters,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "project_each_step": args.project_each_step,
        "projection_floor": args.projection_floor,
        "sinkhorn_iters": args.sinkhorn_iters,
        "sinkhorn_tol": args.sinkhorn_tol,
        "eval_batches": args.eval_batches,
        "eval_log_interval": args.eval_log_interval,
        "amp_dtype": args.amp_dtype,
        "num_h_res_modules": len(entries),
        "num_trainable_failed_modules": len(train_entries),
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "eval_loss_delta": None if initial_eval_loss is None or final_eval_loss is None else final_eval_loss - initial_eval_loss,
        "logs": logs,
        "initial_failed_layers": initial_rows,
        "final_failed_layers": final_rows,
    }

    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(REPO_ROOT, "analyze", "h_res_identity_kkt_results")
        os.makedirs(output_dir, exist_ok=True)
        ckpt_tag = os.path.basename(os.path.dirname(args.ckpt.rstrip("/"))) or "checkpoint"
        output_path = os.path.join(
            output_dir,
            f"{ckpt_tag}_failed_h_res_projected_train_{args.train_iters}iter_lr{str(args.lr).replace('.', 'p')}.json",
        )

    with open(output_path + ".tmp", "w") as f:
        json.dump(result, f, indent=2)
    os.replace(output_path + ".tmp", output_path)

    print("\nsummary")
    print(f"checkpoint: {args.ckpt}")
    print(f"trainable failed H_res modules: {len(train_entries)} / {len(entries)}")
    print(f"initial val loss: {initial_eval_loss:.6f}" if initial_eval_loss is not None else "initial val loss: None")
    print(f"final val loss:   {final_eval_loss:.6f}" if final_eval_loss is not None else "final val loss: None")
    if initial_eval_loss is not None and final_eval_loss is not None:
        print(f"val loss delta:   {final_eval_loss - initial_eval_loss:.6e}")
    print("top changed failed layers:")
    for row in final_rows[:max(0, args.top_k)]:
        print(
            f"  {row['module_name']:<30} delta_l2={row['delta_l2']:.3e} "
            f"offdiag_sum={row['offdiag_sum']:.3e} offdiag_max={row['offdiag_max']:.3e} "
            f"diag_mean={row['diag_mean']:.6f} kkt_pair={row['kkt_min_pair_swap_pair']}"
        )
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
