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
import torch.nn.functional as F

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


def build_model_from_checkpoint(checkpoint: dict, device: str) -> GPT:
    checkpoint_model_args = checkpoint["model_args"]
    config_keys = {field.name for field in fields(GPTConfig)}
    model_args = {key: value for key, value in checkpoint_model_args.items() if key in config_keys}
    if "mhc_disable_dynamic_h_res" not in model_args:
        model_args["mhc_disable_dynamic_h_res"] = False
    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(strip_compile_prefix(checkpoint["model"]))
    model.to(device)
    model.eval()
    return model


def forward_pre_reduce_hidden(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    device = idx.device
    bsz, seq_len = idx.size()
    if seq_len > model.config.block_size:
        raise ValueError(f"sequence length {seq_len} exceeds block_size {model.config.block_size}")

    pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    if model.expand_stream is not None:
        x = model.expand_stream(x)
    if model.attn_res_mixer is not None:
        model.attn_res_mixer.reset(x)
    if model.block_depth_memory is not None:
        model.block_depth_memory.reset(x)
    for block in model.transformer.h:
        x = block(x)
    if model.attn_res_mixer is not None:
        x = model.attn_res_mixer.finalize()
    if model.block_depth_memory is not None:
        model.block_depth_memory.clear()
    return model.transformer.ln_f(x)


def get_val_batch(
    val_data: np.memmap,
    *,
    block_size: int,
    batch_size: int,
    device: str,
    device_type: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(val_data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([
        torch.from_numpy((val_data[i:i + block_size]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((val_data[i + 1:i + 1 + block_size]).astype(np.int64))
        for i in ix
    ])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def maybe_sample_positions(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_positions: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    # hidden: (batch, streams, seq, dim), targets: (batch, seq)
    bsz, streams, seq_len, dim = hidden.shape
    hidden_flat = hidden.permute(0, 2, 1, 3).reshape(bsz * seq_len, streams, dim)
    targets_flat = targets.reshape(bsz * seq_len)
    if max_positions <= 0 or hidden_flat.shape[0] <= max_positions:
        return hidden_flat, targets_flat
    indices = torch.randperm(hidden_flat.shape[0], generator=generator, device="cpu")[:max_positions]
    indices = indices.to(hidden_flat.device)
    return hidden_flat.index_select(0, indices), targets_flat.index_select(0, indices)


def update_pairwise_metrics(
    logits: torch.Tensor,
    pair_stats: dict[tuple[int, int], dict[str, float]],
):
    # logits: (positions, streams, vocab)
    positions, streams, vocab = logits.shape
    top1 = logits.argmax(dim=-1)
    for i in range(streams):
        li = logits[:, i, :].float()
        for j in range(i + 1, streams):
            lj = logits[:, j, :].float()
            diff = li - lj
            stats = pair_stats[(i, j)]
            stats["sse"] += diff.square().sum().item()
            stats["sae"] += diff.abs().sum().item()
            stats["max_abs"] = max(stats["max_abs"], diff.abs().max().item())
            stats["dot"] += (li * lj).sum().item()
            stats["norm_i_sq"] += li.square().sum().item()
            stats["norm_j_sq"] += lj.square().sum().item()
            stats["top1_same"] += (top1[:, i] == top1[:, j]).float().sum().item()
            stats["count_logits"] += positions * vocab
            stats["count_positions"] += positions


def update_stream_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    stream_loss_sum: torch.Tensor,
    stream_correct_top1: torch.Tensor,
):
    # logits: (positions, streams, vocab), targets: (positions,)
    positions, streams, _ = logits.shape
    for stream_idx in range(streams):
        stream_logits = logits[:, stream_idx, :].float()
        stream_loss_sum[stream_idx] += F.cross_entropy(stream_logits, targets, reduction="sum").item()
        stream_correct_top1[stream_idx] += (stream_logits.argmax(dim=-1) == targets).float().sum().item()


def finalize_pair_stats(pair_stats: dict[tuple[int, int], dict[str, float]]) -> list[dict[str, float | int]]:
    rows = []
    for (i, j), stats in sorted(pair_stats.items()):
        count_logits = max(stats["count_logits"], 1.0)
        count_positions = max(stats["count_positions"], 1.0)
        cosine = stats["dot"] / math.sqrt(max(stats["norm_i_sq"] * stats["norm_j_sq"], 1e-30))
        rows.append({
            "stream_i": i,
            "stream_j": j,
            "rmse": math.sqrt(stats["sse"] / count_logits),
            "mae": stats["sae"] / count_logits,
            "max_abs": stats["max_abs"],
            "cosine": cosine,
            "top1_agreement": stats["top1_same"] / count_positions,
        })
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


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
        description="Evaluate validation-set differences between per-stream output logits."
    )
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint path, e.g. identity_tanh_offidiag/ckpt.pt")
    parser.add_argument("--data-dir", default="", help="Directory containing val.bin. Defaults to data/<dataset>.")
    parser.add_argument("--dataset", default="", help="Dataset name fallback from checkpoint config, default openwebtext.")
    parser.add_argument("--output-dir", default="analyze/stream_logits_diff_results")
    parser.add_argument("--eval-iters", type=int, default=0, help="Default: checkpoint config eval_iters or 200.")
    parser.add_argument("--batch-size", type=int, default=0, help="Default: checkpoint config batch_size or 1.")
    parser.add_argument("--block-size", type=int, default=0, help="Default: checkpoint model block_size.")
    parser.add_argument("--max-positions-per-batch", type=int, default=256, help="Sample this many token positions per batch; <=0 uses all positions.")
    parser.add_argument("--logit-position-chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="", choices=["", "float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=0, help="Default: checkpoint config seed or 1337.")
    parser.add_argument("--print-pairs-limit", type=int, default=20)
    parser.add_argument("--use-autocast", type=str2bool, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    config = checkpoint.get("config", {})
    model_args = checkpoint.get("model_args", {})
    dataset = args.dataset or config.get("dataset", "openwebtext")
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data" / dataset
    val_path = data_dir / "val.bin"
    if not val_path.exists():
        raise FileNotFoundError(val_path)

    device = args.device
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype_name = args.dtype or config.get("dtype", "float32")
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if args.use_autocast and device_type == "cuda" and dtype_name != "float32"
        else nullcontext()
    )

    eval_iters = args.eval_iters or int(config.get("eval_iters", 200))
    batch_size = args.batch_size or int(config.get("batch_size", 1))
    block_size = args.block_size or int(model_args.get("block_size", config.get("block_size", 1024)))
    seed = args.seed or int(config.get("seed", 1337))

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(seed)
    position_generator = torch.Generator(device="cpu")
    position_generator.manual_seed(seed + 1)

    model = build_model_from_checkpoint(checkpoint, device=device)
    streams = int(model.config.hyper_conn_n if model.reduce_stream is not None else 1)
    if streams <= 1:
        raise ValueError(f"checkpoint has {streams} stream; stream-logit comparison requires >1")

    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    pair_stats = {
        (i, j): {
            "sse": 0.0,
            "sae": 0.0,
            "max_abs": 0.0,
            "dot": 0.0,
            "norm_i_sq": 0.0,
            "norm_j_sq": 0.0,
            "top1_same": 0.0,
            "count_logits": 0.0,
            "count_positions": 0.0,
        }
        for i in range(streams)
        for j in range(i + 1, streams)
    }
    stream_loss_sum = torch.zeros(streams, dtype=torch.float64)
    stream_correct_top1 = torch.zeros(streams, dtype=torch.float64)
    reduced_full_loss_sum = 0.0
    reduced_full_correct_top1 = 0.0
    reduced_sampled_loss_sum = 0.0
    reduced_sampled_correct_top1 = 0.0
    total_positions = 0

    model.eval()
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_val_batch(
                val_data,
                block_size=block_size,
                batch_size=batch_size,
                device=device,
                device_type=device_type,
                generator=batch_generator,
            )
            with ctx:
                pre_reduce_hidden = forward_pre_reduce_hidden(model, x)
                reduced_hidden = model.reduce_stream(pre_reduce_hidden)
                reduced_logits = model.lm_head(reduced_hidden).float()
                reduced_full_loss_sum += F.cross_entropy(
                    reduced_logits.reshape(-1, reduced_logits.size(-1)),
                    y.reshape(-1),
                    reduction="sum",
                ).item()
                reduced_full_correct_top1 += (
                    reduced_logits.argmax(dim=-1).reshape(-1) == y.reshape(-1)
                ).float().sum().item()

                hidden = pre_reduce_hidden.reshape(batch_size, streams, block_size, model.config.n_embd)
                hidden_positions, targets_positions = maybe_sample_positions(
                    hidden,
                    y,
                    max_positions=args.max_positions_per_batch,
                    generator=position_generator,
                )

            # Compute per-stream logits in chunks over sampled positions. This keeps
            # memory bounded for large vocabularies and many streams.
            for start in range(0, hidden_positions.shape[0], args.logit_position_chunk_size):
                end = min(start + args.logit_position_chunk_size, hidden_positions.shape[0])
                hidden_chunk = hidden_positions[start:end]
                targets_chunk = targets_positions[start:end]
                with ctx:
                    logits_chunk = torch.einsum(
                        "psd,vd->psv",
                        hidden_chunk.to(device),
                        model.lm_head.weight,
                    )
                    reduced_hidden_chunk = model.reduce_stream(
                        hidden_chunk.reshape(-1, hidden_chunk.shape[-1]).to(device)
                    )
                    reduced_logits_chunk = model.lm_head(reduced_hidden_chunk)
                logits_chunk = logits_chunk.float()
                reduced_logits_chunk = reduced_logits_chunk.float()
                reduced_sampled_loss_sum += F.cross_entropy(
                    reduced_logits_chunk,
                    targets_chunk.to(reduced_logits_chunk.device),
                    reduction="sum",
                ).item()
                reduced_sampled_correct_top1 += (
                    reduced_logits_chunk.argmax(dim=-1) == targets_chunk.to(reduced_logits_chunk.device)
                ).float().sum().item()
                update_pairwise_metrics(logits_chunk, pair_stats)
                update_stream_losses(
                    logits_chunk,
                    targets_chunk.to(logits_chunk.device),
                    stream_loss_sum,
                    stream_correct_top1,
                )
                total_positions += logits_chunk.shape[0]

    pair_rows = finalize_pair_stats(pair_stats)
    stream_rows = []
    total_positions_safe = max(total_positions, 1)
    for stream_idx in range(streams):
        stream_rows.append({
            "stream": stream_idx,
            "loss": float(stream_loss_sum[stream_idx].item() / total_positions_safe),
            "top1_accuracy": float(stream_correct_top1[stream_idx].item() / total_positions_safe),
        })

    pair_rmse = [float(row["rmse"]) for row in pair_rows]
    pair_mae = [float(row["mae"]) for row in pair_rows]
    pair_cos = [float(row["cosine"]) for row in pair_rows]
    pair_top1 = [float(row["top1_agreement"]) for row in pair_rows]
    stream_losses = [float(row["loss"]) for row in stream_rows]
    reduced_positions = eval_iters * batch_size * block_size
    summary = {
        "checkpoint": str(ckpt_path),
        "iter_num": checkpoint.get("iter_num", ""),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "mode": model.config.mhc_h_res_mode,
        "hyper_conn_type": model.config.hyper_conn_type,
        "streams": streams,
        "reduce_stream_mode": model.config.hyper_conn_reduce_stream_mode,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "block_size": block_size,
        "sampled_positions": total_positions,
        "reduced_positions": reduced_positions,
        "reduced_loss_full_positions": reduced_full_loss_sum / max(reduced_positions, 1),
        "reduced_top1_accuracy_full_positions": reduced_full_correct_top1 / max(reduced_positions, 1),
        "reduced_loss_sampled_positions": reduced_sampled_loss_sum / max(total_positions, 1),
        "reduced_top1_accuracy_sampled_positions": reduced_sampled_correct_top1 / max(total_positions, 1),
        "stream_loss_mean": mean(stream_losses),
        "stream_loss_min": min(stream_losses),
        "stream_loss_max": max(stream_losses),
        "stream_loss_range": max(stream_losses) - min(stream_losses),
        "pairwise_logit_rmse_mean": mean(pair_rmse),
        "pairwise_logit_rmse_min": min(pair_rmse),
        "pairwise_logit_rmse_max": max(pair_rmse),
        "pairwise_logit_mae_mean": mean(pair_mae),
        "pairwise_logit_cosine_mean": mean(pair_cos),
        "pairwise_logit_cosine_min": min(pair_cos),
        "pairwise_top1_agreement_mean": mean(pair_top1),
        "pairwise_top1_agreement_min": min(pair_top1),
        "pairwise_logit_max_abs_max": max(float(row["max_abs"]) for row in pair_rows),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.parent.name or ckpt_path.stem
    summary_path = output_dir / f"{stem}_stream_logits_summary.json"
    pair_path = output_dir / f"{stem}_stream_pair_metrics.csv"
    stream_path = output_dir / f"{stem}_stream_losses.csv"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_csv(pair_path, pair_rows)
    write_csv(stream_path, stream_rows)

    print("\nSummary")
    print("| metric | value |")
    print("| --- | ---: |")
    for key in [
        "iter_num",
        "best_val_loss",
        "mode",
        "streams",
        "reduce_stream_mode",
        "eval_iters",
        "sampled_positions",
        "reduced_loss_sampled_positions",
        "reduced_loss_full_positions",
        "stream_loss_mean",
        "stream_loss_range",
        "pairwise_logit_rmse_mean",
        "pairwise_logit_rmse_max",
        "pairwise_logit_cosine_mean",
        "pairwise_logit_cosine_min",
        "pairwise_top1_agreement_mean",
        "pairwise_logit_max_abs_max",
    ]:
        value = summary[key]
        if isinstance(value, float):
            value = f"{value:.6g}"
        print(f"| {key} | {value} |")

    print("\nPer-stream loss")
    print("| stream | loss | top1_accuracy |")
    print("| ---: | ---: | ---: |")
    for row in stream_rows:
        print(f"| {row['stream']} | {row['loss']:.6g} | {row['top1_accuracy']:.6g} |")

    print(f"\nTop {min(args.print_pairs_limit, len(pair_rows))} stream-pair differences by RMSE")
    print("| stream_i | stream_j | rmse | mae | cosine | top1_agreement | max_abs |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(pair_rows, key=lambda item: item["rmse"], reverse=True)[:args.print_pairs_limit]:
        print(
            f"| {row['stream_i']} | {row['stream_j']} | {row['rmse']:.6g} | "
            f"{row['mae']:.6g} | {row['cosine']:.6g} | {row['top1_agreement']:.6g} | "
            f"{row['max_abs']:.6g} |"
        )

    print("\nWrote:")
    print(f"  {summary_path}")
    print(f"  {pair_path}")
    print(f"  {stream_path}")


if __name__ == "__main__":
    main()
