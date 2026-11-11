from __future__ import annotations

import argparse
import csv
import json
import math
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
from hyper_conn.mhc import ExpandStreams


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
    # The compressed iHC fast path returns an IdentityStreamState wrapper between
    # layers. Analysis routines need explicit stream tensors, so switch to the
    # equivalent materialized repeat path after restoring the identical weights.
    if model.identity_stream_compressed:
        model.expand_stream = ExpandStreams(
            model.config.hyper_conn_n,
            mode=model.config.hyper_conn_expand_stream_mode,
            stream_dtype=model.config.mhc_stream_dtype,
        )
        for module in model.modules():
            if hasattr(module, "mhc_identity_stream_compressed"):
                module.mhc_identity_stream_compressed = False
        model.identity_stream_compressed = False
    model.to(device)
    model.eval()
    return model


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


def sample_position_indices(
    *,
    batch_size: int,
    block_size: int,
    max_positions: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    total_positions = batch_size * block_size
    if max_positions <= 0 or total_positions <= max_positions:
        return torch.arange(total_positions, device=device)
    indices = torch.randperm(total_positions, generator=generator, device="cpu")[:max_positions]
    return indices.to(device)


def select_hidden_positions(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    position_indices: torch.Tensor,
    *,
    batch_size: int,
    streams: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # hidden comes from the MHC stream-major batch layout: (batch * streams, seq, dim).
    hidden = hidden.reshape(batch_size, streams, block_size, hidden.shape[-1])
    hidden = hidden.permute(0, 2, 1, 3).reshape(batch_size * block_size, streams, hidden.shape[-1])
    targets = targets.reshape(batch_size * block_size)
    return hidden.index_select(0, position_indices), targets.index_select(0, position_indices)


def compute_per_stream_logits(hidden: torch.Tensor, model: GPT) -> torch.Tensor:
    # hidden: (positions, streams, dim). The model's lm_head is bias-free today,
    # but this keeps the helper correct if a future checkpoint adds a bias.
    logits = torch.einsum("psd,vd->psv", hidden, model.lm_head.weight)
    if model.lm_head.bias is not None:
        logits = logits + model.lm_head.bias
    return logits


def empty_pair_stats(streams: int) -> dict[tuple[int, int], dict[str, float]]:
    return {
        (i, j): {
            "sse": 0.0,
            "sae": 0.0,
            "max_abs": 0.0,
            "dot": 0.0,
            "norm_i_sq": 0.0,
            "norm_j_sq": 0.0,
            "centered_dot": 0.0,
            "centered_norm_i_sq": 0.0,
            "centered_norm_j_sq": 0.0,
            "top1_same": 0.0,
            "count_logits": 0.0,
            "count_positions": 0.0,
        }
        for i in range(streams)
        for j in range(i + 1, streams)
    }


def update_pairwise_metrics(
    logits: torch.Tensor,
    pair_stats: dict[tuple[int, int], dict[str, float]],
):
    # logits: (positions, streams, vocab)
    positions, streams, vocab = logits.shape
    logits = logits.float()
    centered = logits - logits.mean(dim=-1, keepdim=True)
    top1 = logits.argmax(dim=-1)

    for i in range(streams):
        li = logits[:, i, :]
        ci = centered[:, i, :]
        for j in range(i + 1, streams):
            lj = logits[:, j, :]
            cj = centered[:, j, :]
            diff = li - lj
            stats = pair_stats[(i, j)]
            stats["sse"] += diff.square().sum().item()
            stats["sae"] += diff.abs().sum().item()
            stats["max_abs"] = max(stats["max_abs"], diff.abs().max().item())
            stats["dot"] += (li * lj).sum().item()
            stats["norm_i_sq"] += li.square().sum().item()
            stats["norm_j_sq"] += lj.square().sum().item()
            stats["centered_dot"] += (ci * cj).sum().item()
            stats["centered_norm_i_sq"] += ci.square().sum().item()
            stats["centered_norm_j_sq"] += cj.square().sum().item()
            stats["top1_same"] += (top1[:, i] == top1[:, j]).float().sum().item()
            stats["count_logits"] += positions * vocab
            stats["count_positions"] += positions


def update_stream_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    stream_loss_sum: torch.Tensor,
    stream_correct_top1: torch.Tensor,
):
    # logits: (positions, streams, vocab), targets: (positions,)
    _, streams, _ = logits.shape
    for stream_idx in range(streams):
        stream_logits = logits[:, stream_idx, :].float()
        stream_loss_sum[stream_idx] += F.cross_entropy(stream_logits, targets, reduction="sum").item()
        stream_correct_top1[stream_idx] += (stream_logits.argmax(dim=-1) == targets).float().sum().item()


def finalize_pair_stats(
    layer: int,
    pair_stats: dict[tuple[int, int], dict[str, float]],
) -> list[dict[str, float | int]]:
    rows = []
    for (i, j), stats in sorted(pair_stats.items()):
        count_logits = max(stats["count_logits"], 1.0)
        count_positions = max(stats["count_positions"], 1.0)
        cosine = stats["dot"] / math.sqrt(max(stats["norm_i_sq"] * stats["norm_j_sq"], 1e-30))
        centered_cosine = stats["centered_dot"] / math.sqrt(
            max(stats["centered_norm_i_sq"] * stats["centered_norm_j_sq"], 1e-30)
        )
        rows.append({
            "layer": layer,
            "stream_i": i,
            "stream_j": j,
            "rmse": math.sqrt(stats["sse"] / count_logits),
            "mae": stats["sae"] / count_logits,
            "max_abs": stats["max_abs"],
            "cosine": cosine,
            "centered_cosine": centered_cosine,
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


def maybe_dump_topk(
    rows: list[dict],
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    position_indices: torch.Tensor,
    layer: int,
    eval_iter: int,
    block_size: int,
    dump_positions: int,
    top_k: int,
):
    if dump_positions <= 0 or top_k <= 0:
        return
    positions_to_dump = min(dump_positions, logits.shape[0])
    values, token_ids = torch.topk(logits[:positions_to_dump].float(), k=top_k, dim=-1)
    for pos_idx in range(positions_to_dump):
        flat_position = int(position_indices[pos_idx].item())
        batch_index = flat_position // block_size
        seq_pos = flat_position % block_size
        target_id = int(targets[pos_idx].item())
        for stream_idx in range(logits.shape[1]):
            for rank in range(top_k):
                rows.append({
                    "eval_iter": eval_iter,
                    "layer": layer,
                    "batch_index": batch_index,
                    "seq_pos": seq_pos,
                    "target_id": target_id,
                    "stream": stream_idx,
                    "rank": rank + 1,
                    "token_id": int(token_ids[pos_idx, stream_idx, rank].item()),
                    "logit": float(values[pos_idx, stream_idx, rank].item()),
                })


def parse_layers(value: str, n_layer: int) -> list[int]:
    if value.strip().lower() in {"", "all"}:
        return list(range(n_layer))

    layers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))

    result = sorted(layers)
    invalid = [layer for layer in result if layer < 0 or layer >= n_layer]
    if invalid:
        raise ValueError(f"invalid layers {invalid}; model has layers 0..{n_layer - 1}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate per-layer per-stream vocabulary-logit similarity for an MHC checkpoint. "
            "By default this is a logit-lens pass: each intermediate layer output is normalized "
            "with the model's final ln_f before applying lm_head."
        )
    )
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint path, e.g. idt_4_small/ckpt.pt")
    parser.add_argument("--data-dir", default="", help="Directory containing val.bin. Defaults to data/<dataset>.")
    parser.add_argument("--dataset", default="", help="Dataset name fallback from checkpoint config, default openwebtext.")
    parser.add_argument("--output-dir", default="analyze/layer_stream_logits_similarity_results")
    parser.add_argument("--eval-iters", type=int, default=0, help="Default: checkpoint config eval_iters or 200.")
    parser.add_argument("--batch-size", type=int, default=0, help="Default: checkpoint config batch_size or 1.")
    parser.add_argument("--block-size", type=int, default=0, help="Default: checkpoint model block_size.")
    parser.add_argument("--layers", default="all", help="Comma/range list, e.g. all, 0,3,7, or 0-11.")
    parser.add_argument("--max-positions-per-batch", type=int, default=128, help="Sample token positions per batch; <=0 uses all positions.")
    parser.add_argument("--logit-position-chunk-size", type=int, default=32)
    parser.add_argument("--apply-final-ln", type=str2bool, default=True, help="Apply model.transformer.ln_f before lm_head at every layer.")
    parser.add_argument("--dump-topk-positions", type=int, default=0, help="Optional top-k logits dump for this many sampled positions per eval batch.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k vocab ids per stream when --dump-topk-positions > 0.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="", choices=["", "float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=0, help="Default: checkpoint config seed or 1337.")
    parser.add_argument("--print-layers-limit", type=int, default=50)
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
    if model.config.hyper_conn_type not in {"mhc", "mhc_lite", "analysis", "hc"}:
        raise ValueError(f"checkpoint hyper_conn_type={model.config.hyper_conn_type!r} has no multi-stream MHC/HC path")

    layers = parse_layers(args.layers, model.config.n_layer)
    layer_set = set(layers)
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    pair_stats_by_layer = {layer: empty_pair_stats(streams) for layer in layers}
    stream_loss_sum_by_layer = {layer: torch.zeros(streams, dtype=torch.float64) for layer in layers}
    stream_correct_top1_by_layer = {layer: torch.zeros(streams, dtype=torch.float64) for layer in layers}
    topk_rows: list[dict] = []
    total_positions_by_layer = {layer: 0 for layer in layers}

    model.eval()
    with torch.no_grad():
        for eval_iter in range(eval_iters):
            idx, targets = get_val_batch(
                val_data,
                block_size=block_size,
                batch_size=batch_size,
                device=device,
                device_type=device_type,
                generator=batch_generator,
            )
            position_indices = sample_position_indices(
                batch_size=batch_size,
                block_size=block_size,
                max_positions=args.max_positions_per_batch,
                generator=position_generator,
                device=targets.device,
            )

            pos = torch.arange(0, block_size, dtype=torch.long, device=idx.device)
            with ctx:
                tok_emb = model.transformer.wte(idx)
                pos_emb = model.transformer.wpe(pos)
                hidden = model.transformer.drop(tok_emb + pos_emb)
                if model.expand_stream is not None:
                    hidden = model.expand_stream(hidden)
                if model.attn_res_mixer is not None:
                    model.attn_res_mixer.reset(hidden)
                if model.block_depth_memory is not None:
                    model.block_depth_memory.reset(hidden)

                for layer, block in enumerate(model.transformer.h):
                    hidden = block(hidden)
                    if layer not in layer_set:
                        continue

                    logit_hidden = model.transformer.ln_f(hidden) if args.apply_final_ln else hidden
                    hidden_positions, target_positions = select_hidden_positions(
                        logit_hidden,
                        targets,
                        position_indices,
                        batch_size=batch_size,
                        streams=streams,
                        block_size=block_size,
                    )

                    for start in range(0, hidden_positions.shape[0], args.logit_position_chunk_size):
                        end = min(start + args.logit_position_chunk_size, hidden_positions.shape[0])
                        hidden_chunk = hidden_positions[start:end]
                        targets_chunk = target_positions[start:end]
                        logits_chunk = compute_per_stream_logits(hidden_chunk, model)
                        update_pairwise_metrics(logits_chunk, pair_stats_by_layer[layer])
                        update_stream_metrics(
                            logits_chunk,
                            targets_chunk.to(logits_chunk.device),
                            stream_loss_sum_by_layer[layer],
                            stream_correct_top1_by_layer[layer],
                        )
                        total_positions_by_layer[layer] += logits_chunk.shape[0]

                        if start == 0:
                            maybe_dump_topk(
                                topk_rows,
                                logits=logits_chunk,
                                targets=targets_chunk,
                                position_indices=position_indices[start:end],
                                layer=layer,
                                eval_iter=eval_iter,
                                block_size=block_size,
                                dump_positions=args.dump_topk_positions,
                                top_k=args.top_k,
                            )

                if model.attn_res_mixer is not None:
                    model.attn_res_mixer.finalize()
                if model.block_depth_memory is not None:
                    model.block_depth_memory.clear()

    pair_rows: list[dict] = []
    stream_rows: list[dict] = []
    layer_rows: list[dict] = []
    for layer in layers:
        layer_pair_rows = finalize_pair_stats(layer, pair_stats_by_layer[layer])
        pair_rows.extend(layer_pair_rows)

        total_positions = max(total_positions_by_layer[layer], 1)
        losses = []
        accuracies = []
        for stream_idx in range(streams):
            loss = float(stream_loss_sum_by_layer[layer][stream_idx].item() / total_positions)
            top1_accuracy = float(stream_correct_top1_by_layer[layer][stream_idx].item() / total_positions)
            losses.append(loss)
            accuracies.append(top1_accuracy)
            stream_rows.append({
                "layer": layer,
                "stream": stream_idx,
                "loss": loss,
                "top1_accuracy": top1_accuracy,
            })

        layer_rmse = [float(row["rmse"]) for row in layer_pair_rows]
        layer_mae = [float(row["mae"]) for row in layer_pair_rows]
        layer_cos = [float(row["cosine"]) for row in layer_pair_rows]
        layer_centered_cos = [float(row["centered_cosine"]) for row in layer_pair_rows]
        layer_top1 = [float(row["top1_agreement"]) for row in layer_pair_rows]
        layer_rows.append({
            "layer": layer,
            "sampled_positions": total_positions_by_layer[layer],
            "stream_loss_mean": mean(losses),
            "stream_loss_min": min(losses),
            "stream_loss_max": max(losses),
            "stream_loss_range": max(losses) - min(losses),
            "stream_top1_accuracy_mean": mean(accuracies),
            "pairwise_logit_rmse_mean": mean(layer_rmse),
            "pairwise_logit_rmse_min": min(layer_rmse),
            "pairwise_logit_rmse_max": max(layer_rmse),
            "pairwise_logit_mae_mean": mean(layer_mae),
            "pairwise_logit_cosine_mean": mean(layer_cos),
            "pairwise_logit_cosine_min": min(layer_cos),
            "pairwise_logit_centered_cosine_mean": mean(layer_centered_cos),
            "pairwise_logit_centered_cosine_min": min(layer_centered_cos),
            "pairwise_top1_agreement_mean": mean(layer_top1),
            "pairwise_top1_agreement_min": min(layer_top1),
            "pairwise_logit_max_abs_max": max(float(row["max_abs"]) for row in layer_pair_rows),
        })

    summary = {
        "checkpoint": str(ckpt_path),
        "iter_num": checkpoint.get("iter_num", ""),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "hyper_conn_type": model.config.hyper_conn_type,
        "mode": getattr(model.config, "mhc_h_res_mode", ""),
        "mhc_lite_method": getattr(model.config, "mhc_lite_method", ""),
        "streams": streams,
        "reduce_stream_mode": model.config.hyper_conn_reduce_stream_mode,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "block_size": block_size,
        "layers": layers,
        "apply_final_ln": args.apply_final_ln,
        "max_positions_per_batch": args.max_positions_per_batch,
        "total_sampled_positions_per_layer": {
            str(layer): total_positions_by_layer[layer]
            for layer in layers
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.parent.name or ckpt_path.stem
    prefix = f"{stem}_layer_stream_logits"
    summary_path = output_dir / f"{prefix}_summary.json"
    layer_path = output_dir / f"{prefix}_layer_summary.csv"
    pair_path = output_dir / f"{prefix}_pair_metrics.csv"
    stream_path = output_dir / f"{prefix}_stream_losses.csv"
    topk_path = output_dir / f"{prefix}_topk.csv"

    summary_path.write_text(json.dumps(summary, indent=2))
    write_csv(layer_path, layer_rows)
    write_csv(pair_path, pair_rows)
    write_csv(stream_path, stream_rows)
    if topk_rows:
        write_csv(topk_path, topk_rows)

    print("\nSummary")
    print("| metric | value |")
    print("| --- | ---: |")
    for key in [
        "iter_num",
        "best_val_loss",
        "hyper_conn_type",
        "mode",
        "mhc_lite_method",
        "streams",
        "reduce_stream_mode",
        "eval_iters",
        "batch_size",
        "block_size",
        "apply_final_ln",
        "max_positions_per_batch",
    ]:
        value = summary[key]
        if isinstance(value, float):
            value = f"{value:.6g}"
        print(f"| {key} | {value} |")

    print("\nLayer summary")
    print("| layer | positions | loss_mean | rmse_mean | cosine_mean | centered_cosine_mean | top1_agreement_mean |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in layer_rows[:args.print_layers_limit]:
        print(
            f"| {row['layer']} | {row['sampled_positions']} | {row['stream_loss_mean']:.6g} | "
            f"{row['pairwise_logit_rmse_mean']:.6g} | {row['pairwise_logit_cosine_mean']:.6g} | "
            f"{row['pairwise_logit_centered_cosine_mean']:.6g} | {row['pairwise_top1_agreement_mean']:.6g} |"
        )

    if len(layer_rows) > args.print_layers_limit:
        print(f"... omitted {len(layer_rows) - args.print_layers_limit} layers from stdout")

    print("\nWrote:")
    print(f"  {summary_path}")
    print(f"  {layer_path}")
    print(f"  {pair_path}")
    print(f"  {stream_path}")
    if topk_rows:
        print(f"  {topk_path}")


if __name__ == "__main__":
    main()
