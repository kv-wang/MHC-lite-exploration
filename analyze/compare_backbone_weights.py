from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch


DEFAULT_REF_CKPT = "out-owt-small-no-mhc-10000iter/ckpt.pt"
DEFAULT_CMP_CKPT = "out-owt-small-mhc-alm-nonnegative-32streams-10000iter/ckpt.pt"
DEFAULT_OUTPUT_DIR = "analyze/backbone_weight_comparison_results"


def is_backbone_key(key: str) -> bool:
    if ".hc_" in key or key.startswith("reduce_stream."):
        return False
    return (
        key in {"transformer.wte.weight", "transformer.wpe.weight", "transformer.ln_f.weight", "lm_head.weight"}
        or ".branch_attn." in key
        or ".branch_mlp." in key
    )


def key_sort_key(key: str):
    layer = parse_layer(key)
    if layer is None:
        prefix_order = {
            "transformer.wte.weight": -3,
            "transformer.wpe.weight": -2,
            "transformer.ln_f.weight": 10**8,
            "lm_head.weight": 10**8 + 1,
        }
        return (prefix_order.get(key, 10**9), key)
    component = parse_component(key)
    component_order = {"attn": 0, "mlp": 1}.get(component, 9)
    return (layer, component_order, key)


def parse_layer(key: str):
    match = re.search(r"transformer\.h\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def parse_component(key: str):
    if ".branch_attn." in key:
        return "attn"
    if ".branch_mlp." in key:
        return "mlp"
    if key.startswith("transformer.wte") or key.startswith("transformer.wpe"):
        return "embedding"
    if key.startswith("transformer.ln_f"):
        return "ln_f"
    if key.startswith("lm_head"):
        return "lm_head"
    return "other"


def parse_param_type(key: str):
    if key.endswith(".c_attn.weight"):
        return "attn.c_attn.weight"
    if key.endswith(".c_proj.weight") and ".branch_attn." in key:
        return "attn.c_proj.weight"
    if key.endswith(".c_fc.weight"):
        return "mlp.c_fc.weight"
    if key.endswith(".c_proj.weight") and ".branch_mlp." in key:
        return "mlp.c_proj.weight"
    if key.endswith(".0.weight") and ".branch_attn." in key:
        return "attn.norm.weight"
    if key.endswith(".0.weight") and ".branch_mlp." in key:
        return "mlp.norm.weight"
    return key


def new_accumulator():
    return {
        "num_tensors": 0,
        "numel": 0,
        "ref_sum": 0.0,
        "cmp_sum": 0.0,
        "ref_sq": 0.0,
        "cmp_sq": 0.0,
        "dot": 0.0,
        "diff_sq": 0.0,
        "abs_diff": 0.0,
        "max_abs_diff": 0.0,
    }


def update_accumulator(acc: dict, ref: torch.Tensor, cmp: torch.Tensor):
    ref_f = ref.detach().float().reshape(-1)
    cmp_f = cmp.detach().float().reshape(-1)
    diff = cmp_f - ref_f
    abs_diff = diff.abs()

    acc["num_tensors"] += 1
    acc["numel"] += ref_f.numel()
    acc["ref_sum"] += ref_f.sum().item()
    acc["cmp_sum"] += cmp_f.sum().item()
    acc["ref_sq"] += ref_f.square().sum().item()
    acc["cmp_sq"] += cmp_f.square().sum().item()
    acc["dot"] += (ref_f * cmp_f).sum().item()
    acc["diff_sq"] += diff.square().sum().item()
    acc["abs_diff"] += abs_diff.sum().item()
    acc["max_abs_diff"] = max(acc["max_abs_diff"], abs_diff.max().item())


def finalize_accumulator(acc: dict):
    n = acc["numel"]
    if n == 0:
        return {}

    ref_norm = math.sqrt(acc["ref_sq"])
    cmp_norm = math.sqrt(acc["cmp_sq"])
    diff_norm = math.sqrt(acc["diff_sq"])
    cosine = acc["dot"] / (ref_norm * cmp_norm) if ref_norm > 0 and cmp_norm > 0 else float("nan")

    ref_mean = acc["ref_sum"] / n
    cmp_mean = acc["cmp_sum"] / n
    centered_ref_sq = acc["ref_sq"] - n * ref_mean * ref_mean
    centered_cmp_sq = acc["cmp_sq"] - n * cmp_mean * cmp_mean
    centered_dot = acc["dot"] - n * ref_mean * cmp_mean
    if centered_ref_sq > 0 and centered_cmp_sq > 0:
        centered_cosine = centered_dot / math.sqrt(centered_ref_sq * centered_cmp_sq)
    else:
        centered_cosine = float("nan")

    return {
        "num_tensors": acc["num_tensors"],
        "numel": n,
        "ref_norm": ref_norm,
        "cmp_norm": cmp_norm,
        "diff_norm": diff_norm,
        "rel_l2": diff_norm / ref_norm if ref_norm > 0 else float("nan"),
        "cosine": cosine,
        "centered_cosine": centered_cosine,
        "rmse": math.sqrt(acc["diff_sq"] / n),
        "mae": acc["abs_diff"] / n,
        "max_abs_diff": acc["max_abs_diff"],
        "ref_mean": ref_mean,
        "cmp_mean": cmp_mean,
        "norm_ratio_cmp_over_ref": cmp_norm / ref_norm if ref_norm > 0 else float("nan"),
    }


def tensor_metrics(key: str, ref: torch.Tensor, cmp: torch.Tensor):
    acc = new_accumulator()
    update_accumulator(acc, ref, cmp)
    out = finalize_accumulator(acc)
    out.update({
        "key": key,
        "layer": parse_layer(key) if parse_layer(key) is not None else "",
        "component": parse_component(key),
        "param_type": parse_param_type(key),
        "shape": "x".join(str(dim) for dim in ref.shape),
        "ndim": ref.ndim,
        "is_matrix": ref.ndim == 2,
    })
    return out


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


def fmt(value):
    if isinstance(value, int):
        return str(value)
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value):
        return "nan"
    if abs(value) >= 100 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.6f}"


def print_table(title: str, rows: list[dict], columns: list[str]):
    print(f"\n{title}")
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare GPT backbone weights between a no-mHC checkpoint and an mHC checkpoint."
    )
    parser.add_argument("--ref", default=DEFAULT_REF_CKPT, help="Reference checkpoint path.")
    parser.add_argument("--cmp", default=DEFAULT_CMP_CKPT, help="Comparison checkpoint path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--matrix-only",
        action="store_true",
        help="Only compare 2D matrix tensors in per-tensor and aggregate reports.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ref_path = Path(args.ref)
    cmp_path = Path(args.cmp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_ckpt = torch.load(ref_path, map_location="cpu", mmap=True, weights_only=False)
    cmp_ckpt = torch.load(cmp_path, map_location="cpu", mmap=True, weights_only=False)
    ref_state = ref_ckpt["model"]
    cmp_state = cmp_ckpt["model"]

    ref_keys = {key for key in ref_state if is_backbone_key(key)}
    cmp_keys = {key for key in cmp_state if is_backbone_key(key)}
    common_keys = sorted(ref_keys & cmp_keys, key=key_sort_key)
    if args.matrix_only:
        common_keys = [
            key for key in common_keys
            if hasattr(ref_state[key], "ndim") and ref_state[key].ndim == 2
        ]

    missing_in_cmp = sorted(ref_keys - cmp_keys, key=key_sort_key)
    missing_in_ref = sorted(cmp_keys - ref_keys, key=key_sort_key)
    shape_mismatches = []

    per_tensor_rows = []
    global_acc = new_accumulator()
    layer_acc = defaultdict(new_accumulator)
    component_acc = defaultdict(new_accumulator)
    param_type_acc = defaultdict(new_accumulator)

    for key in common_keys:
        ref_tensor = ref_state[key]
        cmp_tensor = cmp_state[key]
        if tuple(ref_tensor.shape) != tuple(cmp_tensor.shape):
            shape_mismatches.append({
                "key": key,
                "ref_shape": tuple(ref_tensor.shape),
                "cmp_shape": tuple(cmp_tensor.shape),
            })
            continue

        row = tensor_metrics(key, ref_tensor, cmp_tensor)
        per_tensor_rows.append(row)

        update_accumulator(global_acc, ref_tensor, cmp_tensor)
        layer = parse_layer(key)
        if layer is not None:
            update_accumulator(layer_acc[layer], ref_tensor, cmp_tensor)
        update_accumulator(component_acc[parse_component(key)], ref_tensor, cmp_tensor)
        update_accumulator(param_type_acc[parse_param_type(key)], ref_tensor, cmp_tensor)

    summary = {
        "ref_checkpoint": str(ref_path),
        "cmp_checkpoint": str(cmp_path),
        "ref_iter_num": ref_ckpt.get("iter_num", ""),
        "cmp_iter_num": cmp_ckpt.get("iter_num", ""),
        "ref_best_val_loss": float(ref_ckpt.get("best_val_loss", float("nan"))),
        "cmp_best_val_loss": float(cmp_ckpt.get("best_val_loss", float("nan"))),
        "num_common_backbone_tensors": len(per_tensor_rows),
        "num_missing_in_cmp": len(missing_in_cmp),
        "num_missing_in_ref": len(missing_in_ref),
        "num_shape_mismatches": len(shape_mismatches),
        **finalize_accumulator(global_acc),
    }

    layer_rows = [
        {"layer": layer, **finalize_accumulator(acc)}
        for layer, acc in sorted(layer_acc.items())
    ]
    component_rows = [
        {"component": component, **finalize_accumulator(acc)}
        for component, acc in sorted(component_acc.items())
    ]
    param_type_rows = [
        {"param_type": param_type, **finalize_accumulator(acc)}
        for param_type, acc in sorted(param_type_acc.items())
    ]

    write_csv(output_dir / "backbone_weight_per_tensor_metrics.csv", per_tensor_rows)
    write_csv(output_dir / "backbone_weight_layer_summary.csv", layer_rows)
    write_csv(output_dir / "backbone_weight_component_summary.csv", component_rows)
    write_csv(output_dir / "backbone_weight_param_type_summary.csv", param_type_rows)
    with (output_dir / "backbone_weight_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "backbone_weight_key_audit.json").open("w") as f:
        json.dump({
            "missing_in_cmp": missing_in_cmp,
            "missing_in_ref": missing_in_ref,
            "shape_mismatches": shape_mismatches,
        }, f, indent=2)

    print_table(
        "Global Backbone Weight Similarity",
        [summary],
        [
            "num_common_backbone_tensors",
            "numel",
            "cosine",
            "centered_cosine",
            "rel_l2",
            "rmse",
            "mae",
            "max_abs_diff",
            "norm_ratio_cmp_over_ref",
        ],
    )
    print_table(
        "Layer Summary",
        layer_rows,
        ["layer", "num_tensors", "numel", "cosine", "centered_cosine", "rel_l2", "rmse", "max_abs_diff"],
    )
    print_table(
        "Param Type Summary",
        param_type_rows,
        ["param_type", "num_tensors", "numel", "cosine", "centered_cosine", "rel_l2", "rmse", "max_abs_diff"],
    )

    print(f"\nWrote:\n  {output_dir / 'backbone_weight_summary.json'}")
    print(f"  {output_dir / 'backbone_weight_per_tensor_metrics.csv'}")
    print(f"  {output_dir / 'backbone_weight_layer_summary.csv'}")
    print(f"  {output_dir / 'backbone_weight_component_summary.csv'}")
    print(f"  {output_dir / 'backbone_weight_param_type_summary.csv'}")
    print(f"  {output_dir / 'backbone_weight_key_audit.json'}")


if __name__ == "__main__":
    main()
