from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import torch


DEFAULT_SMALL_ALM_CHECKPOINTS = [
    "out-owt-small-mhc-alm-nonnegative-4streams-10000iter/ckpt.pt",
    "out-owt-small-mhc-alm-nonnegative-8streams-reduce-4mean-10000iter/ckpt.pt",
    "out-owt-small-mhc-alm-nonnegative-16streams-reduce-4mean-10000iter/ckpt.pt",
    "out-owt-small-mhc-alm-nonnegative-32streams-reduce-4mean-10000iter/ckpt.pt",
]

OLD_SUM_SMALL_ALM_CHECKPOINTS = [
    "out-owt-small-mhc-alm-nonnegative-8streams-10000iter/ckpt.pt",
    "out-owt-small-mhc-alm-nonnegative-16streams-10000iter/ckpt.pt",
    "out-owt-small-mhc-alm-nonnegative-32streams-10000iter/ckpt.pt",
]


def static_alpha_sort_key(key: str):
    match = re.search(r"transformer\.h\.(\d+)\.(hc_attn|hc_mlp)\.static_alpha$", key)
    if match is None:
        return (10**9, key)
    block_idx = int(match.group(1))
    component_idx = 0 if match.group(2) == "hc_attn" else 1
    return (block_idx, component_idx, key)


def parse_layer_key(key: str):
    match = re.search(r"transformer\.h\.(\d+)\.(hc_attn|hc_mlp)\.static_alpha$", key)
    if match is None:
        return None, "unknown"
    return int(match.group(1)), "attn" if match.group(2) == "hc_attn" else "mlp"


def as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    if value is None:
        return float("nan")
    return float(value)


def extract_h_res(static_alpha: torch.Tensor):
    rows, cols = static_alpha.shape
    num_input_views = cols - rows
    if num_input_views < 0:
        raise ValueError(f"static_alpha shape {tuple(static_alpha.shape)} cannot contain square H_res")
    return static_alpha[:, num_input_views:].float()


def effective_h_res(static_alpha: torch.Tensor, config: dict, state: dict, key: str):
    h = extract_h_res(static_alpha)
    mode = config.get("mhc_h_res_mode")
    if mode not in {"identity_tanh_offdiag", "identity_clip_offdiag"}:
        return h
    n = h.shape[0]
    eye = torch.eye(n, dtype=h.dtype, device=h.device)
    offdiag_mask = 1. - eye
    gamma = torch.as_tensor(
        float(config.get("mhc_h_res_offdiag_init_scale", 0.05)),
        dtype=h.dtype,
        device=h.device,
    )
    scale_key = key.rsplit(".static_alpha", 1)[0] + ".h_res_offdiag_log_scale"
    if scale_key in state:
        gamma = state[scale_key].detach().float().exp().to(dtype=h.dtype, device=h.device)
    if mode == "identity_tanh_offdiag":
        return eye + gamma * offdiag_mask * h.tanh()
    clipped_offdiag = torch.maximum(torch.minimum(h, gamma), -gamma)
    return eye + offdiag_mask * clipped_offdiag


def h_res_metrics(h_res: torch.Tensor, sparsity_thresholds: tuple[float, ...]):
    if h_res.ndim != 2 or h_res.shape[0] != h_res.shape[1]:
        raise ValueError(f"H_res must be square 2D matrix, got shape {tuple(h_res.shape)}")

    h = h_res.float()
    n = h.shape[0]
    eye = torch.eye(n, dtype=h.dtype, device=h.device)
    eye_mask = eye.bool()
    offdiag = h[~eye_mask]
    diff = h - eye
    abs_diff = diff.abs()

    row_sums = h.sum(dim=-1)
    col_sums = h.sum(dim=-2)
    row_res = row_sums - 1.0
    col_res = col_sums - 1.0
    nonneg_violation = (-h).clamp_min(0.0)

    nonnegative_h = h.clamp_min(0.0)
    row_probs = nonnegative_h / nonnegative_h.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    entropy = -(row_probs * row_probs.clamp_min(1e-30).log()).sum(dim=-1)
    entropy_norm = entropy / math.log(n) if n > 1 else torch.zeros_like(entropy)
    top1_values, top1_indices = row_probs.max(dim=-1)
    top2_values = row_probs.topk(min(2, n), dim=-1).values.sum(dim=-1)
    diag_indices = torch.arange(n, device=h.device)

    total_mass = h.sum()
    diag_mass = h.diag().sum()
    offdiag_mass = offdiag.sum()

    metrics = {
        "n": float(n),
        "identity_fro": torch.linalg.vector_norm(diff).item(),
        "identity_rmse": diff.square().mean().sqrt().item(),
        "identity_mae": abs_diff.mean().item(),
        "identity_max_abs": abs_diff.max().item(),
        "diag_mass": diag_mass.item(),
        "diag_mass_frac": (diag_mass / n).item(),
        "offdiag_mass": offdiag_mass.item(),
        "offdiag_mass_frac": (offdiag_mass / n).item(),
        "total_mass": total_mass.item(),
        "row_entropy_mean": entropy.mean().item(),
        "row_entropy_norm_mean": entropy_norm.mean().item(),
        "row_entropy_norm_max": entropy_norm.max().item(),
        "effective_support_mean": entropy.exp().mean().item(),
        "row_top1_mass_mean": top1_values.mean().item(),
        "row_top1_mass_min": top1_values.min().item(),
        "row_top2_mass_mean": top2_values.mean().item(),
        "row_argmax_is_diag_frac": (top1_indices == diag_indices).float().mean().item(),
        "col_argmax_is_diag_frac": (h.argmax(dim=0) == diag_indices).float().mean().item(),
        "symmetry_fro": torch.linalg.vector_norm(h - h.T).item(),
        "symmetry_rmse": (h - h.T).square().mean().sqrt().item(),
        "row_err_max": row_res.abs().max().item(),
        "row_err_mean": row_res.abs().mean().item(),
        "col_err_max": col_res.abs().max().item(),
        "col_err_mean": col_res.abs().mean().item(),
        "nonneg_violation_max": nonneg_violation.max().item(),
        "nonneg_violation_mean": nonneg_violation.mean().item(),
    }

    for threshold in sparsity_thresholds:
        suffix = f"{threshold:g}".replace("-", "m").replace(".", "p")
        metrics[f"sparsity_abs_lt_{suffix}"] = (h.abs() < threshold).float().mean().item()
        if offdiag.numel() > 0:
            metrics[f"offdiag_sparsity_abs_lt_{suffix}"] = (offdiag.abs() < threshold).float().mean().item()
        else:
            metrics[f"offdiag_sparsity_abs_lt_{suffix}"] = float("nan")

    return metrics


def top_offdiag_edges(h_res: torch.Tensor, k: int):
    h = h_res.float()
    n = h.shape[0]
    if n <= 1 or k <= 0:
        return []
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edges.append((float(h[i, j].item()), i, j))
    edges.sort(reverse=True)
    return [
        {"value": value, "from": i, "to": j}
        for value, i, j in edges[:k]
    ]


def summarize(rows: list[dict], numeric_keys: list[str], prefix: dict):
    out = dict(prefix)
    out["num_layers"] = len(rows)
    for key in numeric_keys:
        values = [row[key] for row in rows if key in row and math.isfinite(float(row[key]))]
        if not values:
            continue
        out[f"{key}_mean"] = sum(values) / len(values)
        out[f"{key}_max"] = max(values)
        out[f"{key}_min"] = min(values)
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


def print_summary(summary_rows: list[dict]):
    columns = [
        "label",
        "iter_num",
        "best_val_loss",
        "n",
        "reduce",
        "num_layers",
        "identity_rmse_mean",
        "identity_rmse_max",
        "diag_mass_frac_mean",
        "offdiag_mass_frac_mean",
        "row_entropy_norm_mean_mean",
        "row_top1_mass_mean_mean",
        "sparsity_abs_lt_0p001_mean",
        "sparsity_abs_lt_0p01_mean",
        "symmetry_rmse_mean",
        "row_err_max_max",
        "col_err_max_max",
    ]
    available_columns = [col for col in columns if any(col in row for row in summary_rows)]
    print("\nSummary")
    print("| " + " | ".join(available_columns) + " |")
    print("| " + " | ".join(["---"] * len(available_columns)) + " |")
    for row in summary_rows:
        print("| " + " | ".join(fmt(row.get(col, "")) for col in available_columns) + " |")


def analyze_checkpoint(path: Path, sparsity_thresholds: tuple[float, ...], top_edges: int):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"]
    config = checkpoint.get("config", {})
    static_keys = sorted(
        [key for key in state if key.endswith(".static_alpha")],
        key=static_alpha_sort_key,
    )
    if not static_keys:
        raise ValueError(f"No static_alpha tensors found in {path}")

    rows = []
    edges = {}
    for key in static_keys:
        static_alpha = state[key]
        h_res = effective_h_res(static_alpha, config, state, key)
        layer_idx, component = parse_layer_key(key)
        metrics = h_res_metrics(h_res, sparsity_thresholds)
        row = {
            "checkpoint": str(path),
            "label": path.parent.name,
            "key": key,
            "layer": layer_idx if layer_idx is not None else "",
            "component": component,
            **metrics,
        }
        rows.append(row)
        edges[key] = top_offdiag_edges(h_res, top_edges)

    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key != "layer"
    ]
    prefix = {
        "checkpoint": str(path),
        "label": path.parent.name,
        "iter_num": checkpoint.get("iter_num", ""),
        "best_val_loss": as_float(checkpoint.get("best_val_loss")),
        "n": config.get("hyper_conn_n", rows[0]["n"]),
        "reduce": config.get("hyper_conn_reduce_stream_mode", ""),
        "model": config.get("out_prefix_model", ""),
        "method": config.get("out_prefix_method", ""),
    }
    summary = summarize(rows, numeric_keys, prefix)

    component_summaries = []
    for component in ("attn", "mlp"):
        component_rows = [row for row in rows if row["component"] == component]
        if component_rows:
            component_summaries.append(summarize(
                component_rows,
                numeric_keys,
                {**prefix, "component": component},
            ))

    return summary, component_summaries, rows, edges


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze static H_res structure from mHC ALM-nonnegative checkpoints."
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        help="Checkpoint paths. Defaults to small n=4,8,16,32 ALM-nonnegative checkpoints.",
    )
    parser.add_argument(
        "--include-old-sum",
        action="store_true",
        help="Also include old small n=8,16,32 checkpoints trained with reduce=sum.",
    )
    parser.add_argument(
        "--output-dir",
        default="analyze/h_res_structure_results",
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--sparsity-thresholds",
        default="1e-4,1e-3,1e-2",
        help="Comma-separated absolute-value thresholds for sparsity metrics.",
    )
    parser.add_argument(
        "--top-edges",
        type=int,
        default=10,
        help="Number of strongest off-diagonal edges to save per layer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_paths = args.checkpoints or DEFAULT_SMALL_ALM_CHECKPOINTS
    if args.include_old_sum:
        checkpoint_paths = list(checkpoint_paths) + OLD_SUM_SMALL_ALM_CHECKPOINTS

    thresholds = tuple(float(item) for item in args.sparsity_thresholds.split(",") if item)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []
    all_component_rows = []
    all_layer_rows = []
    all_edges = {}

    for checkpoint_path in checkpoint_paths:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(path)
        summary, component_rows, layer_rows, edges = analyze_checkpoint(
            path,
            sparsity_thresholds=thresholds,
            top_edges=args.top_edges,
        )
        all_summary_rows.append(summary)
        all_component_rows.extend(component_rows)
        all_layer_rows.extend(layer_rows)
        all_edges[str(path)] = edges

    write_csv(output_dir / "h_res_summary.csv", all_summary_rows)
    write_csv(output_dir / "h_res_component_summary.csv", all_component_rows)
    write_csv(output_dir / "h_res_per_layer_metrics.csv", all_layer_rows)
    with (output_dir / "h_res_top_offdiag_edges.json").open("w") as f:
        json.dump(all_edges, f, indent=2)

    print_summary(all_summary_rows)
    print(f"\nWrote:\n  {output_dir / 'h_res_summary.csv'}")
    print(f"  {output_dir / 'h_res_component_summary.csv'}")
    print(f"  {output_dir / 'h_res_per_layer_metrics.csv'}")
    print(f"  {output_dir / 'h_res_top_offdiag_edges.json'}")


if __name__ == "__main__":
    main()
