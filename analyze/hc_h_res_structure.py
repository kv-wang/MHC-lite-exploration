from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyper_conn.mhc import (
    admm_l2_with_capacity,
    admm_reverse_kl_sprox_alm_with_capacity,
    admm_reverse_kl_with_capacity,
    cayley_orthogonalize,
    sinkhorn_knopps,
    sinkhorn_with_capacity,
)


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
        return "", "unknown"
    return int(match.group(1)), "attn" if match.group(2) == "hc_attn" else "mlp"


def as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def split_static_h_res(static_alpha: torch.Tensor):
    rows, cols = static_alpha.shape
    num_input_views = cols - rows
    if num_input_views < 0:
        raise ValueError(f"static_alpha shape {tuple(static_alpha.shape)} cannot contain square H_res")
    return static_alpha[:, num_input_views:].float()


def projected_h_res(static_alpha: torch.Tensor, config: dict):
    h = split_static_h_res(static_alpha)
    n = h.shape[0]
    mode = config.get("mhc_h_res_mode", "sinkhorn")

    if config.get("mhc_identity_h_res", False):
        return torch.eye(n, dtype=h.dtype)

    if mode == "sinkhorn":
        return sinkhorn_knopps(h, int(config.get("sinkhorn_iters", 20))).float()
    if mode == "admm_reverse_kl":
        return admm_reverse_kl_with_capacity(
            h,
            base_streams=int(config.get("mhc_adapter_base_streams", n)),
            cap=float(config.get("mhc_adapter_cap", 1.0)),
            iters=int(config.get("mhc_admm_iters", 20)),
            rho=float(config.get("mhc_admm_rho", 1.0)),
            floor=float(config.get("mhc_adapter_admm_input_floor", 1e-30)),
        ).float()
    if mode == "admm_reverse_kl_sprox_alm":
        return admm_reverse_kl_sprox_alm_with_capacity(
            h,
            base_streams=int(config.get("mhc_adapter_base_streams", n)),
            cap=float(config.get("mhc_adapter_cap", 1.0)),
            iters=int(config.get("mhc_admm_iters", 20)),
            rho=float(config.get("mhc_admm_rho", 1.0)),
            floor=float(config.get("mhc_adapter_admm_input_floor", 1e-30)),
            dual_step=float(config.get("mhc_admm_dual_step", 0.5)),
            prox_weight=config.get("mhc_admm_prox_weight", None),
            smooth_beta=float(config.get("mhc_admm_smooth_beta", 0.5)),
            step_scale=float(config.get("mhc_admm_step_scale", 1.0)),
        ).float()
    if mode == "admm_l2":
        return admm_l2_with_capacity(
            h,
            base_streams=int(config.get("mhc_adapter_base_streams", n)),
            cap=float(config.get("mhc_adapter_cap", 1.0)),
            iters=int(config.get("mhc_admm_iters", 20)),
            rho=float(config.get("mhc_admm_rho", 1.0)),
            floor=float(config.get("mhc_adapter_admm_input_floor", 1e-30)),
        ).float()
    if mode == "adapter_cap":
        return sinkhorn_with_capacity(
            h,
            base_streams=int(config.get("mhc_adapter_base_streams", n)),
            cap=float(config.get("mhc_adapter_cap", 1.0)),
            iters=int(config.get("sinkhorn_iters", 20)),
        ).float()
    if mode == "adapter_cap_admm":
        return admm_reverse_kl_with_capacity(
            h,
            base_streams=int(config.get("mhc_adapter_base_streams", n)),
            cap=float(config.get("mhc_adapter_cap", 1.0)),
            iters=int(config.get("mhc_admm_iters", 20)),
            rho=float(config.get("mhc_admm_rho", 1.0)),
            floor=float(config.get("mhc_adapter_admm_input_floor", 1e-30)),
        ).float()
    if mode == "cayley":
        return cayley_orthogonalize(h).float()

    # ALM/S-prox modes train H_res directly in static_alpha.
    if mode in {"alm_signed", "alm_nonnegative", "alm_nonnegative_cap", "alm_signed_sprox", "alm_spectral_sprox"}:
        return h

    # Fallback: report raw residual block rather than silently failing.
    return h


def h_res_metrics(h_res: torch.Tensor, thresholds: tuple[float, ...]):
    h = h_res.float()
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"H_res must be square, got {tuple(h.shape)}")

    n = h.shape[0]
    eye = torch.eye(n, dtype=h.dtype, device=h.device)
    eye_mask = eye.bool()
    offdiag = h[~eye_mask]
    diff = h - eye
    abs_h = h.abs()
    abs_offdiag = offdiag.abs()
    abs_diff = diff.abs()
    row_sums = h.sum(dim=1)
    col_sums = h.sum(dim=0)
    row_res = row_sums - 1.0
    col_res = col_sums - 1.0
    nonneg_violation = (-h).clamp_min(0.0)
    fro_h = torch.linalg.vector_norm(h).clamp_min(1e-30)
    fro_eye = torch.linalg.vector_norm(eye)
    identity_cosine = (h * eye).sum() / (fro_h * fro_eye)
    diag_sum = h.diag().sum()
    total_sum = h.sum()
    abs_total_sum = abs_h.sum().clamp_min(1e-30)
    abs_diag_sum = h.diag().abs().sum()
    abs_offdiag_sum = abs_offdiag.sum()

    nonnegative_h = h.clamp_min(0.0)
    row_probs = nonnegative_h / nonnegative_h.sum(dim=1, keepdim=True).clamp_min(1e-30)
    entropy = -(row_probs * row_probs.clamp_min(1e-30).log()).sum(dim=1)
    entropy_norm = entropy / math.log(n) if n > 1 else torch.zeros_like(entropy)
    top1_values, top1_indices = row_probs.max(dim=1)
    top2_values = row_probs.topk(min(2, n), dim=1).values.sum(dim=1)
    diag_indices = torch.arange(n, device=h.device)

    out = {
        "n": float(n),
        "identity_fro": torch.linalg.vector_norm(diff).item(),
        "identity_rmse": diff.square().mean().sqrt().item(),
        "identity_mae": abs_diff.mean().item(),
        "identity_max_abs": abs_diff.max().item(),
        "identity_cosine": identity_cosine.item(),
        "diag_mass": diag_sum.item(),
        "diag_mass_frac_total": (diag_sum / total_sum.clamp_min(1e-30)).item(),
        "diag_abs_mass_frac": (abs_diag_sum / abs_total_sum).item(),
        "offdiag_abs_mass_frac": (abs_offdiag_sum / abs_total_sum).item(),
        "offdiag_abs_mean": abs_offdiag.mean().item() if offdiag.numel() else float("nan"),
        "offdiag_abs_max": abs_offdiag.max().item() if offdiag.numel() else float("nan"),
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

    for threshold in thresholds:
        suffix = f"{threshold:g}".replace("-", "m").replace(".", "p")
        out[f"sparsity_abs_lt_{suffix}"] = (abs_h < threshold).float().mean().item()
        out[f"offdiag_sparsity_abs_lt_{suffix}"] = (
            (abs_offdiag < threshold).float().mean().item() if offdiag.numel() else float("nan")
        )
    return out


def top_offdiag_edges(h_res: torch.Tensor, k: int):
    h = h_res.float()
    n = h.shape[0]
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((abs(float(h[i, j])), float(h[i, j]), i, j))
    edges.sort(reverse=True)
    return [{"abs_value": abs_value, "value": value, "from": i, "to": j} for abs_value, value, i, j in edges[:k]]


def summarize(rows: list[dict], numeric_keys: list[str], prefix: dict):
    out = dict(prefix)
    out["num_h_res"] = len(rows)
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
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
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value):
        return "nan"
    if abs(value) >= 100 or (0 < abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.6f}"


def print_summary(summary_rows: list[dict]):
    columns = [
        "label",
        "iter_num",
        "best_val_loss",
        "mode",
        "n",
        "reduce",
        "num_h_res",
        "identity_cosine_mean",
        "identity_rmse_mean",
        "identity_max_abs_max",
        "diag_abs_mass_frac_mean",
        "offdiag_abs_mass_frac_mean",
        "offdiag_abs_max_max",
        "row_entropy_norm_mean_mean",
        "row_top1_mass_mean_mean",
        "row_argmax_is_diag_frac_mean",
        "offdiag_sparsity_abs_lt_0p001_mean",
        "offdiag_sparsity_abs_lt_0p01_mean",
        "row_err_max_max",
        "col_err_max_max",
        "nonneg_violation_max_max",
    ]
    available = [col for col in columns if any(col in row for row in summary_rows)]
    print("\nSummary")
    print("| " + " | ".join(available) + " |")
    print("| " + " | ".join(["---"] * len(available)) + " |")
    for row in summary_rows:
        print("| " + " | ".join(fmt(row.get(col, "")) for col in available) + " |")


def analyze_checkpoint(path: Path, thresholds: tuple[float, ...], top_edges: int):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"]
    config = checkpoint.get("config", {})
    keys = sorted([key for key in state if key.endswith(".static_alpha")], key=static_alpha_sort_key)
    if not keys:
        raise ValueError(f"No .static_alpha tensors found in {path}")

    rows = []
    edges = {}
    for key in keys:
        h_res = projected_h_res(state[key], config)
        layer, component = parse_layer_key(key)
        metrics = h_res_metrics(h_res, thresholds)
        row = {
            "checkpoint": str(path),
            "label": path.parent.name,
            "key": key,
            "layer": layer,
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
        "mode": config.get("mhc_h_res_mode", "sinkhorn"),
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
            component_summaries.append(
                summarize(component_rows, numeric_keys, {**prefix, "component": component})
            )
    return summary, component_summaries, rows, edges


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze trained HC/mHC static H_res structure and identity similarity."
    )
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths to analyze.")
    parser.add_argument("--output-dir", default="analyze/hc_h_res_structure_results")
    parser.add_argument("--sparsity-thresholds", default="1e-4,1e-3,1e-2")
    parser.add_argument("--top-edges", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = tuple(float(item) for item in args.sparsity_thresholds.split(",") if item)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    component_rows = []
    layer_rows = []
    all_edges = {}
    for checkpoint in args.checkpoints:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        summary, components, layers, edges = analyze_checkpoint(path, thresholds, args.top_edges)
        summary_rows.append(summary)
        component_rows.extend(components)
        layer_rows.extend(layers)
        all_edges[str(path)] = edges

    write_csv(output_dir / "hc_h_res_summary.csv", summary_rows)
    write_csv(output_dir / "hc_h_res_component_summary.csv", component_rows)
    write_csv(output_dir / "hc_h_res_per_layer_metrics.csv", layer_rows)
    with (output_dir / "hc_h_res_top_offdiag_edges.json").open("w") as f:
        json.dump(all_edges, f, indent=2)

    print_summary(summary_rows)
    print(f"\nWrote:\n  {output_dir / 'hc_h_res_summary.csv'}")
    print(f"  {output_dir / 'hc_h_res_component_summary.csv'}")
    print(f"  {output_dir / 'hc_h_res_per_layer_metrics.csv'}")
    print(f"  {output_dir / 'hc_h_res_top_offdiag_edges.json'}")


if __name__ == "__main__":
    main()
