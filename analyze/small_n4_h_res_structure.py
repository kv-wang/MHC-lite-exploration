from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyper_conn.mhc import (
    admm_doubly_stochastic,
    admm_l2_with_capacity,
    admm_reverse_kl_sprox_alm_with_capacity,
    admm_reverse_kl_with_capacity,
    sinkhorn_knopps,
)


DEFAULT_SMALL_N4_CHECKPOINTS = [
    "out-owt-small-mhc-sinkhorn-4streams-10000iter/ckpt.pt",
    "out-owt-small-mhc-identity-h-res-4streams-10000iter/ckpt.pt",
    "out-owt-small-mhc-alm-nonnegative-4streams-10000iter/ckpt.pt",
    "out-owt-small-mhc-alm-signed-sprox-4streams-10000iter/ckpt.pt",
    "out-owt-small-mhc-admm-reverse-kl-4streams-10000iter/ckpt.pt",
    "out-owt-small-mhc-admm-l2-4streams-10000iter/ckpt.pt",
    "out-owt-small-mhc-admm-reverse-kl-sprox-alm-4streams-10000iter/ckpt.pt",
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


def extract_h_res_logits(static_alpha: torch.Tensor):
    rows, cols = static_alpha.shape
    num_input_views = cols - rows
    if num_input_views < 0:
        raise ValueError(f"static_alpha shape {tuple(static_alpha.shape)} cannot contain square H_res")
    return static_alpha[:, num_input_views:].float()


def infer_h_res_mode(path: Path, config: dict):
    if config.get("mhc_identity_h_res", False):
        return "identity"
    mode = config.get("mhc_h_res_mode")
    if mode:
        return mode

    label = path.parent.name
    if "identity-h-res" in label:
        return "identity"
    if "admm-reverse-kl-sprox-alm" in label:
        return "admm_reverse_kl_sprox_alm"
    if "admm-reverse-kl" in label:
        return "admm_reverse_kl"
    if "admm-l2" in label:
        return "admm_l2"
    if "alm-nonnegative" in label:
        return "alm_nonnegative"
    if "alm-spectral-sprox" in label:
        return "alm_spectral_sprox"
    if "alm-signed-sprox" in label:
        return "alm_signed_sprox"
    if "sinkhorn" in label:
        return "sinkhorn"
    return "sinkhorn"


def projected_h_res(
    h_res_logits: torch.Tensor,
    mode: str,
    config: dict,
):
    n = h_res_logits.shape[-1]
    sinkhorn_iters = int(config.get("sinkhorn_iters", 20))
    admm_iters = int(config.get("mhc_admm_iters", 20))
    admm_rho = float(config.get("mhc_admm_rho", 1.0))
    floor = float(config.get("mhc_adapter_admm_input_floor", 1e-30))
    cap = float(config.get("mhc_adapter_cap", 1.0))
    base_streams = int(config.get("mhc_adapter_base_streams", n))

    if mode == "identity":
        return torch.eye(n, dtype=h_res_logits.dtype, device=h_res_logits.device)
    if mode == "sinkhorn":
        return sinkhorn_knopps(h_res_logits, sinkhorn_iters)
    if mode == "admm_reverse_kl":
        return admm_reverse_kl_with_capacity(
            h_res_logits,
            base_streams=base_streams,
            cap=cap,
            iters=admm_iters,
            rho=admm_rho,
            floor=floor,
        )
    if mode == "admm_l2":
        return admm_l2_with_capacity(
            h_res_logits,
            base_streams=base_streams,
            cap=cap,
            iters=admm_iters,
            rho=admm_rho,
            floor=floor,
        )
    if mode == "admm_reverse_kl_sprox_alm":
        return admm_reverse_kl_sprox_alm_with_capacity(
            h_res_logits,
            base_streams=base_streams,
            cap=cap,
            iters=admm_iters,
            rho=admm_rho,
            floor=floor,
            dual_step=float(config.get("mhc_admm_dual_step", 0.5)),
            prox_weight=config.get("mhc_admm_prox_weight"),
            smooth_beta=float(config.get("mhc_admm_smooth_beta", 0.5)),
            step_scale=float(config.get("mhc_admm_step_scale", 1.0)),
        )
    if mode in {"alm_nonnegative", "alm_signed", "alm_signed_sprox", "alm_spectral_sprox"}:
        return h_res_logits
    if mode == "admm":
        return admm_doubly_stochastic(h_res_logits, iters=admm_iters, rho=admm_rho)

    raise ValueError(f"Unsupported mhc_h_res_mode for analysis: {mode}")


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

    diag_mass = h.diag().sum()
    offdiag_mass = offdiag.sum()
    total_mass = h.sum()
    abs_offdiag = offdiag.abs()

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
        "offdiag_abs_mean": abs_offdiag.mean().item() if abs_offdiag.numel() else float("nan"),
        "offdiag_abs_max": abs_offdiag.max().item() if abs_offdiag.numel() else float("nan"),
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
        metrics[f"offdiag_sparsity_abs_lt_{suffix}"] = (
            (offdiag.abs() < threshold).float().mean().item()
            if offdiag.numel() > 0
            else float("nan")
        )

    return metrics


def top_offdiag_edges(h_res: torch.Tensor, k: int):
    h = h_res.float()
    n = h.shape[0]
    if n <= 1 or k <= 0:
        return []
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((float(h[i, j].item()), i, j))
    edges.sort(key=lambda item: abs(item[0]), reverse=True)
    return [{"value": value, "from": i, "to": j} for value, i, j in edges[:k]]


def summarize(rows: list[dict], numeric_keys: list[str], prefix: dict):
    out = dict(prefix)
    out["num_layers"] = len(rows)
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
        "h_res_mode",
        "iter_num",
        "best_val_loss",
        "num_layers",
        "identity_rmse_mean",
        "identity_rmse_max",
        "identity_max_abs_mean",
        "diag_mass_frac_mean",
        "offdiag_abs_mean_mean",
        "offdiag_abs_max_mean",
        "row_entropy_norm_mean_mean",
        "row_top1_mass_mean_mean",
        "offdiag_sparsity_abs_lt_0p001_mean",
        "offdiag_sparsity_abs_lt_0p01_mean",
        "row_err_max_max",
        "col_err_max_max",
        "nonneg_violation_max_max",
    ]
    available_columns = [col for col in columns if any(col in row for row in summary_rows)]
    print("\nSummary")
    print("| " + " | ".join(available_columns) + " |")
    print("| " + " | ".join(["---"] * len(available_columns)) + " |")
    for row in summary_rows:
        print("| " + " | ".join(fmt(row.get(col, "")) for col in available_columns) + " |")


def analyze_checkpoint(path: Path, sparsity_thresholds: tuple[float, ...], top_edges: int):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["model"]
    config = checkpoint.get("config", {})
    mode = infer_h_res_mode(path, config)
    static_keys = sorted(
        [key for key in state if key.endswith(".static_alpha")],
        key=static_alpha_sort_key,
    )
    if not static_keys:
        raise ValueError(f"No static_alpha tensors found in {path}")

    rows = []
    edges = {}
    for key in static_keys:
        h_res_logits = extract_h_res_logits(state[key])
        h_res = projected_h_res(h_res_logits, mode=mode, config=config)
        if torch.isnan(h_res).any() or torch.isinf(h_res).any():
            raise ValueError(f"NaN/Inf in projected H_res for {path}: {key}")
        layer_idx, component = parse_layer_key(key)
        metrics = h_res_metrics(h_res, sparsity_thresholds)
        row = {
            "checkpoint": str(path),
            "label": path.parent.name,
            "h_res_mode": mode,
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
        "h_res_mode": mode,
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

    del checkpoint
    gc.collect()

    return summary, component_summaries, rows, edges


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze whether small n=4 mHC static H_res checkpoints are identity-like and sparse."
    )
    parser.add_argument(
        "checkpoints",
        nargs="*",
        help="Checkpoint paths. Defaults to known small n=4 mHC checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="analyze/small_n4_h_res_structure_results",
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
        help="Number of strongest absolute off-diagonal edges to save per layer.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a default checkpoint path is missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_paths = args.checkpoints or DEFAULT_SMALL_N4_CHECKPOINTS
    thresholds = tuple(float(item) for item in args.sparsity_thresholds.split(",") if item)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []
    all_component_rows = []
    all_layer_rows = []
    all_edges = {}
    missing = []

    for checkpoint_path in checkpoint_paths:
        path = Path(checkpoint_path)
        if not path.exists():
            missing.append(str(path))
            if args.strict or args.checkpoints:
                raise FileNotFoundError(path)
            continue
        summary, component_rows, layer_rows, edges = analyze_checkpoint(
            path,
            sparsity_thresholds=thresholds,
            top_edges=args.top_edges,
        )
        all_summary_rows.append(summary)
        all_component_rows.extend(component_rows)
        all_layer_rows.extend(layer_rows)
        all_edges[str(path)] = edges

    write_csv(output_dir / "small_n4_h_res_summary.csv", all_summary_rows)
    write_csv(output_dir / "small_n4_h_res_component_summary.csv", all_component_rows)
    write_csv(output_dir / "small_n4_h_res_per_layer_metrics.csv", all_layer_rows)
    with (output_dir / "small_n4_h_res_top_offdiag_edges.json").open("w") as f:
        json.dump(all_edges, f, indent=2)

    if missing:
        print("Skipped missing default checkpoints:")
        for path in missing:
            print(f"  {path}")
    print_summary(all_summary_rows)
    print(f"\nWrote:\n  {output_dir / 'small_n4_h_res_summary.csv'}")
    print(f"  {output_dir / 'small_n4_h_res_component_summary.csv'}")
    print(f"  {output_dir / 'small_n4_h_res_per_layer_metrics.csv'}")
    print(f"  {output_dir / 'small_n4_h_res_top_offdiag_edges.json'}")


if __name__ == "__main__":
    main()
