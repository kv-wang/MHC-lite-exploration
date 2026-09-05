#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from itertools import product

import numpy as np
import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import GPT, GPTConfig
from hyper_conn.mhc import (
    admm_l2_with_capacity,
    admm_reverse_kl_sprox_alm_with_capacity,
    admm_reverse_kl_with_capacity,
    admm_with_capacity,
    cayley_orthogonalize,
    sinkhorn_knopps,
    sinkhorn_with_capacity,
    sinkhorn_with_marginals,
)


def strip_compile_prefix(state_dict):
    prefix = "_orig_mod."
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check first-order KKT conditions for H_res=I in the H-space "
            "doubly-stochastic subproblem, holding all non-H_res weights fixed."
        )
    )
    parser.add_argument(
        "--ckpt",
        default="out-owt-medium-mhc-sinkhorn-4streams-10000iter/ckpt.pt",
        help="Checkpoint path.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset override. Defaults to checkpoint config.")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=0, help="Defaults to checkpoint block_size.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--kkt-tol", type=float, default=1e-5)
    parser.add_argument(
        "--h-res-test-point",
        choices=("identity", "checkpoint", "checkpoint-raw"),
        default="checkpoint",
        help=(
            "H_res point to test. Default 'checkpoint' tests each module at its learned/effective "
            "checkpoint H_res; 'identity' preserves the older forced-identity analysis; "
            "'checkpoint-raw' uses the raw static H_res/logits block."
        ),
    )
    parser.add_argument(
        "--kkt-check-mode",
        choices=(
            "analytic",
            "random-sinkhorn",
            "projected-gradient",
            "global-random-sinkhorn",
            "single-layer-random-sinkhorn",
            "both",
        ),
        default="projected-gradient",
        help=(
            "Which pass/fail criterion to report as kkt_pass. 'analytic' uses exact "
            "first-order KKT inequalities; 'random-sinkhorn' uses sampled feasible "
            "checkpoint-H_res + positive random noise projected by Sinkhorn; "
            "'projected-gradient' uses the projected-gradient fixed-point residual; "
            "'global-random-sinkhorn' runs true forward passes with all layers "
            "perturbed at once; 'single-layer-random-sinkhorn' randomly selects one "
            "layer and runs true forward passes with only that layer perturbed; "
            "'both' requires analytic and per-layer random checks."
        ),
    )
    parser.add_argument(
        "--random-sinkhorn-samples",
        type=int,
        default=0,
        help="Number of random Sinkhorn feasible perturbations sampled per H_res module.",
    )
    parser.add_argument(
        "--random-sinkhorn-eps",
        type=float,
        default=1e-2,
        help="Scale of the positive random matrix added to the tested H_res before Sinkhorn.",
    )
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=1000,
        help="Number of row/column normalizations for random Sinkhorn perturbations.",
    )
    parser.add_argument(
        "--sinkhorn-tol",
        type=float,
        default=1e-7,
        help="Early-stop tolerance for row/column sum errors during Sinkhorn.",
    )
    parser.add_argument(
        "--global-random-sinkhorn-trials",
        type=int,
        default=-1,
        help=(
            "Number of true forward trials where every layer's H_res is perturbed "
            "simultaneously. Default: 64 for global-random-sinkhorn mode, otherwise 0."
        ),
    )
    parser.add_argument(
        "--single-layer-random-sinkhorn-trials",
        type=int,
        default=-1,
        help=(
            "Number of true forward trials where one selected layer's H_res is "
            "perturbed and all other layers stay at their tested checkpoint H_res. Default: 512 for "
            "single-layer-random-sinkhorn mode, otherwise 0."
        ),
    )
    parser.add_argument(
        "--single-layer-module-ordinal",
        type=int,
        default=-1,
        help=(
            "MHC module ordinal to perturb in single-layer-random-sinkhorn mode. "
            "Default -1 selects one module reproducibly from --seed."
        ),
    )
    parser.add_argument(
        "--loss-decrease-tol",
        type=float,
        default=0.0,
        help="Tolerance for counting a perturbed forward pass as a loss decrease.",
    )
    parser.add_argument(
        "--projected-gradient-step-size",
        type=float,
        default=0.01,
        help="Step size eta for the projected-gradient KKT residual.",
    )
    parser.add_argument(
        "--projected-gradient-update",
        choices=("dykstra", "sprox-alm", "alm-projection"),
        default="sprox-alm",
        help=(
            "Update map used by projected-gradient KKT residual. 'dykstra' uses "
            "Euclidean projection onto the Birkhoff polytope; 'sprox-alm' uses "
            "the S-prox-ALM primal update with row/column constraints handled by "
            "dual/penalty terms and nonnegativity handled by clamping; "
            "'alm-projection' projects H_res - eta * grad with an ALM inner loop."
        ),
    )
    parser.add_argument(
        "--projected-gradient-tol",
        type=float,
        default=-1.0,
        help="Tolerance for projected-gradient KKT residual. Default: --kkt-tol.",
    )
    parser.add_argument(
        "--birkhoff-projection-iters",
        type=int,
        default=1000,
        help="Maximum Dykstra iterations for Euclidean projection onto the Birkhoff polytope.",
    )
    parser.add_argument(
        "--birkhoff-projection-tol",
        type=float,
        default=1e-9,
        help="Convergence tolerance for Birkhoff projection feasibility and iterate changes.",
    )
    parser.add_argument(
        "--sprox-alm-iters",
        type=int,
        default=500,
        help=(
            "Number of S-prox-ALM primal-dual updates for --projected-gradient-update=sprox-alm, "
            "or ALM inner-loop updates for --projected-gradient-update=alm-projection."
        ),
    )
    parser.add_argument(
        "--sprox-alm-rho",
        type=float,
        default=5.0,
        help="Augmented-Lagrangian penalty rho for S-prox-ALM and ALM-projection updates.",
    )
    parser.add_argument(
        "--sprox-alm-dual-step",
        type=float,
        default=1.0,
        help="Dual ascent step alpha for S-prox-ALM and ALM-projection updates.",
    )
    parser.add_argument(
        "--sprox-alm-prox-weight",
        type=float,
        default=1.0,
        help=(
            "Proximal/objective weight p for S-prox-ALM and ALM-projection updates. "
            "Negative means use --sprox-alm-rho."
        ),
    )
    parser.add_argument(
        "--alm-projection-step-size",
        type=float,
        default=-1.0,
        help=(
            "Primal step size for --projected-gradient-update=alm-projection. "
            "Default: --projected-gradient-step-size."
        ),
    )
    parser.add_argument(
        "--sprox-alm-smooth-beta",
        type=float,
        default=0.5,
        help="Smoothing beta for the auxiliary z update in --projected-gradient-update=sprox-alm.",
    )
    parser.add_argument(
        "--sprox-alm-dual-init",
        choices=("analytic", "zero"),
        default="analytic",
        help=(
            "Initial row/column equality multipliers for the S-prox-ALM fixed-point check. "
            "'analytic' solves for multipliers on the current positive support of H_res."
        ),
    )
    parser.add_argument(
        "--trial-eval-mode",
        choices=("forward-loss", "first-order"),
        default="forward-loss",
        help=(
            "How global/single-layer random Sinkhorn trials are evaluated. "
            "'forward-loss' runs a forward pass and compares loss deltas; "
            "'first-order' does not forward perturbed H_res and instead checks "
            "the directional derivative <G,D>."
        ),
    )
    parser.add_argument("--finite-diff-top-k", type=int, default=0)
    parser.add_argument("--finite-diff-eps", type=float, default=1e-3)
    parser.add_argument(
        "--finite-diff-direction",
        choices=("auto", "pair-swap", "random-sinkhorn"),
        default="auto",
        help="Which perturbation family to finite-difference check.",
    )
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="none")
    parser.add_argument("--save-gradients", action="store_true")
    parser.add_argument("--output", default="", help="Output JSON path.")
    parser.add_argument(
        "--sweep-seeds",
        default="",
        help="Comma/space/slash-separated seeds for a robustness sweep, e.g. '1337,42,2024,9001'.",
    )
    parser.add_argument(
        "--sweep-splits",
        default="",
        help="Comma/space/slash-separated splits for a robustness sweep, e.g. 'train,val'.",
    )
    parser.add_argument(
        "--sweep-batch-sizes",
        default="",
        help="Comma/space/slash-separated batch sizes for a robustness sweep, e.g. '2,4' or '2/4'.",
    )
    parser.add_argument(
        "--sweep-output",
        default="",
        help="Aggregate JSON output path for sweep mode. Individual run JSON files are still written.",
    )
    return parser.parse_args()


def parse_list_arg(value):
    if not value:
        return []
    normalized = value.replace("/", ",").replace(";", ",")
    parts = []
    for chunk in normalized.split(","):
        parts.extend(item for item in chunk.split() if item)
    return parts


def parse_int_list_arg(value, name):
    values = []
    for item in parse_list_arg(value):
        try:
            parsed = int(item)
        except ValueError as exc:
            raise ValueError(f"{name} expects integers, got {item!r}") from exc
        if parsed <= 0:
            raise ValueError(f"{name} values must be positive, got {parsed}")
        values.append(parsed)
    return values


def parse_split_list_arg(value):
    splits = parse_list_arg(value)
    valid = {"train", "val"}
    invalid = [split for split in splits if split not in valid]
    if invalid:
        raise ValueError(f"--sweep-splits only supports train/val, got {invalid}")
    return splits


def sweep_values(args):
    seeds = parse_int_list_arg(args.sweep_seeds, "--sweep-seeds") or [args.seed]
    splits = parse_split_list_arg(args.sweep_splits) or [args.split]
    batch_sizes = parse_int_list_arg(args.sweep_batch_sizes, "--sweep-batch-sizes") or [args.batch_size]
    return seeds, splits, batch_sizes


def load_batches(data_dir, split, batch_size, seq_len, num_batches, seed, device):
    data_path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    if len(data) <= seq_len + 1:
        raise ValueError(f"{data_path} is too short for seq_len={seq_len}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches = []
    for _ in range(num_batches):
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
        batches.append((x, y))
    return batches


def iter_mhc_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "static_h_res") and hasattr(module, "num_residual_streams"):
            yield name, module


def projected_static_checkpoint_h_res(module, n, device):
    if not hasattr(module, "static_h_res"):
        raise RuntimeError("checkpoint H_res testing requires module.static_h_res().")

    with torch.no_grad():
        h = module.static_h_res().detach().float().to(device=device)
        if h.ndim != 2 or tuple(h.shape) != (n, n):
            raise ValueError(
                "checkpoint static H_res/logits block must be a square 2D matrix "
                f"with shape {(n, n)}, got {tuple(h.shape)}"
            )

        hyper_conn_type = getattr(module, "hyper_conn_type", "mhc")
        mode = getattr(module, "mhc_h_res_mode", "sinkhorn")

        if hyper_conn_type == "hc":
            return h
        if getattr(module, "mhc_identity_h_res", False):
            return torch.eye(n, device=device, dtype=torch.float32)
        if mode in {"identity_tanh_offdiag", "identity_clip_offdiag"}:
            if not hasattr(module, "effective_static_h_res"):
                raise RuntimeError(f"{mode} requires module.effective_static_h_res().")
            return module.effective_static_h_res().detach().float().to(device=device)
        if mode == "sinkhorn":
            return sinkhorn_knopps(h, int(getattr(module, "sinkhorn_iters", 20))).float()
        if mode == "admm_reverse_kl":
            return admm_reverse_kl_with_capacity(
                h,
                base_streams=int(getattr(module, "mhc_adapter_base_streams", n)),
                cap=float(getattr(module, "mhc_adapter_cap", 1.0)),
                iters=int(getattr(module, "mhc_admm_iters", 20)),
                rho=float(getattr(module, "mhc_admm_rho", 1.0)),
                floor=float(getattr(module, "mhc_adapter_admm_input_floor", 1e-30)),
            ).float()
        if mode == "admm_reverse_kl_sprox_alm":
            return admm_reverse_kl_sprox_alm_with_capacity(
                h,
                base_streams=int(getattr(module, "mhc_adapter_base_streams", n)),
                cap=float(getattr(module, "mhc_adapter_cap", 1.0)),
                iters=int(getattr(module, "mhc_admm_iters", 20)),
                rho=float(getattr(module, "mhc_admm_rho", 1.0)),
                floor=float(getattr(module, "mhc_adapter_admm_input_floor", 1e-30)),
                dual_step=float(getattr(module, "mhc_admm_dual_step", 0.5)),
                prox_weight=getattr(module, "mhc_admm_prox_weight", None),
                smooth_beta=float(getattr(module, "mhc_admm_smooth_beta", 0.5)),
                step_scale=float(getattr(module, "mhc_admm_step_scale", 1.0)),
            ).float()
        if mode == "admm_l2":
            return admm_l2_with_capacity(
                h,
                base_streams=int(getattr(module, "mhc_adapter_base_streams", n)),
                cap=float(getattr(module, "mhc_adapter_cap", 1.0)),
                iters=int(getattr(module, "mhc_admm_iters", 20)),
                rho=float(getattr(module, "mhc_admm_rho", 1.0)),
                floor=float(getattr(module, "mhc_adapter_admm_input_floor", 1e-30)),
            ).float()
        if mode == "adapter_epsilon":
            target = torch.full((n,), float(getattr(module, "mhc_adapter_epsilon", 0.1)), device=device)
            base_streams = min(int(getattr(module, "mhc_adapter_base_streams", n)), n)
            target[:base_streams] = 1.0
            return sinkhorn_with_marginals(
                h,
                row_targets=target,
                col_targets=target,
                iters=int(getattr(module, "sinkhorn_iters", 20)),
            ).float()
        if mode == "adapter_cap":
            return sinkhorn_with_capacity(
                h,
                base_streams=min(int(getattr(module, "mhc_adapter_base_streams", n)), n),
                cap=float(getattr(module, "mhc_adapter_cap", 1.0)),
                iters=int(getattr(module, "sinkhorn_iters", 20)),
            ).float()
        if mode == "adapter_cap_admm":
            return admm_with_capacity(
                h,
                base_streams=min(int(getattr(module, "mhc_adapter_base_streams", n)), n),
                cap=float(getattr(module, "mhc_adapter_cap", 1.0)),
                iters=int(getattr(module, "mhc_admm_iters", 20)),
                rho=float(getattr(module, "mhc_admm_rho", 1.0)),
            ).float()
        if mode == "cayley":
            return cayley_orthogonalize(h).float()

        return h


def h_res_test_point_matrix(module, n, device, test_point):
    if test_point == "identity":
        return torch.eye(n, device=device, dtype=torch.float32)
    if test_point == "checkpoint":
        return projected_static_checkpoint_h_res(module, n=n, device=device)
    if test_point == "checkpoint-raw":
        if not hasattr(module, "effective_static_h_res"):
            raise RuntimeError("checkpoint H_res testing requires module.effective_static_h_res().")
        with torch.no_grad():
            h_res = module.effective_static_h_res().detach().float().to(device=device)
        if h_res.ndim != 2 or tuple(h_res.shape) != (n, n):
            raise ValueError(
                "checkpoint effective H_res must be a square 2D matrix with shape "
                f"{(n, n)}, got {tuple(h_res.shape)}"
            )
        return h_res
    raise ValueError(f"unsupported H_res test point: {test_point}")


def set_forced_h_res(model, device, test_point):
    entries = []
    for name, module in iter_mhc_modules(model):
        n = int(module.num_residual_streams) * int(getattr(module, "num_fracs", 1))
        base_h_res = h_res_test_point_matrix(module, n=n, device=device, test_point=test_point)
        param = nn.Parameter(base_h_res.clone())
        param._kkt_base_h_res = base_h_res.detach().clone()
        module._kkt_forced_h_res = param
        entries.append((name, module, param))
    if not entries:
        raise RuntimeError("No mHC modules with static_h_res() were found.")
    return entries


def forced_param_base_h_res(param):
    base_h_res = getattr(param, "_kkt_base_h_res", None)
    if base_h_res is None:
        base_h_res = param.detach()
    return base_h_res.detach().to(device=param.device, dtype=param.dtype)


def reset_forced_params_to_base(entries):
    with torch.no_grad():
        for _, _, param in entries:
            param.copy_(forced_param_base_h_res(param))


def autocast_context(device, amp_dtype):
    if "cuda" not in device or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def compute_loss(model, batches, device, amp_dtype, backward=False):
    total = 0.0
    ctx = autocast_context(device, amp_dtype)
    grad_ctx = nullcontext() if backward else torch.no_grad()
    with grad_ctx:
        for x, y in batches:
            with ctx:
                _, loss = model(x, y)
                scaled = loss / len(batches)
            if backward:
                scaled.backward()
            total += float(loss.detach().float().item()) / len(batches)
    return total


def kkt_stats_from_gradient(grad, tol):
    g = grad.detach().float().cpu()
    n = g.shape[0]
    diag = g.diag()

    pair_min = math.inf
    pair_argmin = None
    for i in range(n):
        for j in range(i + 1, n):
            val = float(g[i, j] + g[j, i] - g[i, i] - g[j, j])
            if val < pair_min:
                pair_min = val
                pair_argmin = [i, j]

    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Difference constraint for stationarity multipliers:
            # a_j <= a_i + G_ij - G_jj.
            edges.append((i, j, float(g[i, j] - diag[j])))

    dist = [0.0] * n
    for _ in range(max(0, n - 1)):
        changed = False
        for i, j, w in edges:
            cand = dist[i] + w
            if cand < dist[j]:
                dist[j] = cand
                changed = True
        if not changed:
            break

    negative_cycle = False
    worst_relaxation = 0.0
    worst_edge = None
    for i, j, w in edges:
        violation = dist[j] - (dist[i] + w)
        if violation > worst_relaxation:
            worst_relaxation = violation
            worst_edge = [i, j]
        if violation > tol:
            negative_cycle = True

    a = torch.tensor(dist, dtype=g.dtype)
    b = -diag - a
    slack = g + a[:, None] + b[None, :]
    offdiag_mask = ~torch.eye(n, dtype=torch.bool)
    offdiag_slack = slack[offdiag_mask]

    return {
        "n": n,
        "kkt_pass": bool(not negative_cycle and float(offdiag_slack.min()) >= -tol),
        "tol": tol,
        "negative_cycle_detected": bool(negative_cycle),
        "worst_difference_constraint_violation": float(worst_relaxation),
        "worst_difference_constraint_edge": worst_edge,
        "min_offdiag_kkt_slack": float(offdiag_slack.min().item()),
        "mean_offdiag_kkt_slack": float(offdiag_slack.mean().item()),
        "min_pair_swap_derivative": pair_min,
        "min_pair_swap_pair": pair_argmin,
        "grad_l2_norm": float(torch.linalg.vector_norm(g).item()),
        "grad_abs_mean": float(g.abs().mean().item()),
        "grad_abs_max": float(g.abs().max().item()),
        "diag_grad_mean": float(diag.mean().item()),
        "diag_grad_min": float(diag.min().item()),
        "diag_grad_max": float(diag.max().item()),
        "offdiag_grad_abs_mean": float(g[offdiag_mask].abs().mean().item()),
        "multipliers_a": [float(x) for x in a.tolist()],
        "multipliers_b": [float(x) for x in b.tolist()],
    }


def support_kkt_multipliers(h_res, grad, tol):
    h = h_res.detach().float().cpu()
    g = grad.detach().float().cpu()
    n = g.shape[0]
    if h.shape != g.shape:
        raise ValueError(f"H_res point shape {tuple(h.shape)} does not match gradient shape {tuple(g.shape)}")

    row_sum_abs_err = float((h.sum(dim=1) - 1.0).abs().max().item())
    col_sum_abs_err = float((h.sum(dim=0) - 1.0).abs().max().item())
    min_entry = float(h.min().item())
    nonneg_violation = max(0.0, -min_entry)

    positive = h > tol
    num_nodes = 2 * n
    adjacency = [[] for _ in range(num_nodes)]
    for i in range(n):
        for j in range(n):
            if positive[i, j]:
                row_node = i
                col_node = n + j
                value = -float(g[i, j].item())
                adjacency[row_node].append((col_node, value))
                adjacency[col_node].append((row_node, value))

    base = [math.nan] * num_nodes
    component = [-1] * num_nodes
    inconsistent_equalities = False
    equality_violation = 0.0
    comp_count = 0
    for start in range(num_nodes):
        if component[start] >= 0:
            continue
        component[start] = comp_count
        base[start] = 0.0
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt, edge_value in adjacency[node]:
                expected = edge_value - base[node]
                if component[nxt] < 0:
                    component[nxt] = comp_count
                    base[nxt] = expected
                    stack.append(nxt)
                else:
                    violation = abs(base[nxt] - expected)
                    equality_violation = max(equality_violation, violation)
                    if violation > tol:
                        inconsistent_equalities = True
        comp_count += 1

    edges = []
    same_component_ineq_violation = 0.0
    for i in range(n):
        row_comp = component[i]
        row_base = base[i]
        for j in range(n):
            if positive[i, j]:
                continue
            col_comp = component[n + j]
            col_base = base[n + j]
            slack_without_offsets = row_base + col_base + float(g[i, j].item())
            if row_comp == col_comp:
                same_component_ineq_violation = max(same_component_ineq_violation, -slack_without_offsets)
            else:
                # Need row_base + delta_row + col_base - delta_col + G_ij >= 0,
                # i.e. delta_col <= delta_row + row_base + col_base + G_ij.
                edges.append((row_comp, col_comp, slack_without_offsets))

    dist = [0.0] * comp_count
    for _ in range(max(0, comp_count - 1)):
        changed = False
        for src, dst, weight in edges:
            cand = dist[src] + weight
            if cand < dist[dst]:
                dist[dst] = cand
                changed = True
        if not changed:
            break

    difference_violation = 0.0
    worst_edge = None
    for src, dst, weight in edges:
        violation = dist[dst] - (dist[src] + weight)
        if violation > difference_violation:
            difference_violation = violation
            worst_edge = [src, dst]

    a = torch.empty(n, dtype=g.dtype)
    b = torch.empty(n, dtype=g.dtype)
    for i in range(n):
        a[i] = base[i] + dist[component[i]]
    for j in range(n):
        b[j] = base[n + j] - dist[component[n + j]]

    slack = g + a[:, None] + b[None, :]
    zero_slack = slack[~positive]
    positive_abs_slack = slack[positive].abs()
    min_zero_slack = float(zero_slack.min().item()) if zero_slack.numel() else math.inf
    max_positive_abs_slack = float(positive_abs_slack.max().item()) if positive_abs_slack.numel() else 0.0
    pass_check = (
        row_sum_abs_err <= tol
        and col_sum_abs_err <= tol
        and nonneg_violation <= tol
        and not inconsistent_equalities
        and equality_violation <= tol
        and same_component_ineq_violation <= tol
        and difference_violation <= tol
        and min_zero_slack >= -tol
        and max_positive_abs_slack <= tol
    )

    return {
        "support_kkt_pass": bool(pass_check),
        "support_kkt_row_sum_abs_err": row_sum_abs_err,
        "support_kkt_col_sum_abs_err": col_sum_abs_err,
        "support_kkt_min_entry": min_entry,
        "support_kkt_nonneg_violation": nonneg_violation,
        "support_kkt_positive_count": int(positive.sum().item()),
        "support_kkt_equality_violation": float(equality_violation),
        "support_kkt_same_component_ineq_violation": float(same_component_ineq_violation),
        "support_kkt_difference_violation": float(difference_violation),
        "support_kkt_worst_difference_edge": worst_edge,
        "support_kkt_min_zero_slack": min_zero_slack,
        "support_kkt_max_positive_abs_slack": max_positive_abs_slack,
        "multipliers_a": [float(x) for x in a.tolist()],
        "multipliers_b": [float(x) for x in b.tolist()],
    }


def kkt_stats_from_h_res(h_res, grad, tol):
    legacy_identity_stats = kkt_stats_from_gradient(grad, tol)
    support_stats = support_kkt_multipliers(h_res, grad, tol)

    min_zero_slack = support_stats["support_kkt_min_zero_slack"]
    if math.isinf(min_zero_slack):
        min_zero_slack = math.inf
    stationarity_margin = min(
        min_zero_slack,
        -support_stats["support_kkt_max_positive_abs_slack"],
    )
    if math.isinf(stationarity_margin):
        stationarity_margin = 0.0
    worst_violation = max(
        support_stats["support_kkt_row_sum_abs_err"],
        support_stats["support_kkt_col_sum_abs_err"],
        support_stats["support_kkt_nonneg_violation"],
        support_stats["support_kkt_equality_violation"],
        support_stats["support_kkt_same_component_ineq_violation"],
        support_stats["support_kkt_difference_violation"],
        max(0.0, -support_stats["support_kkt_min_zero_slack"])
        if not math.isinf(support_stats["support_kkt_min_zero_slack"])
        else 0.0,
        support_stats["support_kkt_max_positive_abs_slack"],
    )

    stats = dict(legacy_identity_stats)
    stats.update({
        "identity_analytic_kkt_pass": legacy_identity_stats["kkt_pass"],
        "identity_min_offdiag_kkt_slack": legacy_identity_stats["min_offdiag_kkt_slack"],
        "identity_mean_offdiag_kkt_slack": legacy_identity_stats["mean_offdiag_kkt_slack"],
        "identity_negative_cycle_detected": legacy_identity_stats["negative_cycle_detected"],
        "kkt_pass": support_stats["support_kkt_pass"],
        "negative_cycle_detected": bool(worst_violation > tol),
        "worst_difference_constraint_violation": worst_violation,
        "worst_difference_constraint_edge": support_stats["support_kkt_worst_difference_edge"],
        "min_offdiag_kkt_slack": stationarity_margin,
        "mean_offdiag_kkt_slack": stationarity_margin,
        "multipliers_a": support_stats["multipliers_a"],
        "multipliers_b": support_stats["multipliers_b"],
        **support_stats,
    })
    return stats


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


def projected_gradient_effective_tol(args):
    return args.kkt_tol if args.projected_gradient_tol < 0.0 else args.projected_gradient_tol


def project_affine_doubly_stochastic(matrix):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Birkhoff projection expects a square matrix, got {tuple(matrix.shape)}")
    n = matrix.shape[0]
    row_excess = matrix.sum(dim=1) - 1.0
    col_excess = matrix.sum(dim=0) - 1.0
    total_excess = matrix.sum() - float(n)
    return (
        matrix
        - row_excess[:, None] / float(n)
        - col_excess[None, :] / float(n)
        + total_excess / float(n * n)
    )


def project_birkhoff_dykstra(matrix, max_iters, tol):
    if max_iters <= 0:
        raise ValueError("--birkhoff-projection-iters must be positive")

    x = matrix.detach().double().cpu().clone()
    p_affine = torch.zeros_like(x)
    p_nonneg = torch.zeros_like(x)
    converged = False
    row_err = math.inf
    col_err = math.inf
    min_entry = -math.inf
    max_abs_change = math.inf

    for iter_idx in range(max_iters):
        affine_input = x + p_affine
        affine_projected = project_affine_doubly_stochastic(affine_input)
        p_affine = affine_input - affine_projected

        nonneg_input = affine_projected + p_nonneg
        x_next = nonneg_input.clamp_min(0.0)
        p_nonneg = nonneg_input - x_next

        max_abs_change = float((x_next - x).abs().max().item())
        x = x_next

        if tol > 0.0 and ((iter_idx + 1) % 10 == 0 or iter_idx + 1 == max_iters):
            row_err = float((x.sum(dim=1) - 1.0).abs().max().item())
            col_err = float((x.sum(dim=0) - 1.0).abs().max().item())
            min_entry = float(x.min().item())
            if max(row_err, col_err, max(0.0, -min_entry), max_abs_change) <= tol:
                converged = True
                break

    if not math.isfinite(row_err):
        row_err = float((x.sum(dim=1) - 1.0).abs().max().item())
        col_err = float((x.sum(dim=0) - 1.0).abs().max().item())
        min_entry = float(x.min().item())

    return x.float(), {
        "iters": iter_idx + 1,
        "converged": bool(converged),
        "row_sum_abs_err": row_err,
        "col_sum_abs_err": col_err,
        "min_entry": min_entry,
        "max_abs_change": max_abs_change,
    }


def alm_projection_primal_step_size(args):
    step_size = float(args.alm_projection_step_size)
    if step_size < 0.0:
        step_size = float(args.projected_gradient_step_size)
    if step_size <= 0.0:
        raise ValueError("--alm-projection-step-size must be positive when set")
    return step_size


def project_birkhoff_alm(matrix, max_iters, tol, rho, dual_step, primal_step, prox_weight):
    if max_iters <= 0:
        raise ValueError("--sprox-alm-iters must be positive")
    if rho < 0.0:
        raise ValueError("--sprox-alm-rho must be nonnegative")
    if dual_step < 0.0:
        raise ValueError("--sprox-alm-dual-step must be nonnegative")
    if primal_step <= 0.0:
        raise ValueError("--alm-projection-step-size must be positive")
    if prox_weight < 0.0:
        prox_weight = rho
    if prox_weight <= 0.0:
        raise ValueError("--sprox-alm-prox-weight must be positive, or negative with positive --sprox-alm-rho")

    target = matrix.detach().double().cpu().clone()
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError(f"Birkhoff projection expects a square matrix, got {tuple(target.shape)}")

    x = target.clamp_min(0.0)
    row_dual = torch.zeros((target.shape[0], 1), dtype=target.dtype)
    col_dual = torch.zeros((1, target.shape[1]), dtype=target.dtype)
    rho = torch.as_tensor(float(rho), dtype=target.dtype)
    dual_step = torch.as_tensor(float(dual_step), dtype=target.dtype)
    primal_step = torch.as_tensor(float(primal_step), dtype=target.dtype)
    prox_weight = torch.as_tensor(float(prox_weight), dtype=target.dtype)

    converged = False
    row_err = math.inf
    col_err = math.inf
    min_entry = -math.inf
    max_abs_change = math.inf

    for iter_idx in range(max_iters):
        row_res = x.sum(dim=1, keepdim=True) - 1.0
        col_res = x.sum(dim=0, keepdim=True) - 1.0
        row_dual = row_dual + dual_step * row_res
        col_dual = col_dual + dual_step * col_res

        grad_k = (
            prox_weight * (x - target)
            + row_dual
            + col_dual
            + rho * (row_res + col_res)
        )
        x_next = (x - primal_step * grad_k).clamp_min(0.0)
        max_abs_change = float((x_next - x).abs().max().item())
        x = x_next

        if tol > 0.0 and ((iter_idx + 1) % 10 == 0 or iter_idx + 1 == max_iters):
            row_err = float((x.sum(dim=1) - 1.0).abs().max().item())
            col_err = float((x.sum(dim=0) - 1.0).abs().max().item())
            min_entry = float(x.min().item())
            if max(row_err, col_err, max(0.0, -min_entry), max_abs_change) <= tol:
                converged = True
                break

    if not math.isfinite(row_err):
        row_err = float((x.sum(dim=1) - 1.0).abs().max().item())
        col_err = float((x.sum(dim=0) - 1.0).abs().max().item())
        min_entry = float(x.min().item())

    return x.float(), {
        "iters": iter_idx + 1,
        "converged": bool(converged),
        "row_sum_abs_err": row_err,
        "col_sum_abs_err": col_err,
        "min_entry": min_entry,
        "max_abs_change": max_abs_change,
        "alm_projection_step_size": float(primal_step.item()),
        "alm_projection_prox_weight": float(prox_weight.item()),
    }


def alm_projection_stats_from_gradient(h_res, grad, args):
    eta = float(args.projected_gradient_step_size)
    if eta <= 0.0:
        raise ValueError("--projected-gradient-step-size must be positive")

    h = h_res.detach().float().cpu()
    g = grad.detach().float().cpu()
    n = g.shape[0]
    if h.shape != g.shape:
        raise ValueError(f"H_res point shape {tuple(h.shape)} does not match gradient shape {tuple(g.shape)}")

    prox_weight_value = float(args.sprox_alm_prox_weight)
    prox_weight = float(args.sprox_alm_rho) if prox_weight_value < 0.0 else prox_weight_value
    projected, projection_stats = project_birkhoff_alm(
        h - eta * g,
        max_iters=args.sprox_alm_iters,
        tol=args.birkhoff_projection_tol,
        rho=float(args.sprox_alm_rho),
        dual_step=float(args.sprox_alm_dual_step),
        primal_step=alm_projection_primal_step_size(args),
        prox_weight=prox_weight,
    )
    fixed_point_delta = h - projected
    residual = fixed_point_delta / eta
    error = float(torch.linalg.vector_norm(residual).item())
    offdiag_mask = ~torch.eye(n, dtype=torch.bool)
    tol = projected_gradient_effective_tol(args)
    projected_min_entry = projection_stats["min_entry"]
    nonneg_violation = max(0.0, -projected_min_entry)
    row_err = projection_stats["row_sum_abs_err"]
    col_err = projection_stats["col_sum_abs_err"]

    return {
        "projected_gradient_kkt_pass": bool(
            error <= tol
            and row_err <= tol
            and col_err <= tol
            and nonneg_violation <= tol
        ),
        "projected_gradient_update": "alm-projection",
        "projected_gradient_tol": tol,
        "projected_gradient_step_size": eta,
        "projected_gradient_kkt_error": error,
        "projected_gradient_fixed_point_distance": float(torch.linalg.vector_norm(fixed_point_delta).item()),
        "projected_gradient_residual_abs_max": float(residual.abs().max().item()),
        "projected_gradient_projection_iters": projection_stats["iters"],
        "projected_gradient_projection_converged": projection_stats["converged"],
        "projected_gradient_projection_row_sum_abs_err": row_err,
        "projected_gradient_projection_col_sum_abs_err": col_err,
        "projected_gradient_projection_min_entry": projected_min_entry,
        "projected_gradient_projection_nonneg_violation": nonneg_violation,
        "projected_gradient_projection_max_abs_change": projection_stats["max_abs_change"],
        "projected_gradient_projection_max_offdiag": float(projected[offdiag_mask].max().item()),
        "alm_projection_step_size": projection_stats["alm_projection_step_size"],
        "alm_projection_prox_weight": projection_stats["alm_projection_prox_weight"],
        "sprox_alm_rho": float(args.sprox_alm_rho),
        "sprox_alm_dual_step": float(args.sprox_alm_dual_step),
    }


def sprox_alm_fixed_point_stats_from_gradient(h_res, grad, args, analytic_stats=None):
    eta = float(args.projected_gradient_step_size)
    if eta <= 0.0:
        raise ValueError("--projected-gradient-step-size must be positive")
    if args.sprox_alm_iters <= 0:
        raise ValueError("--sprox-alm-iters must be positive")

    h = h_res.detach().float().cpu()
    g = grad.detach().float().cpu()
    n = g.shape[0]
    if h.shape != g.shape:
        raise ValueError(f"H_res point shape {tuple(h.shape)} does not match gradient shape {tuple(g.shape)}")

    x = h.clone()
    z = h.clone()
    support_dual_stats = None
    if args.sprox_alm_dual_init == "analytic":
        support_dual_stats = support_kkt_multipliers(h, g, projected_gradient_effective_tol(args))
        row_dual = torch.tensor(support_dual_stats["multipliers_a"], dtype=x.dtype).view(n, 1)
        col_dual = torch.tensor(support_dual_stats["multipliers_b"], dtype=x.dtype).view(1, n)
    else:
        row_dual = torch.zeros((n, 1), dtype=x.dtype)
        col_dual = torch.zeros((1, n), dtype=x.dtype)

    rho = torch.as_tensor(float(args.sprox_alm_rho), dtype=x.dtype)
    dual_step = torch.as_tensor(float(args.sprox_alm_dual_step), dtype=x.dtype)
    prox_weight_value = float(args.sprox_alm_prox_weight)
    prox_weight = rho if prox_weight_value < 0.0 else torch.as_tensor(prox_weight_value, dtype=x.dtype)
    smooth_beta = torch.as_tensor(float(args.sprox_alm_smooth_beta), dtype=x.dtype)
    max_abs_change = math.inf

    for iter_idx in range(args.sprox_alm_iters):
        row_res = x.sum(dim=1, keepdim=True) - 1.0
        col_res = x.sum(dim=0, keepdim=True) - 1.0
        row_dual = row_dual + dual_step * row_res
        col_dual = col_dual + dual_step * col_res

        grad_k = (
            g
            + row_dual
            + col_dual
            + rho * (row_res + col_res)
            + prox_weight * (x - z)
        )
        x_next = (x - eta * grad_k).clamp_min(0.0)
        max_abs_change = float((x_next - x).abs().max().item())
        z = z + smooth_beta * (x_next - z)
        x = x_next

    fixed_point_delta = h - x
    residual = fixed_point_delta / eta
    error = float(torch.linalg.vector_norm(residual).item())
    offdiag_mask = ~torch.eye(n, dtype=torch.bool)
    tol = projected_gradient_effective_tol(args)
    row_err = float((x.sum(dim=1) - 1.0).abs().max().item())
    col_err = float((x.sum(dim=0) - 1.0).abs().max().item())
    min_entry = float(x.min().item())
    nonneg_violation = max(0.0, -min_entry)
    pass_check = bool(
        error <= tol
        and row_err <= tol
        and col_err <= tol
        and nonneg_violation <= tol
    )
    return {
        "projected_gradient_kkt_pass": pass_check,
        "projected_gradient_update": "sprox-alm",
        "projected_gradient_tol": tol,
        "projected_gradient_step_size": eta,
        "projected_gradient_kkt_error": error,
        "projected_gradient_fixed_point_distance": float(torch.linalg.vector_norm(fixed_point_delta).item()),
        "projected_gradient_residual_abs_max": float(residual.abs().max().item()),
        "projected_gradient_projection_iters": args.sprox_alm_iters,
        "projected_gradient_projection_converged": None,
        "projected_gradient_projection_row_sum_abs_err": row_err,
        "projected_gradient_projection_col_sum_abs_err": col_err,
        "projected_gradient_projection_min_entry": min_entry,
        "projected_gradient_projection_nonneg_violation": nonneg_violation,
        "projected_gradient_projection_max_abs_change": max_abs_change,
        "projected_gradient_projection_max_offdiag": float(x[offdiag_mask].max().item()),
        "sprox_alm_dual_init": args.sprox_alm_dual_init,
        "sprox_alm_support_kkt_pass": (
            None if support_dual_stats is None else support_dual_stats["support_kkt_pass"]
        ),
        "sprox_alm_support_kkt_positive_count": (
            None if support_dual_stats is None else support_dual_stats["support_kkt_positive_count"]
        ),
        "sprox_alm_support_kkt_row_sum_abs_err": (
            None if support_dual_stats is None else support_dual_stats["support_kkt_row_sum_abs_err"]
        ),
        "sprox_alm_support_kkt_col_sum_abs_err": (
            None if support_dual_stats is None else support_dual_stats["support_kkt_col_sum_abs_err"]
        ),
        "sprox_alm_support_kkt_nonneg_violation": (
            None if support_dual_stats is None else support_dual_stats["support_kkt_nonneg_violation"]
        ),
        "sprox_alm_support_kkt_equality_violation": (
            None if support_dual_stats is None else support_dual_stats["support_kkt_equality_violation"]
        ),
        "sprox_alm_support_kkt_min_zero_slack": (
            None if support_dual_stats is None else support_dual_stats["support_kkt_min_zero_slack"]
        ),
        "sprox_alm_support_kkt_max_positive_abs_slack": (
            None if support_dual_stats is None else support_dual_stats["support_kkt_max_positive_abs_slack"]
        ),
        "sprox_alm_rho": float(rho.item()),
        "sprox_alm_dual_step": float(dual_step.item()),
        "sprox_alm_prox_weight": float(prox_weight.item()),
        "sprox_alm_smooth_beta": float(smooth_beta.item()),
    }


def projected_gradient_stats_from_gradient(h_res, grad, args, analytic_stats=None):
    eta = float(args.projected_gradient_step_size)
    if eta <= 0.0:
        raise ValueError("--projected-gradient-step-size must be positive")
    if args.projected_gradient_update == "sprox-alm":
        return sprox_alm_fixed_point_stats_from_gradient(
            h_res=h_res,
            grad=grad,
            args=args,
            analytic_stats=analytic_stats,
        )
    if args.projected_gradient_update == "alm-projection":
        return alm_projection_stats_from_gradient(
            h_res=h_res,
            grad=grad,
            args=args,
        )

    h = h_res.detach().float().cpu()
    g = grad.detach().float().cpu()
    n = g.shape[0]
    if h.shape != g.shape:
        raise ValueError(f"H_res point shape {tuple(h.shape)} does not match gradient shape {tuple(g.shape)}")
    projected, projection_stats = project_birkhoff_dykstra(
        h - eta * g,
        max_iters=args.birkhoff_projection_iters,
        tol=args.birkhoff_projection_tol,
    )
    fixed_point_delta = h - projected
    residual = fixed_point_delta / eta
    error = float(torch.linalg.vector_norm(residual).item())
    offdiag_mask = ~torch.eye(n, dtype=torch.bool)
    tol = projected_gradient_effective_tol(args)
    projected_min_entry = projection_stats["min_entry"]

    return {
        "projected_gradient_kkt_pass": bool(error <= tol),
        "projected_gradient_update": "dykstra",
        "projected_gradient_tol": tol,
        "projected_gradient_step_size": eta,
        "projected_gradient_kkt_error": error,
        "projected_gradient_fixed_point_distance": float(torch.linalg.vector_norm(fixed_point_delta).item()),
        "projected_gradient_residual_abs_max": float(residual.abs().max().item()),
        "projected_gradient_projection_iters": projection_stats["iters"],
        "projected_gradient_projection_converged": projection_stats["converged"],
        "projected_gradient_projection_row_sum_abs_err": projection_stats["row_sum_abs_err"],
        "projected_gradient_projection_col_sum_abs_err": projection_stats["col_sum_abs_err"],
        "projected_gradient_projection_min_entry": projected_min_entry,
        "projected_gradient_projection_nonneg_violation": max(0.0, -projected_min_entry),
        "projected_gradient_projection_max_abs_change": projection_stats["max_abs_change"],
        "projected_gradient_projection_max_offdiag": float(projected[offdiag_mask].max().item()),
    }


def random_sinkhorn_seed(base_seed, module_ordinal):
    return int(base_seed + 1_000_003 * (module_ordinal + 1))


def global_random_sinkhorn_seed(base_seed, module_ordinal, trial_index):
    return int(base_seed + 1_000_003 * (module_ordinal + 1) + 104_729 * (trial_index + 1))


def build_random_sinkhorn_perturbations(base_h_res, args, module_ordinal, num_samples):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_sinkhorn_seed(args.seed, module_ordinal))
    base = base_h_res.detach().float().cpu()
    n = base.shape[0]
    tiny = torch.finfo(base.dtype).tiny
    sinkhorn_input = base.clamp_min(tiny)
    noise = torch.rand((num_samples, n, n), generator=generator, dtype=torch.float32)
    candidates = sinkhorn_project(
        sinkhorn_input.unsqueeze(0) + args.random_sinkhorn_eps * noise,
        args.sinkhorn_iters,
        args.sinkhorn_tol,
    )
    directions = (candidates - base.unsqueeze(0)) / args.random_sinkhorn_eps
    return candidates, directions


def build_random_sinkhorn_perturbation(base_h_res, args, module_ordinal, sample_index):
    candidates, directions = build_random_sinkhorn_perturbations(
        base_h_res=base_h_res,
        args=args,
        module_ordinal=module_ordinal,
        num_samples=sample_index + 1,
    )
    return candidates[sample_index].clone(), directions[sample_index].clone()


def build_global_random_sinkhorn_perturbation(base_h_res, args, module_ordinal, trial_index):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(global_random_sinkhorn_seed(args.seed, module_ordinal, trial_index))
    base = base_h_res.detach().float().cpu()
    n = base.shape[0]
    tiny = torch.finfo(base.dtype).tiny
    sinkhorn_input = base.clamp_min(tiny)
    noise = torch.rand((n, n), generator=generator, dtype=torch.float32)
    candidate = sinkhorn_project(
        sinkhorn_input + args.random_sinkhorn_eps * noise,
        args.sinkhorn_iters,
        args.sinkhorn_tol,
    )
    direction = (candidate - base) / args.random_sinkhorn_eps
    return candidate, direction


def random_sinkhorn_stats_from_gradient(h_res, grad, args, module_ordinal):
    num_samples = max(0, args.random_sinkhorn_samples)
    g = grad.detach().float().cpu()
    if num_samples == 0:
        return {
            "random_sinkhorn_sampled_pass": None,
            "random_sinkhorn_samples": 0,
            "random_sinkhorn_eps": args.random_sinkhorn_eps,
            "random_sinkhorn_iters": args.sinkhorn_iters,
            "random_sinkhorn_min_derivative": None,
            "random_sinkhorn_mean_derivative": None,
            "random_sinkhorn_num_negative_derivatives": None,
            "random_sinkhorn_best_sample_index": None,
            "random_sinkhorn_best_row_sum_abs_err": None,
            "random_sinkhorn_best_col_sum_abs_err": None,
            "random_sinkhorn_best_min_entry": None,
            "random_sinkhorn_best_max_offdiag": None,
        }

    candidates, directions = build_random_sinkhorn_perturbations(
        base_h_res=h_res,
        args=args,
        module_ordinal=module_ordinal,
        num_samples=num_samples,
    )
    n = g.shape[0]
    derivatives_tensor = (directions * g.unsqueeze(0)).sum(dim=(-2, -1))
    min_derivative, best_sample_index_tensor = torch.min(derivatives_tensor, dim=0)
    min_derivative = float(min_derivative.item())
    best_sample_index = int(best_sample_index_tensor.item())
    best_candidate = candidates[best_sample_index]
    offdiag_mask = ~torch.eye(n, dtype=torch.bool)

    return {
        "random_sinkhorn_sampled_pass": bool(min_derivative >= -args.kkt_tol),
        "random_sinkhorn_samples": num_samples,
        "random_sinkhorn_eps": args.random_sinkhorn_eps,
        "random_sinkhorn_iters": args.sinkhorn_iters,
        "random_sinkhorn_tol": args.sinkhorn_tol,
        "random_sinkhorn_min_derivative": min_derivative,
        "random_sinkhorn_mean_derivative": float(derivatives_tensor.mean().item()),
        "random_sinkhorn_num_negative_derivatives": int((derivatives_tensor < -args.kkt_tol).sum().item()),
        "random_sinkhorn_best_sample_index": best_sample_index,
        "random_sinkhorn_best_row_sum_abs_err": float((best_candidate.sum(dim=1) - 1.0).abs().max().item()),
        "random_sinkhorn_best_col_sum_abs_err": float((best_candidate.sum(dim=0) - 1.0).abs().max().item()),
        "random_sinkhorn_best_min_entry": float(best_candidate.min().item()),
        "random_sinkhorn_best_max_offdiag": float(best_candidate[offdiag_mask].max().item()),
    }


def select_kkt_pass(args, analytic_pass, random_sinkhorn_pass, projected_gradient_pass):
    if args.kkt_check_mode == "analytic":
        return bool(analytic_pass)
    if args.kkt_check_mode == "random-sinkhorn":
        if random_sinkhorn_pass is None:
            raise ValueError(f"--random-sinkhorn-samples must be > 0 when --kkt-check-mode={args.kkt_check_mode}")
        return bool(random_sinkhorn_pass)
    if args.kkt_check_mode == "projected-gradient":
        return bool(projected_gradient_pass)
    if args.kkt_check_mode in ("global-random-sinkhorn", "single-layer-random-sinkhorn"):
        return bool(random_sinkhorn_pass) if random_sinkhorn_pass is not None else bool(analytic_pass)
    if random_sinkhorn_pass is None:
        raise ValueError("--random-sinkhorn-samples must be > 0 when --kkt-check-mode=both")
    return bool(analytic_pass and random_sinkhorn_pass)


def finite_diff_direction_mode(args):
    if args.finite_diff_direction != "auto":
        return args.finite_diff_direction
    if args.kkt_check_mode == "analytic":
        return "pair-swap"
    return "random-sinkhorn"


def effective_global_random_sinkhorn_trials(args):
    if args.global_random_sinkhorn_trials >= 0:
        return args.global_random_sinkhorn_trials
    return 64 if args.kkt_check_mode == "global-random-sinkhorn" else 0


def effective_single_layer_random_sinkhorn_trials(args):
    if args.single_layer_random_sinkhorn_trials >= 0:
        return args.single_layer_random_sinkhorn_trials
    return 512 if args.kkt_check_mode == "single-layer-random-sinkhorn" else 0


def fmt_optional(value, spec=".3e"):
    if value is None:
        return "None"
    return format(value, spec)


def layer_sort_key(row, args):
    if args.kkt_check_mode == "projected-gradient":
        return (
            -row["projected_gradient_kkt_error"],
            row["projected_gradient_projection_row_sum_abs_err"],
            row["projected_gradient_projection_col_sum_abs_err"],
        )
    if args.kkt_check_mode in ("random-sinkhorn", "global-random-sinkhorn", "single-layer-random-sinkhorn"):
        random_derivative = row["random_sinkhorn_min_derivative"]
        return (
            math.inf if random_derivative is None else random_derivative,
            row["min_offdiag_kkt_slack"],
        )
    if args.kkt_check_mode == "both":
        random_derivative = row["random_sinkhorn_min_derivative"]
        random_derivative = math.inf if random_derivative is None else random_derivative
        return (
            min(row["min_offdiag_kkt_slack"], random_derivative),
            row["min_offdiag_kkt_slack"],
            random_derivative,
        )
    return (
        row["min_offdiag_kkt_slack"],
        row["min_pair_swap_derivative"],
    )


def set_all_layers_to_random_sinkhorn_trial(entries, args, trial_index):
    trial_row_sum_abs_err = 0.0
    trial_col_sum_abs_err = 0.0
    trial_min_entry = math.inf
    trial_max_offdiag = -math.inf

    with torch.no_grad():
        for module_ordinal, (_, _, param) in enumerate(entries):
            n = param.shape[0]
            candidate, _ = build_global_random_sinkhorn_perturbation(
                base_h_res=forced_param_base_h_res(param),
                args=args,
                module_ordinal=module_ordinal,
                trial_index=trial_index,
            )
            offdiag_mask = ~torch.eye(n, dtype=torch.bool)
            row_sum_abs_err = float((candidate.sum(dim=1) - 1.0).abs().max().item())
            col_sum_abs_err = float((candidate.sum(dim=0) - 1.0).abs().max().item())
            trial_row_sum_abs_err = max(trial_row_sum_abs_err, row_sum_abs_err)
            trial_col_sum_abs_err = max(trial_col_sum_abs_err, col_sum_abs_err)
            trial_min_entry = min(trial_min_entry, float(candidate.min().item()))
            trial_max_offdiag = max(trial_max_offdiag, float(candidate[offdiag_mask].max().item()))
            param.copy_(candidate.to(device=param.device, dtype=param.dtype))

    return {
        "row_sum_abs_err_max": trial_row_sum_abs_err,
        "col_sum_abs_err_max": trial_col_sum_abs_err,
        "min_entry": trial_min_entry,
        "max_offdiag": trial_max_offdiag,
    }


def directional_derivative_from_param(param, direction):
    if param.grad is None:
        raise RuntimeError("Missing H_res gradient for first-order trial evaluation.")
    grad = param.grad.detach().float().cpu()
    return float((grad * direction.detach().float().cpu()).sum().item())


def all_layers_random_sinkhorn_directional_trial(entries, args, trial_index):
    trial_row_sum_abs_err = 0.0
    trial_col_sum_abs_err = 0.0
    trial_min_entry = math.inf
    trial_max_offdiag = -math.inf
    directional_derivative = 0.0

    for module_ordinal, (_, _, param) in enumerate(entries):
        n = param.shape[0]
        candidate, direction = build_global_random_sinkhorn_perturbation(
            base_h_res=forced_param_base_h_res(param),
            args=args,
            module_ordinal=module_ordinal,
            trial_index=trial_index,
        )
        offdiag_mask = ~torch.eye(n, dtype=torch.bool)
        trial_row_sum_abs_err = max(
            trial_row_sum_abs_err,
            float((candidate.sum(dim=1) - 1.0).abs().max().item()),
        )
        trial_col_sum_abs_err = max(
            trial_col_sum_abs_err,
            float((candidate.sum(dim=0) - 1.0).abs().max().item()),
        )
        trial_min_entry = min(trial_min_entry, float(candidate.min().item()))
        trial_max_offdiag = max(trial_max_offdiag, float(candidate[offdiag_mask].max().item()))
        directional_derivative += directional_derivative_from_param(param, direction)

    return {
        "row_sum_abs_err_max": trial_row_sum_abs_err,
        "col_sum_abs_err_max": trial_col_sum_abs_err,
        "min_entry": trial_min_entry,
        "max_offdiag": trial_max_offdiag,
    }, directional_derivative


def collect_global_random_sinkhorn_h_res_matrices(entries, args, trial_index):
    if trial_index is None:
        return []

    matrices = []
    for module_ordinal, (name, _, param) in enumerate(entries):
        n = param.shape[0]
        candidate, _ = build_global_random_sinkhorn_perturbation(
            base_h_res=forced_param_base_h_res(param),
            args=args,
            module_ordinal=module_ordinal,
            trial_index=trial_index,
        )
        layer_index, component = layer_info_from_module_name(name)
        offdiag_mask = ~torch.eye(n, dtype=torch.bool)
        matrices.append({
            "module_name": name,
            "module_ordinal": module_ordinal,
            "layer_index": layer_index,
            "component": component,
            "trial_index": trial_index,
            "row_sum_abs_err_max": float((candidate.sum(dim=1) - 1.0).abs().max().item()),
            "col_sum_abs_err_max": float((candidate.sum(dim=0) - 1.0).abs().max().item()),
            "min_entry": float(candidate.min().item()),
            "max_offdiag": float(candidate[offdiag_mask].max().item()),
            "h_res": candidate.tolist(),
        })
    return matrices


def empty_single_layer_random_sinkhorn_result(args):
    return {
        "enabled": False,
        "selected_module_ordinal": None,
        "selected_module_name": None,
        "selected_layer_index": None,
        "selected_component": None,
        "num_trials": 0,
        "trial_eval_mode": args.trial_eval_mode,
        "num_loss_decreases": None,
        "loss_decrease_rate": None,
        "num_directional_decreases": None,
        "directional_decrease_rate": None,
        "sampled_pass": None,
        "loss_decrease_tol": args.loss_decrease_tol,
        "directional_decrease_tol": args.kkt_tol,
        "min_loss_delta": None,
        "mean_loss_delta": None,
        "max_loss_delta": None,
        "min_directional_derivative": None,
        "mean_directional_derivative": None,
        "max_directional_derivative": None,
        "best_trial_index": None,
        "best_trial_loss": None,
        "best_trial_directional_derivative": None,
        "best_trial_first_order_loss_delta": None,
        "best_trial_row_sum_abs_err_max": None,
        "best_trial_col_sum_abs_err_max": None,
        "best_trial_min_entry": None,
        "best_trial_max_offdiag": None,
        "best_h_res_matrix": None,
        "trials": [],
    }


def select_single_layer_module_ordinal(entries, args):
    if not entries:
        raise RuntimeError("No MHC modules are available for single-layer perturbation.")
    if args.single_layer_module_ordinal >= 0:
        if args.single_layer_module_ordinal >= len(entries):
            raise ValueError(
                f"--single-layer-module-ordinal={args.single_layer_module_ordinal} "
                f"is out of range for {len(entries)} MHC modules."
            )
        return args.single_layer_module_ordinal

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed + 5_300_017))
    return int(torch.randint(len(entries), (1,), generator=generator).item())


def single_layer_candidate_stats(candidate):
    n = candidate.shape[0]
    offdiag_mask = ~torch.eye(n, dtype=torch.bool)
    return {
        "row_sum_abs_err_max": float((candidate.sum(dim=1) - 1.0).abs().max().item()),
        "col_sum_abs_err_max": float((candidate.sum(dim=0) - 1.0).abs().max().item()),
        "min_entry": float(candidate.min().item()),
        "max_offdiag": float(candidate[offdiag_mask].max().item()),
    }


def build_single_layer_best_h_res_matrix(entries, args, selected_ordinal, trial_index):
    if selected_ordinal is None or trial_index is None:
        return None

    name, _, param = entries[selected_ordinal]
    n = param.shape[0]
    candidate, _ = build_global_random_sinkhorn_perturbation(
        base_h_res=forced_param_base_h_res(param),
        args=args,
        module_ordinal=selected_ordinal,
        trial_index=trial_index,
    )
    layer_index, component = layer_info_from_module_name(name)
    return {
        "module_name": name,
        "module_ordinal": selected_ordinal,
        "layer_index": layer_index,
        "component": component,
        "trial_index": trial_index,
        **single_layer_candidate_stats(candidate),
        "h_res": candidate.tolist(),
    }


def compute_single_layer_random_sinkhorn_loss_trials(model, batches, entries, args, base_loss):
    num_trials = effective_single_layer_random_sinkhorn_trials(args)
    if num_trials <= 0:
        return empty_single_layer_random_sinkhorn_result(args)

    selected_ordinal = select_single_layer_module_ordinal(entries, args)
    selected_name, _, selected_param = entries[selected_ordinal]
    selected_layer_index, selected_component = layer_info_from_module_name(selected_name)
    n = selected_param.shape[0]

    trials = []
    best_trial = None
    try:
        reset_forced_params_to_base(entries)
        for trial_index in range(num_trials):
            candidate, direction = build_global_random_sinkhorn_perturbation(
                base_h_res=forced_param_base_h_res(selected_param),
                args=args,
                module_ordinal=selected_ordinal,
                trial_index=trial_index,
            )
            constraint_stats = single_layer_candidate_stats(candidate)
            if args.trial_eval_mode == "first-order":
                directional_derivative = directional_derivative_from_param(selected_param, direction)
                trial = {
                    "trial_index": trial_index,
                    "loss": None,
                    "loss_delta": None,
                    "loss_decreased": None,
                    "directional_derivative": directional_derivative,
                    "first_order_loss_delta": args.random_sinkhorn_eps * directional_derivative,
                    "directional_decreased": bool(directional_derivative < -args.kkt_tol),
                    **constraint_stats,
                }
                trial_metric = directional_derivative
            else:
                with torch.no_grad():
                    selected_param.copy_(candidate.to(device=selected_param.device, dtype=selected_param.dtype))
                perturbed_loss = compute_loss(
                    model=model,
                    batches=batches,
                    device=args.device,
                    amp_dtype=args.amp_dtype,
                    backward=False,
                )
                loss_delta = perturbed_loss - base_loss
                trial = {
                    "trial_index": trial_index,
                    "loss": perturbed_loss,
                    "loss_delta": loss_delta,
                    "loss_decreased": bool(loss_delta < -args.loss_decrease_tol),
                    "directional_derivative": None,
                    "first_order_loss_delta": None,
                    "directional_decreased": None,
                    **constraint_stats,
                }
                trial_metric = loss_delta
            trials.append(trial)
            if best_trial is None or trial_metric < best_trial["_selection_metric"]:
                trial["_selection_metric"] = trial_metric
                best_trial = trial
    finally:
        reset_forced_params_to_base(entries)

    if args.trial_eval_mode == "first-order":
        directional_derivatives = torch.tensor(
            [trial["directional_derivative"] for trial in trials],
            dtype=torch.float32,
        )
        num_loss_decreases = None
        num_directional_decreases = sum(trial["directional_decreased"] for trial in trials)
        loss_decrease_rate = None
        directional_decrease_rate = float(num_directional_decreases / num_trials)
        sampled_pass = bool(num_directional_decreases == 0)
        min_loss_delta = mean_loss_delta = max_loss_delta = None
        min_directional_derivative = float(directional_derivatives.min().item())
        mean_directional_derivative = float(directional_derivatives.mean().item())
        max_directional_derivative = float(directional_derivatives.max().item())
    else:
        loss_deltas = torch.tensor([trial["loss_delta"] for trial in trials], dtype=torch.float32)
        num_loss_decreases = sum(trial["loss_decreased"] for trial in trials)
        num_directional_decreases = None
        loss_decrease_rate = float(num_loss_decreases / num_trials)
        directional_decrease_rate = None
        sampled_pass = bool(num_loss_decreases == 0)
        min_loss_delta = float(loss_deltas.min().item())
        mean_loss_delta = float(loss_deltas.mean().item())
        max_loss_delta = float(loss_deltas.max().item())
        min_directional_derivative = mean_directional_derivative = max_directional_derivative = None

    for trial in trials:
        trial.pop("_selection_metric", None)

    return {
        "enabled": True,
        "selected_module_ordinal": selected_ordinal,
        "selected_module_name": selected_name,
        "selected_layer_index": selected_layer_index,
        "selected_component": selected_component,
        "num_trials": num_trials,
        "trial_eval_mode": args.trial_eval_mode,
        "num_loss_decreases": None if num_loss_decreases is None else int(num_loss_decreases),
        "loss_decrease_rate": loss_decrease_rate,
        "num_directional_decreases": (
            None if num_directional_decreases is None else int(num_directional_decreases)
        ),
        "directional_decrease_rate": directional_decrease_rate,
        "sampled_pass": sampled_pass,
        "loss_decrease_tol": args.loss_decrease_tol,
        "directional_decrease_tol": args.kkt_tol,
        "min_loss_delta": min_loss_delta,
        "mean_loss_delta": mean_loss_delta,
        "max_loss_delta": max_loss_delta,
        "min_directional_derivative": min_directional_derivative,
        "mean_directional_derivative": mean_directional_derivative,
        "max_directional_derivative": max_directional_derivative,
        "best_trial_index": best_trial["trial_index"],
        "best_trial_loss": best_trial["loss"],
        "best_trial_directional_derivative": best_trial["directional_derivative"],
        "best_trial_first_order_loss_delta": best_trial["first_order_loss_delta"],
        "best_trial_row_sum_abs_err_max": best_trial["row_sum_abs_err_max"],
        "best_trial_col_sum_abs_err_max": best_trial["col_sum_abs_err_max"],
        "best_trial_min_entry": best_trial["min_entry"],
        "best_trial_max_offdiag": best_trial["max_offdiag"],
        "best_h_res_matrix": build_single_layer_best_h_res_matrix(
            entries=entries,
            args=args,
            selected_ordinal=selected_ordinal,
            trial_index=best_trial["trial_index"],
        ),
        "trials": trials,
    }


def compute_global_random_sinkhorn_loss_trials(model, batches, entries, args, base_loss):
    num_trials = effective_global_random_sinkhorn_trials(args)
    if num_trials <= 0:
        return {
            "enabled": False,
            "num_trials": 0,
            "trial_eval_mode": args.trial_eval_mode,
            "num_loss_decreases": None,
            "loss_decrease_rate": None,
            "num_directional_decreases": None,
            "directional_decrease_rate": None,
            "sampled_pass": None,
            "loss_decrease_tol": args.loss_decrease_tol,
            "directional_decrease_tol": args.kkt_tol,
            "min_loss_delta": None,
            "mean_loss_delta": None,
            "max_loss_delta": None,
            "min_directional_derivative": None,
            "mean_directional_derivative": None,
            "max_directional_derivative": None,
            "best_trial_index": None,
            "best_trial_loss": None,
            "best_trial_directional_derivative": None,
            "best_trial_first_order_loss_delta": None,
            "best_trial_row_sum_abs_err_max": None,
            "best_trial_col_sum_abs_err_max": None,
            "best_trial_min_entry": None,
            "best_trial_max_offdiag": None,
            "trials": [],
        }

    trials = []
    best_trial = None
    try:
        for trial_index in range(num_trials):
            if args.trial_eval_mode == "first-order":
                constraint_stats, directional_derivative = all_layers_random_sinkhorn_directional_trial(
                    entries=entries,
                    args=args,
                    trial_index=trial_index,
                )
                trial = {
                    "trial_index": trial_index,
                    "loss": None,
                    "loss_delta": None,
                    "loss_decreased": None,
                    "directional_derivative": directional_derivative,
                    "first_order_loss_delta": args.random_sinkhorn_eps * directional_derivative,
                    "directional_decreased": bool(directional_derivative < -args.kkt_tol),
                    **constraint_stats,
                }
                trial_metric = directional_derivative
            else:
                constraint_stats = set_all_layers_to_random_sinkhorn_trial(entries, args, trial_index)
                perturbed_loss = compute_loss(
                    model=model,
                    batches=batches,
                    device=args.device,
                    amp_dtype=args.amp_dtype,
                    backward=False,
                )
                loss_delta = perturbed_loss - base_loss
                trial = {
                    "trial_index": trial_index,
                    "loss": perturbed_loss,
                    "loss_delta": loss_delta,
                    "loss_decreased": bool(loss_delta < -args.loss_decrease_tol),
                    "directional_derivative": None,
                    "first_order_loss_delta": None,
                    "directional_decreased": None,
                    **constraint_stats,
                }
                trial_metric = loss_delta
            trials.append(trial)
            if best_trial is None or trial_metric < best_trial["_selection_metric"]:
                trial["_selection_metric"] = trial_metric
                best_trial = trial
    finally:
        reset_forced_params_to_base(entries)

    if args.trial_eval_mode == "first-order":
        directional_derivatives = torch.tensor(
            [trial["directional_derivative"] for trial in trials],
            dtype=torch.float32,
        )
        num_loss_decreases = None
        num_directional_decreases = sum(trial["directional_decreased"] for trial in trials)
        loss_decrease_rate = None
        directional_decrease_rate = float(num_directional_decreases / num_trials)
        sampled_pass = bool(num_directional_decreases == 0)
        min_loss_delta = mean_loss_delta = max_loss_delta = None
        min_directional_derivative = float(directional_derivatives.min().item())
        mean_directional_derivative = float(directional_derivatives.mean().item())
        max_directional_derivative = float(directional_derivatives.max().item())
    else:
        loss_deltas = torch.tensor([trial["loss_delta"] for trial in trials], dtype=torch.float32)
        num_loss_decreases = sum(trial["loss_decreased"] for trial in trials)
        num_directional_decreases = None
        loss_decrease_rate = float(num_loss_decreases / num_trials)
        directional_decrease_rate = None
        sampled_pass = bool(num_loss_decreases == 0)
        min_loss_delta = float(loss_deltas.min().item())
        mean_loss_delta = float(loss_deltas.mean().item())
        max_loss_delta = float(loss_deltas.max().item())
        min_directional_derivative = mean_directional_derivative = max_directional_derivative = None

    for trial in trials:
        trial.pop("_selection_metric", None)

    return {
        "enabled": True,
        "num_trials": num_trials,
        "trial_eval_mode": args.trial_eval_mode,
        "num_loss_decreases": None if num_loss_decreases is None else int(num_loss_decreases),
        "loss_decrease_rate": loss_decrease_rate,
        "num_directional_decreases": (
            None if num_directional_decreases is None else int(num_directional_decreases)
        ),
        "directional_decrease_rate": directional_decrease_rate,
        "sampled_pass": sampled_pass,
        "loss_decrease_tol": args.loss_decrease_tol,
        "directional_decrease_tol": args.kkt_tol,
        "min_loss_delta": min_loss_delta,
        "mean_loss_delta": mean_loss_delta,
        "max_loss_delta": max_loss_delta,
        "min_directional_derivative": min_directional_derivative,
        "mean_directional_derivative": mean_directional_derivative,
        "max_directional_derivative": max_directional_derivative,
        "best_trial_index": best_trial["trial_index"],
        "best_trial_loss": best_trial["loss"],
        "best_trial_directional_derivative": best_trial["directional_derivative"],
        "best_trial_first_order_loss_delta": best_trial["first_order_loss_delta"],
        "best_trial_row_sum_abs_err_max": best_trial["row_sum_abs_err_max"],
        "best_trial_col_sum_abs_err_max": best_trial["col_sum_abs_err_max"],
        "best_trial_min_entry": best_trial["min_entry"],
        "best_trial_max_offdiag": best_trial["max_offdiag"],
        "trials": trials,
    }


def compute_finite_differences(model, batches, entries, layer_rows, args, base_loss):
    out = []
    entry_by_name = {name: (module, param) for name, module, param in entries}
    direction_mode = finite_diff_direction_mode(args)

    if direction_mode == "pair-swap":
        candidates = [
            row for row in layer_rows
            if row["min_pair_swap_pair"] is not None
        ]
        candidates.sort(key=lambda row: row["min_pair_swap_derivative"])

        for row in candidates[:max(0, args.finite_diff_top_k)]:
            name = row["module_name"]
            _, param = entry_by_name[name]
            i, j = row["min_pair_swap_pair"]
            n = param.shape[0]
            base_h_res = forced_param_base_h_res(param)
            direction = torch.zeros_like(param)
            direction[i, j] = 1.0
            direction[j, i] = 1.0
            direction[i, i] = -1.0
            direction[j, j] = -1.0
            with torch.no_grad():
                param.copy_(base_h_res + args.finite_diff_eps * direction)
            perturbed_loss = compute_loss(
                model,
                batches,
                args.device,
                args.amp_dtype,
                backward=False,
            )
            with torch.no_grad():
                param.copy_(base_h_res)
            out.append({
                "module_name": name,
                "layer_index": row["layer_index"],
                "component": row["component"],
                "perturbation": "pair-swap",
                "pair": [i, j],
                "epsilon": args.finite_diff_eps,
                "base_loss": base_loss,
                "perturbed_loss": perturbed_loss,
                "finite_difference_derivative": (perturbed_loss - base_loss) / args.finite_diff_eps,
                "predicted_derivative": row["min_pair_swap_derivative"],
            })
        return out

    candidates = [
        row for row in layer_rows
        if row["random_sinkhorn_best_sample_index"] is not None
        and row["random_sinkhorn_min_derivative"] is not None
    ]
    candidates.sort(key=lambda row: row["random_sinkhorn_min_derivative"])

    for row in candidates[:max(0, args.finite_diff_top_k)]:
        name = row["module_name"]
        _, param = entry_by_name[name]
        base_h_res = forced_param_base_h_res(param)
        sample_index = row["random_sinkhorn_best_sample_index"]
        candidate, _ = build_random_sinkhorn_perturbation(
            base_h_res=base_h_res,
            args=args,
            module_ordinal=row["module_ordinal"],
            sample_index=sample_index,
        )
        with torch.no_grad():
            param.copy_(candidate.to(device=param.device, dtype=param.dtype))
        perturbed_loss = compute_loss(
            model,
            batches,
            args.device,
            args.amp_dtype,
            backward=False,
        )
        with torch.no_grad():
            param.copy_(base_h_res)
        out.append({
            "module_name": name,
            "layer_index": row["layer_index"],
            "component": row["component"],
            "perturbation": "random-sinkhorn",
            "sample_index": sample_index,
            "epsilon": args.random_sinkhorn_eps,
            "sinkhorn_iters": args.sinkhorn_iters,
            "base_loss": base_loss,
            "perturbed_loss": perturbed_loss,
            "finite_difference_derivative": (perturbed_loss - base_loss) / args.random_sinkhorn_eps,
            "predicted_derivative": row["random_sinkhorn_min_derivative"],
        })
    return out


def layer_info_from_module_name(name):
    parts = name.split(".")
    layer_index = None
    component = "unknown"
    if len(parts) >= 4 and parts[0] == "transformer" and parts[1] == "h":
        try:
            block = int(parts[2])
            if "hc_attn" in parts[3]:
                component = "attn"
                layer_index = 2 * block
            elif "hc_mlp" in parts[3]:
                component = "mlp"
                layer_index = 2 * block + 1
        except ValueError:
            pass
    return layer_index, component


def write_json_atomic(path, payload):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(path + ".tmp", path)


def run_single(args, print_report=True):
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

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset)

    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(strip_compile_prefix(checkpoint["model"]), strict=True)
    model.to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    entries = set_forced_h_res(model, args.device, args.h_res_test_point)
    batches = load_batches(
        data_dir=data_dir,
        split=args.split,
        batch_size=args.batch_size,
        seq_len=seq_len,
        num_batches=args.num_batches,
        seed=args.seed,
        device=args.device,
    )

    for _, _, param in entries:
        param.grad = None
    base_loss = compute_loss(model, batches, args.device, args.amp_dtype, backward=True)

    layer_rows = []
    for module_ordinal, (name, _, param) in enumerate(entries):
        if param.grad is None:
            raise RuntimeError(f"Missing gradient for forced H_res parameter at {name}")
        layer_index, component = layer_info_from_module_name(name)
        stats = kkt_stats_from_h_res(param, param.grad, args.kkt_tol)
        analytic_kkt_pass = stats["kkt_pass"]
        random_stats = random_sinkhorn_stats_from_gradient(
            h_res=param,
            grad=param.grad,
            args=args,
            module_ordinal=module_ordinal,
        )
        projected_gradient_stats = projected_gradient_stats_from_gradient(
            h_res=param,
            grad=param.grad,
            args=args,
            analytic_stats=stats,
        )
        row = {
            "module_name": name,
            "module_ordinal": module_ordinal,
            "layer_index": layer_index,
            "component": component,
            **stats,
            "analytic_kkt_pass": analytic_kkt_pass,
            **random_stats,
            **projected_gradient_stats,
        }
        row["kkt_pass"] = select_kkt_pass(
            args=args,
            analytic_pass=analytic_kkt_pass,
            random_sinkhorn_pass=random_stats["random_sinkhorn_sampled_pass"],
            projected_gradient_pass=projected_gradient_stats["projected_gradient_kkt_pass"],
        )
        if args.save_gradients:
            row["gradient"] = param.grad.detach().float().cpu().tolist()
        layer_rows.append(row)

    reset_forced_params_to_base(entries)
    finite_differences = compute_finite_differences(
        model=model,
        batches=batches,
        entries=entries,
        layer_rows=layer_rows,
        args=args,
        base_loss=base_loss,
    )
    global_random_sinkhorn_loss_trials = compute_global_random_sinkhorn_loss_trials(
        model=model,
        batches=batches,
        entries=entries,
        args=args,
        base_loss=base_loss,
    )
    global_random_sinkhorn_best_h_res_matrices = collect_global_random_sinkhorn_h_res_matrices(
        entries=entries,
        args=args,
        trial_index=global_random_sinkhorn_loss_trials["best_trial_index"],
    )
    single_layer_random_sinkhorn_loss_trials = compute_single_layer_random_sinkhorn_loss_trials(
        model=model,
        batches=batches,
        entries=entries,
        args=args,
        base_loss=base_loss,
    )

    failed_rows = [row for row in layer_rows if not row["kkt_pass"]]
    layer_rows_sorted = sorted(
        layer_rows,
        key=lambda row: layer_sort_key(row, args),
    )
    if args.kkt_check_mode == "global-random-sinkhorn":
        sampled_kkt_pass = global_random_sinkhorn_loss_trials["sampled_pass"]
    elif args.kkt_check_mode == "single-layer-random-sinkhorn":
        sampled_kkt_pass = single_layer_random_sinkhorn_loss_trials["sampled_pass"]
    else:
        sampled_kkt_pass = len(failed_rows) == 0

    result = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "checkpoint": args.ckpt,
        "checkpoint_iter": int(checkpoint.get("iter_num", -1)),
        "dataset": dataset,
        "split": args.split,
        "device": args.device,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "seed": args.seed,
        "amp_dtype": args.amp_dtype,
        "kkt_tol": args.kkt_tol,
        "h_res_test_point": args.h_res_test_point,
        "kkt_check_mode": args.kkt_check_mode,
        "trial_eval_mode": args.trial_eval_mode,
        "random_sinkhorn_samples": args.random_sinkhorn_samples,
        "random_sinkhorn_eps": args.random_sinkhorn_eps,
        "sinkhorn_iters": args.sinkhorn_iters,
        "sinkhorn_tol": args.sinkhorn_tol,
        "projected_gradient_step_size": args.projected_gradient_step_size,
        "projected_gradient_update": args.projected_gradient_update,
        "projected_gradient_tol": projected_gradient_effective_tol(args),
        "birkhoff_projection_iters": args.birkhoff_projection_iters,
        "birkhoff_projection_tol": args.birkhoff_projection_tol,
        "sprox_alm_iters": args.sprox_alm_iters,
        "sprox_alm_rho": args.sprox_alm_rho,
        "sprox_alm_dual_step": args.sprox_alm_dual_step,
        "sprox_alm_prox_weight": args.sprox_alm_prox_weight,
        "sprox_alm_smooth_beta": args.sprox_alm_smooth_beta,
        "sprox_alm_dual_init": args.sprox_alm_dual_init,
        "alm_projection_step_size": alm_projection_primal_step_size(args),
        "global_random_sinkhorn_trials_requested": args.global_random_sinkhorn_trials,
        "global_random_sinkhorn_trials": global_random_sinkhorn_loss_trials["num_trials"],
        "global_random_sinkhorn_sampled_pass": global_random_sinkhorn_loss_trials["sampled_pass"],
        "global_random_sinkhorn_trial_eval_mode": global_random_sinkhorn_loss_trials["trial_eval_mode"],
        "global_random_sinkhorn_num_loss_decreases": global_random_sinkhorn_loss_trials["num_loss_decreases"],
        "global_random_sinkhorn_loss_decrease_rate": global_random_sinkhorn_loss_trials["loss_decrease_rate"],
        "global_random_sinkhorn_num_directional_decreases": (
            global_random_sinkhorn_loss_trials["num_directional_decreases"]
        ),
        "global_random_sinkhorn_directional_decrease_rate": (
            global_random_sinkhorn_loss_trials["directional_decrease_rate"]
        ),
        "global_random_sinkhorn_min_loss_delta": global_random_sinkhorn_loss_trials["min_loss_delta"],
        "global_random_sinkhorn_mean_loss_delta": global_random_sinkhorn_loss_trials["mean_loss_delta"],
        "global_random_sinkhorn_max_loss_delta": global_random_sinkhorn_loss_trials["max_loss_delta"],
        "global_random_sinkhorn_min_directional_derivative": (
            global_random_sinkhorn_loss_trials["min_directional_derivative"]
        ),
        "global_random_sinkhorn_mean_directional_derivative": (
            global_random_sinkhorn_loss_trials["mean_directional_derivative"]
        ),
        "global_random_sinkhorn_max_directional_derivative": (
            global_random_sinkhorn_loss_trials["max_directional_derivative"]
        ),
        "global_random_sinkhorn_best_trial_index": global_random_sinkhorn_loss_trials["best_trial_index"],
        "global_random_sinkhorn_best_trial_loss": global_random_sinkhorn_loss_trials["best_trial_loss"],
        "global_random_sinkhorn_best_trial_directional_derivative": (
            global_random_sinkhorn_loss_trials["best_trial_directional_derivative"]
        ),
        "global_random_sinkhorn_best_trial_first_order_loss_delta": (
            global_random_sinkhorn_loss_trials["best_trial_first_order_loss_delta"]
        ),
        "global_random_sinkhorn_best_trial_row_sum_abs_err_max": (
            global_random_sinkhorn_loss_trials["best_trial_row_sum_abs_err_max"]
        ),
        "global_random_sinkhorn_best_trial_col_sum_abs_err_max": (
            global_random_sinkhorn_loss_trials["best_trial_col_sum_abs_err_max"]
        ),
        "global_random_sinkhorn_best_trial_min_entry": global_random_sinkhorn_loss_trials["best_trial_min_entry"],
        "global_random_sinkhorn_best_trial_max_offdiag": global_random_sinkhorn_loss_trials["best_trial_max_offdiag"],
        "single_layer_random_sinkhorn_trials_requested": args.single_layer_random_sinkhorn_trials,
        "single_layer_module_ordinal_requested": args.single_layer_module_ordinal,
        "single_layer_random_sinkhorn_selected_module_ordinal": (
            single_layer_random_sinkhorn_loss_trials["selected_module_ordinal"]
        ),
        "single_layer_random_sinkhorn_selected_module_name": (
            single_layer_random_sinkhorn_loss_trials["selected_module_name"]
        ),
        "single_layer_random_sinkhorn_selected_layer_index": (
            single_layer_random_sinkhorn_loss_trials["selected_layer_index"]
        ),
        "single_layer_random_sinkhorn_selected_component": (
            single_layer_random_sinkhorn_loss_trials["selected_component"]
        ),
        "single_layer_random_sinkhorn_trials": single_layer_random_sinkhorn_loss_trials["num_trials"],
        "single_layer_random_sinkhorn_sampled_pass": single_layer_random_sinkhorn_loss_trials["sampled_pass"],
        "single_layer_random_sinkhorn_trial_eval_mode": (
            single_layer_random_sinkhorn_loss_trials["trial_eval_mode"]
        ),
        "single_layer_random_sinkhorn_num_loss_decreases": (
            single_layer_random_sinkhorn_loss_trials["num_loss_decreases"]
        ),
        "single_layer_random_sinkhorn_loss_decrease_rate": (
            single_layer_random_sinkhorn_loss_trials["loss_decrease_rate"]
        ),
        "single_layer_random_sinkhorn_num_directional_decreases": (
            single_layer_random_sinkhorn_loss_trials["num_directional_decreases"]
        ),
        "single_layer_random_sinkhorn_directional_decrease_rate": (
            single_layer_random_sinkhorn_loss_trials["directional_decrease_rate"]
        ),
        "single_layer_random_sinkhorn_min_loss_delta": single_layer_random_sinkhorn_loss_trials["min_loss_delta"],
        "single_layer_random_sinkhorn_mean_loss_delta": single_layer_random_sinkhorn_loss_trials["mean_loss_delta"],
        "single_layer_random_sinkhorn_max_loss_delta": single_layer_random_sinkhorn_loss_trials["max_loss_delta"],
        "single_layer_random_sinkhorn_min_directional_derivative": (
            single_layer_random_sinkhorn_loss_trials["min_directional_derivative"]
        ),
        "single_layer_random_sinkhorn_mean_directional_derivative": (
            single_layer_random_sinkhorn_loss_trials["mean_directional_derivative"]
        ),
        "single_layer_random_sinkhorn_max_directional_derivative": (
            single_layer_random_sinkhorn_loss_trials["max_directional_derivative"]
        ),
        "single_layer_random_sinkhorn_best_trial_index": single_layer_random_sinkhorn_loss_trials["best_trial_index"],
        "single_layer_random_sinkhorn_best_trial_loss": single_layer_random_sinkhorn_loss_trials["best_trial_loss"],
        "single_layer_random_sinkhorn_best_trial_directional_derivative": (
            single_layer_random_sinkhorn_loss_trials["best_trial_directional_derivative"]
        ),
        "single_layer_random_sinkhorn_best_trial_first_order_loss_delta": (
            single_layer_random_sinkhorn_loss_trials["best_trial_first_order_loss_delta"]
        ),
        "single_layer_random_sinkhorn_best_trial_row_sum_abs_err_max": (
            single_layer_random_sinkhorn_loss_trials["best_trial_row_sum_abs_err_max"]
        ),
        "single_layer_random_sinkhorn_best_trial_col_sum_abs_err_max": (
            single_layer_random_sinkhorn_loss_trials["best_trial_col_sum_abs_err_max"]
        ),
        "single_layer_random_sinkhorn_best_trial_min_entry": (
            single_layer_random_sinkhorn_loss_trials["best_trial_min_entry"]
        ),
        "single_layer_random_sinkhorn_best_trial_max_offdiag": (
            single_layer_random_sinkhorn_loss_trials["best_trial_max_offdiag"]
        ),
        "loss_decrease_tol": args.loss_decrease_tol,
        "directional_decrease_tol": args.kkt_tol,
        "finite_diff_direction": finite_diff_direction_mode(args),
        "loss_at_forced_h_res": base_loss,
        "loss_at_forced_identity": base_loss,
        "num_h_res_modules": len(layer_rows),
        "sampled_kkt_pass": sampled_kkt_pass,
        "num_layer_kkt_failures": len(failed_rows),
        "num_kkt_failures": len(failed_rows),
        "all_modules_kkt_pass": len(failed_rows) == 0,
        "num_analytic_kkt_failures": sum(not row["analytic_kkt_pass"] for row in layer_rows),
        "num_random_sinkhorn_sampled_failures": sum(
            row["random_sinkhorn_sampled_pass"] is False
            for row in layer_rows
        ),
        "num_projected_gradient_kkt_failures": sum(
            not row["projected_gradient_kkt_pass"]
            for row in layer_rows
        ),
        "worst_min_offdiag_kkt_slack": min(row["min_offdiag_kkt_slack"] for row in layer_rows),
        "worst_pair_swap_derivative": min(row["min_pair_swap_derivative"] for row in layer_rows),
        "worst_random_sinkhorn_derivative": min(
            row["random_sinkhorn_min_derivative"]
            for row in layer_rows
            if row["random_sinkhorn_min_derivative"] is not None
        ) if args.random_sinkhorn_samples > 0 else None,
        "worst_projected_gradient_kkt_error": max(
            row["projected_gradient_kkt_error"] for row in layer_rows
        ),
        "worst_projected_gradient_fixed_point_distance": max(
            row["projected_gradient_fixed_point_distance"] for row in layer_rows
        ),
        "worst_projected_gradient_projection_row_sum_abs_err": max(
            row["projected_gradient_projection_row_sum_abs_err"] for row in layer_rows
        ),
        "worst_projected_gradient_projection_col_sum_abs_err": max(
            row["projected_gradient_projection_col_sum_abs_err"] for row in layer_rows
        ),
        "num_projected_gradient_projection_nonconverged": sum(
            row["projected_gradient_projection_converged"] is False
            for row in layer_rows
        ),
        "layers": layer_rows_sorted,
        "global_random_sinkhorn_loss_trials_detail": global_random_sinkhorn_loss_trials["trials"],
        "global_random_sinkhorn_best_h_res_matrices": global_random_sinkhorn_best_h_res_matrices,
        "single_layer_random_sinkhorn_loss_trials_detail": single_layer_random_sinkhorn_loss_trials["trials"],
        "single_layer_random_sinkhorn_best_h_res_matrix": single_layer_random_sinkhorn_loss_trials[
            "best_h_res_matrix"
        ],
        "finite_differences": finite_differences,
    }

    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "h_res_checkpoint_kkt_results",
        )
        os.makedirs(output_dir, exist_ok=True)
        ckpt_tag = os.path.basename(os.path.dirname(args.ckpt.rstrip("/"))) or "checkpoint"
        output_path = os.path.join(
            output_dir,
            f"{ckpt_tag}_{args.split}_{args.num_batches}x{args.batch_size}_seq{seq_len}_seed{args.seed}.json",
        )

    write_json_atomic(output_path, result)

    if not print_report:
        return result, output_path

    print(f"checkpoint: {args.ckpt}")
    print(f"forced H_res test point: {args.h_res_test_point}")
    print(f"forced-H_res loss: {base_loss:.6f}")
    print(f"H_res modules checked: {len(layer_rows)}")
    print(f"KKT check mode: {args.kkt_check_mode}")
    print(f"trial eval mode: {args.trial_eval_mode}")
    print(f"sampled KKT pass: {sampled_kkt_pass}")
    if global_random_sinkhorn_loss_trials["enabled"]:
        if args.trial_eval_mode == "first-order":
            print(
                "global random Sinkhorn negative <G,D>: "
                f"{global_random_sinkhorn_loss_trials['num_directional_decreases']} / "
                f"{global_random_sinkhorn_loss_trials['num_trials']}"
            )
            print(
                "global random Sinkhorn <G,D>: "
                f"min={global_random_sinkhorn_loss_trials['min_directional_derivative']:.6e} "
                f"mean={global_random_sinkhorn_loss_trials['mean_directional_derivative']:.6e} "
                f"max={global_random_sinkhorn_loss_trials['max_directional_derivative']:.6e}"
            )
            print(
                "best global random Sinkhorn trial: "
                f"idx={global_random_sinkhorn_loss_trials['best_trial_index']} "
                f"<G,D>={global_random_sinkhorn_loss_trials['best_trial_directional_derivative']:.6e} "
                f"first_order_delta={global_random_sinkhorn_loss_trials['best_trial_first_order_loss_delta']:.6e} "
                f"row_err={global_random_sinkhorn_loss_trials['best_trial_row_sum_abs_err_max']:.3e} "
                f"col_err={global_random_sinkhorn_loss_trials['best_trial_col_sum_abs_err_max']:.3e}"
            )
        else:
            print(
                "global random Sinkhorn loss decreases: "
                f"{global_random_sinkhorn_loss_trials['num_loss_decreases']} / "
                f"{global_random_sinkhorn_loss_trials['num_trials']}"
            )
            print(
                "global random Sinkhorn loss delta: "
                f"min={global_random_sinkhorn_loss_trials['min_loss_delta']:.6e} "
                f"mean={global_random_sinkhorn_loss_trials['mean_loss_delta']:.6e} "
                f"max={global_random_sinkhorn_loss_trials['max_loss_delta']:.6e}"
            )
            print(
                "best global random Sinkhorn trial: "
                f"idx={global_random_sinkhorn_loss_trials['best_trial_index']} "
                f"loss={global_random_sinkhorn_loss_trials['best_trial_loss']:.6f} "
                f"row_err={global_random_sinkhorn_loss_trials['best_trial_row_sum_abs_err_max']:.3e} "
                f"col_err={global_random_sinkhorn_loss_trials['best_trial_col_sum_abs_err_max']:.3e}"
            )
    if single_layer_random_sinkhorn_loss_trials["enabled"]:
        print(
            "single-layer random Sinkhorn target: "
            f"ordinal={single_layer_random_sinkhorn_loss_trials['selected_module_ordinal']} "
            f"layer={single_layer_random_sinkhorn_loss_trials['selected_layer_index']} "
            f"{single_layer_random_sinkhorn_loss_trials['selected_component']} "
            f"{single_layer_random_sinkhorn_loss_trials['selected_module_name']}"
        )
        if args.trial_eval_mode == "first-order":
            print(
                "single-layer random Sinkhorn negative <G,D>: "
                f"{single_layer_random_sinkhorn_loss_trials['num_directional_decreases']} / "
                f"{single_layer_random_sinkhorn_loss_trials['num_trials']}"
            )
            print(
                "single-layer random Sinkhorn <G,D>: "
                f"min={single_layer_random_sinkhorn_loss_trials['min_directional_derivative']:.6e} "
                f"mean={single_layer_random_sinkhorn_loss_trials['mean_directional_derivative']:.6e} "
                f"max={single_layer_random_sinkhorn_loss_trials['max_directional_derivative']:.6e}"
            )
            print(
                "best single-layer random Sinkhorn trial: "
                f"idx={single_layer_random_sinkhorn_loss_trials['best_trial_index']} "
                f"<G,D>={single_layer_random_sinkhorn_loss_trials['best_trial_directional_derivative']:.6e} "
                f"first_order_delta={single_layer_random_sinkhorn_loss_trials['best_trial_first_order_loss_delta']:.6e} "
                f"row_err={single_layer_random_sinkhorn_loss_trials['best_trial_row_sum_abs_err_max']:.3e} "
                f"col_err={single_layer_random_sinkhorn_loss_trials['best_trial_col_sum_abs_err_max']:.3e}"
            )
        else:
            print(
                "single-layer random Sinkhorn loss decreases: "
                f"{single_layer_random_sinkhorn_loss_trials['num_loss_decreases']} / "
                f"{single_layer_random_sinkhorn_loss_trials['num_trials']}"
            )
            print(
                "single-layer random Sinkhorn loss delta: "
                f"min={single_layer_random_sinkhorn_loss_trials['min_loss_delta']:.6e} "
                f"mean={single_layer_random_sinkhorn_loss_trials['mean_loss_delta']:.6e} "
                f"max={single_layer_random_sinkhorn_loss_trials['max_loss_delta']:.6e}"
            )
            print(
                "best single-layer random Sinkhorn trial: "
                f"idx={single_layer_random_sinkhorn_loss_trials['best_trial_index']} "
                f"loss={single_layer_random_sinkhorn_loss_trials['best_trial_loss']:.6f} "
                f"row_err={single_layer_random_sinkhorn_loss_trials['best_trial_row_sum_abs_err_max']:.3e} "
                f"col_err={single_layer_random_sinkhorn_loss_trials['best_trial_col_sum_abs_err_max']:.3e}"
            )
    print(f"layer sampled KKT failures: {len(failed_rows)} / {len(layer_rows)}")
    print(f"analytic KKT failures: {result['num_analytic_kkt_failures']} / {len(layer_rows)}")
    print(f"random Sinkhorn sampled failures: {result['num_random_sinkhorn_sampled_failures']} / {len(layer_rows)}")
    print(
        "projected-gradient KKT failures: "
        f"{result['num_projected_gradient_kkt_failures']} / {len(layer_rows)}"
    )
    print(f"worst min offdiag KKT slack: {result['worst_min_offdiag_kkt_slack']:.6e}")
    print(f"worst pair-swap derivative: {result['worst_pair_swap_derivative']:.6e}")
    if result["worst_random_sinkhorn_derivative"] is not None:
        print(f"worst random Sinkhorn derivative: {result['worst_random_sinkhorn_derivative']:.6e}")
    print(f"worst projected-gradient KKT error: {result['worst_projected_gradient_kkt_error']:.6e}")
    print(
        "worst projected-gradient fixed-point distance: "
        f"{result['worst_projected_gradient_fixed_point_distance']:.6e}"
    )
    print(f"projected-gradient update: {result['projected_gradient_update']}")
    if result["projected_gradient_update"] in ("dykstra", "alm-projection"):
        print(
            "projected-gradient projection nonconverged: "
            f"{result['num_projected_gradient_projection_nonconverged']} / {len(layer_rows)}"
        )
    print("worst modules:")
    for row in layer_rows_sorted[:10]:
        print(
            f"  layer={row['layer_index']} {row['component']:<4} "
            f"pass={row['kkt_pass']} "
            f"analytic={row['analytic_kkt_pass']} "
            f"rand={row['random_sinkhorn_sampled_pass']} "
            f"pg={row['projected_gradient_kkt_pass']} "
            f"min_slack={row['min_offdiag_kkt_slack']:.3e} "
            f"pair_deriv={row['min_pair_swap_derivative']:.3e} "
            f"rand_deriv={fmt_optional(row['random_sinkhorn_min_derivative'])} "
            f"pg_err={row['projected_gradient_kkt_error']:.3e} "
            f"pg_dist={row['projected_gradient_fixed_point_distance']:.3e} "
            f"pair={row['min_pair_swap_pair']} "
            f"grad_l2={row['grad_l2_norm']:.3e}"
        )
    if finite_differences:
        print("finite differences:")
        for row in finite_differences:
            detail = (
                f"pair={row['pair']}"
                if row["perturbation"] == "pair-swap"
                else f"sample={row['sample_index']}"
            )
            print(
                f"  layer={row['layer_index']} {row['component']:<4} "
                f"{row['perturbation']} {detail} "
                f"fd={row['finite_difference_derivative']:.3e} "
                f"pred={row['predicted_derivative']:.3e} "
                f"loss_delta={row['perturbed_loss'] - row['base_loss']:.3e}"
            )
    print(f"wrote: {output_path}")
    return result, output_path


def compact_layer_row(row):
    return {
        "module_name": row["module_name"],
        "module_ordinal": row["module_ordinal"],
        "layer_index": row["layer_index"],
        "component": row["component"],
        "kkt_pass": row["kkt_pass"],
        "analytic_kkt_pass": row["analytic_kkt_pass"],
        "random_sinkhorn_sampled_pass": row["random_sinkhorn_sampled_pass"],
        "projected_gradient_kkt_pass": row["projected_gradient_kkt_pass"],
        "min_offdiag_kkt_slack": row["min_offdiag_kkt_slack"],
        "min_pair_swap_derivative": row["min_pair_swap_derivative"],
        "random_sinkhorn_min_derivative": row["random_sinkhorn_min_derivative"],
        "projected_gradient_kkt_error": row["projected_gradient_kkt_error"],
        "projected_gradient_projection_converged": row["projected_gradient_projection_converged"],
        "grad_l2_norm": row["grad_l2_norm"],
    }


def update_module_aggregate(aggregate, layer):
    key = layer["module_name"]
    current = aggregate.setdefault(
        key,
        {
            "module_name": layer["module_name"],
            "module_ordinal": layer["module_ordinal"],
            "layer_index": layer["layer_index"],
            "component": layer["component"],
            "num_runs": 0,
            "num_kkt_failures": 0,
            "num_analytic_failures": 0,
            "num_random_sinkhorn_failures": 0,
            "num_projected_gradient_failures": 0,
            "num_projected_gradient_projection_nonconverged": 0,
            "min_offdiag_kkt_slack_min": math.inf,
            "min_pair_swap_derivative_min": math.inf,
            "random_sinkhorn_min_derivative_min": math.inf,
            "projected_gradient_kkt_error_max": -math.inf,
            "grad_l2_norm_mean_sum": 0.0,
            "grad_l2_norm_max": -math.inf,
        },
    )
    current["num_runs"] += 1
    current["num_kkt_failures"] += int(not layer["kkt_pass"])
    current["num_analytic_failures"] += int(not layer["analytic_kkt_pass"])
    current["num_random_sinkhorn_failures"] += int(layer["random_sinkhorn_sampled_pass"] is False)
    current["num_projected_gradient_failures"] += int(not layer["projected_gradient_kkt_pass"])
    current["num_projected_gradient_projection_nonconverged"] += int(
        layer["projected_gradient_projection_converged"] is False
    )
    current["min_offdiag_kkt_slack_min"] = min(
        current["min_offdiag_kkt_slack_min"], layer["min_offdiag_kkt_slack"]
    )
    current["min_pair_swap_derivative_min"] = min(
        current["min_pair_swap_derivative_min"], layer["min_pair_swap_derivative"]
    )
    if layer["random_sinkhorn_min_derivative"] is not None:
        current["random_sinkhorn_min_derivative_min"] = min(
            current["random_sinkhorn_min_derivative_min"], layer["random_sinkhorn_min_derivative"]
        )
    current["projected_gradient_kkt_error_max"] = max(
        current["projected_gradient_kkt_error_max"], layer["projected_gradient_kkt_error"]
    )
    current["grad_l2_norm_mean_sum"] += layer["grad_l2_norm"]
    current["grad_l2_norm_max"] = max(current["grad_l2_norm_max"], layer["grad_l2_norm"])


def finish_module_aggregate(aggregate):
    rows = []
    for row in aggregate.values():
        row = dict(row)
        row["kkt_failure_rate"] = row["num_kkt_failures"] / row["num_runs"]
        row["analytic_failure_rate"] = row["num_analytic_failures"] / row["num_runs"]
        row["random_sinkhorn_failure_rate"] = row["num_random_sinkhorn_failures"] / row["num_runs"]
        row["projected_gradient_failure_rate"] = row["num_projected_gradient_failures"] / row["num_runs"]
        row["grad_l2_norm_mean"] = row.pop("grad_l2_norm_mean_sum") / row["num_runs"]
        if math.isinf(row["random_sinkhorn_min_derivative_min"]):
            row["random_sinkhorn_min_derivative_min"] = None
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["num_kkt_failures"],
            row["min_offdiag_kkt_slack_min"],
            -row["projected_gradient_kkt_error_max"],
        )
    )
    return rows


def default_sweep_output_path(args, summary):
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "h_res_checkpoint_kkt_results",
    )
    ckpt_tag = os.path.basename(os.path.dirname(args.ckpt.rstrip("/"))) or "checkpoint"
    split_tag = "-".join(summary["splits"])
    batch_tag = "-".join(str(v) for v in summary["batch_sizes"])
    seed_tag = "-".join(str(v) for v in summary["seeds"])
    return os.path.join(
        output_dir,
        f"{ckpt_tag}_sweep_{split_tag}_{args.num_batches}batches_bs{batch_tag}"
        f"_seq{summary['seq_len']}_seeds{seed_tag}.json",
    )


def summarize_sweep(args, run_records, module_aggregate):
    if not run_records:
        raise ValueError("No sweep runs were executed.")
    num_runs = len(run_records)
    total_layer_checks = sum(run["num_h_res_modules"] for run in run_records)
    sampled_pass_values = [run["sampled_kkt_pass"] for run in run_records if run["sampled_kkt_pass"] is not None]
    summary = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "checkpoint": args.ckpt,
        "checkpoint_iter": run_records[0]["checkpoint_iter"],
        "dataset": run_records[0]["dataset"],
        "device": args.device,
        "seeds": sorted({run["seed"] for run in run_records}),
        "splits": sorted({run["split"] for run in run_records}),
        "batch_sizes": sorted({run["batch_size"] for run in run_records}),
        "num_batches": args.num_batches,
        "seq_len": run_records[0]["seq_len"],
        "amp_dtype": args.amp_dtype,
        "kkt_check_mode": args.kkt_check_mode,
        "trial_eval_mode": args.trial_eval_mode,
        "kkt_tol": args.kkt_tol,
        "projected_gradient_tol": projected_gradient_effective_tol(args),
        "projected_gradient_step_size": args.projected_gradient_step_size,
        "projected_gradient_update": args.projected_gradient_update,
        "birkhoff_projection_iters": args.birkhoff_projection_iters,
        "birkhoff_projection_tol": args.birkhoff_projection_tol,
        "sprox_alm_iters": args.sprox_alm_iters,
        "sprox_alm_rho": args.sprox_alm_rho,
        "sprox_alm_dual_step": args.sprox_alm_dual_step,
        "sprox_alm_prox_weight": args.sprox_alm_prox_weight,
        "sprox_alm_smooth_beta": args.sprox_alm_smooth_beta,
        "sprox_alm_dual_init": args.sprox_alm_dual_init,
        "alm_projection_step_size": alm_projection_primal_step_size(args),
        "num_runs": num_runs,
        "total_layer_checks": total_layer_checks,
        "all_runs_all_modules_kkt_pass": all(run["all_modules_kkt_pass"] for run in run_records),
        "num_runs_with_layer_kkt_failures": sum(run["num_layer_kkt_failures"] > 0 for run in run_records),
        "total_layer_kkt_failures": sum(run["num_layer_kkt_failures"] for run in run_records),
        "total_analytic_kkt_failures": sum(run["num_analytic_kkt_failures"] for run in run_records),
        "total_random_sinkhorn_sampled_failures": sum(
            run["num_random_sinkhorn_sampled_failures"] for run in run_records
        ),
        "total_projected_gradient_kkt_failures": sum(
            run["num_projected_gradient_kkt_failures"] for run in run_records
        ),
        "total_projected_gradient_projection_nonconverged": sum(
            run["num_projected_gradient_projection_nonconverged"] for run in run_records
        ),
        "sampled_kkt_pass_rate": (
            None if not sampled_pass_values else sum(bool(v) for v in sampled_pass_values) / len(sampled_pass_values)
        ),
        "worst_min_offdiag_kkt_slack": min(run["worst_min_offdiag_kkt_slack"] for run in run_records),
        "worst_pair_swap_derivative": min(run["worst_pair_swap_derivative"] for run in run_records),
        "worst_projected_gradient_kkt_error": max(
            run["worst_projected_gradient_kkt_error"] for run in run_records
        ),
        "worst_projected_gradient_fixed_point_distance": max(
            run["worst_projected_gradient_fixed_point_distance"] for run in run_records
        ),
        "runs": run_records,
        "modules": finish_module_aggregate(module_aggregate),
    }
    random_derivatives = [
        run["worst_random_sinkhorn_derivative"]
        for run in run_records
        if run["worst_random_sinkhorn_derivative"] is not None
    ]
    summary["worst_random_sinkhorn_derivative"] = (
        min(random_derivatives) if random_derivatives else None
    )
    return summary


def run_sweep(args, seeds, splits, batch_sizes):
    if args.output:
        raise ValueError("--output is for single-run mode; use --sweep-output for sweep mode.")

    run_records = []
    module_aggregate = {}
    combinations = list(product(splits, batch_sizes, seeds))
    print(
        "KKT sweep: "
        f"{len(combinations)} runs, splits={splits}, batch_sizes={batch_sizes}, "
        f"seeds={seeds}, num_batches={args.num_batches}"
    )
    for run_index, (split, batch_size, seed) in enumerate(combinations, start=1):
        run_args = argparse.Namespace(**vars(args))
        run_args.split = split
        run_args.batch_size = batch_size
        run_args.seed = seed
        run_args.output = ""
        print(
            f"[{run_index}/{len(combinations)}] split={split} "
            f"batch_size={batch_size} seed={seed}"
        )
        result, output_path = run_single(run_args, print_report=False)
        compact_layers = [compact_layer_row(row) for row in result["layers"]]
        for layer in compact_layers:
            update_module_aggregate(module_aggregate, layer)
        run_record = {
            "output_path": output_path,
            "checkpoint_iter": result["checkpoint_iter"],
            "dataset": result["dataset"],
            "split": result["split"],
            "seed": result["seed"],
            "batch_size": result["batch_size"],
            "num_batches": result["num_batches"],
            "seq_len": result["seq_len"],
            "loss_at_forced_h_res": result["loss_at_forced_h_res"],
            "loss_at_forced_identity": result["loss_at_forced_identity"],
            "sampled_kkt_pass": result["sampled_kkt_pass"],
            "all_modules_kkt_pass": result["all_modules_kkt_pass"],
            "num_h_res_modules": result["num_h_res_modules"],
            "num_layer_kkt_failures": result["num_layer_kkt_failures"],
            "num_analytic_kkt_failures": result["num_analytic_kkt_failures"],
            "num_random_sinkhorn_sampled_failures": result["num_random_sinkhorn_sampled_failures"],
            "num_projected_gradient_kkt_failures": result["num_projected_gradient_kkt_failures"],
            "num_projected_gradient_projection_nonconverged": result[
                "num_projected_gradient_projection_nonconverged"
            ],
            "worst_min_offdiag_kkt_slack": result["worst_min_offdiag_kkt_slack"],
            "worst_pair_swap_derivative": result["worst_pair_swap_derivative"],
            "worst_random_sinkhorn_derivative": result["worst_random_sinkhorn_derivative"],
            "worst_projected_gradient_kkt_error": result["worst_projected_gradient_kkt_error"],
            "worst_projected_gradient_fixed_point_distance": result[
                "worst_projected_gradient_fixed_point_distance"
            ],
            "layers": compact_layers,
        }
        run_records.append(run_record)
        print(
            f"  failures={result['num_layer_kkt_failures']}/{result['num_h_res_modules']} "
            f"analytic={result['num_analytic_kkt_failures']} "
            f"pg={result['num_projected_gradient_kkt_failures']} "
            f"worst_pg={result['worst_projected_gradient_kkt_error']:.3e} "
            f"wrote={output_path}"
        )

    summary = summarize_sweep(args, run_records, module_aggregate)
    output_path = args.sweep_output or default_sweep_output_path(args, summary)
    write_json_atomic(output_path, summary)
    print("sweep summary:")
    print(f"  runs: {summary['num_runs']}")
    print(f"  total layer checks: {summary['total_layer_checks']}")
    print(
        "  total KKT failures: "
        f"{summary['total_layer_kkt_failures']} / {summary['total_layer_checks']}"
    )
    print(
        "  total projected-gradient failures: "
        f"{summary['total_projected_gradient_kkt_failures']} / {summary['total_layer_checks']}"
    )
    print(f"  worst projected-gradient error: {summary['worst_projected_gradient_kkt_error']:.6e}")
    print("  worst modules:")
    for row in summary["modules"][:10]:
        print(
            f"    layer={row['layer_index']} {row['component']:<4} "
            f"failures={row['num_kkt_failures']}/{row['num_runs']} "
            f"pg_failures={row['num_projected_gradient_failures']} "
            f"min_slack={row['min_offdiag_kkt_slack_min']:.3e} "
            f"pg_err_max={row['projected_gradient_kkt_error_max']:.3e}"
        )
    print(f"wrote sweep summary: {output_path}")
    return summary, output_path


def main():
    args = parse_args()
    seeds, splits, batch_sizes = sweep_values(args)
    sweep_requested = bool(args.sweep_seeds or args.sweep_splits or args.sweep_batch_sizes or args.sweep_output)
    if sweep_requested or len(seeds) > 1 or len(splits) > 1 or len(batch_sizes) > 1:
        run_sweep(args, seeds=seeds, splits=splits, batch_sizes=batch_sizes)
    else:
        run_single(args, print_report=True)


if __name__ == "__main__":
    main()
