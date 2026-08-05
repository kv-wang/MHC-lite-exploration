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
        "--kkt-check-mode",
        choices=(
            "analytic",
            "random-sinkhorn",
            "global-random-sinkhorn",
            "single-layer-random-sinkhorn",
            "both",
        ),
        default="global-random-sinkhorn",
        help=(
            "Which pass/fail criterion to report as kkt_pass. 'analytic' uses exact "
            "first-order KKT inequalities; 'random-sinkhorn' uses sampled feasible "
            "identity + positive random noise projected by Sinkhorn; "
            "'global-random-sinkhorn' runs true forward passes with all layers "
            "perturbed at once; 'single-layer-random-sinkhorn' randomly selects one "
            "layer and runs true forward passes with only that layer perturbed; "
            "'both' requires analytic and per-layer random checks."
        ),
    )
    parser.add_argument(
        "--random-sinkhorn-samples",
        type=int,
        default=64,
        help="Number of random Sinkhorn feasible perturbations sampled per H_res module.",
    )
    parser.add_argument(
        "--random-sinkhorn-eps",
        type=float,
        default=1e-2,
        help="Scale of the positive random matrix added to identity before Sinkhorn.",
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
            "perturbed and all other layers stay identity. Default: 512 for "
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
    return parser.parse_args()


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


def set_forced_identity_h_res(model, device):
    entries = []
    for name, module in iter_mhc_modules(model):
        n = int(module.num_residual_streams) * int(getattr(module, "num_fracs", 1))
        param = nn.Parameter(torch.eye(n, device=device, dtype=torch.float32))
        module._kkt_forced_h_res = param
        entries.append((name, module, param))
    if not entries:
        raise RuntimeError("No mHC modules with static_h_res() were found.")
    return entries


def reset_forced_params_to_identity(entries):
    with torch.no_grad():
        for _, _, param in entries:
            param.copy_(torch.eye(param.shape[0], device=param.device, dtype=param.dtype))


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


def random_sinkhorn_seed(base_seed, module_ordinal):
    return int(base_seed + 1_000_003 * (module_ordinal + 1))


def global_random_sinkhorn_seed(base_seed, module_ordinal, trial_index):
    return int(base_seed + 1_000_003 * (module_ordinal + 1) + 104_729 * (trial_index + 1))


def build_random_sinkhorn_perturbations(n, args, module_ordinal, num_samples):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_sinkhorn_seed(args.seed, module_ordinal))
    eye = torch.eye(n, dtype=torch.float32)
    noise = torch.rand((num_samples, n, n), generator=generator, dtype=torch.float32)
    candidates = sinkhorn_project(
        eye.unsqueeze(0) + args.random_sinkhorn_eps * noise,
        args.sinkhorn_iters,
        args.sinkhorn_tol,
    )
    directions = (candidates - eye.unsqueeze(0)) / args.random_sinkhorn_eps
    return candidates, directions


def build_random_sinkhorn_perturbation(n, args, module_ordinal, sample_index):
    candidates, directions = build_random_sinkhorn_perturbations(
        n=n,
        args=args,
        module_ordinal=module_ordinal,
        num_samples=sample_index + 1,
    )
    return candidates[sample_index].clone(), directions[sample_index].clone()


def build_global_random_sinkhorn_perturbation(n, args, module_ordinal, trial_index):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(global_random_sinkhorn_seed(args.seed, module_ordinal, trial_index))
    eye = torch.eye(n, dtype=torch.float32)
    noise = torch.rand((n, n), generator=generator, dtype=torch.float32)
    candidate = sinkhorn_project(
        eye + args.random_sinkhorn_eps * noise,
        args.sinkhorn_iters,
        args.sinkhorn_tol,
    )
    direction = (candidate - eye) / args.random_sinkhorn_eps
    return candidate, direction


def random_sinkhorn_stats_from_gradient(grad, args, module_ordinal):
    num_samples = max(0, args.random_sinkhorn_samples)
    g = grad.detach().float().cpu()
    n = g.shape[0]
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
        n=n,
        args=args,
        module_ordinal=module_ordinal,
        num_samples=num_samples,
    )
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


def select_kkt_pass(args, analytic_pass, random_sinkhorn_pass):
    if args.kkt_check_mode == "analytic":
        return bool(analytic_pass)
    if args.kkt_check_mode == "random-sinkhorn":
        if random_sinkhorn_pass is None:
            raise ValueError(f"--random-sinkhorn-samples must be > 0 when --kkt-check-mode={args.kkt_check_mode}")
        return bool(random_sinkhorn_pass)
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
                n=n,
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
            n=n,
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
            n=n,
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
        n=n,
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
        reset_forced_params_to_identity(entries)
        for trial_index in range(num_trials):
            candidate, direction = build_global_random_sinkhorn_perturbation(
                n=n,
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
        reset_forced_params_to_identity(entries)

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
        reset_forced_params_to_identity(entries)

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
            direction = torch.zeros_like(param)
            direction[i, j] = 1.0
            direction[j, i] = 1.0
            direction[i, i] = -1.0
            direction[j, j] = -1.0
            with torch.no_grad():
                param.copy_(torch.eye(n, device=param.device, dtype=param.dtype) + args.finite_diff_eps * direction)
            perturbed_loss = compute_loss(
                model,
                batches,
                args.device,
                args.amp_dtype,
                backward=False,
            )
            with torch.no_grad():
                param.copy_(torch.eye(n, device=param.device, dtype=param.dtype))
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
        n = param.shape[0]
        sample_index = row["random_sinkhorn_best_sample_index"]
        candidate, _ = build_random_sinkhorn_perturbation(
            n=n,
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
            param.copy_(torch.eye(n, device=param.device, dtype=param.dtype))
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


def main():
    args = parse_args()
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

    entries = set_forced_identity_h_res(model, args.device)
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
        stats = kkt_stats_from_gradient(param.grad, args.kkt_tol)
        analytic_kkt_pass = stats["kkt_pass"]
        random_stats = random_sinkhorn_stats_from_gradient(
            grad=param.grad,
            args=args,
            module_ordinal=module_ordinal,
        )
        row = {
            "module_name": name,
            "module_ordinal": module_ordinal,
            "layer_index": layer_index,
            "component": component,
            **stats,
            "analytic_kkt_pass": analytic_kkt_pass,
            **random_stats,
        }
        row["kkt_pass"] = select_kkt_pass(
            args=args,
            analytic_pass=analytic_kkt_pass,
            random_sinkhorn_pass=random_stats["random_sinkhorn_sampled_pass"],
        )
        if args.save_gradients:
            row["gradient"] = param.grad.detach().float().cpu().tolist()
        layer_rows.append(row)

    reset_forced_params_to_identity(entries)
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
        "kkt_check_mode": args.kkt_check_mode,
        "trial_eval_mode": args.trial_eval_mode,
        "random_sinkhorn_samples": args.random_sinkhorn_samples,
        "random_sinkhorn_eps": args.random_sinkhorn_eps,
        "sinkhorn_iters": args.sinkhorn_iters,
        "sinkhorn_tol": args.sinkhorn_tol,
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
        "worst_min_offdiag_kkt_slack": min(row["min_offdiag_kkt_slack"] for row in layer_rows),
        "worst_pair_swap_derivative": min(row["min_pair_swap_derivative"] for row in layer_rows),
        "worst_random_sinkhorn_derivative": min(
            row["random_sinkhorn_min_derivative"]
            for row in layer_rows
            if row["random_sinkhorn_min_derivative"] is not None
        ) if args.random_sinkhorn_samples > 0 else None,
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
            "h_res_identity_kkt_results",
        )
        os.makedirs(output_dir, exist_ok=True)
        ckpt_tag = os.path.basename(os.path.dirname(args.ckpt.rstrip("/"))) or "checkpoint"
        output_path = os.path.join(
            output_dir,
            f"{ckpt_tag}_{args.split}_{args.num_batches}x{args.batch_size}_seq{seq_len}_seed{args.seed}.json",
        )

    with open(output_path + ".tmp", "w") as f:
        json.dump(result, f, indent=2)
    os.replace(output_path + ".tmp", output_path)

    print(f"checkpoint: {args.ckpt}")
    print(f"forced-identity loss: {base_loss:.6f}")
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
    print(f"worst min offdiag KKT slack: {result['worst_min_offdiag_kkt_slack']:.6e}")
    print(f"worst pair-swap derivative: {result['worst_pair_swap_derivative']:.6e}")
    if result["worst_random_sinkhorn_derivative"] is not None:
        print(f"worst random Sinkhorn derivative: {result['worst_random_sinkhorn_derivative']:.6e}")
    print("worst modules:")
    for row in layer_rows_sorted[:10]:
        print(
            f"  layer={row['layer_index']} {row['component']:<4} "
            f"pass={row['kkt_pass']} "
            f"analytic={row['analytic_kkt_pass']} "
            f"rand={row['random_sinkhorn_sampled_pass']} "
            f"min_slack={row['min_offdiag_kkt_slack']:.3e} "
            f"pair_deriv={row['min_pair_swap_derivative']:.3e} "
            f"rand_deriv={fmt_optional(row['random_sinkhorn_min_derivative'])} "
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


if __name__ == "__main__":
    main()
