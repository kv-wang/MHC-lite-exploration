from __future__ import annotations
from typing import Callable

from functools import partial
import math
from random import randrange

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
"""

# helper functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def add(x, y):
    return x + y


def make_equal_diag_offdiag_doubly_stochastic(size, diag_mass_frac=1.):
    if size < 1:
        raise ValueError("size must be >= 1")

    diag_value = float(diag_mass_frac)
    if size == 1:
        return torch.ones((1, 1))

    offdiag_value = (1. - diag_value) / (size - 1)
    matrix = torch.full((size, size), offdiag_value)
    matrix.fill_diagonal_(diag_value)
    return matrix


DYNAMIC_H_RES_DISABLED_MODES = {
    "unconstrained",
    "identity_tanh_offdiag",
    "alm_nonnegative",
    "alm_nonnegative_cap",
    "alm_signed_sprox",
    "alm_spectral_sprox",
}


class Scale(Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale))

    def forward(self, residuals):
        return residuals * self.scale.to(device=residuals.device, dtype=residuals.dtype)


class SoftmaxWeightedReduceStreams(Module):
    def __init__(self, num_streams, scale=4.):
        super().__init__()
        self.num_streams = num_streams
        self.scale = scale
        self.logits = nn.Parameter(torch.zeros(num_streams))

    def forward(self, residuals):
        residuals = rearrange(residuals, '(b s) ... -> b s ...', s=self.num_streams)
        weights = self.logits.softmax(dim=0).to(device=residuals.device, dtype=residuals.dtype)
        view_shape = (1, self.num_streams) + (1,) * (residuals.ndim - 2)
        return (residuals * weights.view(view_shape)).sum(dim=1) * self.scale

# sinkhorn

def l1norm(t, dim):
    return F.normalize(t, p = 1, dim = dim)

# def sinkhorn_knopps(log_alpha, iters = 20):
#     log_alpha = log_alpha - log_alpha.amax(dim = -2, keepdim = True).detach()

#     alpha = log_alpha.exp()

#     for _ in range(iters):
#         alpha = l1norm(alpha, dim = -2)
#         alpha = l1norm(alpha, dim = -1)

#     return alpha

def sinkhorn_knopps(log_alpha, iters=20):

    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)

    return log_alpha.exp()

def sinkhorn_with_marginals(log_alpha, row_targets, col_targets, iters=20):
    row_targets = row_targets.to(device=log_alpha.device, dtype=log_alpha.dtype)
    col_targets = col_targets.to(device=log_alpha.device, dtype=log_alpha.dtype)
    log_row_targets = row_targets.clamp_min(torch.finfo(log_alpha.dtype).tiny).log()
    log_col_targets = col_targets.clamp_min(torch.finfo(log_alpha.dtype).tiny).log()

    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        log_alpha = log_alpha + log_col_targets.view(*((1,) * (log_alpha.ndim - 1)), -1)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha + log_row_targets.view(*((1,) * (log_alpha.ndim - 2)), -1, 1)

    return log_alpha.exp()

def sinkhorn_with_capacity(log_alpha, base_streams, cap=1., iters=20):
    # Base rows / cols use equality constraints; added rows / cols use capacity constraints.
    cap = torch.as_tensor(cap, device=log_alpha.device, dtype=log_alpha.dtype)
    log_cap = cap.clamp_min(torch.finfo(log_alpha.dtype).tiny).log()
    rows = log_alpha.shape[-2]
    cols = log_alpha.shape[-1]
    base_streams = min(base_streams, rows, cols)

    row_is_base = torch.arange(rows, device=log_alpha.device) < base_streams
    row_is_base = row_is_base.view(*((1,) * (log_alpha.ndim - 2)), rows, 1)
    col_is_base = torch.arange(cols, device=log_alpha.device) < base_streams
    col_is_base = col_is_base.view(*((1,) * (log_alpha.ndim - 1)), cols)

    for _ in range(iters):
        row_logsum = torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        row_cap_scale = torch.minimum(torch.zeros_like(row_logsum), log_cap - row_logsum)
        log_alpha = log_alpha + torch.where(row_is_base, -row_logsum, row_cap_scale)

        col_logsum = torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        col_cap_scale = torch.minimum(torch.zeros_like(col_logsum), log_cap - col_logsum)
        log_alpha = log_alpha + torch.where(col_is_base, -col_logsum, col_cap_scale)

    return log_alpha.exp()

def project_simplex(x, dim=-1, target=1.):
    dim = dim if dim >= 0 else x.ndim + dim
    target = torch.as_tensor(target, device=x.device, dtype=x.dtype)
    x = x.movedim(dim, -1)
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])

    sorted_x, _ = torch.sort(x_flat, dim=-1, descending=True)
    cssv = sorted_x.cumsum(dim=-1) - target
    ind = torch.arange(1, shape[-1] + 1, device=x.device, dtype=x.dtype)
    support = sorted_x - cssv / ind > 0
    rho = support.sum(dim=-1, keepdim=True).clamp_min(1)
    theta = cssv.gather(-1, rho - 1) / rho.to(dtype=x.dtype)

    projected = (x_flat - theta).clamp_min(0)
    return projected.reshape(shape).movedim(-1, dim)

def project_nonnegative_l1_ball(x, cap=1., dim=-1):
    cap = torch.as_tensor(cap, device=x.device, dtype=x.dtype)
    x_pos = x.clamp_min(0)
    x_sum = x_pos.sum(dim=dim, keepdim=True)
    projected = project_simplex(x, dim=dim, target=cap)
    return torch.where(x_sum <= cap, x_pos, projected)

def project_rows(x):
    return project_simplex(x, dim=-1)

def project_cols(x):
    return project_simplex(x.mT, dim=-1).mT

def project_mixed_rows(x, base_streams, cap=1.):
    rows = x.shape[-2]
    base_streams = min(base_streams, rows)
    if base_streams == rows:
        return project_simplex(x, dim=-1)
    row_is_base = torch.arange(rows, device=x.device) < base_streams
    row_is_base = row_is_base.view(*((1,) * (x.ndim - 2)), rows, 1)
    equality = project_simplex(x, dim=-1)
    capacity = project_nonnegative_l1_ball(x, cap=cap, dim=-1)
    return torch.where(row_is_base, equality, capacity)

def project_mixed_cols(x, base_streams, cap=1.):
    return project_mixed_rows(x.mT, base_streams=base_streams, cap=cap).mT

def admm_doubly_stochastic(log_alpha, iters=20, rho=1.):
    a = torch.softmax(log_alpha, dim=-1)
    x1 = project_rows(a)
    x2 = project_cols(a)
    u = torch.zeros_like(a)

    for _ in range(iters):
        x1 = project_rows((a + rho * (x2 - u)) / (1. + rho))
        x2 = project_cols(x1 + u)
        u = u + x1 - x2

    return 0.5 * (x1 + x2)

def admm_with_capacity(log_alpha, base_streams, cap=1., iters=20, rho=1.):
    orig_dtype = log_alpha.dtype
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    log_alpha = log_alpha.to(work_dtype)
    rows = log_alpha.shape[-2]
    cols = log_alpha.shape[-1]

    row_targets = torch.ones((rows,), device=log_alpha.device, dtype=work_dtype)
    row_targets = row_targets.view(*((1,) * (log_alpha.ndim - 2)), rows, 1)
    col_targets = torch.ones((cols,), device=log_alpha.device, dtype=work_dtype)
    col_targets = col_targets.view(*((1,) * (log_alpha.ndim - 1)), cols)

    row_dual = torch.zeros_like(row_targets).expand(*log_alpha.shape[:-1], 1)
    col_dual = torch.zeros_like(col_targets).expand(*log_alpha.shape[:-2], 1, cols)
    log_scale = torch.zeros_like(log_alpha)
    u = torch.zeros_like(log_alpha)
    rho = torch.as_tensor(rho, device=log_alpha.device, dtype=work_dtype)

    for _ in range(iters):
        target_scale = row_dual + col_dual - u
        next_log_scale = log_scale
        for _ in range(8):
            alpha = (log_alpha + next_log_scale).exp()
            residual = alpha + rho * (next_log_scale - target_scale)
            curvature = alpha + rho
            next_log_scale = next_log_scale - residual / curvature
        log_scale = next_log_scale

        row_update = (log_scale - col_dual + u).mean(dim=-1, keepdim=True)
        row_update = row_update + row_targets / (rho * cols)
        row_dual = row_update

        col_update = (log_scale - row_dual + u).mean(dim=-2, keepdim=True)
        col_update = col_update + col_targets / (rho * rows)
        col_dual = col_update

        u = u + log_scale - row_dual - col_dual

    return (log_alpha + log_scale).exp().to(orig_dtype)

def admm_reverse_kl_with_capacity(log_alpha, base_streams, cap=1., iters=20, rho=1., floor=1e-30):
    orig_dtype = log_alpha.dtype
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    log_alpha = log_alpha.to(work_dtype)
    floor = torch.as_tensor(floor, device=log_alpha.device, dtype=work_dtype)
    rho = torch.as_tensor(rho, device=log_alpha.device, dtype=work_dtype)

    # Reverse KL needs a positive reference K. X itself is kept nonnegative by
    # the row / column projections below, not by exponentiating the output.
    k = F.softplus(log_alpha) + floor
    x = k
    y = project_rows(x)
    z = project_cols(x)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)
    tiny = torch.finfo(work_dtype).tiny

    for _ in range(iters):
        q = 0.5 * (y - u + z - v)
        b = 1. - 2. * rho * q
        sqrt_disc = (b.square() + 8. * rho * k).sqrt()
        direct = (-b + sqrt_disc) / (4. * rho)
        stable = (2. * k) / (sqrt_disc + b).clamp_min(tiny)
        x = torch.where(b >= 0., stable, direct)

        y = project_rows(x + u)
        z = project_cols(x + v)
        u = u + x - y
        v = v + x - z

    return (0.5 * (y + z)).to(orig_dtype)

def admm_reverse_kl_sprox_alm_with_capacity(
    log_alpha,
    base_streams,
    cap=1.,
    iters=20,
    rho=1.,
    floor=1e-30,
    dual_step=0.5,
    prox_weight=None,
    smooth_beta=0.5,
    step_scale=1.,
):
    orig_dtype = log_alpha.dtype
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    log_alpha = log_alpha.to(work_dtype)
    floor = torch.as_tensor(floor, device=log_alpha.device, dtype=work_dtype)
    rho = torch.as_tensor(rho, device=log_alpha.device, dtype=work_dtype)

    rows = log_alpha.shape[-2]
    cols = log_alpha.shape[-1]

    row_targets = torch.ones((rows,), device=log_alpha.device, dtype=work_dtype)
    row_targets = row_targets.view(*((1,) * (log_alpha.ndim - 2)), rows, 1)
    col_targets = torch.ones((cols,), device=log_alpha.device, dtype=work_dtype)
    col_targets = col_targets.view(*((1,) * (log_alpha.ndim - 1)), cols)

    k = F.softplus(log_alpha) + floor
    x = k.clamp_min(floor)
    z_aux = x
    row_eq_dual = torch.zeros_like(x.sum(dim=-1, keepdim=True))
    col_eq_dual = torch.zeros_like(x.sum(dim=-2, keepdim=True))
    dual_step = torch.as_tensor(dual_step, device=log_alpha.device, dtype=work_dtype)
    prox_weight = rho if prox_weight is None else torch.as_tensor(prox_weight, device=log_alpha.device, dtype=work_dtype)
    smooth_beta = torch.as_tensor(smooth_beta, device=log_alpha.device, dtype=work_dtype)
    step_scale = torch.as_tensor(step_scale, device=log_alpha.device, dtype=work_dtype)
    smooth_floor = torch.maximum(floor, torch.as_tensor(1e-8, device=log_alpha.device, dtype=work_dtype))
    x = x.clamp_min(smooth_floor)

    for _ in range(iters):
        row_res = x.sum(dim=-1, keepdim=True) - row_targets
        col_res = x.sum(dim=-2, keepdim=True) - col_targets

        row_eq_dual = row_eq_dual + dual_step * row_res
        col_eq_dual = col_eq_dual + dual_step * col_res

        row_term = row_eq_dual + rho * row_res
        col_term = col_eq_dual + rho * col_res

        x_safe = x.clamp_min(smooth_floor)
        grad = 1. - k / x_safe
        grad = grad + row_term + col_term + prox_weight * (x - z_aux)

        local_lip = (k / x_safe.square()).clamp_min(1.)
        step_size = step_scale / (local_lip + rho * (rows + cols) + prox_weight)
        next_x = (x - step_size * grad).clamp_min(smooth_floor)
        z_aux = z_aux + smooth_beta * (next_x - z_aux)
        x = next_x

    return x.to(orig_dtype)

def admm_l2_with_capacity(log_alpha, base_streams, cap=1., iters=20, rho=1., floor=1e-30):
    orig_dtype = log_alpha.dtype
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    log_alpha = log_alpha.to(work_dtype)
    floor = torch.as_tensor(floor, device=log_alpha.device, dtype=work_dtype)
    rho = torch.as_tensor(rho, device=log_alpha.device, dtype=work_dtype)

    k = F.softplus(log_alpha) + floor
    x = k
    y = project_rows(x)
    z = project_cols(x)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)

    for _ in range(iters):
        x = (k + rho * (y - u + z - v)) / (1. + 2. * rho)
        y = project_rows(x + u)
        z = project_cols(x + v)
        u = u + x - y
        v = v + x - z

    return (0.5 * (y + z)).to(orig_dtype)

def cayley_orthogonalize(matrix):
    orig_dtype = matrix.dtype
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    matrix = matrix.to(work_dtype)
    skew = 0.5 * (matrix - matrix.transpose(-2, -1))
    eye = torch.eye(skew.shape[-1], device=skew.device, dtype=work_dtype)
    eye = eye.expand(skew.shape)
    orth = torch.linalg.solve(eye - skew, eye + skew)
    return orth.to(orig_dtype)

def project_spectral_residual_sphere(matrix, radius=1.):
    orig_dtype = matrix.dtype
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    matrix = matrix.to(work_dtype)
    rows, cols = matrix.shape[-2:]
    if rows != cols:
        raise ValueError(f"spectral H_res projection expects a square matrix, got {tuple(matrix.shape[-2:])}")

    radius = torch.as_tensor(radius, device=matrix.device, dtype=work_dtype)
    j = torch.full((rows, cols), 1. / rows, device=matrix.device, dtype=work_dtype)
    j = j.expand(matrix.shape)
    displacement = matrix - j

    # Orthogonal projection onto {D | D 1 = 0, 1^T D = 0}.
    displacement = (
        displacement
        - displacement.mean(dim=-1, keepdim=True)
        - displacement.mean(dim=-2, keepdim=True)
        + displacement.mean(dim=(-2, -1), keepdim=True)
    )

    u, s, vh = torch.linalg.svd(displacement, full_matrices=False)
    s = s.clamp(max=radius)
    projected = (u * s.unsqueeze(-2)) @ vh
    return (j + projected).to(orig_dtype)

# main functions

def get_expand_reduce_stream_functions(
    num_streams,
    add_stream_embed = False,
    dim = None,
    disable = False,
    reduce_stream_mode = "sum",
    expand_stream_mode = "repeat",
    expand_active_streams = None,
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if reduce_stream_mode not in {"sum", "mean", "4mean", "softmax_4mean"}:
        raise ValueError(f"Invalid reduce_stream_mode: {reduce_stream_mode}")
    if expand_stream_mode not in {"repeat", "split", "repeat_base_zero_rest"}:
        raise ValueError(f"Invalid expand_stream_mode: {expand_stream_mode}")

    if add_stream_embed:
        assert exists(dim), '`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added'

        expand_fn = StreamEmbed(
            num_streams,
            dim,
            expand_to_streams = True,
            expand_stream_mode = expand_stream_mode,
            expand_active_streams = expand_active_streams,
        )
    else:
        expand_fn = ExpandStreams(
            num_streams,
            mode = expand_stream_mode,
            active_streams = expand_active_streams,
        )

    if reduce_stream_mode == "4mean":
        reduce_fn = Sequential(
            Reduce(pattern='(b s) ... -> b ...', reduction='mean', s=num_streams),
            Scale(4.),
        )
    elif reduce_stream_mode == "softmax_4mean":
        reduce_fn = SoftmaxWeightedReduceStreams(num_streams, scale=4.)
    else:
        reduce_fn = Reduce(pattern = '(b s) ... -> b ...', reduction = reduce_stream_mode, s = num_streams)

    return expand_fn, reduce_fn

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    disable = None,
    sinkhorn_iters = 20,
    reduce_stream_mode = "sum",
    expand_stream_mode = "repeat",
    **kwargs
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = ManifoldConstrainedHyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs = num_fracs, sinkhorn_iters = sinkhorn_iters, **kwargs)
    expand_active_streams = None
    if expand_stream_mode == "repeat_base_zero_rest":
        expand_active_streams = kwargs.get("mhc_adapter_base_streams", num_streams)

    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams,
        add_stream_embed = add_stream_embed,
        dim = dim,
        disable = disable,
        reduce_stream_mode = reduce_stream_mode,
        expand_stream_mode = expand_stream_mode,
        expand_active_streams = expand_active_streams,
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)

# norms

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main classes

# residual base class

class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(
        self,
        residuals
    ):
        return residuals, residuals, dict()

    def depth_connection(
        self,
        branch_output,
        residuals,

    ):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

# hyper connection residual streams

class ManifoldConstrainedHyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index = None,
        channel_first = False,
        dropout = 0.,
        residual_transform: Module | None = None, # to support resnet blocks where dimension in not equal to dimension out - usually a residual conv
        add_branch_out_to_residual = True,  # will disable depth connections (weighted residual sum with beta) if set False
        num_input_views = 1,                # allow for the branch module to receive multiple input views, dimension placed on the very left (before batch)
        depth_residual_fn = add,
        num_fracs = 1,                      # https://arxiv.org/abs/2503.14125
        sinkhorn_iters = 20,
        mhc_gate_fn = "sigmoid",
        mhc_zero_init_pre_post_logits = False,
        mhc_identity_h_res = False,
        mhc_h_res_mode = "sinkhorn",
        mhc_admm_iters = 20,
        mhc_admm_rho = 1.,
        mhc_admm_dual_step = 0.5,
        mhc_admm_prox_weight = None,
        mhc_admm_smooth_beta = 0.5,
        mhc_admm_step_scale = 1.,
        mhc_h_res_init_diag_mass_frac = 1.,
        mhc_h_res_cap = 1.5,
        mhc_h_res_offdiag_init_scale = 0.05,
        mhc_disable_dynamic_h_res = False,
        mhc_adapter_base_streams = 4,
        mhc_adapter_epsilon = 0.1,
        mhc_adapter_cap = 1.,
        mhc_adapter_admm_input_mode = "raw_logits",
        mhc_adapter_admm_input_floor = 1e-30,
        mhc_adapter_admm_checkpoint = False,
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()
        valid_h_res_modes = {"sinkhorn", "admm", "admm_reverse_kl", "admm_reverse_kl_sprox_alm", "admm_l2", "unconstrained", "identity_tanh_offdiag", "alm_signed", "alm_nonnegative", "alm_nonnegative_cap", "alm_signed_sprox", "alm_spectral_sprox", "cayley", "adapter_epsilon", "adapter_cap", "adapter_cap_admm"}
        if mhc_h_res_mode not in valid_h_res_modes:
            raise ValueError(f"Invalid mhc_h_res_mode: {mhc_h_res_mode}")
        valid_adapter_admm_input_modes = {"raw_logits", "sinkhorn_base_log"}
        if mhc_adapter_admm_input_mode not in valid_adapter_admm_input_modes:
            raise ValueError(f"Invalid mhc_adapter_admm_input_mode: {mhc_adapter_admm_input_mode}")
        if mhc_admm_iters < 1:
            raise ValueError("mhc_admm_iters must be >= 1")
        if mhc_admm_rho <= 0:
            raise ValueError("mhc_admm_rho must be > 0")
        if mhc_admm_dual_step <= 0:
            raise ValueError("mhc_admm_dual_step must be > 0")
        if mhc_admm_prox_weight is not None and mhc_admm_prox_weight < 0:
            raise ValueError("mhc_admm_prox_weight must be >= 0 when set")
        if mhc_admm_smooth_beta <= 0:
            raise ValueError("mhc_admm_smooth_beta must be > 0")
        if mhc_admm_step_scale <= 0:
            raise ValueError("mhc_admm_step_scale must be > 0")
        if not 0 <= mhc_h_res_init_diag_mass_frac <= 1:
            raise ValueError("mhc_h_res_init_diag_mass_frac must be in [0, 1]")
        if mhc_h_res_cap <= 0:
            raise ValueError("mhc_h_res_cap must be > 0")
        if mhc_h_res_offdiag_init_scale <= 0:
            raise ValueError("mhc_h_res_offdiag_init_scale must be > 0")
        if mhc_disable_dynamic_h_res and mhc_h_res_mode not in DYNAMIC_H_RES_DISABLED_MODES:
            raise ValueError(
                f"mhc_disable_dynamic_h_res=True is only supported for modes "
                f"{sorted(DYNAMIC_H_RES_DISABLED_MODES)}, got {mhc_h_res_mode!r}"
            )
        if mhc_adapter_base_streams < 1:
            raise ValueError("mhc_adapter_base_streams must be >= 1")
        if mhc_adapter_epsilon <= 0:
            raise ValueError("mhc_adapter_epsilon must be > 0")
        if mhc_adapter_cap <= 0:
            raise ValueError("mhc_adapter_cap must be > 0")
        if mhc_adapter_admm_input_floor <= 0:
            raise ValueError("mhc_adapter_admm_input_floor must be > 0")

        self.branch = branch

        # frac-connections paper - num_fracs > 1 will be the `m` in their paper https://arxiv.org/abs/2503.14125

        assert num_fracs >= 1

        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange('b ... (f d) -> b ... f d', f = num_fracs)
        self.merge_fracs = Rearrange('b ... f d -> b ... (f d)')

        assert divisible_by(dim, num_fracs), f'feature dimension ({dim}) must be divisible by the `num_fracs` ({num_fracs})'

        dim //= num_fracs # effective dim handled in dimension is feature dimension divided by num fractions

        # they used layernorm in paper, but rmsnorm is fine given what we know now

        assert num_residual_streams > 0, '`num_residual_streams` must be greater than 0'

        self.num_residual_streams = num_residual_streams
        init_residual_index = default(layer_index, randrange(num_residual_streams)) % num_residual_streams # just choose one random residual stream if layer index not given

        # handle the parameter dimensions, which may require (num_residuals x num_fractions) - generalizing hyper + frac connections

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs

        self.num_fracs = num_fracs

        # width num residual streams

        assert num_input_views >= 1
        self.num_input_views = num_input_views

        # width connection

        self.norm = RMSNorm(dim * num_residual_streams_fracs)

        if mhc_zero_init_pre_post_logits:
            init_alpha0 = torch.zeros((num_residual_streams_fracs, num_input_views_fracs))
        else:
            init_alpha0 = torch.ones((num_residual_streams_fracs, num_input_views_fracs)) * -1
            init_alpha0[init_residual_index, :] = 1.
        if mhc_h_res_mode == "identity_tanh_offdiag":
            init_alpha1 = torch.zeros((num_residual_streams_fracs, num_residual_streams_fracs))
        elif mhc_h_res_mode in {"unconstrained", "alm_signed", "alm_nonnegative", "alm_nonnegative_cap", "alm_signed_sprox", "alm_spectral_sprox"}:
            init_alpha1 = make_equal_diag_offdiag_doubly_stochastic(
                num_residual_streams_fracs,
                diag_mass_frac=mhc_h_res_init_diag_mass_frac,
            )
        else:
            init_alpha1 = torch.ones((num_residual_streams_fracs, num_residual_streams_fracs)) * -8
            init_alpha1.fill_diagonal_(0.)
        self.static_alpha = nn.Parameter(cat((init_alpha0, init_alpha1), dim = 1))

        self.dynamic_h_res_disabled = bool(
            mhc_disable_dynamic_h_res and mhc_h_res_mode in DYNAMIC_H_RES_DISABLED_MODES
        )
        dynamic_alpha_out_dim = num_fracs * num_residual_streams * num_input_views
        if not self.dynamic_h_res_disabled:
            dynamic_alpha_out_dim += num_fracs * num_residual_streams * num_residual_streams

        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(
                dim * num_residual_streams, 
                dynamic_alpha_out_dim
            ) 
        )

        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)
        if mhc_h_res_mode == "identity_tanh_offdiag":
            self.h_res_offdiag_log_scale = nn.Parameter(
                torch.tensor(math.log(float(mhc_h_res_offdiag_init_scale)))
            )

        # depth connection related (beta)

        self.add_branch_out_to_residual = add_branch_out_to_residual

        if add_branch_out_to_residual:
            if mhc_zero_init_pre_post_logits:
                beta_init = torch.zeros(num_residual_streams_fracs)
            else:
                beta_init = torch.ones(num_residual_streams_fracs) * -1.
                beta_init[init_residual_index] = 1.
            self.static_beta = nn.Parameter(beta_init)

            # ------ 
            # XXX
            # same here
            # ------
            dynamic_beta_shape = (
                dim * num_residual_streams,
                num_fracs * num_residual_streams
            ) # preserve backwards compat
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dynamic_beta_shape))

            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        # sinkhorn related

        self.sinkhorn_iters = sinkhorn_iters
        self.mhc_gate_fn = mhc_gate_fn
        self.mhc_zero_init_pre_post_logits = mhc_zero_init_pre_post_logits
        self.mhc_identity_h_res = mhc_identity_h_res
        self.mhc_h_res_mode = mhc_h_res_mode
        self.mhc_admm_iters = mhc_admm_iters
        self.mhc_admm_rho = mhc_admm_rho
        self.mhc_admm_dual_step = mhc_admm_dual_step
        self.mhc_admm_prox_weight = mhc_admm_prox_weight
        self.mhc_admm_smooth_beta = mhc_admm_smooth_beta
        self.mhc_admm_step_scale = mhc_admm_step_scale
        self.mhc_h_res_init_diag_mass_frac = mhc_h_res_init_diag_mass_frac
        self.mhc_h_res_cap = mhc_h_res_cap
        self.mhc_h_res_offdiag_init_scale = mhc_h_res_offdiag_init_scale
        self.mhc_disable_dynamic_h_res = mhc_disable_dynamic_h_res
        self.mhc_adapter_base_streams = mhc_adapter_base_streams
        self.mhc_adapter_epsilon = mhc_adapter_epsilon
        self.mhc_adapter_cap = mhc_adapter_cap
        self.mhc_adapter_admm_input_mode = mhc_adapter_admm_input_mode
        self.mhc_adapter_admm_input_floor = mhc_adapter_admm_input_floor
        self.mhc_adapter_admm_checkpoint = mhc_adapter_admm_checkpoint
        self.collect_h_res_constraint_errors = False
        self.reset_h_res_constraint_errors()

        h_res_shape = (num_residual_streams_fracs, num_residual_streams_fracs)
        h_res_alm_row_dual = torch.zeros(num_residual_streams_fracs, 1)
        h_res_alm_col_dual = torch.zeros(1, num_residual_streams_fracs)
        h_res_alm_nonneg_dual = torch.zeros(h_res_shape)
        h_res_sprox_z = make_equal_diag_offdiag_doubly_stochastic(
            num_residual_streams_fracs,
            diag_mass_frac=mhc_h_res_init_diag_mass_frac,
        )
        self.register_buffer("h_res_alm_row_dual", h_res_alm_row_dual)
        self.register_buffer("h_res_alm_col_dual", h_res_alm_col_dual)
        self.register_buffer("h_res_alm_nonneg_dual", h_res_alm_nonneg_dual)
        self.register_buffer("h_res_sprox_z", h_res_sprox_z)
        self.reset_h_res_alm_forward_loss()
        self.reset_h_res_alm_dual_update_accumulators()

        # dropouts

        self.dropout = nn.Dropout(dropout)

        # channel first option

        self.channel_first = channel_first

        # maybe residual transform

        self.residual_transform = default(residual_transform, nn.Identity())

        # maybe custom depth connection residual function
        # this is to prepare for gating the addition of the branch outputs to the residual streams
        # needed for memory lanes a la RMT / LMM

        self.depth_residual_fn = depth_residual_fn

    def reset_h_res_constraint_errors(self):
        self._h_res_constraint_count = 0
        self._h_res_row_err_sum = 0.
        self._h_res_col_err_sum = 0.
        self._h_res_nonneg_violation_sum = 0.
        self._h_res_spectral_violation_sum = 0.
        self._h_res_row_err_max = 0.
        self._h_res_col_err_max = 0.
        self._h_res_nonneg_violation_max = 0.
        self._h_res_spectral_violation_max = 0.

    def _record_h_res_constraint_errors(self, alpha_residual):
        if not self.collect_h_res_constraint_errors:
            return
        if self.mhc_h_res_mode == "identity_tanh_offdiag":
            return

        with torch.no_grad():
            matrix = alpha_residual.detach().float()
            if self.mhc_h_res_mode == "alm_nonnegative_cap":
                cap = torch.as_tensor(self.mhc_h_res_cap, device=matrix.device, dtype=matrix.dtype)
                row_err = (matrix.sum(dim=-1) - cap).clamp_min(0.).amax().item()
                col_err = (matrix.sum(dim=-2) - cap).clamp_min(0.).amax().item()
            else:
                row_err = (matrix.sum(dim=-1) - 1.).abs().amax().item()
                col_err = (matrix.sum(dim=-2) - 1.).abs().amax().item()
            if self.mhc_h_res_mode in {"alm_signed_sprox", "alm_spectral_sprox"}:
                nonneg_violation = 0.
            else:
                nonneg_violation = (-matrix).clamp_min(0.).amax().item()
            if self.mhc_h_res_mode == "alm_spectral_sprox":
                n = matrix.shape[-1]
                j = torch.full_like(matrix, 1. / n)
                displacement = matrix - j
                spectral_norm = torch.linalg.matrix_norm(displacement, ord=2).item()
                spectral_violation = max(0., spectral_norm - 1.)
            else:
                spectral_violation = 0.

        self._h_res_constraint_count += 1
        self._h_res_row_err_sum += row_err
        self._h_res_col_err_sum += col_err
        self._h_res_nonneg_violation_sum += nonneg_violation
        self._h_res_spectral_violation_sum += spectral_violation
        self._h_res_row_err_max = max(self._h_res_row_err_max, row_err)
        self._h_res_col_err_max = max(self._h_res_col_err_max, col_err)
        self._h_res_nonneg_violation_max = max(self._h_res_nonneg_violation_max, nonneg_violation)
        self._h_res_spectral_violation_max = max(self._h_res_spectral_violation_max, spectral_violation)

    def get_h_res_constraint_errors(self):
        count = self._h_res_constraint_count
        if count <= 0:
            return None

        return {
            "row_err_mean": self._h_res_row_err_sum / count,
            "col_err_mean": self._h_res_col_err_sum / count,
            "nonneg_violation_mean": self._h_res_nonneg_violation_sum / count,
            "spectral_violation_mean": self._h_res_spectral_violation_sum / count,
            "row_err_max": self._h_res_row_err_max,
            "col_err_max": self._h_res_col_err_max,
            "nonneg_violation_max": self._h_res_nonneg_violation_max,
            "spectral_violation_max": self._h_res_spectral_violation_max,
            "count": count,
        }

    def reset_h_res_alm_forward_loss(self):
        self._h_res_alm_loss = None
        self._h_res_sprox_lm_grad_norm = None

    def get_h_res_alm_loss(self):
        return self._h_res_alm_loss

    def static_h_res(self):
        return self.static_alpha[:, self.num_input_views:]

    def identity_tanh_offdiag_h_res(self, h_res_logits):
        n = h_res_logits.shape[-1]
        eye = torch.eye(n, device=h_res_logits.device, dtype=h_res_logits.dtype)
        offdiag_mask = 1. - eye
        log_scale = getattr(self, "h_res_offdiag_log_scale", None)
        if log_scale is None:
            gamma = torch.as_tensor(
                self.mhc_h_res_offdiag_init_scale,
                device=h_res_logits.device,
                dtype=h_res_logits.dtype,
            )
        else:
            gamma = log_scale.to(device=h_res_logits.device, dtype=h_res_logits.dtype).exp()
        return eye + gamma * offdiag_mask * h_res_logits.tanh()

    def effective_static_h_res(self):
        h_res = self.static_h_res()
        if self.mhc_h_res_mode == "identity_tanh_offdiag":
            return self.identity_tanh_offdiag_h_res(h_res)
        return h_res

    def reset_h_res_alm_dual_update_accumulators(self):
        self._h_res_alm_update_count = 0
        self._h_res_alm_row_res_sum = torch.zeros_like(self.h_res_alm_row_dual)
        self._h_res_alm_col_res_sum = torch.zeros_like(self.h_res_alm_col_dual)
        self._h_res_alm_nonneg_res_sum = torch.zeros_like(self.h_res_alm_nonneg_dual)

    def _ensure_h_res_alm_accumulators_on_buffer_device(self):
        if (
            self._h_res_alm_row_res_sum.device != self.h_res_alm_row_dual.device
            or self._h_res_alm_col_res_sum.device != self.h_res_alm_col_dual.device
            or self._h_res_alm_nonneg_res_sum.device != self.h_res_alm_nonneg_dual.device
        ):
            self.reset_h_res_alm_dual_update_accumulators()

    def _mean_over_sample_dims(self, tensor, target_ndim):
        sample_ndim = tensor.ndim - target_ndim
        if sample_ndim <= 0:
            return tensor
        return tensor.mean(dim=tuple(range(sample_ndim)))

    def apply_h_res_alm_loss(self, alpha_residual):
        rows = alpha_residual.shape[-2]
        cols = alpha_residual.shape[-1]
        row_targets = torch.ones(
            *([1] * (alpha_residual.ndim - 2)),
            rows,
            1,
            device=alpha_residual.device,
            dtype=alpha_residual.dtype,
        )
        col_targets = torch.ones(
            *([1] * (alpha_residual.ndim - 2)),
            1,
            cols,
            device=alpha_residual.device,
            dtype=alpha_residual.dtype,
        )

        row_res = alpha_residual.sum(dim=-1, keepdim=True) - row_targets
        col_res = alpha_residual.sum(dim=-2, keepdim=True) - col_targets
        nonneg_res = (-alpha_residual).clamp_min(0.)
        dual_target_ndim = self.h_res_alm_row_dual.ndim
        row_res_mean = self._mean_over_sample_dims(row_res, dual_target_ndim)
        col_res_mean = self._mean_over_sample_dims(col_res, dual_target_ndim)
        nonneg_res_mean = self._mean_over_sample_dims(nonneg_res, dual_target_ndim)
        row_res_square_mean = self._mean_over_sample_dims(row_res.square(), dual_target_ndim)
        col_res_square_mean = self._mean_over_sample_dims(col_res.square(), dual_target_ndim)
        nonneg_res_square_mean = self._mean_over_sample_dims(nonneg_res.square(), dual_target_ndim)

        row_dual = self.h_res_alm_row_dual.to(device=alpha_residual.device, dtype=alpha_residual.dtype)
        col_dual = self.h_res_alm_col_dual.to(device=alpha_residual.device, dtype=alpha_residual.dtype)
        nonneg_dual = self.h_res_alm_nonneg_dual.to(device=alpha_residual.device, dtype=alpha_residual.dtype)
        rho = torch.as_tensor(self.mhc_admm_rho, device=alpha_residual.device, dtype=alpha_residual.dtype)

        dual_loss = (
            (row_dual * row_res_mean).sum()
            + (col_dual * col_res_mean).sum()
            + (nonneg_dual * nonneg_res_mean).sum()
        )
        penalty_loss = 0.5 * rho * (
            row_res_square_mean.sum()
            + col_res_square_mean.sum()
            + nonneg_res_square_mean.sum()
        )
        self._h_res_alm_loss = dual_loss + penalty_loss

        with torch.no_grad():
            self._ensure_h_res_alm_accumulators_on_buffer_device()
            self._h_res_alm_row_res_sum = self._h_res_alm_row_res_sum + row_res_mean.detach().to(
                device=self.h_res_alm_row_dual.device,
                dtype=self.h_res_alm_row_dual.dtype,
            )
            self._h_res_alm_col_res_sum = self._h_res_alm_col_res_sum + col_res_mean.detach().to(
                device=self.h_res_alm_col_dual.device,
                dtype=self.h_res_alm_col_dual.dtype,
            )
            self._h_res_alm_nonneg_res_sum = self._h_res_alm_nonneg_res_sum + nonneg_res_mean.detach().to(
                device=self.h_res_alm_nonneg_dual.device,
                dtype=self.h_res_alm_nonneg_dual.dtype,
            )
            self._h_res_alm_update_count += 1

        return alpha_residual

    @torch.no_grad()
    def update_h_res_alm_duals(self):
        count = self._h_res_alm_update_count
        if count <= 0:
            return None

        row_res = self._h_res_alm_row_res_sum / count
        col_res = self._h_res_alm_col_res_sum / count
        nonneg_res = self._h_res_alm_nonneg_res_sum / count
        rho = torch.as_tensor(self.mhc_admm_rho, device=self.h_res_alm_row_dual.device, dtype=self.h_res_alm_row_dual.dtype)
        self.h_res_alm_row_dual.add_(rho * row_res)
        self.h_res_alm_col_dual.add_(rho * col_res)
        self.h_res_alm_nonneg_dual.add_(rho * nonneg_res).clamp_min_(0.)

        stats = {
            "row_err_max": row_res.abs().amax().item(),
            "col_err_max": col_res.abs().amax().item(),
            "nonneg_violation_max": nonneg_res.abs().amax().item(),
            "row_err_mean": row_res.abs().mean().item(),
            "col_err_mean": col_res.abs().mean().item(),
            "nonneg_violation_mean": nonneg_res.abs().mean().item(),
            "row_dual_norm": torch.linalg.vector_norm(self.h_res_alm_row_dual.float()).item(),
            "col_dual_norm": torch.linalg.vector_norm(self.h_res_alm_col_dual.float()).item(),
            "nonneg_dual_norm": torch.linalg.vector_norm(self.h_res_alm_nonneg_dual.float()).item(),
            "count": count,
        }
        self.reset_h_res_alm_dual_update_accumulators()
        return stats

    def prepare_h_res_sprox_step(self):
        if self.static_alpha.grad is None:
            self._h_res_sprox_grad = torch.zeros_like(self.static_h_res())
        else:
            self._h_res_sprox_grad = self.static_alpha.grad[:, self.num_input_views:].detach().clone()
            self.static_alpha.grad[:, self.num_input_views:].zero_()
        self._h_res_sprox_x = self.static_h_res().detach().clone()
        self._h_res_sprox_lm_grad_norm = torch.linalg.vector_norm(self._h_res_sprox_grad.float()).item()

    @torch.no_grad()
    def step_h_res_sprox_alm(self):
        x = getattr(self, "_h_res_sprox_x", None)
        grad_f = getattr(self, "_h_res_sprox_grad", None)
        if x is None or grad_f is None:
            return None

        x = x.to(device=self.static_alpha.device, dtype=self.static_alpha.dtype)
        grad_f = grad_f.to(device=self.static_alpha.device, dtype=self.static_alpha.dtype)
        if self.h_res_sprox_z.device != self.static_alpha.device:
            self.h_res_sprox_z = self.h_res_sprox_z.to(self.static_alpha.device)

        rho = torch.as_tensor(self.mhc_admm_rho, device=x.device, dtype=x.dtype)
        dual_step = torch.as_tensor(self.mhc_admm_dual_step, device=x.device, dtype=x.dtype)
        prox_weight = self.mhc_admm_rho if self.mhc_admm_prox_weight is None else self.mhc_admm_prox_weight
        prox_weight = torch.as_tensor(prox_weight, device=x.device, dtype=x.dtype)
        primal_step = torch.as_tensor(self.mhc_admm_step_scale, device=x.device, dtype=x.dtype)
        smooth_beta = torch.as_tensor(self.mhc_admm_smooth_beta, device=x.device, dtype=x.dtype)

        if self.mhc_h_res_mode == "alm_nonnegative_cap":
            cap = torch.as_tensor(self.mhc_h_res_cap, device=x.device, dtype=x.dtype)
            row_res = x.sum(dim=-1, keepdim=True) - cap
            col_res = x.sum(dim=-2, keepdim=True) - cap
            row_penalty_res = row_res.clamp_min(0.)
            col_penalty_res = col_res.clamp_min(0.)
            self.h_res_alm_row_dual.add_(dual_step * row_res.to(self.h_res_alm_row_dual.dtype)).clamp_(min=0.)
            self.h_res_alm_col_dual.add_(dual_step * col_res.to(self.h_res_alm_col_dual.dtype)).clamp_(min=0.)
        else:
            row_res = x.sum(dim=-1, keepdim=True) - 1.
            col_res = x.sum(dim=-2, keepdim=True) - 1.
            row_penalty_res = row_res
            col_penalty_res = col_res
            self.h_res_alm_row_dual.add_(dual_step * row_res.to(self.h_res_alm_row_dual.dtype))
            self.h_res_alm_col_dual.add_(dual_step * col_res.to(self.h_res_alm_col_dual.dtype))

        row_dual = self.h_res_alm_row_dual.to(device=x.device, dtype=x.dtype)
        col_dual = self.h_res_alm_col_dual.to(device=x.device, dtype=x.dtype)
        z = self.h_res_sprox_z.to(device=x.device, dtype=x.dtype)

        grad_k = (
            grad_f
            + row_dual
            + col_dual
            + rho * (row_penalty_res + col_penalty_res)
            + prox_weight * (x - z)
        )
        x_next = x - primal_step * grad_k
        if self.mhc_h_res_mode in {"alm_nonnegative", "alm_nonnegative_cap"}:
            x_next = x_next.clamp_min(0.)
        elif self.mhc_h_res_mode == "alm_spectral_sprox":
            x_next = project_spectral_residual_sphere(x_next, radius=1.)
        self.h_res_sprox_z.lerp_(x_next.to(self.h_res_sprox_z.dtype), smooth_beta.to(self.h_res_sprox_z.dtype))
        self.static_h_res().copy_(x_next.to(self.static_alpha.dtype))

        if self.mhc_h_res_mode == "alm_nonnegative_cap":
            cap = torch.as_tensor(self.mhc_h_res_cap, device=x_next.device, dtype=x_next.dtype)
            next_row_res = (x_next.sum(dim=-1, keepdim=True) - cap).clamp_min(0.)
            next_col_res = (x_next.sum(dim=-2, keepdim=True) - cap).clamp_min(0.)
        else:
            next_row_res = x_next.sum(dim=-1, keepdim=True) - 1.
            next_col_res = x_next.sum(dim=-2, keepdim=True) - 1.
        if self.mhc_h_res_mode in {"alm_nonnegative", "alm_nonnegative_cap"}:
            nonneg_violation = (-x_next).clamp_min(0.)
        else:
            nonneg_violation = torch.zeros_like(x_next)
        if self.mhc_h_res_mode == "alm_spectral_sprox":
            n = x_next.shape[-1]
            j = torch.full_like(x_next, 1. / n)
            displacement = x_next - j
            spectral_norm = torch.linalg.matrix_norm(displacement.float(), ord=2).item()
            spectral_violation = max(0., spectral_norm - 1.)
        else:
            spectral_norm = 0.
            spectral_violation = 0.

        stats = {
            "row_err_max": next_row_res.abs().amax().item(),
            "col_err_max": next_col_res.abs().amax().item(),
            "nonneg_violation_max": nonneg_violation.amax().item(),
            "spectral_norm": spectral_norm,
            "spectral_violation_max": spectral_violation,
            "row_err_mean": next_row_res.abs().mean().item(),
            "col_err_mean": next_col_res.abs().mean().item(),
            "nonneg_violation_mean": nonneg_violation.mean().item(),
            "spectral_violation_mean": spectral_violation,
            "row_dual_norm": torch.linalg.vector_norm(self.h_res_alm_row_dual.float()).item(),
            "col_dual_norm": torch.linalg.vector_norm(self.h_res_alm_col_dual.float()).item(),
            "nonneg_dual_norm": 0.,
            "lm_grad_norm": self._h_res_sprox_lm_grad_norm or 0.,
            "count": 1,
        }
        self._h_res_sprox_x = None
        self._h_res_sprox_grad = None
        return stats

    def width_connection(
        self,
        residuals
    ):
        streams = self.num_residual_streams

        maybe_transformed_residuals = self.residual_transform(residuals)

        # width connection

        # handle channel first

        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')

        # split out fractions

        residuals = self.split_fracs(residuals)

        # split out streams

        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = streams)

        # norm
        normed = rearrange(residuals, 'b ... s d -> b ... (s d)', s = streams)
        # normed = F.normalize(normed, dim = -1)
        normed = self.norm(normed)
        if self.mhc_h_res_mode in {"adapter_epsilon", "adapter_cap", "adapter_cap_admm"}:
            base_streams = min(self.mhc_adapter_base_streams, streams)
            if base_streams < streams:
                normed = normed * ((base_streams / streams) ** 0.5)

        pre_branch_scale = repeat(self.pre_branch_scale, '1 -> v', v = self.num_input_views * self.num_fracs)
        residual_scale   = repeat(self.residual_scale  , '1 -> s', s = self.num_fracs * streams)

        # alpha for weighted sum of residuals going into branch. Some H_res
        # modes use static H_res only, so their dynamic network emits H_pre
        # logits only instead of generating then zeroing a dynamic H_res block.
        wc_weight = normed @ self.dynamic_alpha_fn
        if self.dynamic_h_res_disabled:
            wc_weight = rearrange(wc_weight, '... (s v) -> ... s v', s = streams)
            dynamic_pre = wc_weight * pre_branch_scale
            dynamic_residual = dynamic_pre.new_zeros(*dynamic_pre.shape[:-1], self.num_fracs * streams)
            dynamic_alpha = cat((dynamic_pre, dynamic_residual), dim=-1)
        else:
            wc_weight = rearrange(wc_weight, '... (s t) -> ... s t', s = streams)
            alpha_scale = cat((pre_branch_scale, residual_scale))
            dynamic_alpha = wc_weight * alpha_scale

        if self.mhc_h_res_mode in DYNAMIC_H_RES_DISABLED_MODES and not self.dynamic_h_res_disabled:
            dynamic_alpha = dynamic_alpha.clone()
            dynamic_alpha[..., self.num_input_views:] = 0.

        static_alpha = rearrange(self.static_alpha, '(f s) t -> f s t', s = streams)

        alpha = dynamic_alpha + static_alpha

        alpha = self.split_fracs(alpha) # (batch, seq, fracs1, streams, fracs2, input + residual streams)

        # the alpha is now split and "manifold constrained" with sinkhorn and sigmoid

        # (..., 1, s, 1, v) / (..., 1, s, 1, s)
        alpha_pre, alpha_residual = alpha[..., :self.num_input_views], alpha[..., self.num_input_views:]

        if self.mhc_gate_fn == "softmax":
            # Normalize across residual streams, matching H_post's stream-wise routing.
            alpha_pre = F.softmax(alpha_pre, dim=-3)
        else:
            alpha_pre = alpha_pre.sigmoid()

        if self.mhc_identity_h_res:
            streams = self.num_residual_streams
            target_shape = alpha_residual.shape
            alpha_residual = torch.eye(
                self.num_fracs * streams,
                device=alpha_residual.device,
                dtype=alpha_residual.dtype,
            )
            alpha_residual = rearrange(
                alpha_residual,
                '(f1 s1) (f2 s2) -> f1 s1 f2 s2',
                f1=self.num_fracs,
                s1=streams,
                f2=self.num_fracs,
                s2=streams,
            )
            alpha_residual = alpha_residual.expand(target_shape)
        else:
            if self.mhc_h_res_mode == "cayley":
                alpha_residual = rearrange(
                    alpha_residual,
                    '... f1 s1 f2 s2 -> ... (f1 s1) (f2 s2)',
                    f1=self.num_fracs,
                    s1=streams,
                    f2=self.num_fracs,
                    s2=streams,
                )
                alpha_residual = cayley_orthogonalize(alpha_residual)
                alpha_residual = rearrange(
                    alpha_residual,
                    '... (f1 s1) (f2 s2) -> ... f1 s1 f2 s2',
                    f1=self.num_fracs,
                    s1=streams,
                    f2=self.num_fracs,
                    s2=streams,
                )
            else:
                alpha_residual = rearrange(alpha_residual, '... f s g t -> ... f g s t')
                if self.mhc_h_res_mode == "sinkhorn":
                    alpha_residual = sinkhorn_knopps(alpha_residual, self.sinkhorn_iters)
                elif self.mhc_h_res_mode == "admm_reverse_kl":
                    def project_admm_reverse_kl(logits):
                        return admm_reverse_kl_with_capacity(
                            logits,
                            base_streams=streams,
                            cap=self.mhc_adapter_cap,
                            iters=self.mhc_admm_iters,
                            rho=self.mhc_admm_rho,
                            floor=self.mhc_adapter_admm_input_floor,
                        )

                    if (
                        self.mhc_adapter_admm_checkpoint
                        and torch.is_grad_enabled()
                        and alpha_residual.requires_grad
                    ):
                        alpha_residual = activation_checkpoint(
                            project_admm_reverse_kl,
                            alpha_residual,
                            use_reentrant=False,
                        )
                    else:
                        alpha_residual = project_admm_reverse_kl(alpha_residual)
                elif self.mhc_h_res_mode == "admm_reverse_kl_sprox_alm":
                    def project_admm_reverse_kl_sprox_alm(logits):
                        return admm_reverse_kl_sprox_alm_with_capacity(
                            logits,
                            base_streams=streams,
                            cap=self.mhc_adapter_cap,
                            iters=self.mhc_admm_iters,
                            rho=self.mhc_admm_rho,
                            floor=self.mhc_adapter_admm_input_floor,
                            dual_step=self.mhc_admm_dual_step,
                            prox_weight=self.mhc_admm_prox_weight,
                            smooth_beta=self.mhc_admm_smooth_beta,
                            step_scale=self.mhc_admm_step_scale,
                        )

                    if (
                        self.mhc_adapter_admm_checkpoint
                        and torch.is_grad_enabled()
                        and alpha_residual.requires_grad
                    ):
                        alpha_residual = activation_checkpoint(
                            project_admm_reverse_kl_sprox_alm,
                            alpha_residual,
                            use_reentrant=False,
                        )
                    else:
                        alpha_residual = project_admm_reverse_kl_sprox_alm(alpha_residual)
                elif self.mhc_h_res_mode == "admm_l2":
                    def project_admm_l2(logits):
                        return admm_l2_with_capacity(
                            logits,
                            base_streams=streams,
                            cap=self.mhc_adapter_cap,
                            iters=self.mhc_admm_iters,
                            rho=self.mhc_admm_rho,
                            floor=self.mhc_adapter_admm_input_floor,
                        )

                    if (
                        self.mhc_adapter_admm_checkpoint
                        and torch.is_grad_enabled()
                        and alpha_residual.requires_grad
                    ):
                        alpha_residual = activation_checkpoint(
                            project_admm_l2,
                            alpha_residual,
                            use_reentrant=False,
                        )
                    else:
                        alpha_residual = project_admm_l2(alpha_residual)
                elif self.mhc_h_res_mode == "alm_signed":
                    alpha_residual = self.apply_h_res_alm_loss(alpha_residual)
                elif self.mhc_h_res_mode == "identity_tanh_offdiag":
                    alpha_residual = rearrange(
                        alpha_residual,
                        '... f g s t -> ... (f s) (g t)',
                        f=self.num_fracs,
                        g=self.num_fracs,
                        s=streams,
                        t=streams,
                    )
                    alpha_residual = self.identity_tanh_offdiag_h_res(alpha_residual)
                    alpha_residual = rearrange(
                        alpha_residual,
                        '... (f s) (g t) -> ... f g s t',
                        f=self.num_fracs,
                        g=self.num_fracs,
                        s=streams,
                        t=streams,
                    )
                elif self.mhc_h_res_mode in {"unconstrained", "alm_nonnegative", "alm_nonnegative_cap", "alm_signed_sprox", "alm_spectral_sprox"}:
                    pass
                elif self.mhc_h_res_mode == "adapter_epsilon":
                    target = torch.full(
                        (streams,),
                        self.mhc_adapter_epsilon,
                        device=alpha_residual.device,
                        dtype=alpha_residual.dtype,
                    )
                    base_streams = min(self.mhc_adapter_base_streams, streams)
                    target[:base_streams] = 1.
                    alpha_residual = sinkhorn_with_marginals(
                        alpha_residual,
                        row_targets=target,
                        col_targets=target,
                        iters=self.sinkhorn_iters,
                    )
                elif self.mhc_h_res_mode == "adapter_cap":
                    alpha_residual = sinkhorn_with_capacity(
                        alpha_residual,
                        base_streams=min(self.mhc_adapter_base_streams, streams),
                        cap=self.mhc_adapter_cap,
                        iters=self.sinkhorn_iters,
                    )
                elif self.mhc_h_res_mode == "adapter_cap_admm":
                    base_streams = min(self.mhc_adapter_base_streams, streams)
                    if self.mhc_adapter_admm_input_mode == "sinkhorn_base_log":
                        base_logits = alpha_residual[..., :base_streams, :base_streams]
                        base_matrix = sinkhorn_knopps(base_logits, self.sinkhorn_iters)
                        base_logits_for_admm = base_matrix.clamp_min(
                            self.mhc_adapter_admm_input_floor
                        ).log()
                        alpha_residual = alpha_residual.clone()
                        alpha_residual[..., :base_streams, :base_streams] = base_logits_for_admm
                    def project_adapter_cap_admm(logits):
                        return admm_with_capacity(
                            logits,
                            base_streams=base_streams,
                            cap=self.mhc_adapter_cap,
                            iters=self.mhc_admm_iters,
                            rho=self.mhc_admm_rho,
                        )

                    if (
                        self.mhc_adapter_admm_checkpoint
                        and torch.is_grad_enabled()
                        and alpha_residual.requires_grad
                    ):
                        alpha_residual = activation_checkpoint(
                            project_adapter_cap_admm,
                            alpha_residual,
                            use_reentrant=False,
                        )
                    else:
                        alpha_residual = project_adapter_cap_admm(alpha_residual)
                else:
                    alpha_residual = admm_doubly_stochastic(
                        alpha_residual,
                        iters=self.mhc_admm_iters,
                        rho=self.mhc_admm_rho,
                    )
                self._record_h_res_constraint_errors(alpha_residual)
                alpha_residual = rearrange(alpha_residual, '... f g s t -> ... f s g t')

        alpha = cat((alpha_pre, alpha_residual), dim = -1) # (..., f, s, f, s+v)

        # beta for weights from branch output back to residual streams

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn # ... (s f)
            dc_weight = rearrange(dc_weight, '... (s f) -> ... s f', s = streams)

            dynamic_beta = dc_weight * self.h_post_scale

            static_beta = rearrange(self.static_beta, '... (s f) -> ... s f', s = streams)

            beta = dynamic_beta + static_beta
            if self.mhc_gate_fn == "softmax":
                beta = F.softmax(beta, dim=-2)
            else:
                beta = beta.sigmoid() * 2

        mix_h = einsum(alpha, residuals, '... f1 s f2 t, ... f1 s d -> ... f2 t d')

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :self.num_input_views, :], mix_h[..., self.num_input_views:, :]
            branch_input = rearrange(branch_input, 'b ... v d -> v b ... d')

        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')

        branch_input = self.merge_fracs(branch_input)

        residuals = rearrange(residuals, 'b ... s d -> (b s) ... d')
        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta = beta)

    def depth_connection(
        self,
        branch_output,
        residuals,
        *,
        beta
    ):
        assert self.add_branch_out_to_residual

        # maybe split fractions

        branch_output = self.split_fracs(branch_output)

        # 'depth' connection

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d ... -> b ... d')

        output = einsum(branch_output, beta, 'b ... f1 d, b ... f1 s f2 -> b ... f2 s d')

        output = rearrange(output, 'b ... s d -> (b s) ... d')

        # merge merge back fractions

        output = self.merge_fracs(output)

        # channel first

        if self.channel_first:
            output = rearrange(output, 'b ... d -> b d ...')

        residuals = self.depth_residual_fn(output, residuals)

        return self.dropout(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):

            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)

ManifoldConstrainedHyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
ManifoldConstrainedHyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)

# stream embed

class ExpandStreams(Module):
    def __init__(
        self,
        num_streams,
        mode = "repeat",
        active_streams = None,
    ):
        super().__init__()
        self.num_streams = num_streams
        self.mode = mode
        self.active_streams = num_streams if active_streams is None else min(active_streams, num_streams)

    def forward(self, residuals):
        residuals = repeat(residuals, 'b ... -> b s ...', s = self.num_streams)

        if self.mode == "split":
            residuals = residuals / self.num_streams
        elif self.mode == "repeat_base_zero_rest":
            mask = torch.arange(self.num_streams, device=residuals.device) < self.active_streams
            residuals = residuals * mask.view(1, self.num_streams, *([1] * (residuals.ndim - 2)))

        residuals = rearrange(residuals, 'b s ... -> (b s) ...')
        return residuals

class StreamEmbed(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False,
        expand_to_streams = False,
        expand_stream_mode = "repeat",
        expand_active_streams = None,
    ):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.expand_stream_mode = expand_stream_mode
        self.expand_active_streams = num_streams if expand_active_streams is None else min(expand_active_streams, num_streams)
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):

        if self.expand_to_streams:
            residuals = repeat(residuals, 'b ... -> b s ...', s = self.num_streams)
            if self.expand_stream_mode == "split":
                residuals = residuals / self.num_streams
            elif self.expand_stream_mode == "repeat_base_zero_rest":
                mask = torch.arange(self.num_streams, device=residuals.device) < self.expand_active_streams
                residuals = residuals * mask.view(1, self.num_streams, *([1] * (residuals.ndim - 2)))
            residuals = rearrange(residuals, 'b s ... -> (b s) ...')

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... s d -> (b s) d ...', s = self.num_streams)
        else:
            residuals = rearrange(residuals, 'b ... s d -> (b s) ... d', s = self.num_streams)

        return residuals

# attention pool - taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else

class AttentionPoolReduceStream(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False
    ):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first

        self.to_attn_logits = nn.Linear(dim, dim, bias = False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim = -2)

        residuals = reduce(residuals * attn, 'b ... s d -> b ... d', 'sum')

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')

        return residuals
