from __future__ import annotations

from functools import partial
from random import randrange

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from .hyper_connections import Residual, add, get_expand_reduce_stream_functions


def _exists(value):
    return value is not None


def _default(value, fallback):
    return value if _exists(value) else fallback


def _factorize_into_twos(num_streams: int) -> list[int]:
    """Return the 2-factorization used by the current KromHC implementation."""
    if num_streams < 1 or num_streams & (num_streams - 1):
        raise ValueError(
            "KromHC currently requires hyper_conn_n to be a positive power of 2; "
            f"got {num_streams}"
        )
    factors = []
    while num_streams > 1:
        factors.append(2)
        num_streams //= 2
    return factors


def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
    reduce_stream_mode="sum",
    expand_stream_mode="repeat",
    **kwargs,
):
    """Return the KromHC initializer and shared stream expand/reduce modules."""
    disable = _default(disable, num_streams == 1 and num_fracs == 1)
    hyper_conn_class = KromHC if not disable else Residual
    init_hyper_conn_fn = partial(
        hyper_conn_class,
        num_streams,
        num_fracs=num_fracs,
        **kwargs,
    )
    expand_reduce = get_expand_reduce_stream_functions(
        num_streams,
        add_stream_embed=add_stream_embed,
        dim=dim,
        disable=disable,
        reduce_stream_mode=reduce_stream_mode,
        expand_stream_mode=expand_stream_mode,
    )
    if _exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)
    return (init_hyper_conn_fn, *expand_reduce)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


class KromHC(Module):
    """KromHC with a Kronecker product of 2x2 doubly stochastic factors.

    For n=2^K streams, each factor is a convex combination of the identity
    and swap permutation matrices. Their Kronecker product is therefore an
    exactly doubly stochastic n-by-n residual matrix.
    """

    dynamic_alpha_layout = "kromhc"

    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index=None,
        channel_first=False,
        dropout=0.0,
        residual_transform: Module | None = None,
        add_branch_out_to_residual=True,
        num_input_views=1,
        depth_residual_fn=add,
        num_fracs=1,
    ):
        super().__init__()

        if num_fracs != 1:
            raise ValueError("KromHC currently supports num_fracs=1 only")
        if num_input_views != 1:
            raise ValueError("KromHC currently supports num_input_views=1 only")
        if num_residual_streams < 2:
            raise ValueError("KromHC requires at least two residual streams")
        if dim % num_fracs != 0:
            raise ValueError(f"feature dimension ({dim}) must be divisible by num_fracs ({num_fracs})")

        self.branch = branch
        self.num_fracs = num_fracs
        self.has_fracs = False
        self.split_fracs = Rearrange("b ... (f d) -> b ... f d", f=num_fracs)
        self.merge_fracs = Rearrange("b ... f d -> b ... (f d)")
        dim //= num_fracs

        self.num_residual_streams = num_residual_streams
        self.num_input_views = num_input_views
        self.channel_first = channel_first
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim * num_residual_streams)
        self.residual_transform = _default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn

        self.factors = _factorize_into_twos(num_residual_streams)
        self.num_factors = len(self.factors)
        self.factor_perms = [2] * self.num_factors
        self.total_res_coeffs = sum(self.factor_perms)

        init_residual_index = _default(layer_index, randrange(num_residual_streams))
        init_residual_index %= num_residual_streams

        # The official KromHC implementation uses a flat parameter containing
        # [H_pre static logits, one pair of H_res logits per factor].
        init_pre = torch.full((num_residual_streams,), -1.0)
        init_pre[init_residual_index] = 1.0
        init_residual = torch.full((self.total_res_coeffs,), -8.0)
        for factor_idx in range(self.num_factors):
            init_residual[2 * factor_idx] = 0.0
        self.static_alpha = nn.Parameter(torch.cat((init_pre, init_residual)))

        # The dynamic output is [n H_pre coefficients, factor coefficients].
        self.dynamic_alpha_pre_width = num_residual_streams
        self.dynamic_alpha_res_width = self.total_res_coeffs
        self.dynamic_alpha_fn = nn.Parameter(
            torch.zeros(
                dim * num_residual_streams,
                self.dynamic_alpha_pre_width + self.dynamic_alpha_res_width,
            )
        )
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        self.add_branch_out_to_residual = add_branch_out_to_residual
        if add_branch_out_to_residual:
            beta_init = torch.full((num_residual_streams,), -1.0)
            beta_init[init_residual_index] = 1.0
            self.static_beta = nn.Parameter(beta_init)
            self.dynamic_beta_fn = nn.Parameter(
                torch.zeros(dim * num_residual_streams, num_residual_streams)
            )
            self.h_post_scale = nn.Parameter(torch.ones(1) * 1e-2)

    def _build_kronecker_h_res(self, dynamic_coeffs, static_coeffs):
        coefficients = self.residual_scale * dynamic_coeffs + static_coeffs
        batch_shape = coefficients.shape[:-1]
        coefficients = coefficients.reshape(*batch_shape, self.num_factors, 2)
        weights = F.softmax(coefficients, dim=-1)
        p = weights[..., 0]
        one_minus_p = 1.0 - p
        row0 = torch.stack((p, one_minus_p), dim=-1)
        row1 = torch.stack((one_minus_p, p), dim=-1)
        factors = torch.stack((row0, row1), dim=-2)

        result = factors[..., 0, :, :]
        for factor_idx in range(1, self.num_factors):
            factor = factors[..., factor_idx, :, :]
            result = torch.einsum(
                "...ab,...cd->...acbd",
                result,
                factor,
            ).reshape(*batch_shape, result.shape[-2] * 2, result.shape[-1] * 2)
        return result

    def width_connection(self, residuals):
        streams = self.num_residual_streams
        residuals = self.residual_transform(residuals)
        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")

        residuals = self.split_fracs(residuals)
        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)
        normed = rearrange(residuals, "b ... s d -> b ... (s d)", s=streams)
        normed = self.norm(normed)

        if self.add_branch_out_to_residual:
            fused_weights = torch.cat((self.dynamic_alpha_fn, self.dynamic_beta_fn), dim=-1)
        else:
            fused_weights = self.dynamic_alpha_fn
        combined_weight = normed @ fused_weights

        alpha_size = self.dynamic_alpha_fn.shape[-1]
        alpha_weight = combined_weight[..., :alpha_size]
        dynamic_pre = alpha_weight[..., :streams]
        dynamic_residual = alpha_weight[..., streams:]
        static_pre = self.static_alpha[:streams]
        static_residual = self.static_alpha[streams:]

        alpha_residual = self._build_kronecker_h_res(dynamic_residual, static_residual)
        alpha_residual = self.split_fracs(alpha_residual)

        alpha_pre = self.pre_branch_scale * dynamic_pre + static_pre
        alpha_pre = rearrange(alpha_pre, "... (f s v) -> ... s f v", v=1, f=1)
        alpha_pre = alpha_pre.sigmoid()
        alpha = torch.cat((alpha_pre, alpha_residual), dim=-1)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = combined_weight[..., alpha_size:]
            dc_weight = rearrange(dc_weight, "... (s f) -> ... s f", s=streams, f=1)
            dynamic_beta = dc_weight * self.h_post_scale
            static_beta = rearrange(self.static_beta, "(s f) -> s f", s=streams, f=1)
            beta = (dynamic_beta + static_beta).sigmoid() * 2

        mixed = einsum(
            alpha,
            residuals,
            "... f1 s f2 t, ... f1 s d -> ... f2 t d",
        )
        branch_input = mixed[..., 0, :]
        residuals = mixed[..., 1:, :]

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")
        branch_input = self.merge_fracs(branch_input)

        residuals = rearrange(residuals, "b ... s d -> (b s) ... d")
        if self.channel_first:
            residuals = rearrange(residuals, "b ... d -> b d ...")
        residuals = self.merge_fracs(residuals)
        return branch_input, residuals, dict(beta=beta)

    def depth_connection(self, branch_output, residuals, *, beta):
        if not self.add_branch_out_to_residual:
            return branch_output
        branch_output = self.split_fracs(branch_output)
        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")
        output = einsum(
            branch_output,
            beta,
            "b ... f1 d, b ... f1 s f2 -> b ... f2 s d",
        )
        output = rearrange(output, "b ... s d -> (b s) ... d")
        output = self.merge_fracs(output)
        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")
        return self.dropout(self.depth_residual_fn(output, residuals))

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)
            return tree_unflatten((branch_out, *rest), tree_spec)

        if not _exists(self.branch):
            return branch_input, add_residual_fn
        return add_residual_fn(self.branch(branch_input, *branch_args, **branch_kwargs))

    def effective_h_res(self):
        """Return the initialization H_res matrix for diagnostics."""
        zeros = self.static_alpha.new_zeros(self.total_res_coeffs)
        return self._build_kronecker_h_res(
            zeros,
            self.static_alpha[self.num_residual_streams:],
        )


KromHC.get_init_and_expand_reduce_stream_functions = staticmethod(
    get_init_and_expand_reduce_stream_functions
)
