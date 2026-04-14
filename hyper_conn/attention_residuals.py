import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockDepthMemory:
    """
    Shared block-level depth memory for unified Block-Depth mHC variants.

    This module only stores compressed depth history. It does not perform any
    mixing by itself, so the actual readout stays inside the hyper-connection
    operator.
    """

    def __init__(self, num_streams: int, block_size: int):
        if num_streams <= 0:
            raise ValueError("num_streams must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.num_streams = num_streams
        self.block_size = block_size
        self.clear()

    def _summarize(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[0] % self.num_streams != 0:
            raise RuntimeError("state batch dimension is not divisible by num_streams")
        reshaped = state.reshape(state.shape[0] // self.num_streams, self.num_streams, *state.shape[1:])
        return reshaped.mean(dim=1)

    def reset(self, initial_state: torch.Tensor):
        initial_summary = self._summarize(initial_state)
        self.initial_state = initial_summary
        self.completed_blocks = []
        self.current_block_sum = None
        self.current_block_count = 0

    def get_sources(self):
        if self.initial_state is None:
            raise RuntimeError("BlockDepthMemory must be reset before use")
        sources = [self.initial_state]
        sources.extend(self.completed_blocks)
        if self.current_block_sum is not None:
            sources.append(self.current_block_sum)
        return sources

    def record(self, state: torch.Tensor):
        summary = self._summarize(state)
        if self.current_block_sum is None:
            self.current_block_sum = summary
        else:
            self.current_block_sum = self.current_block_sum + summary
        self.current_block_count += 1

        if self.current_block_count >= self.block_size:
            self.completed_blocks.append(self.current_block_sum)
            self.current_block_sum = None
            self.current_block_count = 0

    def clear(self):
        self.initial_state = None
        self.completed_blocks = []
        self.current_block_sum = None
        self.current_block_count = 0


class _BaseAttentionResidualMixer(nn.Module):
    def __init__(self, num_steps: int, dim: int):
        super().__init__()
        self.num_steps = num_steps
        self.dim = dim
        self.queries = nn.Parameter(torch.empty(num_steps + 1, dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.last_stats = {}
        self.collect_stats = False

    def _source_scale(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        return norm.clamp_min(1e-12).reciprocal() * (self.dim ** 0.5)

    def _aggregate_values(
        self,
        step_idx: int,
        values,
        source_scales=None,
    ) -> torch.Tensor:
        if torch.is_tensor(values):
            value_sources = tuple(values.unbind(dim=-2))
        else:
            value_sources = tuple(values)

        if source_scales is None:
            scale_sources = (None,) * len(value_sources)
        elif torch.is_tensor(source_scales):
            scale_sources = tuple(source_scales.unbind(dim=-2))
        else:
            scale_sources = tuple(source_scales)

        if len(value_sources) == 0:
            raise RuntimeError("Attention mixer received no value sources")
        if len(scale_sources) != len(value_sources):
            raise RuntimeError("Attention mixer source scales do not match value sources")

        if len(value_sources) == 1:
            mixed = value_sources[0]
            if self.collect_stats:
                with torch.no_grad():
                    self.last_stats = {
                        "depth_sources": torch.tensor(1.0, device=mixed.device),
                        "attn_entropy": torch.tensor(0.0, device=mixed.device),
                        "attn_max": torch.tensor(1.0, device=mixed.device),
                    }
            return mixed

        query = self.queries[step_idx].to(dtype=value_sources[0].dtype)
        attn_logits = torch.empty(
            *value_sources[0].shape[:-1],
            len(value_sources),
            device=value_sources[0].device,
            dtype=torch.float32,
        )

        for source_idx, (value, source_scale) in enumerate(zip(value_sources, scale_sources)):
            if source_scale is None:
                source_scale = self._source_scale(value)
            logit = torch.matmul(value.reshape(-1, value.shape[-1]), query)
            logit = logit.view(*value.shape[:-1])
            logit = logit * source_scale.squeeze(-1).to(dtype=logit.dtype)
            attn_logits[..., source_idx] = logit.float()

        attn = attn_logits.softmax(dim=-1)

        mixed = torch.zeros_like(value_sources[0])
        for source_idx, value in enumerate(value_sources):
            weight = attn[..., source_idx].to(dtype=value.dtype).unsqueeze(-1)
            mixed = mixed + value * weight

        if self.collect_stats:
            with torch.no_grad():
                entropy = -(attn * attn.clamp_min(1e-9).log()).sum(dim=-1).mean()
                self.last_stats = {
                    "depth_sources": torch.tensor(float(len(value_sources)), device=mixed.device),
                    "attn_entropy": entropy.detach(),
                    "attn_max": attn.max(dim=-1).values.mean().detach(),
                }

        return mixed

    def prepare_input(self, step_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def record(self, state: torch.Tensor):
        raise NotImplementedError

    def apply_branch(self, step_idx: int, branch: nn.Module) -> torch.Tensor:
        branch_input = self.prepare_input(step_idx)
        branch_output = branch(branch_input)
        self.record(branch_output)
        return branch_output

    def finalize(self) -> torch.Tensor:
        output = self.prepare_input(self.num_steps)
        self._clear_state()
        return output

    def _clear_state(self):
        raise NotImplementedError


class AttentionResidualMixer(_BaseAttentionResidualMixer):
    """
    Full Attention Residuals over depth.

    Each sublayer has a learned pseudo-query. The current hidden state is formed
    by attending over the initial embedding plus all previous sublayer outputs.
    """

    def __init__(self, num_steps: int, dim: int):
        super().__init__(num_steps=num_steps, dim=dim)
        self._history = None
        self._history_scales = None

    def reset(self, initial_state: torch.Tensor):
        self._history = [initial_state]
        self._history_scales = [self._source_scale(initial_state)]

    def prepare_input(self, step_idx: int) -> torch.Tensor:
        if self._history is None or len(self._history) == 0:
            raise RuntimeError("AttentionResidualMixer must be reset before use")
        return self._aggregate_values(step_idx, self._history, self._history_scales)

    def record(self, state: torch.Tensor):
        if self._history is None:
            raise RuntimeError("AttentionResidualMixer must be reset before use")
        self._history.append(state)
        self._history_scales.append(self._source_scale(state))

    def _clear_state(self):
        self._history = None
        self._history_scales = None


class BlockAttentionResidualMixer(_BaseAttentionResidualMixer):
    """
    Block Attention Residuals over depth.

    The mixer attends over the initial state, completed block summaries, and the
    running partial sum of the current block.
    """

    def __init__(self, num_steps: int, dim: int, block_size: int):
        super().__init__(num_steps=num_steps, dim=dim)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._initial_state = None
        self._initial_state_scale = None
        self._completed_blocks = None
        self._completed_block_scales = None
        self._current_block_sum = None
        self._current_block_scale = None
        self._current_block_count = 0

    def reset(self, initial_state: torch.Tensor):
        self._initial_state = initial_state
        self._initial_state_scale = self._source_scale(initial_state)
        self._completed_blocks = []
        self._completed_block_scales = []
        self._current_block_sum = None
        self._current_block_scale = None
        self._current_block_count = 0

    def _value_sources(self):
        if self._initial_state is None:
            raise RuntimeError("BlockAttentionResidualMixer must be reset before use")
        values = [self._initial_state]
        values.extend(self._completed_blocks)
        if self._current_block_sum is not None:
            values.append(self._current_block_sum)
        scales = [self._initial_state_scale]
        scales.extend(self._completed_block_scales)
        if self._current_block_scale is not None:
            scales.append(self._current_block_scale)
        return values, scales

    def prepare_input(self, step_idx: int) -> torch.Tensor:
        value_sources, scale_sources = self._value_sources()
        return self._aggregate_values(step_idx, value_sources, scale_sources)

    def record(self, state: torch.Tensor):
        if self._initial_state is None:
            raise RuntimeError("BlockAttentionResidualMixer must be reset before use")

        if self._current_block_sum is None:
            self._current_block_sum = state
        else:
            self._current_block_sum = self._current_block_sum + state
        self._current_block_scale = self._source_scale(self._current_block_sum)

        self._current_block_count += 1
        if self._current_block_count >= self.block_size:
            self._completed_blocks.append(self._current_block_sum)
            self._completed_block_scales.append(self._current_block_scale)
            self._current_block_sum = None
            self._current_block_scale = None
            self._current_block_count = 0

    def _clear_state(self):
        self._initial_state = None
        self._initial_state_scale = None
        self._completed_blocks = None
        self._completed_block_scales = None
        self._current_block_sum = None
        self._current_block_scale = None
        self._current_block_count = 0
