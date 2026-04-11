import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseAttentionResidualMixer(nn.Module):
    def __init__(self, num_steps: int, dim: int):
        super().__init__()
        self.num_steps = num_steps
        self.dim = dim
        self.queries = nn.Parameter(torch.empty(num_steps + 1, dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        self.last_attn_stats = {}

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * (self.dim ** 0.5)

    def _aggregate_values(self, step_idx: int, values) -> torch.Tensor:
        if torch.is_tensor(values):
            value_sources = tuple(values.unbind(dim=-2))
        else:
            value_sources = tuple(values)

        if len(value_sources) == 0:
            raise RuntimeError("Attention mixer received no value sources")

        query = self.queries[step_idx]
        logits = []

        for value in value_sources:
            key = self._rms_norm(value)
            source_query = query.to(dtype=key.dtype)
            logit = torch.einsum("d,btd->bt", source_query, key)
            logits.append(logit.float())

        attn = torch.stack(logits, dim=-1).softmax(dim=-1)

        mixed = torch.zeros_like(value_sources[0])
        for source_idx, value in enumerate(value_sources):
            weight = attn[..., source_idx].to(dtype=value.dtype).unsqueeze(-1)
            mixed = mixed + value * weight

        with torch.no_grad():
            entropy = -(attn * attn.clamp_min(1e-9).log()).sum(dim=-1).mean()
            self.last_attn_stats = {
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

    def reset(self, initial_state: torch.Tensor):
        self._history = [initial_state]

    def prepare_input(self, step_idx: int) -> torch.Tensor:
        if self._history is None or len(self._history) == 0:
            raise RuntimeError("AttentionResidualMixer must be reset before use")
        return self._aggregate_values(step_idx, self._history)

    def record(self, state: torch.Tensor):
        if self._history is None:
            raise RuntimeError("AttentionResidualMixer must be reset before use")
        self._history.append(state)

    def _clear_state(self):
        self._history = None


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
        self._completed_blocks = None
        self._current_block_sum = None
        self._current_block_count = 0

    def reset(self, initial_state: torch.Tensor):
        self._initial_state = initial_state
        self._completed_blocks = []
        self._current_block_sum = None
        self._current_block_count = 0

    def _value_sources(self):
        if self._initial_state is None:
            raise RuntimeError("BlockAttentionResidualMixer must be reset before use")
        values = [self._initial_state]
        values.extend(self._completed_blocks)
        if self._current_block_sum is not None:
            values.append(self._current_block_sum)
        return values

    def prepare_input(self, step_idx: int) -> torch.Tensor:
        return self._aggregate_values(step_idx, self._value_sources())

    def record(self, state: torch.Tensor):
        if self._initial_state is None:
            raise RuntimeError("BlockAttentionResidualMixer must be reset before use")

        if self._current_block_sum is None:
            self._current_block_sum = state
        else:
            self._current_block_sum = self._current_block_sum + state

        self._current_block_count += 1
        if self._current_block_count >= self.block_size:
            self._completed_blocks.append(self._current_block_sum)
            self._current_block_sum = None
            self._current_block_count = 0

    def _clear_state(self):
        self._initial_state = None
        self._completed_blocks = None
        self._current_block_sum = None
        self._current_block_count = 0
