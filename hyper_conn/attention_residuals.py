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

    def _ensure_source_buffer(
        self,
        buffer: torch.Tensor | None,
        example: torch.Tensor,
        num_sources: int,
    ) -> torch.Tensor:
        expected_shape = (num_sources, *example.shape)
        if (
            buffer is None
            or buffer.shape != expected_shape
            or buffer.device != example.device
            or buffer.dtype != example.dtype
        ):
            return example.new_empty(expected_shape)
        return buffer

    def _aggregate_values(
        self,
        step_idx: int,
        values,
        keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(values):
            values = torch.stack(tuple(values), dim=0)

        if values.size(0) == 0:
            raise RuntimeError("Attention mixer received no value sources")

        if keys is None:
            keys = self._rms_norm(values)

        query = self.queries[step_idx]
        source_query = query.to(dtype=keys.dtype)
        logits = (keys * source_query).sum(dim=-1).movedim(0, -1)
        attn = logits.float().softmax(dim=-1)

        weights = attn.to(dtype=values.dtype).unsqueeze(-1)
        mixed = (values.movedim(0, -2) * weights).sum(dim=-2)

        with torch.no_grad():
            entropy = -(attn * attn.clamp_min(1e-9).log()).sum(dim=-1).mean()
            self.last_attn_stats = {
                "depth_sources": torch.tensor(float(values.size(0)), device=mixed.device),
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
        self._history_buffer = None
        self._history_count = 0

    def reset(self, initial_state: torch.Tensor):
        self._history_buffer = self._ensure_source_buffer(
            self._history_buffer,
            initial_state,
            self.num_steps + 1,
        )
        self._history_buffer[0].copy_(initial_state)
        self._history_count = 1

    def prepare_input(self, step_idx: int) -> torch.Tensor:
        if self._history_buffer is None or self._history_count == 0:
            raise RuntimeError("AttentionResidualMixer must be reset before use")
        return self._aggregate_values(
            step_idx,
            self._history_buffer[:self._history_count],
        )

    def record(self, state: torch.Tensor):
        if self._history_buffer is None or self._history_count == 0:
            raise RuntimeError("AttentionResidualMixer must be reset before use")
        self._history_buffer[self._history_count].copy_(state)
        self._history_count += 1

    def _clear_state(self):
        self._history_count = 0


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
        self._max_completed_blocks = (num_steps + block_size - 1) // block_size
        self._values_buffer = None
        self._keys_buffer = None
        self._has_state = False
        self._num_completed_blocks = 0
        self._current_block_count = 0
        self._current_block_active = False

    def _current_block_index(self) -> int:
        return 1 + self._num_completed_blocks

    def _num_sources(self) -> int:
        return 1 + self._num_completed_blocks + int(self._current_block_active)

    def reset(self, initial_state: torch.Tensor):
        num_sources = 1 + self._max_completed_blocks + 1
        self._values_buffer = self._ensure_source_buffer(
            self._values_buffer,
            initial_state,
            num_sources,
        )
        self._keys_buffer = self._ensure_source_buffer(
            self._keys_buffer,
            initial_state,
            num_sources,
        )
        self._values_buffer[0].copy_(initial_state)
        self._keys_buffer[0].copy_(self._rms_norm(initial_state))
        self._has_state = True
        self._num_completed_blocks = 0
        self._current_block_count = 0
        self._current_block_active = False

    def prepare_input(self, step_idx: int) -> torch.Tensor:
        if not self._has_state or self._values_buffer is None or self._keys_buffer is None:
            raise RuntimeError("BlockAttentionResidualMixer must be reset before use")
        num_sources = self._num_sources()
        return self._aggregate_values(
            step_idx,
            self._values_buffer[:num_sources],
            self._keys_buffer[:num_sources],
        )

    def record(self, state: torch.Tensor):
        if not self._has_state or self._values_buffer is None or self._keys_buffer is None:
            raise RuntimeError("BlockAttentionResidualMixer must be reset before use")

        current_idx = self._current_block_index()
        if not self._current_block_active:
            self._values_buffer[current_idx].copy_(state)
            self._current_block_active = True
        else:
            self._values_buffer[current_idx].add_(state)

        self._keys_buffer[current_idx].copy_(
            self._rms_norm(self._values_buffer[current_idx])
        )

        self._current_block_count += 1
        if self._current_block_count >= self.block_size:
            self._num_completed_blocks += 1
            self._current_block_count = 0
            self._current_block_active = False

    def _clear_state(self):
        self._has_state = False
        self._num_completed_blocks = 0
        self._current_block_count = 0
        self._current_block_active = False
