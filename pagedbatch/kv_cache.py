"""The physical KV cache: one flat [num_slots, kv_heads, head_dim] tensor pair per layer.

Slot ``s`` of the pool belongs to block ``s // block_size``. The model writes a
token's K and V with ``index_copy_`` at its slot and reads a sequence's context
by gathering the slots in its block table, so no sequence ever needs contiguous
memory.
"""

from __future__ import annotations

import torch

from .config import ModelConfig


class KVCache:
    def __init__(self, model: ModelConfig, num_blocks: int, block_size: int, dtype: torch.dtype, device: str) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_slots = num_blocks * block_size
        shape = (self.num_slots, model.num_kv_heads, model.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(model.num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(model.num_layers)]

    @property
    def num_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def bytes_per_block(self) -> int:
        return self.num_bytes // self.num_blocks
