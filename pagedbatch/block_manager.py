"""Physical KV-cache blocks: a free list and per-sequence block tables.

The KV cache is one big pool of ``num_blocks`` fixed-size blocks. A sequence
owns a *block table*: the ordered list of physical blocks holding its tokens.
Logical position ``p`` of a sequence lives in ``blocks[p // block_size]`` at
offset ``p % block_size``; the flat *slot* index used by the model to read and
write the cache is ``block * block_size + offset``.

This is the memory-management idea of PagedAttention: sequences grow one block
at a time instead of reserving ``max_model_len`` up front, so internal
fragmentation is bounded by one block per sequence and any sequence can use any
free block.
"""

from __future__ import annotations

from collections import deque


class NoFreeBlocks(RuntimeError):
    """Raised when an allocation cannot be satisfied; the scheduler preempts."""


class BlockAllocator:
    """A free list over ``num_blocks`` physical blocks."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))
        self._allocated: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def can_allocate(self, n: int) -> bool:
        return n <= len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise NoFreeBlocks(f"requested {n} blocks, {len(self._free)} free")
        blocks = [self._free.popleft() for _ in range(n)]
        self._allocated.update(blocks)
        return blocks

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if b not in self._allocated:
                raise ValueError(f"block {b} is not allocated")
            self._allocated.remove(b)
            self._free.append(b)


class BlockTable:
    """The physical blocks of one sequence, in logical order."""

    def __init__(self, allocator: BlockAllocator) -> None:
        self._allocator = allocator
        self.block_size = allocator.block_size
        self.blocks: list[int] = []
        self.num_tokens = 0
        """Logical tokens that have been assigned a slot."""

    def blocks_needed(self, num_new_tokens: int) -> int:
        """Additional blocks required to hold ``num_new_tokens`` more tokens."""
        total = self.num_tokens + num_new_tokens
        return max(0, -(-total // self.block_size) - len(self.blocks))

    def can_append(self, num_new_tokens: int) -> bool:
        return self._allocator.can_allocate(self.blocks_needed(num_new_tokens))

    def append(self, num_new_tokens: int) -> list[int]:
        """Reserve slots for the next ``num_new_tokens`` tokens and return them."""
        need = self.blocks_needed(num_new_tokens)
        if need:
            self.blocks.extend(self._allocator.allocate(need))
        start = self.num_tokens
        self.num_tokens += num_new_tokens
        return self.slots(start, self.num_tokens)

    def slot(self, position: int) -> int:
        if position >= self.num_tokens:
            raise IndexError(f"position {position} beyond {self.num_tokens} assigned tokens")
        return self.blocks[position // self.block_size] * self.block_size + position % self.block_size

    def slots(self, start: int, end: int) -> list[int]:
        return [self.slot(p) for p in range(start, end)]

    def free(self) -> None:
        self._allocator.free(self.blocks)
        self.blocks = []
        self.num_tokens = 0

    @property
    def num_slots_reserved(self) -> int:
        return len(self.blocks) * self.block_size

    @property
    def internal_fragmentation(self) -> int:
        """Reserved slots not holding a token: at most ``block_size - 1``."""
        return self.num_slots_reserved - self.num_tokens
