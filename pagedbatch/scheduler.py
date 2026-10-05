"""Continuous batching: decide, every step, which tokens of which sequences run.

One ``schedule()`` call produces the work for one model forward pass:

1. Running sequences come first. Each decoding sequence needs one slot for its
   next token; a sequence still in (chunked) prefill needs up to
   ``max_num_batched_tokens`` more. If a sequence cannot get a block, the
   lowest-priority running sequence (the most recently admitted) is preempted:
   its blocks are freed and it goes back to the front of the waiting queue to be
   recomputed from scratch, generated tokens included.
2. Waiting sequences are admitted FIFO while the sequence cap, the token budget
   and the block watermark allow. A prompt that does not fit the remaining token
   budget is prefilled in chunks across steps.

Finished sequences release their blocks immediately, so the next step can admit
new work: that is the whole point of continuous batching over static batching.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockAllocator, BlockTable
from .config import EngineConfig
from .sequence import Sequence, SequenceStatus


@dataclass
class ScheduledSequence:
    seq: Sequence
    num_new_tokens: int
    slots: list[int]
    """Physical cache slots for the new tokens, in order."""
    is_prefill: bool
    """More than one token, or a chunk that does not yet reach the end of the prompt."""
    samples_token: bool
    """True when this chunk brings the sequence up to date, so the next token is
    sampled from its last logits. Snapshotted at schedule time: the engine
    mutates ``num_computed_tokens`` after the forward pass."""

    @classmethod
    def plan(cls, seq: Sequence, num_new_tokens: int, slots: list[int]) -> ScheduledSequence:
        up_to_date = seq.num_computed_tokens + num_new_tokens == seq.num_tokens
        # A decode step processes exactly the token sampled last step. Anything else
        # (a prompt, a prompt chunk even of length 1, a recompute) is prefill work.
        is_decode = num_new_tokens == 1 and bool(seq.output_token_ids) and seq.num_computed_tokens == seq.num_tokens - 1
        return cls(seq, num_new_tokens, slots, is_prefill=not is_decode, samples_token=up_to_date)


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(s.num_new_tokens for s in self.scheduled)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(s.num_new_tokens for s in self.scheduled if s.is_prefill)

    @property
    def num_decode_tokens(self) -> int:
        return sum(s.num_new_tokens for s in self.scheduled if not s.is_prefill)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, config: EngineConfig, allocator: BlockAllocator) -> None:
        self.config = config
        self.allocator = allocator
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.watermark_blocks = math.ceil(allocator.num_blocks * config.watermark)
        self.num_preemptions = 0

    # -- queue management -------------------------------------------------

    def add(self, seq: Sequence) -> None:
        max_len = seq.num_tokens + seq.sampling.max_tokens
        if max_len > self.config.max_model_len:
            raise ValueError(f"prompt ({seq.num_tokens}) + max_tokens ({seq.sampling.max_tokens}) exceeds max_model_len={self.config.max_model_len}")
        # A sequence must be able to reach its maximum length with the whole cache
        # to itself; otherwise it would preempt itself forever. Because victims are
        # always taken from the tail of the running list, this also guarantees the
        # head of the queue makes progress, so every admitted request terminates.
        blocks_at_max = math.ceil(max_len / self.allocator.block_size)
        if blocks_at_max > self.allocator.num_blocks - self.watermark_blocks:
            raise ValueError(
                f"request may grow to {max_len} tokens = {blocks_at_max} blocks, but the cache has "
                f"{self.allocator.num_blocks} blocks (watermark {self.watermark_blocks}); raise --kv-cache-mb or lower max_tokens"
            )
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def abort(self, request_id: str) -> bool:
        for seq in list(self.waiting):
            if seq.request_id == request_id:
                self.waiting.remove(seq)
                seq.finish("aborted")
                return True
        for seq in self.running:
            if seq.request_id == request_id:
                self.finish(seq, "aborted")
                return True
        return False

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def finish(self, seq: Sequence, reason: str) -> None:
        """Mark a running sequence finished and release its blocks."""
        if seq.block_table is not None:
            seq.block_table.free()
            seq.block_table = None
        if seq in self.running:
            self.running.remove(seq)
        seq.finish(reason)

    # -- the step ---------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        token_budget = self.config.max_num_batched_tokens
        now = time.perf_counter()

        # 1. Running sequences: decodes and in-flight prefill chunks.
        for seq in list(self.running):
            if seq.status is not SequenceStatus.RUNNING:
                continue  # preempted earlier in this loop
            if token_budget <= 0:
                break
            n = min(seq.num_uncomputed_tokens, token_budget)
            if n <= 0:
                continue
            assert seq.block_table is not None
            while not seq.block_table.can_append(n):
                victim = self.running[-1]
                self._preempt(victim, out)
                if victim is seq:
                    break
            if seq.status is not SequenceStatus.RUNNING:
                continue
            slots = seq.block_table.append(n)
            out.scheduled.append(ScheduledSequence.plan(seq, n, slots))
            token_budget -= n

        # 2. Admit waiting sequences FIFO.
        while self.waiting and token_budget > 0 and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting[0]
            n = min(seq.num_uncomputed_tokens, token_budget)
            if n < seq.num_uncomputed_tokens and not self.config.enable_chunked_prefill:
                break
            table = seq.block_table or BlockTable(self.allocator)
            need = table.blocks_needed(n)
            if self.allocator.num_free - need < self.watermark_blocks:
                break
            self.waiting.popleft()
            seq.block_table = table
            seq.status = SequenceStatus.RUNNING
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = now
            self.running.append(seq)
            slots = table.append(n)
            out.scheduled.append(ScheduledSequence.plan(seq, n, slots))
            token_budget -= n

        return out

    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        """Recompute-style preemption: drop the sequence's cache and requeue it."""
        assert seq.block_table is not None
        seq.block_table.free()
        seq.block_table = None
        seq.num_computed_tokens = 0
        seq.status = SequenceStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.running.remove(seq)
        self.waiting.appendleft(seq)
        out.preempted.append(seq)
