"""One request's lifecycle: WAITING -> RUNNING -> FINISHED (with possible preemption)."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from .block_manager import BlockTable
from .config import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    request_id: str
    prompt_token_ids: list[int]
    sampling: SamplingParams
    arrival_time: float = field(default_factory=time.perf_counter)
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    num_computed_tokens: int = 0
    """Tokens whose K/V are in the cache. Reset to 0 when preempted (recompute)."""
    block_table: BlockTable | None = None
    finish_reason: str | None = None
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    num_preemptions: int = 0

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        """Tokens that still need a forward pass: a whole prompt, a prefill
        chunk, or exactly one token when decoding."""
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def finish(self, reason: str) -> None:
        self.status = SequenceStatus.FINISHED
        self.finish_reason = reason
        self.finish_time = time.perf_counter()
