import pytest

from pagedbatch.block_manager import BlockAllocator
from pagedbatch.config import EngineConfig, SamplingParams
from pagedbatch.scheduler import Scheduler
from pagedbatch.sequence import Sequence, SequenceStatus


def make(cfg: EngineConfig, num_blocks: int):
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=cfg.block_size)
    return Scheduler(cfg, alloc), alloc


def seq(i: int, prompt_len: int, max_tokens: int = 8) -> Sequence:
    return Sequence(request_id=f"r{i}", prompt_token_ids=list(range(prompt_len)), sampling=SamplingParams(max_tokens=max_tokens))


def commit(out):
    """What the engine does after a forward: mark the chunk computed and, if the
    sequence is up to date, append one sampled token."""
    for s in out.scheduled:
        s.seq.num_computed_tokens += s.num_new_tokens
        if s.samples_token:
            s.seq.output_token_ids.append(1)
    return out


def test_fifo_admission_within_budgets():
    cfg = EngineConfig(block_size=4, max_num_seqs=2, max_num_batched_tokens=10, watermark=0.0)
    sched, alloc = make(cfg, num_blocks=100)
    for i, n in enumerate([4, 4, 4]):
        sched.add(seq(i, n))
    out = sched.schedule()
    # two sequences fit the seq cap; the token budget (10) admits both prompts of 4.
    assert [s.seq.request_id for s in out.scheduled] == ["r0", "r1"]
    assert all(s.is_prefill and s.samples_token for s in out.scheduled)
    assert sched.num_running == 2 and sched.num_waiting == 1
    commit(out)
    out = sched.schedule()
    # decodes of r0 and r1 first (1 token each), then nothing new: seq cap reached.
    assert [(s.seq.request_id, s.num_new_tokens) for s in out.scheduled] == [("r0", 1), ("r1", 1)]
    assert out.num_decode_tokens == 2 and out.num_prefill_tokens == 0


def test_chunked_prefill_splits_long_prompt():
    cfg = EngineConfig(block_size=4, max_num_seqs=4, max_num_batched_tokens=6, watermark=0.0)
    sched, _ = make(cfg, num_blocks=100)
    sched.add(seq(0, 10))
    out = sched.schedule()
    assert [(s.num_new_tokens, s.samples_token) for s in out.scheduled] == [(6, False)]
    commit(out)
    out = sched.schedule()
    assert [(s.num_new_tokens, s.samples_token) for s in out.scheduled] == [(4, True)]
    commit(out)
    out = sched.schedule()
    assert [(s.num_new_tokens, s.is_prefill) for s in out.scheduled] == [(1, False)]


def test_chunked_prefill_can_be_disabled():
    cfg = EngineConfig(block_size=4, max_num_seqs=4, max_num_batched_tokens=6, watermark=0.0, enable_chunked_prefill=False)
    sched, _ = make(cfg, num_blocks=100)
    sched.add(seq(0, 10))
    sched.add(seq(1, 3))
    out = sched.schedule()
    assert out.is_empty  # r0 does not fit and FIFO order is preserved, so r1 waits too


def test_decode_allocates_block_only_at_boundary():
    cfg = EngineConfig(block_size=4, max_num_seqs=4, max_num_batched_tokens=64, watermark=0.0)
    sched, alloc = make(cfg, num_blocks=100)
    sched.add(seq(0, 3, max_tokens=10))
    commit(sched.schedule())
    assert alloc.num_used == 1
    commit(sched.schedule())  # token 4 fills block 0
    assert alloc.num_used == 1
    commit(sched.schedule())  # token 5 needs block 1
    assert alloc.num_used == 2


def test_preemption_recomputes_lowest_priority_sequence():
    # 4 blocks of 4 slots = 16 slots in total. r0 and r1 each start with 6 tokens
    # (2 blocks each), leaving nothing free: the first decode that crosses a block
    # boundary must preempt the most recently admitted sequence (r1).
    cfg = EngineConfig(block_size=4, max_num_seqs=4, max_num_batched_tokens=64, watermark=0.0)
    sched, alloc = make(cfg, num_blocks=4)
    sched.add(seq(0, 6, max_tokens=10))  # may grow to 16 tokens = all 4 blocks
    sched.add(seq(1, 6, max_tokens=10))
    commit(sched.schedule())
    assert alloc.num_free == 0
    commit(sched.schedule())  # 7th token of each sequence fits its partial second block
    commit(sched.schedule())  # 8th token fills both blocks exactly
    out = sched.schedule()  # r0 needs a third block: r1 is preempted
    assert [s.request_id for s in out.preempted] == ["r1"]
    r1 = out.preempted[0]
    assert r1.status is SequenceStatus.WAITING and r1.num_computed_tokens == 0 and r1.block_table is None
    assert r1.num_preemptions == 1 and sched.num_preemptions == 1
    assert len(r1.output_token_ids) == 3  # generated tokens are kept for the recompute
    assert [s.seq.request_id for s in out.scheduled] == ["r0"]
    commit(out)
    # r1 is at the front of the waiting queue; it cannot be readmitted until r0 frees blocks.
    assert sched.waiting[0] is r1
    out = sched.schedule()
    assert [s.seq.request_id for s in out.scheduled] == ["r0"]
    commit(out)
    sched.finish(sched.running[0], "stop")
    assert alloc.num_free == 4
    out = sched.schedule()
    # r1 is recomputed from scratch: prompt (6) + the 3 generated tokens in one prefill.
    assert [(s.seq.request_id, s.num_new_tokens, s.samples_token) for s in out.scheduled] == [("r1", 9, True)]


def test_tail_sequence_preempts_itself_when_it_cannot_grow():
    # 3 blocks of 4. r0 and r1 (4-token prompts, up to 8 more) each take one block;
    # r0's growth takes the last free block, so r1's growth finds nothing to steal
    # but itself: it is preempted, and readmitted only once r0 is done.
    cfg = EngineConfig(block_size=4, max_num_seqs=4, max_num_batched_tokens=64, watermark=0.0)
    sched, alloc = make(cfg, num_blocks=3)
    sched.add(seq(0, 4, max_tokens=8))
    sched.add(seq(1, 4, max_tokens=8))
    commit(sched.schedule())
    out = sched.schedule()  # r0 -> block 3; r1 cannot grow: preempts itself
    assert [s.seq.request_id for s in out.scheduled] == ["r0"]
    assert [s.request_id for s in out.preempted] == ["r1"]
    commit(out)
    out = sched.schedule()  # r1 is at the head of waiting but 1 block cannot hold its 5 tokens
    assert [s.seq.request_id for s in out.scheduled] == ["r0"] and sched.num_waiting == 1
    commit(out)
    sched.finish(sched.running[0], "stop")
    out = sched.schedule()
    assert [(s.seq.request_id, s.num_new_tokens) for s in out.scheduled] == [("r1", 5)]


def test_lone_sequence_that_could_never_finish_is_rejected():
    cfg = EngineConfig(block_size=4, max_num_seqs=4, max_num_batched_tokens=64, watermark=0.0)
    sched, _ = make(cfg, num_blocks=1)
    with pytest.raises(ValueError):
        sched.add(seq(0, 3, max_tokens=5))  # 8 tokens need 2 blocks


def test_watermark_holds_back_admission():
    cfg = EngineConfig(block_size=4, max_num_seqs=8, max_num_batched_tokens=64, watermark=0.5)
    sched, alloc = make(cfg, num_blocks=4)  # watermark = 2 blocks
    for i in range(3):
        sched.add(seq(i, 4, max_tokens=4))  # 8 tokens = 2 blocks: fits alone with the watermark kept
    out = sched.schedule()
    assert [s.seq.request_id for s in out.scheduled] == ["r0", "r1"]
    assert alloc.num_free == 2 and sched.num_waiting == 1


def test_rejects_prompts_that_can_never_fit():
    cfg = EngineConfig(block_size=4, max_model_len=8, watermark=0.0)
    sched, _ = make(cfg, num_blocks=2)
    with pytest.raises(ValueError):
        sched.add(seq(0, 9))  # exceeds max_model_len
    cfg2 = EngineConfig(block_size=4, max_model_len=64, watermark=0.0)
    sched2, _ = make(cfg2, num_blocks=2)
    with pytest.raises(ValueError):
        sched2.add(seq(1, 12))  # 12 + 8 tokens need 5 blocks, cache has 2


def test_abort_from_waiting_and_running():
    cfg = EngineConfig(block_size=4, max_num_seqs=1, watermark=0.0)
    sched, alloc = make(cfg, num_blocks=10)
    sched.add(seq(0, 4))
    sched.add(seq(1, 4))
    commit(sched.schedule())
    assert sched.abort("r1") and sched.num_waiting == 0
    assert sched.abort("r0") and sched.num_running == 0 and alloc.num_free == 10
    assert not sched.abort("nope")
