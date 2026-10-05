import torch

from pagedbatch.config import SamplingParams
from pagedbatch.model import ForwardBatch


def test_forward_batch_indexing():
    # Two sequences: seq 0 prefills 3 tokens at positions 0..2 into block 5 (slots 20..22);
    # seq 1 decodes its 6th token (position 5) with blocks [2, 7] (slot 7*4+1 = 29).
    batch = ForwardBatch.build(
        token_chunks=[[10, 11, 12], [99]],
        start_positions=[0, 5],
        slots=[[20, 21, 22], [29]],
        block_tables=[[5], [2, 7]],
        block_size=4,
    )
    assert batch.num_tokens == 4 and batch.num_seqs == 2
    assert batch.query_start_loc.tolist() == [0, 3, 4]
    assert batch.context_lens.tolist() == [3, 6]
    assert batch.last_token_index.tolist() == [2, 3]
    # kv_slot_table: seq 1 positions 0..5 -> block 2 slots 8..11 then block 7 slots 28, 29
    assert batch.kv_slot_table[1].tolist() == [8, 9, 10, 11, 28, 29]
    assert batch.kv_slot_table[0, :3].tolist() == [20, 21, 22]
    mask = batch.attn_mask[:, 0]  # [B, Q_max, L_max]
    # seq 0: causal over its 3 queries
    assert mask[0, :3, :3].tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    # seq 1: its single query (position 5) sees all 6 keys
    assert mask[1, 0, :6].tolist() == [True] * 6
    assert batch.valid_query.tolist() == [[True, True, True], [True, False, False]]
    assert batch.pad_index[0].tolist() == [0, 1, 2] and batch.pad_index[1, 0].item() == 3


def test_token_by_token_prefill_matches_one_shot_prefill(engine_factory):
    """Paged attention consistency: feeding a prompt one token per step must give the
    same final logits as prefilling it in one chunk."""
    ids = [256] + list(range(5, 29))
    one = engine_factory(max_num_batched_tokens=64)
    chunked = engine_factory(max_num_batched_tokens=1)
    logits = []
    for eng in (one, chunked):
        eng.add_request(prompt_token_ids=ids, sampling=SamplingParams(max_tokens=1, temperature=0))
        while True:
            sched = eng.scheduler.schedule()
            batch = eng._build_batch(sched)
            with torch.inference_mode():
                out = eng.model(batch, eng.kv_cache)
            for s in sched.scheduled:
                s.seq.num_computed_tokens += s.num_new_tokens
            if sched.scheduled[0].samples_token:
                logits.append(out[0].clone())
                break
    torch.testing.assert_close(logits[0], logits[1], atol=1e-4, rtol=1e-4)
    assert chunked.metrics.steps_total == 0  # we drove the scheduler by hand
