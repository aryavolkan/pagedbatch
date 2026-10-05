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
    decode, prefill = batch.groups  # decodes first, then prefill chunks
    # decode group = seq 1: positions 0..5 -> block 2 slots 8..11 then block 7 slots 28, 29;
    # its single query (position 5) sees all 6 keys; it owns flat row 3.
    assert decode.kv_slot_table.tolist() == [[8, 9, 10, 11, 28, 29]]
    assert decode.attn_mask[0, 0].tolist() == [[True] * 6]
    assert decode.valid_query.tolist() == [[True]] and decode.rows.tolist() == [3]
    # prefill group = seq 0: three queries, causal, flat rows 0..2, no padding needed.
    assert prefill.kv_slot_table.tolist() == [[20, 21, 22]]
    assert prefill.attn_mask[0, 0].tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    assert prefill.valid_query.tolist() == [[True, True, True]] and prefill.rows.tolist() == [0, 1, 2]


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
