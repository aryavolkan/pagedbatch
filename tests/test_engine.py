import torch

from pagedbatch.config import SamplingParams
from pagedbatch.sampling import sample_token

PROMPTS = ["hello world", "quick brown fox", "x", "paged attention"]
GREEDY = SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True)


def ids_of(outputs):
    return [o.output_token_ids for o in outputs]


def test_preemption_does_not_change_greedy_outputs(engine_factory):
    roomy = engine_factory(num_blocks=1000)
    # 7 blocks of 4 = 28 slots: each prompt (<= 16 tokens + 10 outputs) fits alone, the four together cannot.
    cramped = engine_factory(num_blocks=7, max_num_seqs=4)
    assert ids_of(cramped.generate(PROMPTS, GREEDY)) == ids_of(roomy.generate(PROMPTS, GREEDY))
    assert cramped.metrics.preemptions_total > 0
    assert cramped.allocator.num_used == 0  # everything released at the end


def test_chunked_prefill_does_not_change_greedy_outputs(engine_factory):
    a = engine_factory(max_num_batched_tokens=512)
    b = engine_factory(max_num_batched_tokens=3)
    assert ids_of(a.generate(PROMPTS, GREEDY)) == ids_of(b.generate(PROMPTS, GREEDY))


def test_batched_equals_sequential(engine_factory):
    batched = engine_factory(max_num_seqs=8)
    one_at_a_time = engine_factory(max_num_seqs=1)
    assert ids_of(batched.generate(PROMPTS, GREEDY)) == ids_of(one_at_a_time.generate(PROMPTS, GREEDY))


def test_streaming_deltas_concatenate_to_final_text(engine_factory):
    eng = engine_factory()
    rid = eng.add_request("stream me", GREEDY)
    text, final = "", None
    while eng.has_unfinished_requests:
        for o in eng.step():
            text += o.text_delta
            if o.finished:
                final = o
    assert final is not None and final.request_id == rid
    assert text == eng.tokenizer.decode(final.output_token_ids)
    assert final.finish_reason == "length" and len(final.output_token_ids) == 10
    assert final.ttft is not None and final.e2e_latency is not None


def test_stop_token_and_eos_finish_with_stop(engine_factory):
    eng = engine_factory()
    first = eng.generate(["abc"], GREEDY)[0].output_token_ids[0]
    out = eng.generate(["abc"], SamplingParams(max_tokens=10, temperature=0.0, stop_token_ids=(first,)))[0]
    assert out.finish_reason == "stop" and out.output_token_ids == [first]


def test_seeded_sampling_is_reproducible(engine_factory):
    p = SamplingParams(max_tokens=8, temperature=1.0, top_p=0.9, seed=7, ignore_eos=True)
    a = ids_of(engine_factory().generate(PROMPTS[:2], p))
    b = ids_of(engine_factory().generate(PROMPTS[:2], p))
    assert a == b


def test_request_validation(engine_factory):
    eng = engine_factory(max_model_len=16)
    try:
        eng.add_request("x" * 20, SamplingParams(max_tokens=1))
    except ValueError as e:
        assert "max_model_len" in str(e)
    else:
        raise AssertionError("expected ValueError")
    rid = eng.add_request("ok", SamplingParams(max_tokens=2))
    assert eng.abort_request(rid) and not eng.has_unfinished_requests
    assert not eng.abort_request("missing")


def test_metrics_render_counts(engine_factory):
    eng = engine_factory()
    eng.generate(PROMPTS[:2], GREEDY)
    text = eng.metrics.render()
    assert "pagedbatch_generated_tokens_total 20" in text
    assert "pagedbatch_requests_finished_total 2" in text
    assert "pagedbatch_step_seconds_count" in text


def test_sampling_primitives():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([0.1, 3.0, 0.2, 2.9])
    assert sample_token(logits, SamplingParams(temperature=0.0), g) == 1
    assert sample_token(logits, SamplingParams(temperature=1.0, top_k=1), g) == 1
    assert sample_token(logits, SamplingParams(temperature=1.0, top_p=1e-6), g) == 1
    picks = {sample_token(logits, SamplingParams(temperature=1.0, top_k=2), g) for _ in range(50)}
    assert picks <= {1, 3}
