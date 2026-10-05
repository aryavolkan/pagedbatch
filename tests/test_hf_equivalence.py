"""The checkpoint loader and the paged model must reproduce Hugging Face transformers
exactly: same random Llama weights, same greedy tokens, under small blocks, chunked
prefill and batching."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig  # noqa: E402
from transformers import LlamaForCausalLM as HFLlama  # noqa: E402

from pagedbatch.config import EngineConfig, SamplingParams  # noqa: E402
from pagedbatch.engine import LLMEngine  # noqa: E402
from pagedbatch.loader import load_model  # noqa: E402


@pytest.fixture(scope="module")
def hf_checkpoint(tmp_path_factory):
    cfg = LlamaConfig(
        vocab_size=258, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=True,
        bos_token_id=256, eos_token_id=257, pad_token_id=257,
    )
    torch.manual_seed(1234)
    hf = HFLlama(cfg).eval()
    hf.generation_config.eos_token_id = None  # compare fixed-length greedy continuations
    d = tmp_path_factory.mktemp("ckpt")
    hf.save_pretrained(d, safe_serialization=True)
    return hf, d


def test_loader_reads_hf_safetensors_and_ties_embeddings(hf_checkpoint):
    hf, d = hf_checkpoint
    model, cfg, tok = load_model(str(d))
    assert cfg.num_layers == 2 and cfg.num_kv_heads == 2 and cfg.tie_word_embeddings
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    torch.testing.assert_close(model.model.layers[1].mlp.down_proj.weight, hf.model.layers[1].mlp.down_proj.weight)


def test_prefill_logits_match_hf(hf_checkpoint):
    hf, d = hf_checkpoint
    model, cfg, tok = load_model(str(d))
    engine = LLMEngine(model, cfg, tok, EngineConfig(block_size=4, num_blocks=64, max_num_batched_tokens=64, watermark=0.0))
    ids = [256] + list(range(10, 31))
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0, -1]
    engine.add_request(prompt_token_ids=ids, sampling=SamplingParams(max_tokens=1, temperature=0))
    sched = engine.scheduler.schedule()
    with torch.inference_mode():
        logits = engine.model(engine._build_batch(sched), engine.kv_cache)[0]
    torch.testing.assert_close(logits, ref, atol=1e-4, rtol=1e-4)


def test_greedy_generation_matches_hf_under_paging_chunking_and_batching(hf_checkpoint):
    hf, d = hf_checkpoint
    model, cfg, tok = load_model(str(d))
    prompts = [[256] + list(range(10, 25)), [256] + list(range(40, 43)), [256] + list(range(100, 131))]
    refs = []
    for p in prompts:
        with torch.no_grad():
            out = hf.generate(torch.tensor([p]), max_new_tokens=12, do_sample=False, pad_token_id=257)
        refs.append(out[0, len(p):].tolist())
    # block_size 2 and a 7-token step budget force many blocks per sequence and chunked prefill.
    engine = LLMEngine(model, cfg, tok, EngineConfig(block_size=2, num_blocks=200, max_num_seqs=3, max_num_batched_tokens=7, watermark=0.0))
    ids = [engine.add_request(prompt_token_ids=p, sampling=SamplingParams(max_tokens=12, temperature=0, ignore_eos=True)) for p in prompts]
    finals = {}
    while engine.has_unfinished_requests:
        for o in engine.step():
            if o.finished:
                finals[o.request_id] = o.output_token_ids
    assert [finals[i] for i in ids] == refs
    assert engine.metrics.prompt_tokens_total == sum(len(p) for p in prompts)
