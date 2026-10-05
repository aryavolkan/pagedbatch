# pagedbatch

[![ci](https://github.com/aryavolkan/pagedbatch/actions/workflows/ci.yml/badge.svg)](https://github.com/aryavolkan/pagedbatch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)

A minimal continuous-batching LLM inference server with a paged KV cache, in
about 1,500 lines of PyTorch. It makes the serving-side ideas behind vLLM
concrete and testable: a block allocator with per-sequence block tables, a
scheduler that mixes prefill chunks and decode steps in one forward pass,
preemption by recomputation when the cache runs out, and an OpenAI-compatible
API with streaming and Prometheus metrics. No custom kernels: every mechanism
is readable Python.

Correctness is pinned to Hugging Face `transformers`: on the same Llama weights
the engine's greedy output is identical token for token with 2-token blocks,
chunked prefill and batched requests ([`tests/test_hf_equivalence.py`](tests/test_hf_equivalence.py)).

## The problem it solves

A naive server runs one request at a time and reserves `max_model_len` worth of
KV cache for each. Two ideas fix that, and this repo implements both:

- **Continuous batching.** Every engine step admits new requests as others
  finish, so the decode batch stays full instead of waiting for the longest
  request in a static batch. Prompts that do not fit the per-step token budget
  are prefilled in chunks alongside running decodes.
- **Paged KV cache.** Sequences grow one fixed-size block at a time out of a
  shared pool, so memory waste is bounded by one block per sequence, any
  sequence can use any free block, and a full cache is handled by preempting
  the lowest-priority sequence and recomputing it later instead of failing.

## What is inside

| Module | Role |
|---|---|
| [`block_manager.py`](pagedbatch/block_manager.py) | Free list over physical blocks; per-sequence block tables; slot = `block * block_size + offset` |
| [`scheduler.py`](pagedbatch/scheduler.py) | One step's plan: running sequences first (a decode token or the next prefill chunk), then FIFO admission under the token budget, the sequence cap and a block watermark; recompute preemption from the tail of the running list |
| [`model.py`](pagedbatch/model.py) | Llama-family transformer (RMSNorm, RoPE, grouped-query attention, SwiGLU). One flat token batch per step; attention writes K/V at slots and gathers each sequence's context through its block table |
| [`kv_cache.py`](pagedbatch/kv_cache.py) | Per-layer `[num_slots, kv_heads, head_dim]` pools |
| [`engine.py`](pagedbatch/engine.py) | `step()` = schedule, build the batch, forward, sample, finish and free |
| [`server.py`](pagedbatch/server.py) | FastAPI front end; the engine runs on its own thread and receives requests through a queue; SSE streaming; `/metrics` |
| [`sampling.py`](pagedbatch/sampling.py), [`tokenizer.py`](pagedbatch/tokenizer.py), [`loader.py`](pagedbatch/loader.py) | Greedy, temperature, top-k, top-p with seeded generators; HF `tokenizers` with a byte-level fallback; safetensors loader with tied embeddings |
| [`bench/offline.py`](bench/offline.py), [`bench/load.py`](bench/load.py) | Engine-level and HTTP benchmarks (throughput, TTFT, time per output token, KV memory) |

## Request lifecycle

```
client ──POST /v1/completions──► handler ──request queue──► engine thread, each step:
                                                             1 scheduler.schedule()   admit / preempt, assign cache slots
  ◄── SSE chunk per token ─── asyncio queue ◄── RequestOutput  2 ForwardBatch.build()   flat tokens + block tables → slot table, mask
                                                             3 model.forward()        write K/V at slots, gather per sequence, SDPA
                                                             4 sample, finish          free the blocks of finished sequences
```

The flat batch is what makes mixed prefill and decode cheap: projections and the
MLP run once over every scheduled token of every sequence, and only attention
cares which sequence a token belongs to.

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[hub,dev]"

pagedbatch serve --model HuggingFaceTB/SmolLM2-135M-Instruct          # downloads ~270 MB once
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":64}'
```

Any Llama-architecture checkpoint on the Hub or on disk works (`config.json` +
safetensors + `tokenizer.json`). Without a download:

```bash
pagedbatch serve --model tiny                 # random two-layer model: gibberish, but the whole engine runs
pagedbatch serve --model smollm2-135m-random  # the real 135M shape with random weights, for benchmarks
pagedbatch generate --model tiny --max-tokens 16 "hello"
```

Engine knobs mirror vLLM's: `--block-size`, `--kv-cache-mb` (or `--num-blocks`),
`--max-num-seqs`, `--max-num-batched-tokens`, `--max-model-len`, `--dtype`, `--device`.

Docker:

```bash
docker build -t pagedbatch .
docker run --rm -p 8000:8000 pagedbatch                                   # tiny model
docker run --rm -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
    pagedbatch serve --model HuggingFaceTB/SmolLM2-135M-Instruct --host 0.0.0.0
```

Tests and lint: `pytest` and `ruff check .` (CI runs both, plus a benchmark
sanity run and a Docker build with an HTTP smoke test).

## Results

<!-- RESULTS -->

## Design notes

**What matches vLLM.** Block tables and slot mapping, the free-block watermark,
recompute preemption of the most recently admitted sequence, FIFO admission,
`max_num_seqs` and `max_num_batched_tokens` as the two budgets, chunked prefill,
the OpenAI endpoints and the metric names.

**What is deliberately simpler.** Attention is the gather-based form of paged
attention (`index_select` into a padded tensor plus
`scaled_dot_product_attention`), not a fused kernel. There is no prefix caching,
no swap-to-CPU preemption, no speculative decoding, no tensor parallelism and
no CUDA graphs. One model family (Llama: SmolLM2, Llama 3, Qwen2-style
checkpoints that use the Llama architecture). Default dtype is float32 on CPU;
`--dtype bfloat16 --device cuda` work but are not tuned.

**Two decisions worth knowing.**

- A request is admitted only if its *maximum* length (prompt plus
  `max_tokens`) fits the cache on its own. Victims are always taken from the
  tail of the running list, so the head always makes progress and every admitted
  request terminates; without the rule a lone oversized request would preempt
  itself forever.
- Scheduled chunks snapshot their `samples_token` and `is_prefill` flags at
  schedule time. The engine mutates sequence state after the forward pass, and
  computing those flags lazily was the first bug the tests caught.

## License

[MIT](LICENSE).
