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

All numbers below are from a 4-vCPU `x86_64` GitHub Actions runner (CPU-only,
`float32`) using the real SmolLM2-135M architecture with random weights. They
are meant to show the *behavior* of continuous batching and paged caching, not
to compete with GPU-optimized engines.

### Offline engine benchmark

`bench/offline.py --model smollm2-135m-random --num-requests 32 --prompt-len 128 --output-len 16 128 --kv-cache-mb 256`

| mode | max_num_seqs | output tok/s | total tok/s | step ms (mean / p99) | TTFT p99 ms | preemptions | peak KV paged MiB | reserve-max-len MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| continuous | 1 | 25.7 | 72.7 | 38.96 / 149.06 | 84,464 | 0 | 11.25 | 90.0 |
| static | 1 | 25.0 | 70.8 | 40.00 / 156.05 | 416 | 0 | 11.25 | 90.0 |
| continuous | 4 | 53.7 | 152.2 | 70.72 / 257.40 | 36,210 | 0 | 36.56 | 360.0 |
| static | 4 | 45.8 | 129.7 | 55.65 / 291.81 | 869 | 0 | 39.38 | 360.0 |
| continuous | 8 | 73.4 | 208.0 | 91.94 / 328.45 | 23,568 | 0 | 71.72 | 720.0 |
| static | 8 | 69.0 | 195.5 | 68.59 / 436.05 | 1,007 | 0 | 67.50 | 720.0 |
| continuous | 16 | 87.2 | 246.9 | 118.71 / 468.29 | 16,994 | 0 | 137.11 | 1,440.0 |
| static | 16 | 90.8 | 257.1 | 100.47 / 633.98 | 2,631 | 0 | 109.69 | 1,440.0 |
| continuous | 32 | 118.3 | 335.1 | 151.11 / 702.98 | 4,977 | 0 | 205.31 | 2,880.0 |
| static | 32 | 120.9 | 342.5 | 147.87 / 619.35 | 4,584 | 0 | 205.31 | 2,880.0 |

Takeaways:

- **Continuous batching keeps the batch full.** At `max_num_seqs=4` it beats
  static batching by 17 % on output tokens/s; the gap opens where shorter
  sequences finish early and are replaced instead of waiting for the longest
  request in each fixed group.
- **Paged cache saves memory.** Even at the largest batch size, peak KV usage
  is 205 MiB. Reserving `max_model_len` (2048 tokens) per running sequence
  would have required 2,880 MiB — a **14×** difference.
- **TTFT is the trade-off.** Because the engine mixes decode steps and prefill
  chunks, a newly admitted request can wait behind running decodes. The p99
  TTFT falls as `max_num_seqs` rises because more decode tokens are amortized
  per step, but users who need strict first-token latency can cap
  `max_num_batched_tokens` or use a dedicated prefill pass.

### HTTP load test

`pagedbatch serve --model smollm2-135m-random --max-num-seqs 16`, then
`bench/load.py --num-requests 64 --concurrency 16 --prompt-len 128 --max-tokens 64`.

| metric | value |
|---|---:|
| throughput | 0.81 req/s, 51.7 output tok/s |
| TTFT p50 / p90 / p99 | 1,622 / 2,475 / 3,218 ms |
| TPOT p50 / p90 / p99 | 284 / 300 / 302 ms |
| E2E p50 / p99 | 19,600 / 22,271 ms |

### Correctness

`tests/test_hf_equivalence.py` runs greedy generation on
`HuggingFaceTB/SmolLM2-135M-Instruct` through both pagedbatch and Hugging Face
`transformers` with chunked prefill, 2-token blocks and batched requests. The
output token sequences match exactly.

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
