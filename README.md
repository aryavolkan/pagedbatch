# pagedbatch

[![ci](https://github.com/aryavolkan/pagedbatch/actions/workflows/ci.yml/badge.svg)](https://github.com/aryavolkan/pagedbatch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)

A minimal continuous-batching LLM inference server with a paged KV cache, in
under 2,000 lines of PyTorch. It makes the serving-side ideas behind vLLM
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

All numbers below were measured on a 4-vCPU `x86_64` cloud container (CPU only,
`float32`, PyTorch 2.14.1) with the real SmolLM2-135M architecture and random
weights: 30 layers, hidden 576, 3 KV heads. Throughput depends on the shapes,
not on the weight values, so the numbers carry over to the real checkpoint; text
quality does not. They show the *behavior* of continuous batching and a paged
cache, not a contest with GPU engines. Raw JSON for every table is in
[`bench/results/`](bench/results/).

### Offline engine benchmark

`bench/offline.py --model smollm2-135m-random --num-requests 32 --prompt-len 128 --output-len 16 128 --max-num-seqs 1 4 8 16 32 --static --kv-cache-mb 256`

32 requests, 128-token prompts, output lengths uniform in 16–128 tokens with
`ignore_eos`, block size 16. All requests are queued at once, so this run measures
throughput; latency is measured over HTTP below.

| mode | max_num_seqs | output tok/s | total tok/s | step ms (mean / p99) | preemptions | peak KV paged MiB | reserve-max-len MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| continuous | 1 | 23.7 | 67.2 | 42.16 / 164.82 | 0 | 11.25 | 90.0 |
| static | 1 | 23.7 | 67.2 | 42.16 / 148.96 | 0 | 11.25 | 90.0 |
| continuous | 4 | 55.6 | 157.4 | 68.40 / 212.76 | 0 | 36.56 | 360.0 |
| static | 4 | 48.7 | 138.0 | 52.30 / 280.75 | 0 | 39.38 | 360.0 |
| continuous | 8 | 78.0 | 220.9 | 86.59 / 260.86 | 0 | 71.72 | 720.0 |
| static | 8 | 68.2 | 193.1 | 69.45 / 421.11 | 0 | 67.50 | 720.0 |
| continuous | 16 | 113.6 | 321.8 | 91.06 / 385.78 | 0 | 137.11 | 1,440.0 |
| static | 16 | 98.5 | 279.1 | 92.56 / 421.01 | 0 | 109.69 | 1,440.0 |
| continuous | 32 | 131.5 | 372.5 | 135.94 / 486.81 | 0 | 205.31 | 2,880.0 |
| static | 32 | 131.7 | 373.0 | 135.76 / 467.97 | 0 | 205.31 | 2,880.0 |

Takeaways:

- **Batching scales throughput.** From batch 1 to 32, output tokens/s rise 5.5×,
  because the weights are read once per step for every sequence in the batch.
- **Continuous batching keeps the batch full.** It beats static batching by 14 % at
  `max_num_seqs=4` and by 15 % at 16: shorter sequences finish early and are replaced
  at once instead of waiting for the longest request in each fixed group. At 32 the
  whole workload is one batch either way.
- **Paged cache saves memory.** Even at the largest batch size, peak KV usage is
  205 MiB. Reserving `max_model_len` (2048 tokens) per running sequence
  would have required 2,880.0 MiB — a **14×** difference.
- **First-token latency is the trade-off.** A newly admitted request waits behind the
  running decodes, and a prefill chunk slows the decode step it joins. Users who need
  strict first-token latency can cap `max_num_batched_tokens` or run a dedicated
  prefill pass; the HTTP table below has the measured latencies.

### Under memory pressure

Same 32 requests with `--kv-cache-mb 96` (a 136-block pool, fully used at peak):
10 preemptions, each a recompute from scratch, and 54.6 output tok/s against
131.5 with the larger cache. The engine degrades instead of failing, and
`tests/test_engine.py` checks that preempted requests produce exactly the tokens they
would have produced without preemption.

### Decode and prefill attend in separate groups

The first version padded every sequence's queries to the longest prefill chunk in the
step, so a mixed step did up to `max_num_seqs` times the attention work it needed.
Grouping (#1) fixed that. Before/after on the same workload
(`bench/results/offline_padded_attention.json` vs `offline.json`):

| max_num_seqs | padded queries: tok/s | grouped: tok/s | change |
|---|---:|---:|---:|
| 1 | 25.7 | 23.7 | -8 % (one group either way: run-to-run variance) |
| 4 | 53.7 | 55.6 | +4 % |
| 8 | 73.4 | 78.0 | +6 % |
| 16 | 87.2 | 113.6 | +30 % |
| 32 | 118.3 | 131.5 | +11 % |

### HTTP load test

`pagedbatch serve --model smollm2-135m-random --max-num-seqs 16`, then
`bench/load.py --num-requests 64 --concurrency 16 --prompt-len 128 --max-tokens 64`
and the same with `--num-requests 16 --concurrency 1`. The load generator shared the
server's four cores, so these numbers sit below the offline engine throughput.

| metric | 1 in flight | 16 in flight |
|---|---:|---:|
| throughput | 0.21 req/s, 13.7 output tok/s | 0.78 req/s, 50.1 output tok/s |
| TTFT p50 / p90 / p99 | 255.2 / 284.9 / 290.0 ms | 1,402.1 / 2,752.6 / 3,588.6 ms |
| TPOT p50 / p90 / p99 | 70.5 / 73.7 / 75.4 ms | 301.2 / 307.6 / 314.1 ms |
| E2E p50 / p99 | 4,726.1 / 4,993.5 ms | 20,225.3 / 22,514.2 ms |

Per-token latency rises with the batch because the cores are shared sixteen ways;
aggregate throughput rises because the weights are read once per step for all
sixteen sequences. On a GPU, where decode is bound by memory bandwidth rather than
compute, the same trade is far steeper, which is why continuous batching is the
default there. Choosing the operating point within a fixed memory budget is the job
of an inference scheduler.

### Correctness

`tests/test_hf_equivalence.py` builds a Llama model with random weights in Hugging
Face `transformers`, saves it as a checkpoint, loads that checkpoint through
pagedbatch's own loader, and runs greedy generation through both engines with
2-token blocks, chunked prefill and batched requests. The output token sequences
match exactly. The test needs no download, so CI runs it on every push; the same
loader reads `HuggingFaceTB/SmolLM2-135M-Instruct` for real text.

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
