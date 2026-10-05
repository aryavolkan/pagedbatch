# CLAUDE.md

Guidance for Claude Code in this repository.

## Communication with the user

Explain things in ASD-STE100 (Simplified Technical English):

- Use short sentences (20 words or fewer). Put one topic in each sentence.
- Use the active voice and the present tense.
- Use simple words. Use one word for one meaning. Do not use slang or idioms.
- Use "must" for a requirement and "can" for a possibility.
- Use lists for steps and for items. Keep a paragraph to 6 sentences or fewer.
- Technical names (for example: KV cache, scheduler, token) are permitted.

## Project

pagedbatch is a minimal continuous-batching LLM inference server with a paged
KV cache. The model is a Llama-family transformer in plain PyTorch. The point
of the project is the control plane: the block allocator, the scheduler, the
preemption path and the OpenAI-compatible server. Read the modules in this
order: `block_manager.py`, `sequence.py`, `scheduler.py`, `model.py`,
`engine.py`, `server.py`.

## Commands

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel
pip install -e ".[dev]"
ruff check .                                   # lint
pytest                                         # 33 tests; includes the transformers equivalence test
pagedbatch serve --model tiny                  # random two-layer model, no download
pagedbatch serve --model HuggingFaceTB/SmolLM2-135M-Instruct
python bench/offline.py --model smollm2-135m-random --max-num-seqs 1 4 8 16 32 --static --json bench/results/offline.json
python bench/load.py --num-requests 64 --concurrency 16            # against a running server
```

## Rules

- Greedy output must stay identical to Hugging Face `transformers` for the same
  weights. `tests/test_hf_equivalence.py` checks this under 2-token blocks,
  chunked prefill and batching. Do not weaken that test.
- Preemption, chunked prefill and batching must not change outputs
  (`tests/test_engine.py`).
- A request is admitted only if its maximum length fits the cache alone. This
  guarantees that every admitted request terminates. Keep this rule.
- `ScheduledSequence` flags are snapshots taken at schedule time. The engine
  changes sequence state after the forward pass, so do not compute them lazily.
- Benchmark numbers in the README come from `bench/results/*.json`. Re-run the
  benchmarks and update both when the engine changes performance.
- Tests and CI must not need a model download. Use `tiny` or a random
  `transformers` checkpoint saved to a temporary directory.
