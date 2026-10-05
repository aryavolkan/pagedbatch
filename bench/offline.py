#!/usr/bin/env python3
"""Engine-level benchmark: throughput against batch size, continuous against static batching, KV memory.

Drives ``LLMEngine`` directly (no HTTP) with synthetic requests, so the numbers isolate
the scheduler and the paged cache from network and tokenization.

    python bench/offline.py --model smollm2-135m-random --num-requests 32 --prompt-len 128 \
        --output-len 16 128 --max-num-seqs 1 4 8 16 32 --static --json bench/results/offline.json

Prompts are random token ids; output lengths are drawn uniformly from the given range
and enforced with ``ignore_eos`` so every run does the same work.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from pagedbatch.config import EngineConfig, SamplingParams  # noqa: E402
from pagedbatch.engine import LLMEngine  # noqa: E402
from pagedbatch.loader import load_model  # noqa: E402


def make_requests(n: int, prompt_len: int, output_len: list[int], seed: int) -> list[tuple[list[int], int]]:
    rng = random.Random(seed)
    lo, hi = (output_len[0], output_len[-1])
    return [([rng.randrange(0, 256) for _ in range(prompt_len)], rng.randint(lo, hi)) for _ in range(n)]


def run(engine: LLMEngine, requests: list[tuple[list[int], int]], static: bool) -> dict:
    """Run all requests; ``static`` admits them in fixed groups and waits for each group."""
    bytes_per_block = engine.kv_cache.bytes_per_block()
    bytes_per_token = engine.model_config.kv_bytes_per_token(engine.config.dtype)
    groups = [requests[i : i + engine.config.max_num_seqs] for i in range(0, len(requests), engine.config.max_num_seqs)] if static else [requests]
    step_times: list[float] = []
    ttfts: list[float] = []
    peak_blocks = peak_running = 0
    output_tokens = 0
    t0 = time.perf_counter()
    for group in groups:
        for ids, out_len in group:
            engine.add_request(prompt_token_ids=ids, sampling=SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True))
        while engine.has_unfinished_requests:
            ts = time.perf_counter()
            outs = engine.step()
            step_times.append(time.perf_counter() - ts)
            peak_blocks = max(peak_blocks, engine.allocator.num_used)
            peak_running = max(peak_running, engine.scheduler.num_running)
            for o in outs:
                output_tokens += len(o.new_token_ids)
                if o.ttft is not None and len(o.output_token_ids) == 1:
                    ttfts.append(o.ttft)
    seconds = time.perf_counter() - t0
    prompt_tokens = sum(len(ids) for ids, _ in requests)
    return {
        "mode": "static" if static else "continuous",
        "max_num_seqs": engine.config.max_num_seqs,
        "seconds": round(seconds, 3),
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "output_tok_s": round(output_tokens / seconds, 1),
        "total_tok_s": round((output_tokens + prompt_tokens) / seconds, 1),
        "steps": len(step_times),
        "step_ms_mean": round(1000 * statistics.mean(step_times), 2),
        "step_ms_p99": round(1000 * sorted(step_times)[int(0.99 * (len(step_times) - 1))], 2),
        "ttft_ms_mean": round(1000 * statistics.mean(ttfts), 1) if ttfts else None,
        "ttft_ms_p99": round(1000 * sorted(ttfts)[int(0.99 * (len(ttfts) - 1))], 1) if ttfts else None,
        "preemptions": engine.metrics.preemptions_total,
        "peak_running": peak_running,
        "peak_blocks": peak_blocks,
        "kv_paged_peak_mib": round(peak_blocks * bytes_per_block / 2**20, 2),
        "kv_reserve_max_len_mib": round(peak_running * engine.config.max_model_len * bytes_per_token / 2**20, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--num-requests", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, nargs="+", default=[64], help="fixed length, or 'lo hi' for a uniform range")
    ap.add_argument("--max-num-seqs", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--static", action="store_true", help="also run static batching at each batch size")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-cache-mb", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="write all results here")
    a = ap.parse_args()

    model, cfg, tok = load_model(a.model, seed=a.seed)
    requests = make_requests(a.num_requests, a.prompt_len, a.output_len, a.seed)
    rows = []
    for n in a.max_num_seqs:
        for static in ([False, True] if a.static else [False]):
            engine = LLMEngine(model, cfg, tok, EngineConfig(block_size=a.block_size, kv_cache_bytes=a.kv_cache_mb * 2**20, max_num_seqs=n,
                                                            max_num_batched_tokens=a.max_num_batched_tokens, max_model_len=a.max_model_len, seed=a.seed))
            row = run(engine, requests, static)
            rows.append(row)
            print(f"{row['mode']:>10} max_num_seqs={n:<3} {row['output_tok_s']:>7.1f} out tok/s  {row['step_ms_mean']:>7.2f} ms/step  "
                  f"ttft p99 {row['ttft_ms_p99']:>7} ms  preempt={row['preemptions']}  peak blocks={row['peak_blocks']}", file=sys.stderr)

    env = {"model": a.model, "layers": cfg.num_layers, "hidden": cfg.hidden_size, "kv_heads": cfg.num_kv_heads, "machine": platform.machine(),
           "cpu_count": os.cpu_count(), "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "num_requests": a.num_requests,
           "prompt_len": a.prompt_len, "output_len": a.output_len, "block_size": a.block_size, "kv_cache_mb": a.kv_cache_mb, "max_model_len": a.max_model_len}
    print("\n| mode | max_num_seqs | output tok/s | step ms (mean / p99) | TTFT p99 ms | preemptions | peak KV paged MiB | reserve-max-len MiB |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['mode']} | {r['max_num_seqs']} | {r['output_tok_s']} | {r['step_ms_mean']} / {r['step_ms_p99']} | {r['ttft_ms_p99']} | "
              f"{r['preemptions']} | {r['kv_paged_peak_mib']} | {r['kv_reserve_max_len_mib']} |")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"env": env, "rows": rows}, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
