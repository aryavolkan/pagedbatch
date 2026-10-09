#!/usr/bin/env python3
"""HTTP load generator for a running server: TTFT, per-token latency and throughput under load.

    pagedbatch serve --model smollm2-135m-random --max-num-seqs 16 &
    python bench/load.py --num-requests 64 --concurrency 16 --prompt-len 128 --max-tokens 64

Closed loop by default (``--concurrency`` requests in flight); ``--rate`` switches to an
open loop with Poisson arrivals at that many requests per second. Every request streams
``/v1/completions``; one SSE chunk is one sampled token, so chunk timestamps give TTFT and
the time per output token (TPOT) directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import httpx


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


async def one(client: httpx.AsyncClient, url: str, prompt: list[int], max_tokens: int, temperature: float) -> dict:
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "ignore_eos": True, "stream": True}
    t0 = time.perf_counter()
    stamps: list[float] = []
    async with client.stream("POST", f"{url}/v1/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            stamps.append(time.perf_counter())
    e2e = time.perf_counter() - t0
    ttft = stamps[0] - t0 if stamps else float("nan")
    tpot = (stamps[-1] - stamps[0]) / (len(stamps) - 1) if len(stamps) > 1 else float("nan")
    return {"ttft": ttft, "tpot": tpot, "e2e": e2e, "tokens": len(stamps)}


async def main_async(a: argparse.Namespace) -> dict:
    rng = random.Random(a.seed)
    prompts = [[rng.randrange(0, 256) for _ in range(a.prompt_len)] for _ in range(a.num_requests)]
    results: list[dict] = []
    limits = httpx.Limits(max_connections=max(a.concurrency, 64))
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=limits) as client:
        (await client.get(f"{a.url}/health")).raise_for_status()
        t0 = time.perf_counter()
        if a.rate:

            async def fire(p):
                results.append(await one(client, a.url, p, a.max_tokens, a.temperature))

            tasks = []
            for p in prompts:
                tasks.append(asyncio.create_task(fire(p)))
                await asyncio.sleep(rng.expovariate(a.rate))
            await asyncio.gather(*tasks)
        else:
            sem = asyncio.Semaphore(a.concurrency)

            async def guarded(p):
                async with sem:
                    results.append(await one(client, a.url, p, a.max_tokens, a.temperature))

            await asyncio.gather(*(guarded(p) for p in prompts))
        wall = time.perf_counter() - t0
    tokens = sum(r["tokens"] for r in results)
    ttft = [r["ttft"] for r in results]
    tpot = [r["tpot"] for r in results if r["tpot"] == r["tpot"]]
    e2e = [r["e2e"] for r in results]
    return {
        "config": {k: getattr(a, k) for k in ("url", "num_requests", "concurrency", "rate", "prompt_len", "max_tokens", "temperature")},
        "wall_seconds": round(wall, 3),
        "requests_per_s": round(len(results) / wall, 2),
        "output_tok_s": round(tokens / wall, 1),
        "ttft_ms": {"p50": round(1000 * pct(ttft, 50), 1), "p90": round(1000 * pct(ttft, 90), 1), "p99": round(1000 * pct(ttft, 99), 1)},
        "tpot_ms": {
            "p50": round(1000 * pct(tpot, 50), 1),
            "p90": round(1000 * pct(tpot, 90), 1),
            "p99": round(1000 * pct(tpot, 99), 1),
            "mean": round(1000 * statistics.mean(tpot), 1) if tpot else None,
        },
        "e2e_ms": {"p50": round(1000 * pct(e2e, 50), 1), "p99": round(1000 * pct(e2e, 99), 1)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rate", type=float, default=None, help="open-loop Poisson arrivals per second (overrides --concurrency)")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    summary = asyncio.run(main_async(a))
    c = summary["config"]
    load = f"rate={c['rate']}/s" if c["rate"] else f"concurrency={c['concurrency']}"
    print(f"{c['num_requests']} requests, {load}, prompt {c['prompt_len']} tokens, {c['max_tokens']} output tokens each")
    print(f"throughput: {summary['requests_per_s']} req/s, {summary['output_tok_s']} output tok/s over {summary['wall_seconds']} s")
    print(f"TTFT ms  p50 {summary['ttft_ms']['p50']}  p90 {summary['ttft_ms']['p90']}  p99 {summary['ttft_ms']['p99']}")
    print(f"TPOT ms  p50 {summary['tpot_ms']['p50']}  p90 {summary['tpot_ms']['p90']}  p99 {summary['tpot_ms']['p99']}")
    print(f"E2E  ms  p50 {summary['e2e_ms']['p50']}  p99 {summary['e2e_ms']['p99']}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(summary, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
