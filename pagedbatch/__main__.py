"""Command line: ``pagedbatch serve`` and ``pagedbatch generate``."""

from __future__ import annotations

import argparse
import sys

from .config import EngineConfig, SamplingParams


def add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="tiny", help="'tiny', 'smollm2-135m-random', a checkpoint directory, or a Hugging Face model id")
    p.add_argument("--block-size", type=int, default=16, help="tokens per KV block")
    p.add_argument("--num-blocks", type=int, default=None, help="physical KV blocks (default: derived from --kv-cache-mb)")
    p.add_argument("--kv-cache-mb", type=int, default=256, help="KV cache budget in MiB when --num-blocks is not given")
    p.add_argument("--max-num-seqs", type=int, default=16, help="max sequences per step")
    p.add_argument("--max-num-batched-tokens", type=int, default=512, help="token budget per step (chunked prefill)")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--no-chunked-prefill", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=0)


def engine_config_from_args(a: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        block_size=a.block_size,
        num_blocks=a.num_blocks,
        kv_cache_bytes=a.kv_cache_mb * 2**20,
        max_num_seqs=a.max_num_seqs,
        max_num_batched_tokens=a.max_num_batched_tokens,
        max_model_len=a.max_model_len,
        enable_chunked_prefill=not a.no_chunked_prefill,
        device=a.device,
        dtype=a.dtype,
        seed=a.seed,
    )


def build_engine(a: argparse.Namespace):
    from .engine import LLMEngine
    from .loader import load_model

    model, cfg, tok = load_model(a.model, device=a.device, dtype=a.dtype, seed=a.seed)
    engine = LLMEngine(model, cfg, tok, engine_config_from_args(a))
    print(
        f"pagedbatch: model={a.model} layers={cfg.num_layers} hidden={cfg.hidden_size} "
        f"kv_heads={cfg.num_kv_heads} | blocks={engine.num_blocks} x {engine.config.block_size} tokens "
        f"({engine.kv_cache.num_bytes / 2**20:.0f} MiB) | max_num_seqs={engine.config.max_num_seqs} "
        f"max_num_batched_tokens={engine.config.max_num_batched_tokens}",
        file=sys.stderr,
    )
    return engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pagedbatch", description="Minimal continuous-batching LLM inference server with a paged KV cache.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the OpenAI-compatible HTTP server")
    add_engine_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", default=None, help="name reported by /v1/models (default: --model)")

    gen = sub.add_parser("generate", help="run prompts offline and print the completions")
    add_engine_args(gen)
    gen.add_argument("prompts", nargs="+")
    gen.add_argument("--max-tokens", type=int, default=32)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)

    a = parser.parse_args(argv)
    if a.command == "serve":
        import uvicorn

        from .server import create_app

        engine = build_engine(a)
        uvicorn.run(create_app(engine, a.served_model_name or a.model), host=a.host, port=a.port, log_level="info")
        return 0

    engine = build_engine(a)
    params = SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature, top_p=a.top_p)
    for prompt, out in zip(a.prompts, engine.generate(a.prompts, params), strict=True):
        print(f"--- {prompt!r} [{out.finish_reason}, {len(out.output_token_ids)} tokens, ttft {out.ttft * 1000:.0f} ms]")
        print(out.text_delta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
