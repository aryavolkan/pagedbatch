"""Configuration dataclasses: model shape, engine limits, per-request sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}


@dataclass(frozen=True)
class ModelConfig:
    """Shape of a Llama-family transformer (RMSNorm, RoPE, GQA, SwiGLU)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def kv_bytes_per_token(self, dtype: str = "float32") -> int:
        """Bytes of KV cache one token occupies across all layers (K and V)."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * DTYPE_BYTES[dtype]

    @classmethod
    def tiny(cls, vocab_size: int = 258) -> ModelConfig:
        """A two-layer toy model for tests and CI; weights are random."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=64,
            intermediate_size=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
            bos_token_id=256,
            eos_token_id=257,
        )

    @classmethod
    def smollm2_135m(cls) -> ModelConfig:
        """The shape of HuggingFaceTB/SmolLM2-135M, for shape-faithful benchmarks."""
        return cls(
            vocab_size=49152,
            hidden_size=576,
            intermediate_size=1536,
            num_layers=30,
            num_heads=9,
            num_kv_heads=3,
            max_position_embeddings=8192,
            rms_norm_eps=1e-5,
            rope_theta=100000.0,
            tie_word_embeddings=True,
            bos_token_id=1,
            eos_token_id=2,
        )

    @classmethod
    def from_hf_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Build from a Hugging Face ``config.json`` of a Llama-architecture model."""
        arch = d.get("architectures") or []
        if arch and not any("Llama" in a for a in arch):
            raise ValueError(f"unsupported architecture {arch}; pagedbatch implements the Llama family")
        eos = d.get("eos_token_id")
        if isinstance(eos, list):
            eos = eos[0]
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_hidden_layers"],
            num_heads=d["num_attention_heads"],
            num_kv_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
            max_position_embeddings=d.get("max_position_embeddings", 2048),
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            rope_theta=d.get("rope_theta", 10000.0),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            bos_token_id=d.get("bos_token_id"),
            eos_token_id=eos,
        )


@dataclass
class EngineConfig:
    """Engine-wide limits. The names follow vLLM so the mapping is obvious."""

    block_size: int = 16
    num_blocks: int | None = None
    """Physical KV blocks. ``None`` derives the count from ``kv_cache_bytes``."""
    kv_cache_bytes: int = 256 * 2**20
    max_num_seqs: int = 16
    """Maximum sequences in one forward pass (the decode batch size)."""
    max_num_batched_tokens: int = 512
    """Token budget per step; prompts longer than this are prefilled in chunks."""
    max_model_len: int = 2048
    watermark: float = 0.01
    """Fraction of blocks kept free when admitting new sequences, so running
    sequences can usually grow without preemption."""
    enable_chunked_prefill: bool = True
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 0

    def resolve_num_blocks(self, model: ModelConfig) -> int:
        if self.num_blocks is not None:
            return self.num_blocks
        per_block = model.kv_bytes_per_token(self.dtype) * self.block_size
        return max(1, self.kv_cache_bytes // per_block)

    @property
    def watermark_blocks_for(self) -> Any:  # pragma: no cover - convenience only
        return lambda num_blocks: max(1, math.ceil(num_blocks * self.watermark))


@dataclass
class SamplingParams:
    """Per-request decoding parameters (a subset of the OpenAI API surface)."""

    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    ignore_eos: bool = False
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0
