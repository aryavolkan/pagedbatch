import pytest
import torch

from pagedbatch.config import EngineConfig, ModelConfig
from pagedbatch.engine import LLMEngine
from pagedbatch.loader import load_model


@pytest.fixture(scope="session")
def tiny():
    """(model, config, tokenizer) for the random two-layer model; built once per session."""
    torch.manual_seed(0)
    return load_model("tiny")


def make_engine(tiny, **overrides) -> LLMEngine:
    model, cfg, tok = tiny
    defaults = dict(block_size=4, num_blocks=256, max_num_seqs=8, max_num_batched_tokens=64, max_model_len=512, watermark=0.0)
    defaults.update(overrides)
    return LLMEngine(model, cfg, tok, EngineConfig(**defaults))


@pytest.fixture
def engine_factory(tiny):
    return lambda **kw: make_engine(tiny, **kw)


__all__ = ["ModelConfig"]
