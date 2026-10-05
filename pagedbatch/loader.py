"""Model loading: a tiny random model, a shape-faithful random model, or a Hugging Face checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import ModelConfig
from .model import LlamaForCausalLM
from .tokenizer import ByteTokenizer, HFTokenizer, TokenizerLike

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

HF_FILES = ["config.json", "*.safetensors", "*.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]


def load_model(spec: str, device: str = "cpu", dtype: str = "float32", seed: int = 0) -> tuple[LlamaForCausalLM, ModelConfig, TokenizerLike]:
    """``spec`` is ``tiny``, ``smollm2-135m-random``, a local checkpoint directory, or a Hub id."""
    torch_dtype = DTYPES[dtype]
    if spec == "tiny":
        cfg = ModelConfig.tiny()
        model = LlamaForCausalLM(cfg).init_random(seed)
        tokenizer: TokenizerLike = ByteTokenizer()
    elif spec == "smollm2-135m-random":
        cfg = ModelConfig.smollm2_135m()
        model = LlamaForCausalLM(cfg).init_random(seed)
        tokenizer = ByteTokenizer()
    else:
        path = Path(spec)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(spec, allow_patterns=HF_FILES))
        cfg = ModelConfig.from_hf_dict(json.loads((path / "config.json").read_text()))
        model = LlamaForCausalLM(cfg)
        load_safetensors_into(model, path)
        if (path / "tokenizer.json").exists():
            tokenizer = HFTokenizer(path, cfg.bos_token_id, cfg.eos_token_id)
        else:  # weights-only checkpoint: still usable with raw token ids or bytes
            tokenizer = ByteTokenizer()
    model = model.to(device=device, dtype=torch_dtype).eval()
    return model, cfg, tokenizer


def load_safetensors_into(model: LlamaForCausalLM, path: Path) -> None:
    from safetensors.torch import load_file

    index = path / "model.safetensors.index.json"
    if index.exists():
        files = sorted({path / f for f in json.loads(index.read_text())["weight_map"].values()})
    else:
        files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {path}")
    state: dict[str, torch.Tensor] = {}
    for f in files:
        state.update(load_file(str(f)))
    if "lm_head.weight" not in state:
        if not model.config.tie_word_embeddings:
            raise KeyError("checkpoint has no lm_head.weight and the config does not tie embeddings")
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if not m.endswith("rope.cos") and not m.endswith("rope.sin")]
    if missing or unexpected:
        raise KeyError(f"checkpoint mismatch; missing={missing[:5]} unexpected={unexpected[:5]}")
    if model.config.tie_word_embeddings:
        model.tie_weights()
