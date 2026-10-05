"""Next-token sampling: greedy, temperature, top-k and top-p, with seeded generators."""

from __future__ import annotations

import torch

from .config import SamplingParams


def sample_token(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator) -> int:
    """Pick one token from a ``[vocab]`` logits row."""
    if params.greedy:
        return int(torch.argmax(logits))
    logits = logits.float() / params.temperature
    if params.top_k > 0 and params.top_k < logits.shape[-1]:
        kth = torch.topk(logits, params.top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if params.top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # Drop every token whose predecessors already cover top_p; keep at least one.
        drop = cumulative - probs > params.top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
