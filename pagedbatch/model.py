"""A Llama-architecture transformer whose attention runs over the paged KV cache.

Every step is one call to ``LlamaForCausalLM.forward`` with a *flat* token batch:
the new tokens of every scheduled sequence concatenated into one ``[T, hidden]``
tensor (prompt chunks and single decode tokens side by side). The dense parts of
the network (projections, MLP) run once over the whole flat batch; attention
gathers each sequence's K/V from the cache through its block table.

The attention here is the gather-based form of paged attention: per layer the
needed slots are gathered into a padded ``[B, L_max, heads, dim]`` tensor and
handed to ``scaled_dot_product_attention`` with a causal-plus-padding mask.
Production engines fuse the gather into a custom kernel; the data layout and
the block-table indirection are the same.

Sequences are attended in two groups, decodes (one query each) and prefill
chunks, because padding every query to the longest chunk in the step made a
mixed step do up to ``max_num_seqs`` times the attention work it needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .kv_cache import KVCache


@dataclass
class AttentionGroup:
    """Sequences attended together: decodes (Q_max = 1) or prefill chunks."""

    kv_slot_table: torch.Tensor
    """[b, L_max] physical slot of every cached position per sequence (padding: 0)."""
    attn_mask: torch.Tensor
    """[b, 1, Q_max, L_max] boolean; True where query may attend key."""
    pad_index: torch.Tensor
    """[b, Q_max] flat row index of each padded query position."""
    valid_query: torch.Tensor
    """[b, Q_max] True for real (non-padding) query positions."""
    rows: torch.Tensor
    """[n] flat rows this group covers, in the order ``valid_query`` yields them."""


@dataclass
class ForwardBatch:
    """One step's inputs. Built by the engine from the scheduler output."""

    input_ids: torch.Tensor
    """[T] new token ids, sequences concatenated in schedule order."""
    positions: torch.Tensor
    """[T] position of each new token within its sequence."""
    slot_mapping: torch.Tensor
    """[T] physical cache slot each new token's K/V is written to."""
    query_start_loc: torch.Tensor
    """[B+1] cumulative query lengths; sequence b owns rows [qsl[b], qsl[b+1])."""
    context_lens: torch.Tensor
    """[B] total tokens in the cache per sequence after this step."""
    last_token_index: torch.Tensor
    """[B] flat row of each sequence's final new token (its logits feed sampling)."""
    groups: list[AttentionGroup]
    """Decode group first (if any), then the prefill group (if any)."""

    @property
    def num_tokens(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def num_seqs(self) -> int:
        return int(self.context_lens.shape[0])

    @staticmethod
    def build(
        token_chunks: list[list[int]],
        start_positions: list[int],
        slots: list[list[int]],
        block_tables: list[list[int]],
        block_size: int,
        device: str = "cpu",
    ) -> ForwardBatch:
        """Assemble a batch from per-sequence pieces (pure indexing, no model work)."""
        q_lens = [len(c) for c in token_chunks]
        num_seqs = len(token_chunks)
        total = sum(q_lens)
        if total == 0:
            raise ValueError("empty batch")
        input_ids = torch.tensor([t for c in token_chunks for t in c], dtype=torch.long)
        positions = torch.tensor([s + i for s, c in zip(start_positions, token_chunks, strict=True) for i in range(len(c))], dtype=torch.long)
        slot_mapping = torch.tensor([s for ss in slots for s in ss], dtype=torch.long)
        qsl = torch.zeros(num_seqs + 1, dtype=torch.long)
        qsl[1:] = torch.cumsum(torch.tensor(q_lens), 0)
        context_lens = torch.tensor([s + n for s, n in zip(start_positions, q_lens, strict=True)], dtype=torch.long)
        last_token_index = qsl[1:] - 1

        decode = [i for i, n in enumerate(q_lens) if n == 1]
        prefill = [i for i, n in enumerate(q_lens) if n > 1]
        groups = [ForwardBatch._group(idx, q_lens, qsl, context_lens, block_tables, block_size, total) for idx in (decode, prefill) if idx]
        to = lambda t: t.to(device)  # noqa: E731
        return ForwardBatch(
            input_ids=to(input_ids),
            positions=to(positions),
            slot_mapping=to(slot_mapping),
            query_start_loc=to(qsl),
            context_lens=to(context_lens),
            last_token_index=to(last_token_index),
            groups=[AttentionGroup(to(g.kv_slot_table), to(g.attn_mask), to(g.pad_index), to(g.valid_query), to(g.rows)) for g in groups],
        )

    @staticmethod
    def _group(
        idx: list[int],
        q_lens: list[int],
        qsl: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: list[list[int]],
        block_size: int,
        total: int,
    ) -> AttentionGroup:
        b = len(idx)
        ctx = context_lens[idx]
        q_lens_t = torch.tensor([q_lens[i] for i in idx], dtype=torch.long)
        max_blocks = max(len(block_tables[i]) for i in idx)
        bt = torch.zeros(b, max_blocks, dtype=torch.long)
        for row, i in enumerate(idx):
            bt[row, : len(block_tables[i])] = torch.tensor(block_tables[i], dtype=torch.long)
        l_max = int(ctx.max())
        key_pos = torch.arange(l_max, dtype=torch.long)
        blk = (key_pos // block_size).clamp(max=max_blocks - 1).unsqueeze(0).expand(b, l_max)
        kv_slot_table = bt.gather(1, blk) * block_size + (key_pos % block_size).unsqueeze(0)

        q_max = int(q_lens_t.max())
        q_idx = torch.arange(q_max, dtype=torch.long)
        valid = q_idx.unsqueeze(0) < q_lens_t.unsqueeze(1)
        q_pos = (ctx - q_lens_t).unsqueeze(1) + q_idx.unsqueeze(0)  # [b, Q_max]
        attn_mask = (key_pos.view(1, 1, l_max) <= q_pos.unsqueeze(2)).unsqueeze(1)  # causal, incl. the token itself
        starts = qsl[:-1][idx]
        pad_index = (starts.unsqueeze(1) + q_idx.unsqueeze(0)).clamp(max=total - 1)
        rows = pad_index[valid]
        return AttentionGroup(kv_slot_table, attn_mask, pad_index, valid, rows)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


class RotaryEmbedding(nn.Module):
    """Llama/HF rotary embedding: rotate-half convention, cos/sin over the full head dim."""

    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_positions, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[positions].unsqueeze(1).to(q.dtype)  # [T, 1, D]
        sin = self.sin[positions].unsqueeze(1).to(q.dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, batch: ForwardBatch, rope: RotaryEmbedding, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)
        q, k = rope(q, k, batch.positions)

        # Write the new tokens' K/V to their slots, then read each sequence's whole
        # context (new tokens included) through its block table.
        k_cache.index_copy_(0, batch.slot_mapping, k)
        v_cache.index_copy_(0, batch.slot_mapping, v)
        out = q.new_empty(t, self.num_heads, self.head_dim)
        for g in batch.groups:
            keys = k_cache[g.kv_slot_table]  # [b, L_max, Hkv, D]
            values = v_cache[g.kv_slot_table]
            if self.n_rep > 1:  # grouped-query attention: share each KV head across n_rep query heads
                keys = keys.repeat_interleave(self.n_rep, dim=2)
                values = values.repeat_interleave(self.n_rep, dim=2)
            keys = keys.transpose(1, 2)  # [b, H, L_max, D]
            values = values.transpose(1, 2)
            queries = q[g.pad_index].transpose(1, 2)  # [b, H, Q_max, D]
            o = F.scaled_dot_product_attention(queries, keys, values, attn_mask=g.attn_mask)
            out[g.rows] = o.transpose(1, 2)[g.valid_query]  # back to the flat batch rows
        return self.o_proj(out.reshape(t, self.num_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, batch: ForwardBatch, rope: RotaryEmbedding, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), batch, rope, k_cache, v_cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)

    def forward(self, batch: ForwardBatch, cache: KVCache) -> torch.Tensor:
        x = self.embed_tokens(batch.input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, batch, self.rope, cache.k[i], cache.v[i])
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    """State-dict keys match Hugging Face's ``LlamaForCausalLM`` so checkpoints load as is."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.model = LlamaModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, batch: ForwardBatch, cache: KVCache) -> torch.Tensor:
        """Returns ``[B, vocab]`` logits for the last new token of each sequence."""
        hidden = self.model(batch, cache)
        return self.lm_head(hidden[batch.last_token_index])

    @torch.no_grad()
    def init_random(self, seed: int = 0, std: float = 0.02) -> LlamaForCausalLM:
        g = torch.Generator().manual_seed(seed)
        for name, p in self.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                p.fill_(1.0)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * std)
        return self
