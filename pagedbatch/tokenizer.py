"""Tokenizers: a Hugging Face ``tokenizers`` wrapper and a dependency-free byte tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TokenizerLike(Protocol):
    bos_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, add_bos: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def apply_chat_template(self, messages: list[dict[str, str]]) -> str: ...


class ByteTokenizer:
    """256 byte tokens plus BOS (256) and EOS (257). Used by the tiny test model."""

    vocab_size = 258

    def __init__(self) -> None:
        self.bos_token_id = 256
        self.eos_token_id = 257

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [self.bos_token_id] + ids if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"


class HFTokenizer:
    """Wraps a ``tokenizer.json``; renders chats in the ChatML format SmolLM2 uses."""

    def __init__(self, path: str | Path, bos_token_id: int | None, eos_token_id: int | None) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = self._tok.get_vocab_size()

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_bos and self.bos_token_id is not None and (not ids or ids[0] != self.bos_token_id):
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        return "".join(parts) + "<|im_start|>assistant\n"


class IncrementalDecoder:
    """Turns a growing token list into text deltas without splitting multi-byte characters."""

    def __init__(self, tokenizer: TokenizerLike) -> None:
        self._tokenizer = tokenizer
        self._emitted = 0

    def step(self, output_token_ids: list[int]) -> str:
        text = self._tokenizer.decode(output_token_ids)
        if text.endswith("�"):  # incomplete UTF-8 sequence: wait for more tokens
            return ""
        delta = text[self._emitted :]
        self._emitted = len(text)
        return delta

    def flush(self, output_token_ids: list[int]) -> str:
        """Final delta: emit everything not yet sent, replacement characters included."""
        text = self._tokenizer.decode(output_token_ids)
        delta = text[self._emitted :]
        self._emitted = len(text)
        return delta
