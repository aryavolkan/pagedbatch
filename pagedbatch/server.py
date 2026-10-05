"""OpenAI-compatible HTTP front end: /v1/completions, /v1/chat/completions, /metrics.

The engine runs on one background thread. HTTP handlers never touch engine state
directly: they enqueue (request, subscriber) pairs that the engine thread drains
at the start of each step, and receive ``RequestOutput`` objects back through an
asyncio queue. The event loop therefore never waits on a model forward pass.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import SamplingParams
from .engine import LLMEngine, RequestOutput


@dataclass
class _Pending:
    request_id: str
    prompt_token_ids: list[int]
    sampling: SamplingParams
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[RequestOutput | Exception]


class AsyncEngine:
    """Thread-owned engine with an asyncio-facing ``generate``."""

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self._cv = threading.Condition()
        self._pending: deque[_Pending] = deque()
        self._aborts: deque[str] = deque()
        self._subscribers: dict[str, _Pending] = {}
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="pagedbatch-engine", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stop and not self._pending and not self._aborts and not self.engine.has_unfinished_requests:
                    self._cv.wait()
                if self._stop:
                    return
                pending = list(self._pending)
                self._pending.clear()
                aborts = list(self._aborts)
                self._aborts.clear()
            for rid in aborts:
                self.engine.abort_request(rid)
                self._subscribers.pop(rid, None)
            for p in pending:
                try:
                    self.engine.add_request(prompt_token_ids=p.prompt_token_ids, sampling=p.sampling, request_id=p.request_id)
                except ValueError as e:
                    p.loop.call_soon_threadsafe(p.queue.put_nowait, e)
                    continue
                self._subscribers[p.request_id] = p
            for out in self.engine.step():
                sub = self._subscribers.get(out.request_id)
                if sub is not None:
                    sub.loop.call_soon_threadsafe(sub.queue.put_nowait, out)
                    if out.finished:
                        del self._subscribers[out.request_id]

    async def generate(self, prompt_token_ids: list[int], sampling: SamplingParams) -> AsyncIterator[RequestOutput]:
        rid = uuid.uuid4().hex
        p = _Pending(rid, prompt_token_ids, sampling, asyncio.get_running_loop(), asyncio.Queue())
        with self._cv:
            self._pending.append(p)
            self._cv.notify()
        finished = False
        try:
            while not finished:
                item = await p.queue.get()
                if isinstance(item, Exception):
                    raise item
                finished = item.finished
                yield item
        finally:
            if not finished:
                with self._cv:
                    self._aborts.append(rid)
                    self._cv.notify()


# -- API schemas (the subset of OpenAI's that the engine supports) --------------


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[int]
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = None
    stream: bool = False
    ignore_eos: bool = False

    def sampling(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, seed=self.seed, ignore_eos=self.ignore_eos
        )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = None
    stream: bool = False
    ignore_eos: bool = False

    def sampling(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, seed=self.seed, ignore_eos=self.ignore_eos
        )


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def create_app(engine: LLMEngine, model_name: str = "pagedbatch") -> FastAPI:
    async_engine = AsyncEngine(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async_engine.start()
        try:
            yield
        finally:
            async_engine.stop()

    app = FastAPI(title="pagedbatch", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.async_engine = async_engine
    tokenizer = engine.tokenizer

    def encode(prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, list):
            if not prompt:
                raise HTTPException(400, "empty prompt")
            return prompt
        return tokenizer.encode(prompt)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "pagedbatch"}]}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return engine.metrics.render()

    async def run(prompt_ids: list[int], sampling: SamplingParams) -> AsyncIterator[RequestOutput]:
        try:
            async for out in async_engine.generate(prompt_ids, sampling):
                yield out
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> Any:
        prompt_ids = encode(req.prompt)
        sampling = req.sampling()
        cid = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        if req.stream:

            async def events() -> AsyncIterator[str]:
                async for out in run(prompt_ids, sampling):
                    yield _sse({"id": cid, "object": "text_completion", "created": created, "model": model_name,
                                "choices": [{"index": 0, "text": out.text_delta, "finish_reason": out.finish_reason}]})
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        text, final = "", None
        async for out in run(prompt_ids, sampling):
            text += out.text_delta
            final = out
        assert final is not None
        return {"id": cid, "object": "text_completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": final.finish_reason}],
                "usage": _usage(len(prompt_ids), len(final.output_token_ids))}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        prompt = tokenizer.apply_chat_template([m.model_dump() for m in req.messages])
        prompt_ids = tokenizer.encode(prompt)
        sampling = req.sampling()
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        if req.stream:

            async def events() -> AsyncIterator[str]:
                yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
                async for out in run(prompt_ids, sampling):
                    yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": out.text_delta}, "finish_reason": out.finish_reason}]})
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        text, final = "", None
        async for out in run(prompt_ids, sampling):
            text += out.text_delta
            final = out
        assert final is not None
        return {"id": cid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": final.finish_reason}],
                "usage": _usage(len(prompt_ids), len(final.output_token_ids))}

    return app
