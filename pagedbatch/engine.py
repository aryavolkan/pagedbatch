"""The engine: scheduler + model + cache. ``step()`` runs one forward pass."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import torch

from .block_manager import BlockAllocator
from .config import EngineConfig, ModelConfig, SamplingParams
from .kv_cache import KVCache
from .loader import DTYPES
from .metrics import EngineMetrics
from .model import ForwardBatch, LlamaForCausalLM
from .sampling import sample_token
from .scheduler import Scheduler, SchedulerOutput
from .sequence import Sequence
from .tokenizer import IncrementalDecoder, TokenizerLike


@dataclass
class RequestOutput:
    """What one step produced for one request."""

    request_id: str
    new_token_ids: list[int]
    text_delta: str
    finished: bool
    finish_reason: str | None
    output_token_ids: list[int]
    num_prompt_tokens: int
    ttft: float | None = None
    e2e_latency: float | None = None
    num_preemptions: int = 0


@dataclass
class _RequestState:
    seq: Sequence
    decoder: IncrementalDecoder
    generator: torch.Generator
    last_token_time: float | None = None
    text: str = field(default="")


class LLMEngine:
    def __init__(self, model: LlamaForCausalLM, model_config: ModelConfig, tokenizer: TokenizerLike, config: EngineConfig) -> None:
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.config = config
        self.num_blocks = config.resolve_num_blocks(model_config)
        self.allocator = BlockAllocator(self.num_blocks, config.block_size)
        self.scheduler = Scheduler(config, self.allocator)
        self.kv_cache = KVCache(model_config, self.num_blocks, config.block_size, DTYPES[config.dtype], config.device)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.metrics = EngineMetrics(blocks_total=self.num_blocks)
        self._requests: dict[str, _RequestState] = {}

    # -- requests ---------------------------------------------------------

    def add_request(
        self,
        prompt: str | None = None,
        sampling: SamplingParams | None = None,
        *,
        prompt_token_ids: list[int] | None = None,
        request_id: str | None = None,
    ) -> str:
        if prompt_token_ids is None:
            if prompt is None:
                raise ValueError("prompt or prompt_token_ids is required")
            prompt_token_ids = self.tokenizer.encode(prompt)
        if not prompt_token_ids:
            raise ValueError("empty prompt")
        sampling = sampling or SamplingParams()
        request_id = request_id or uuid.uuid4().hex
        if request_id in self._requests:
            raise ValueError(f"duplicate request id {request_id}")
        seq = Sequence(request_id=request_id, prompt_token_ids=list(prompt_token_ids), sampling=sampling)
        self.scheduler.add(seq)
        gen = torch.Generator().manual_seed(sampling.seed) if sampling.seed is not None else self.generator
        self._requests[request_id] = _RequestState(seq=seq, decoder=IncrementalDecoder(self.tokenizer), generator=gen)
        self.metrics.requests_total += 1
        return request_id

    def abort_request(self, request_id: str) -> bool:
        state = self._requests.pop(request_id, None)
        if state is None:
            return False
        return self.scheduler.abort(request_id)

    @property
    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished

    # -- one step ---------------------------------------------------------

    def step(self) -> list[RequestOutput]:
        t0 = time.perf_counter()
        sched = self.scheduler.schedule()
        self.metrics.preemptions_total += len(sched.preempted)
        if sched.is_empty:
            self._refresh_gauges()
            return []
        batch = self._build_batch(sched)
        with torch.inference_mode():
            logits = self.model(batch, self.kv_cache)
        outputs = self._postprocess(sched, logits, t0)
        self.metrics.steps_total += 1
        self.metrics.step_seconds.observe(time.perf_counter() - t0)
        self.metrics.prompt_tokens_total += sched.num_prefill_tokens
        self._refresh_gauges()
        return outputs

    def _build_batch(self, sched: SchedulerOutput) -> ForwardBatch:
        chunks, starts, slots, tables = [], [], [], []
        for s in sched.scheduled:
            seq = s.seq
            start = seq.num_computed_tokens
            chunks.append(seq.all_token_ids[start : start + s.num_new_tokens])
            starts.append(start)
            slots.append(s.slots)
            assert seq.block_table is not None
            tables.append(seq.block_table.blocks)
        return ForwardBatch.build(chunks, starts, slots, tables, self.config.block_size, self.config.device)

    def _postprocess(self, sched: SchedulerOutput, logits: torch.Tensor, step_start: float) -> list[RequestOutput]:
        outputs: list[RequestOutput] = []
        now = time.perf_counter()
        for i, s in enumerate(sched.scheduled):
            seq = s.seq
            seq.num_computed_tokens += s.num_new_tokens
            if not s.samples_token:
                continue
            state = self._requests[seq.request_id]
            token = sample_token(logits[i], seq.sampling, state.generator)
            seq.output_token_ids.append(token)
            self.metrics.generated_tokens_total += 1
            if seq.first_token_time is None:
                seq.first_token_time = now
                self.metrics.time_to_first_token_seconds.observe(now - seq.arrival_time)
            elif state.last_token_time is not None:
                self.metrics.time_per_output_token_seconds.observe(now - state.last_token_time)
            state.last_token_time = now

            reason = self._finish_reason(seq, token)
            if reason is not None:
                self.scheduler.finish(seq, reason)
                self.metrics.requests_finished += 1
                self.metrics.e2e_latency_seconds.observe(now - seq.arrival_time)
                del self._requests[seq.request_id]
            delta = state.decoder.flush(seq.output_token_ids) if reason is not None else state.decoder.step(seq.output_token_ids)
            state.text += delta
            outputs.append(
                RequestOutput(
                    request_id=seq.request_id,
                    new_token_ids=[token],
                    text_delta=delta,
                    finished=reason is not None,
                    finish_reason=reason,
                    output_token_ids=list(seq.output_token_ids),
                    num_prompt_tokens=len(seq.prompt_token_ids),
                    ttft=(seq.first_token_time - seq.arrival_time) if seq.first_token_time else None,
                    e2e_latency=(now - seq.arrival_time) if reason is not None else None,
                    num_preemptions=seq.num_preemptions,
                )
            )
        return outputs

    def _finish_reason(self, seq: Sequence, token: int) -> str | None:
        p = seq.sampling
        eos = self.model_config.eos_token_id
        if not p.ignore_eos and ((eos is not None and token == eos) or token in p.stop_token_ids):
            return "stop"
        if len(seq.output_token_ids) >= p.max_tokens:
            return "length"
        if seq.num_tokens >= self.config.max_model_len:
            return "length"
        return None

    def _refresh_gauges(self) -> None:
        self.metrics.num_running = self.scheduler.num_running
        self.metrics.num_waiting = self.scheduler.num_waiting
        self.metrics.blocks_used = self.allocator.num_used

    # -- offline convenience ---------------------------------------------

    def generate(self, prompts: list[str], sampling: SamplingParams | list[SamplingParams] | None = None) -> list[RequestOutput]:
        """Run every prompt to completion and return the final output per prompt, in order."""
        params = sampling if isinstance(sampling, list) else [sampling or SamplingParams()] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, params, strict=True)]
        finals: dict[str, RequestOutput] = {}
        texts: dict[str, str] = dict.fromkeys(ids, "")
        while self.has_unfinished_requests:
            for out in self.step():
                texts[out.request_id] += out.text_delta
                if out.finished:
                    out.text_delta = texts[out.request_id]
                    finals[out.request_id] = out
        return [finals[i] for i in ids]
