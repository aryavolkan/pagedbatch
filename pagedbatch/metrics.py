"""Engine metrics with a Prometheus text exposition, no client library needed."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


@dataclass
class Histogram:
    buckets: tuple[float, ...] = LATENCY_BUCKETS
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, value)] += 1
        self.total += value
        self.count += 1

    def render(self, name: str, help_text: str) -> str:
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        cumulative = 0
        for bound, c in zip(self.buckets, self.counts, strict=False):
            cumulative += c
            lines.append(f'{name}_bucket{{le="{bound}"}} {cumulative}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {self.count}')
        lines.append(f"{name}_sum {self.total:.6f}")
        lines.append(f"{name}_count {self.count}")
        return "\n".join(lines)


@dataclass
class EngineMetrics:
    requests_total: int = 0
    requests_finished: int = 0
    prompt_tokens_total: int = 0
    generated_tokens_total: int = 0
    preemptions_total: int = 0
    steps_total: int = 0
    num_running: int = 0
    num_waiting: int = 0
    blocks_used: int = 0
    blocks_total: int = 0
    step_seconds: Histogram = field(default_factory=Histogram)
    time_to_first_token_seconds: Histogram = field(default_factory=Histogram)
    time_per_output_token_seconds: Histogram = field(default_factory=Histogram)
    e2e_latency_seconds: Histogram = field(default_factory=Histogram)

    def render(self) -> str:
        counters = {
            "pagedbatch_requests_total": ("Requests accepted.", self.requests_total),
            "pagedbatch_requests_finished_total": ("Requests finished.", self.requests_finished),
            "pagedbatch_prompt_tokens_total": ("Prompt tokens prefilled (recomputes included).", self.prompt_tokens_total),
            "pagedbatch_generated_tokens_total": ("Tokens sampled.", self.generated_tokens_total),
            "pagedbatch_preemptions_total": ("Sequences preempted for recompute.", self.preemptions_total),
            "pagedbatch_steps_total": ("Engine steps (model forward passes).", self.steps_total),
        }
        gauges = {
            "pagedbatch_num_running": ("Sequences in the running batch.", self.num_running),
            "pagedbatch_num_waiting": ("Sequences waiting for admission.", self.num_waiting),
            "pagedbatch_kv_blocks_used": ("KV cache blocks in use.", self.blocks_used),
            "pagedbatch_kv_blocks_total": ("KV cache blocks available.", self.blocks_total),
        }
        out = []
        for name, (help_text, value) in counters.items():
            out.append(f"# HELP {name} {help_text}\n# TYPE {name} counter\n{name} {value}")
        for name, (help_text, value) in gauges.items():
            out.append(f"# HELP {name} {help_text}\n# TYPE {name} gauge\n{name} {value}")
        out.append(self.step_seconds.render("pagedbatch_step_seconds", "Wall time of one engine step."))
        out.append(self.time_to_first_token_seconds.render("pagedbatch_time_to_first_token_seconds", "Arrival to first sampled token."))
        out.append(self.time_per_output_token_seconds.render("pagedbatch_time_per_output_token_seconds", "Gap between consecutive sampled tokens."))
        out.append(self.e2e_latency_seconds.render("pagedbatch_e2e_latency_seconds", "Arrival to finish."))
        return "\n".join(out) + "\n"
