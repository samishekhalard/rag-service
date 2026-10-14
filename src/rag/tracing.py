"""
Tracing.

Every stage of a query opens a span: embed, dense search, sparse search,
fusion, rerank, generate. Each span records its latency and, where it applies,
tokens and cost. Spans are collected locally in all cases and additionally
shipped to Langfuse when keys are configured.

Local collection is the point. A trace that only exists in a SaaS dashboard
cannot be asserted on in a test or summarised in an eval table; here the same
spans drive the /query response, the eval report and the Langfuse view.

When Langfuse keys are absent — tests, CI, a clone without a .env — the
exporter degrades to a no-op instead of raising.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .config import Settings


@dataclass
class Span:
    name: str
    duration_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Trace:
    name: str
    spans: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return round(sum(s.duration_ms for s in self.spans), 2)

    @property
    def tokens_in(self) -> int:
        return sum(s.tokens_in for s in self.spans)

    @property
    def tokens_out(self) -> int:
        return sum(s.tokens_out for s in self.spans)

    def stage(self, name: str) -> Span | None:
        for span in self.spans:
            if span.name == name:
                return span
        return None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    **({"metadata": s.metadata} if s.metadata else {}),
                }
                for s in self.spans
            ],
        }


class Tracer:
    """Collects spans locally; mirrors them to Langfuse when configured."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = self._connect(settings)

    @staticmethod
    def _connect(settings: Settings):
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            return None
        try:
            from langfuse import Langfuse

            return Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception:
            # Tracing must never take the service down with it.
            return None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def trace(self, name: str, **metadata):
        trace = Trace(name=name)
        started = time.perf_counter()
        try:
            yield trace
        finally:
            if self._client is not None:
                self._export(trace, metadata, (time.perf_counter() - started) * 1000)

    @contextmanager
    def span(self, trace: Trace, name: str, **metadata):
        span = Span(name=name, duration_ms=0.0, metadata=dict(metadata))
        started = time.perf_counter()
        try:
            yield span
        finally:
            span.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            trace.spans.append(span)

    def _export(self, trace: Trace, metadata: dict, total_ms: float) -> None:
        try:
            root = self._client.trace(name=trace.name, metadata={**metadata, "total_ms": total_ms})
            for span in trace.spans:
                root.span(
                    name=span.name,
                    metadata={
                        **span.metadata,
                        "duration_ms": span.duration_ms,
                        "tokens_in": span.tokens_in,
                        "tokens_out": span.tokens_out,
                    },
                )
        except Exception:
            return


def cost_usd(settings: Settings, embed_tokens: int, tokens_in: int, tokens_out: int) -> float:
    """
    API spend for one query, in USD.

    Only the halves that actually hit a paid API are charged. A local model
    bills nothing, so applying the configured prices to it would report a
    cost that was never incurred — and the two halves can use different
    providers, so they are priced separately: embedding locally while
    generating with a cloud model charges for generation alone.

    This is API spend only. It says nothing about the hardware a local model
    runs on, or the electricity it uses.
    """
    million = 1_000_000
    total = 0.0
    if settings.resolved_embed_provider == "cloud":
        total += embed_tokens / million * settings.price_embed_per_mtok
    if settings.resolved_chat_provider == "cloud":
        total += (
            tokens_in / million * settings.price_input_per_mtok
            + tokens_out / million * settings.price_output_per_mtok
        )
    return round(total, 6)
