"""Lightweight tracing (OpenTelemetry-style spans) with pluggable exporters.

* JsonlExporter     - always on, writes data/p11_traces.jsonl
* LangSmithExporter - enabled automatically when LANGSMITH_API_KEY is set and
                      the ``langsmith`` package is installed
Every LLM call made while a span is active is recorded as a child "llm" span
with model, tokens, cost and latency (via common.llm listeners).
"""
from __future__ import annotations

import contextvars
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import config
from common import llm as llm_mod
from common.utils import append_jsonl

_current: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("current_span", default=None)


@dataclass
class Span:
    name: str
    kind: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.time()) - self.start) * 1000

    def set(self, **attrs) -> None:
        self.attrs.update(attrs)


class JsonlExporter:
    def __init__(self, path: Path = config.DATA_DIR / "p11_traces.jsonl"):
        self.path = path

    def export(self, span: Span) -> None:
        append_jsonl(self.path, {**asdict(span), "duration_ms": round(span.duration_ms, 2)})


class LangSmithExporter:
    """Sends spans as LangSmith runs (parent/child preserved)."""

    def __init__(self, project: str | None = None):
        from langsmith import Client  # optional dependency

        self.client = Client()
        self.project = project or os.getenv("LANGSMITH_PROJECT", "genai-agentic-projects")
        self._ids: dict[str, str] = {}
        self._buffer: dict[str, list[Span]] = {}

    def _rid(self, sid: str) -> str:
        return self._ids.setdefault(sid, str(uuid.uuid4()))

    def export(self, span: Span) -> None:
        # buffer until the root span ends, then send parents before children
        self._buffer.setdefault(span.trace_id, []).append(span)
        if span.parent_id is None:
            for sp in sorted(self._buffer.pop(span.trace_id), key=lambda x: x.start):
                self._send(sp)

    def _send(self, span: Span) -> None:
        run_type = {"llm": "llm", "tool": "tool"}.get(span.kind, "chain")
        try:
            self.client.create_run(
                id=self._rid(span.span_id), name=span.name, run_type=run_type, project_name=self.project,
                parent_run_id=self._rid(span.parent_id) if span.parent_id else None,
                inputs={k: v for k, v in span.attrs.items() if k in ("input", "query", "messages")} or {"_": ""},
                outputs={k: v for k, v in span.attrs.items() if k not in ("input", "query", "messages")},
                start_time=datetime.fromtimestamp(span.start, timezone.utc),
                end_time=datetime.fromtimestamp(span.end or time.time(), timezone.utc),
                error=span.error)
        except Exception as e:  # noqa: BLE001 - observability must never break the app
            print(f"[langsmith] export failed: {e}")


class Tracer:
    def __init__(self, exporters: list | None = None):
        self.exporters = exporters if exporters is not None else default_exporters()
        llm_mod.add_listener(self._on_llm)

    def _emit(self, span: Span) -> None:
        for ex in self.exporters:
            ex.export(span)

    def span(self, name: str, kind: str = "internal", **attrs) -> "_SpanCtx":
        return _SpanCtx(self, name, kind, attrs)

    def _on_llm(self, usage, messages, resp) -> None:
        parent = _current.get()
        if parent is None:
            return
        now = time.time()
        s = Span(f"llm.{usage.tag or usage.model}", "llm", parent.trace_id, parent_id=parent.span_id,
                 start=now - usage.latency_ms / 1000, end=now,
                 attrs={"model": usage.model, "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens, "reasoning_tokens": usage.reasoning_tokens,
                        "cost_usd": round(usage.cost_usd, 8), "finish_reason": resp.finish_reason})
        # roll usage up to every ancestor that tracks totals
        parent.attrs["llm_calls"] = parent.attrs.get("llm_calls", 0) + 1
        parent.attrs["tokens"] = parent.attrs.get("tokens", 0) + usage.total_tokens
        parent.attrs["cost_usd"] = round(parent.attrs.get("cost_usd", 0.0) + usage.cost_usd, 8)
        self._emit(s)

    def close(self) -> None:
        llm_mod.remove_listener(self._on_llm)


class _SpanCtx:
    def __init__(self, tracer: Tracer, name: str, kind: str, attrs: dict):
        self.tracer, self.name, self.kind, self.attrs = tracer, name, kind, attrs

    def __enter__(self) -> Span:
        parent = _current.get()
        self.span = Span(self.name, self.kind, parent.trace_id if parent else uuid.uuid4().hex[:16],
                         parent_id=parent.span_id if parent else None, attrs=dict(self.attrs))
        self.token = _current.set(self.span)
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.span.end = time.time()
        if exc is not None:
            self.span.status, self.span.error = "error", f"{exc_type.__name__}: {exc}"
        _current.reset(self.token)
        parent = _current.get()
        if parent is not None:  # propagate totals upward
            for k in ("tokens", "cost_usd", "llm_calls"):
                if k in self.span.attrs:
                    parent.attrs[k] = round(parent.attrs.get(k, 0) + self.span.attrs[k], 8)
        self.tracer._emit(self.span)
        return False


def default_exporters() -> list:
    exps: list = [JsonlExporter()]
    if os.getenv("LANGSMITH_API_KEY"):
        try:
            exps.append(LangSmithExporter())
        except ImportError:
            print("[tracing] LANGSMITH_API_KEY set but `pip install langsmith` missing - JSONL only")
    return exps
