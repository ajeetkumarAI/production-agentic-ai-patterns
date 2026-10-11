"""Workflows triggered by events, and the worker that executes them.

Each workflow is a list of named steps run through ``StepContext.step`` which
records completed steps - when a job is retried after a failure, completed
steps are *skipped* (their stored result is reused), so side effects such as
"open case" or "send SMS" happen exactly once per event.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from common.llm import BaseLLM
from common.structured import generate_structured
from common.utils import append_jsonl
from common import config

from .queue_store import Job, JobQueue

EVENT_LOG = config.DATA_DIR / "p08_events.jsonl"


class PermanentError(Exception):
    """Non-retryable (bad payload, business rule) -> straight to dead-letter."""


class StepContext:
    def __init__(self, queue: JobQueue, job: Job):
        self.queue, self.job = queue, job
        self.results: dict[str, Any] = {}

    def step(self, name: str, fn: Callable[[], Any]) -> Any:
        done, result = self.queue.step_result(self.job.event_id, name)
        if done:
            self.results[name] = result
            append_jsonl(EVENT_LOG, {"event_id": self.job.event_id, "step": name, "status": "skipped_already_done"})
            return result
        result = fn()
        self.queue.record_step(self.job.event_id, name, result)
        self.results[name] = result
        append_jsonl(EVENT_LOG, {"event_id": self.job.event_id, "step": name, "status": "done"})
        return result


Workflow = Callable[[StepContext, dict], Any]


# ------------------------------------------------------------------ example workflow
class TransactionEvent(BaseModel):
    txn_id: str
    customer_id: str
    amount: float = Field(gt=0)
    currency: str = "INR"
    merchant: str
    ip_country: str = "IN"


class RiskVerdict(BaseModel):
    risk: Literal["low", "medium", "high"]
    reason: str


@dataclass
class Integrations:
    """External side effects. ``flaky_failures`` simulates a notification
    provider that is down for the first N calls (to exercise retries)."""
    cases: list[dict] = field(default_factory=list)
    notifications: list[dict] = field(default_factory=list)
    flaky_failures: int = 0
    _calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def open_case(self, txn: TransactionEvent, verdict: RiskVerdict) -> dict:
        case = {"case_id": f"CASE-{len(self.cases) + 1}", "txn_id": txn.txn_id, "risk": verdict.risk}
        self.cases.append(case)
        return case

    def notify(self, customer_id: str, text: str) -> dict:
        with self._lock:
            self._calls += 1
            if self._calls <= self.flaky_failures:
                raise ConnectionError(f"SMS gateway timeout (call {self._calls})")
            self.notifications.append({"customer_id": customer_id, "text": text})
            return {"delivered": True}


def make_transaction_workflow(llm: BaseLLM, integrations: Integrations) -> Workflow:
    def workflow(ctx: StepContext, payload: dict) -> dict:
        try:
            txn = TransactionEvent.model_validate(payload)
        except ValidationError as e:
            raise PermanentError(f"invalid payload: {e.errors()[:2]}") from e

        verdict = RiskVerdict.model_validate(ctx.step("score_risk", lambda: generate_structured(llm, [
            {"role": "system", "content": "Classify fraud risk of this card transaction. high = likely fraud."},
            {"role": "user", "content": txn.model_dump_json()}], RiskVerdict, tag="p08.score").value.model_dump()))

        if verdict.risk == "high" or txn.amount >= 50_000:
            ctx.step("open_case", lambda: integrations.open_case(txn, verdict))
            ctx.step("notify_customer", lambda: integrations.notify(
                txn.customer_id, f"We paused a {txn.currency} {txn.amount:,.0f} payment at {txn.merchant}. "
                                 "Reply YES if this was you."))
        return {"risk": verdict.risk, "steps": list(ctx.results)}
    return workflow


# ------------------------------------------------------------------ worker
class Worker:
    def __init__(self, queue: JobQueue, workflows: dict[str, Workflow], poll_s: float = 0.2):
        self.queue, self.workflows, self.poll_s = queue, workflows, poll_s
        self._stop = threading.Event()
        self.processed = 0

    def run_once(self) -> bool:
        job = self.queue.claim()
        if job is None:
            return False
        wf = self.workflows.get(job.type)
        try:
            if wf is None:
                raise PermanentError(f"no workflow registered for event type '{job.type}'")
            result = wf(StepContext(self.queue, job), job.payload)
            self.queue.complete(job)
            append_jsonl(EVENT_LOG, {"event_id": job.event_id, "status": "completed", "result": result})
        except PermanentError as e:
            self.queue.fail(job, str(e), permanent=True)
            append_jsonl(EVENT_LOG, {"event_id": job.event_id, "status": "dead_letter", "error": str(e)})
        except Exception as e:  # noqa: BLE001 - transient by default -> retry with backoff
            outcome = self.queue.fail(job, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}")
            append_jsonl(EVENT_LOG, {"event_id": job.event_id, "status": outcome, "attempt": job.attempts + 1,
                                     "error": f"{type(e).__name__}: {e}"})
        self.processed += 1
        return True

    def run_until_idle(self, timeout_s: float = 60.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not self.run_once():
                if self.queue.pending() == 0:
                    return
                time.sleep(self.poll_s)

    def start(self, n_threads: int = 2) -> list[threading.Thread]:
        def loop():
            while not self._stop.is_set():
                if not self.run_once():
                    self._stop.wait(self.poll_s)
        threads = [threading.Thread(target=loop, daemon=True, name=f"worker-{i}") for i in range(n_threads)]
        for t in threads:
            t.start()
        return threads

    def stop(self) -> None:
        self._stop.set()


def event_id_for(payload: dict, header_key: str | None) -> str:
    """Idempotency key: explicit header > provider event id > content hash."""
    if header_key:
        return header_key
    if payload.get("id"):
        return str(payload["id"])
    import hashlib
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
