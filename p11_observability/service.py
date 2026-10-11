"""Production wrapper: versioned agent + tracing + metrics + alerts + canary."""
from __future__ import annotations

import time
from dataclasses import dataclass

from common.llm import BaseLLM
from common.tools import Tool
from p03_react_planning.agent import ReActAgent, default_tools

from .ops import Alerter, DeploymentManager, MetricsStore
from .tracing import Tracer


@dataclass
class AgentVersion:
    name: str
    max_iterations: int = 8
    reasoning_effort: str | None = None
    self_critique: bool = True


def traced_tools(tracer: Tracer) -> list[Tool]:
    """Wrap every tool so each call becomes a 'tool' span."""
    out = []
    for t in default_tools():
        def make(fn, name):
            def wrapped(**kw):
                with tracer.span(f"tool.{name}", kind="tool", input=kw) as s:
                    res = fn(**kw)
                    s.set(output=str(res)[:500])
                    return res
            return wrapped
        out.append(Tool(t.name, t.description, make(t.func, t.name), t.args_model, t.output_model))
    return out


class ProductionAgent:
    def __init__(self, llm: BaseLLM, versions: dict[str, AgentVersion], tracer: Tracer | None = None,
                 metrics: MetricsStore | None = None, alerter: Alerter | None = None,
                 deployment: DeploymentManager | None = None):
        self.llm = llm
        self.versions = versions
        self.tracer = tracer or Tracer()
        self.metrics = metrics or MetricsStore()
        self.alerter = alerter or Alerter()
        self.deployment = deployment or DeploymentManager(stable=next(iter(versions)))

    def handle(self, user_id: str, query: str) -> dict:
        version = self.deployment.route(user_id)
        cfg = self.versions[version]
        t0 = time.perf_counter()
        status, answer, steps, reason, error = "failed", "", 0, "", None
        with self.tracer.span("agent.request", kind="agent", query=query, user=user_id, version=version) as root:
            try:
                agent = ReActAgent(self.llm, tools=traced_tools(self.tracer), max_iterations=cfg.max_iterations,
                                   self_critique=cfg.self_critique)
                res = agent.run(query)
                status, answer, steps, reason = res.status, res.answer, res.iterations, res.reason
                root.set(status=status, steps=steps, reason=reason, answer=answer[:500])
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                root.set(status="failed", error=error)
        latency = (time.perf_counter() - t0) * 1000
        record = {"trace_id": root.trace_id, "version": version, "user": user_id, "status": status,
                  "latency_ms": latency, "cost": root.attrs.get("cost_usd", 0.0), "tokens": root.attrs.get("tokens", 0),
                  "steps": steps, "quality": {"success": 1.0, "degraded": 0.4}.get(status, 0.0), "error": error,
                  "loop": "loop detected" in reason, "reason": reason}
        self.metrics.record(**record)
        alerts = self.alerter.evaluate(record, self.metrics.summary(version, last_n=20))
        return {**record, "answer": answer, "alerts": [a.rule for a in alerts]}
