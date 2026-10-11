"""Ops dashboard:  uvicorn p11_observability.dashboard:app --port 8011  -> http://localhost:8011/

GET  /             HTML dashboard (latency / cost / errors per version, alerts, deployment state)
GET  /metrics      JSON metrics          GET /traces/{trace_id}  spans of one request
POST /ask          {"user": "...", "query": "..."}
POST /canary/start?version=v2&percent=25   POST /canary/evaluate   POST /canary/rollback
"""
from __future__ import annotations

import html
import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from common import config
from common.llm import get_llm
from common.utils import read_jsonl

from .service import AgentVersion, ProductionAgent

prod = ProductionAgent(get_llm(), {"v1": AgentVersion("v1", max_iterations=8),
                                   "v2": AgentVersion("v2", max_iterations=2, self_critique=False)})
app = FastAPI(title="Agent Observability")


class Ask(BaseModel):
    user: str
    query: str


@app.post("/ask")
def ask(a: Ask):
    return prod.handle(a.user, a.query)


@app.get("/metrics")
def metrics():
    return {"overall": prod.metrics.summary(), **{v: prod.metrics.summary(v) for v in prod.versions},
            "timeseries": prod.metrics.timeseries(), "deployment": prod.deployment.state}


@app.get("/traces/{trace_id}")
def trace(trace_id: str):
    return [s for s in read_jsonl(config.DATA_DIR / "p11_traces.jsonl") if s["trace_id"] == trace_id]


@app.post("/canary/start")
def canary_start(version: str = "v2", percent: int = 25):
    prod.deployment.start_canary(version, percent)
    return prod.deployment.state


@app.post("/canary/evaluate")
def canary_eval():
    return {"decision": prod.deployment.evaluate(prod.metrics, alerter=prod.alerter), "state": prod.deployment.state}


@app.post("/canary/rollback")
def canary_rollback(reason: str = "manual"):
    prod.deployment.rollback(reason)
    return prod.deployment.state


def _bars(series: list[dict], key: str, color: str, fmt: str) -> str:
    if not series:
        return "<p class=muted>no data yet</p>"
    mx = max(s[key] for s in series) or 1
    w = 560 / len(series)
    bars = "".join(f'<rect x="{i * w + 2:.1f}" y="{120 - 110 * s[key] / mx:.1f}" width="{w - 4:.1f}" '
                   f'height="{110 * s[key] / mx:.1f}" fill="{color}"><title>{fmt.format(s[key])}</title></rect>'
                   for i, s in enumerate(series))
    return f'<svg viewBox="0 0 560 122" width="100%" height="122">{bars}</svg>'


@app.get("/", response_class=HTMLResponse)
def dashboard():
    m = metrics()
    ts = m["timeseries"]
    alerts = list(read_jsonl(config.DATA_DIR / "p11_alerts.jsonl"))[-10:][::-1]
    rows = "".join(
        f"<tr><td>{v}</td>" + "".join(f"<td>{m[v].get(k, '-')}</td>" for k in
                                      ("requests", "error_rate", "degraded_rate", "p50_latency_ms", "p95_latency_ms",
                                       "avg_cost_usd", "avg_tokens", "avg_steps", "avg_quality")) + "</tr>"
        for v in ["overall", *prod.versions])
    alert_rows = "".join(f"<li><b>[{html.escape(a['severity'])}] {html.escape(a['rule'])}</b> "
                         f"{html.escape(a['message'])}</li>" for a in alerts) or "<li class=muted>none</li>"
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Agent Ops</title><meta http-equiv=refresh content=10><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f5;color:#1d1d1b}}
table{{border-collapse:collapse;width:100%;background:#fff}}td,th{{border:1px solid #ddd;padding:6px 8px;
font-size:13px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px}}.muted{{color:#888}}
pre{{white-space:pre-wrap;font-size:12px}}</style></head><body>
<h1>Agent operations</h1>
<table><tr><th>version</th><th>requests</th><th>error rate</th><th>degraded</th><th>p50 ms</th><th>p95 ms</th>
<th>avg $</th><th>avg tokens</th><th>avg steps</th><th>quality</th></tr>{rows}</table>
<div class=grid style="margin-top:16px">
<div class=card><h3>Latency per minute</h3>{_bars(ts, 'avg_latency_ms', '#3b6ea8', '{:.0f} ms')}</div>
<div class=card><h3>Cost per minute</h3>{_bars(ts, 'cost_usd', '#2f8f5b', '${:.5f}')}</div>
<div class=card><h3>Errors per minute</h3>{_bars(ts, 'errors', '#c0392b', '{} errors')}</div>
<div class=card><h3>Recent alerts</h3><ul>{alert_rows}</ul></div>
<div class=card><h3>Deployment</h3><pre>{html.escape(json.dumps(m['deployment'], indent=1)[:1500])}</pre></div>
</div></body></html>"""
