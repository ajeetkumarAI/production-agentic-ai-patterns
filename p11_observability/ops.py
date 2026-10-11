"""Metrics store, alerting rules and canary deployment manager."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from common import config
from common.utils import append_jsonl


# ------------------------------------------------------------------ metrics
class MetricsStore:
    def __init__(self, path: str | Path = config.DATA_DIR / "p11_metrics.db"):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS requests (trace_id TEXT, ts REAL, version TEXT, user TEXT, "
                        "status TEXT, latency_ms REAL, cost REAL, tokens INT, steps INT, quality REAL, error TEXT)")
        self.db.commit()

    def record(self, **r) -> None:
        with self.lock:
            self.db.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (r["trace_id"], time.time(), r["version"], r["user"], r["status"], r["latency_ms"],
                             r["cost"], r["tokens"], r["steps"], r["quality"], r.get("error")))
            self.db.commit()

    def summary(self, version: str | None = None, last_n: int | None = None, since: float | None = None) -> dict:
        q, args = "SELECT status, latency_ms, cost, tokens, steps, quality FROM requests WHERE 1=1", []
        if version:
            q += " AND version=?"
            args.append(version)
        if since:
            q += " AND ts>=?"
            args.append(since)
        q += " ORDER BY ts DESC"
        if last_n:
            q += f" LIMIT {int(last_n)}"
        with self.lock:
            rows = self.db.execute(q, args).fetchall()
        n = len(rows)
        if not n:
            return {"requests": 0}
        lat = sorted(r[1] for r in rows)
        pct = lambda p: round(lat[min(n - 1, int(p * n))], 1)  # noqa: E731
        return {"requests": n, "error_rate": round(sum(r[0] == "failed" for r in rows) / n, 3),
                "degraded_rate": round(sum(r[0] == "degraded" for r in rows) / n, 3),
                "p50_latency_ms": pct(0.5), "p95_latency_ms": pct(0.95),
                "avg_cost_usd": round(sum(r[2] for r in rows) / n, 7), "total_cost_usd": round(sum(r[2] for r in rows), 6),
                "avg_tokens": round(sum(r[3] for r in rows) / n), "avg_steps": round(sum(r[4] for r in rows) / n, 2),
                "avg_quality": round(sum(r[5] for r in rows) / n, 3)}

    def timeseries(self, bucket_s: int = 60, limit: int = 30) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT CAST(ts / ? AS INT) * ? AS b, COUNT(*), AVG(latency_ms), SUM(cost), "
                "SUM(status='failed') FROM requests GROUP BY b ORDER BY b DESC LIMIT ?", (bucket_s, bucket_s, limit)).fetchall()
        return [{"t": r[0], "requests": r[1], "avg_latency_ms": round(r[2], 1), "cost_usd": round(r[3], 6),
                 "errors": r[4]} for r in reversed(rows)]


# ------------------------------------------------------------------ alerting
@dataclass
class Alert:
    rule: str
    severity: str
    message: str
    ts: float = field(default_factory=time.time)


@dataclass
class AlertRule:
    name: str
    severity: str
    check: Callable[[dict, dict], str | None]   # (request, window_summary) -> message or None
    cooldown_s: float = 300


def default_rules(max_steps: int = 6, max_error_rate: float = 0.2, max_p95_ms: float = 30_000,
                  max_cost: float = 0.01) -> list[AlertRule]:
    return [
        AlertRule("agent_loop", "critical", lambda r, w: (
            f"trace {r['trace_id']}: {r.get('reason')}" if r.get("loop") or r["steps"] > max_steps else None), 0),
        AlertRule("error_rate", "critical", lambda r, w: (
            f"error rate {w['error_rate']:.0%} over last {w['requests']} requests"
            if w.get("requests", 0) >= 5 and w["error_rate"] > max_error_rate else None)),
        AlertRule("latency_p95", "warning", lambda r, w: (
            f"p95 latency {w['p95_latency_ms']:.0f}ms > {max_p95_ms:.0f}ms"
            if w.get("requests", 0) >= 5 and w["p95_latency_ms"] > max_p95_ms else None)),
        AlertRule("cost_spike", "warning", lambda r, w: (
            f"request cost ${r['cost']:.5f} > ${max_cost}" if r["cost"] > max_cost else None), 60),
    ]


class Alerter:
    def __init__(self, rules: list[AlertRule] | None = None, log: Path = config.DATA_DIR / "p11_alerts.jsonl",
                 webhook_url: str | None = None, echo: bool = True):
        self.rules = rules or default_rules()
        self.log = log
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")  # e.g. Slack incoming webhook
        self.echo = echo
        self._last: dict[str, float] = {}
        self.fired: list[Alert] = []

    def evaluate(self, request: dict, window: dict) -> list[Alert]:
        out = []
        for rule in self.rules:
            try:
                msg = rule.check(request, window)
            except Exception:  # noqa: BLE001
                continue
            if msg and time.time() - self._last.get(rule.name, 0) >= rule.cooldown_s:
                self._last[rule.name] = time.time()
                out.append(self.fire(rule.name, rule.severity, msg))
        return out

    def fire(self, rule: str, severity: str, message: str) -> Alert:
        a = Alert(rule, severity, message)
        self.fired.append(a)
        append_jsonl(self.log, a.__dict__)
        if self.echo:
            print(f"  !! ALERT [{severity}] {rule}: {message}")
        if self.webhook_url:
            try:
                req = urllib.request.Request(self.webhook_url, data=json.dumps(
                    {"text": f"[{severity}] {rule}: {message}"}).encode(), headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:  # noqa: BLE001
                print(f"  (alert webhook failed: {e})")
        return a


# ------------------------------------------------------------------ canary deployments
class DeploymentManager:
    """Canary rollout with automatic promotion or rollback.
    State is a JSON file so it survives restarts and can be edited by ops."""

    STEPS = [10, 25, 50, 100]

    def __init__(self, path: Path = config.DATA_DIR / "p11_deployment.json", stable: str = "v1"):
        self.path = path
        self.state = json.loads(path.read_text()) if path.exists() else {
            "stable": stable, "canary": None, "canary_percent": 0, "status": "stable", "history": []}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=1))

    def _event(self, kind: str, **data) -> None:
        self.state["history"].append({"ts": time.time(), "event": kind, **data})
        self._save()

    def start_canary(self, version: str, percent: int = 10) -> None:
        self.state.update(canary=version, canary_percent=percent, status="canary", canary_started=time.time())
        self._event("canary_started", version=version, percent=percent)

    def route(self, user_id: str) -> str:
        """Sticky routing: a user always lands in the same bucket (hash of user id)."""
        if not self.state["canary"] or self.state["canary_percent"] <= 0:
            return self.state["stable"]
        bucket = int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 100
        return self.state["canary"] if bucket < self.state["canary_percent"] else self.state["stable"]

    def rollback(self, reason: str) -> None:
        bad = self.state["canary"]
        self.state.update(canary=None, canary_percent=0, status="rolled_back")
        self._event("rollback", version=bad, reason=reason)

    def evaluate(self, metrics: MetricsStore, min_requests: int = 8, max_error_delta: float = 0.1,
                 max_quality_drop: float = 0.15, max_latency_ratio: float = 1.5, alerter: Alerter | None = None) -> str:
        """Compare canary vs stable since the canary started -> 'rollback' | 'promote' | 'wait'."""
        canary = self.state["canary"]
        if not canary:
            return "no_canary"
        since = self.state.get("canary_started")
        c = metrics.summary(canary, since=since)
        s = metrics.summary(self.state["stable"], since=since)
        if c.get("requests", 0) < min_requests or s.get("requests", 0) < min_requests:
            return "wait"
        reasons = []
        if c["error_rate"] - s["error_rate"] > max_error_delta:
            reasons.append(f"error rate {c['error_rate']:.0%} vs {s['error_rate']:.0%}")
        if s["avg_quality"] - c["avg_quality"] > max_quality_drop:
            reasons.append(f"quality {c['avg_quality']:.2f} vs {s['avg_quality']:.2f}")
        if s["p95_latency_ms"] and c["p95_latency_ms"] > max_latency_ratio * s["p95_latency_ms"]:
            reasons.append(f"p95 {c['p95_latency_ms']:.0f}ms vs {s['p95_latency_ms']:.0f}ms")
        if reasons:
            self.rollback("; ".join(reasons))
            if alerter:
                alerter.fire("canary_rollback", "critical", f"{canary} rolled back: {'; '.join(reasons)}")
            return "rollback"
        nxt = next((p for p in self.STEPS if p > self.state["canary_percent"]), None)
        if nxt is None or nxt >= 100:
            self.state.update(stable=canary, canary=None, canary_percent=0, status="stable")
            self._event("promoted", version=canary)
        else:
            self.state["canary_percent"] = nxt
            self.state["canary_started"] = time.time()
            self._event("canary_increased", version=canary, percent=nxt)
        return "promote"
