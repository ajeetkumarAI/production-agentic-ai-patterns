"""Durable SQLite job queue with idempotent enqueue, leases, retries with
exponential backoff + jitter, and a dead-letter state with replay."""
from __future__ import annotations

import json
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, event_id TEXT UNIQUE NOT NULL, source TEXT, type TEXT, payload TEXT,
  status TEXT NOT NULL DEFAULT 'queued',   -- queued | running | done | dead
  attempts INTEGER DEFAULT 0, max_attempts INTEGER DEFAULT 5, next_run_at REAL, lease_until REAL,
  last_error TEXT, created REAL, updated REAL);
CREATE INDEX IF NOT EXISTS ix_jobs_ready ON jobs(status, next_run_at);
CREATE TABLE IF NOT EXISTS step_runs (
  event_id TEXT, step TEXT, result TEXT, finished REAL, PRIMARY KEY (event_id, step));
"""


@dataclass
class Job:
    id: str
    event_id: str
    source: str
    type: str
    payload: dict
    attempts: int
    max_attempts: int


class JobQueue:
    def __init__(self, path: str | Path, base_backoff_s: float = 1.0, max_backoff_s: float = 300.0,
                 lease_s: float = 60.0, max_attempts: int = 5):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)  # autocommit
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()
        self.base_backoff_s, self.max_backoff_s = base_backoff_s, max_backoff_s
        self.lease_s, self.max_attempts = lease_s, max_attempts

    # ---------------------------------------------------------------- producer
    def enqueue(self, event_id: str, type_: str, payload: dict, source: str = "internal") -> tuple[str, bool]:
        """Idempotent: the same event_id is accepted only once. Returns (job_id, duplicate)."""
        now = time.time()
        jid = uuid.uuid4().hex[:12]
        with self.lock:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO jobs (id,event_id,source,type,payload,max_attempts,next_run_at,created,updated) "
                "VALUES (?,?,?,?,?,?,?,?,?)", (jid, event_id, source, type_, json.dumps(payload), self.max_attempts,
                                              now, now, now))
            if cur.rowcount == 0:
                existing = self.db.execute("SELECT id FROM jobs WHERE event_id=?", (event_id,)).fetchone()
                return existing[0], True
        return jid, False

    # ---------------------------------------------------------------- consumer
    def claim(self) -> Job | None:
        """Atomically lease the next ready job. Expired leases (crashed worker) are reclaimed."""
        now = time.time()
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                row = self.db.execute(
                    "SELECT id,event_id,source,type,payload,attempts,max_attempts FROM jobs "
                    "WHERE (status='queued' AND next_run_at<=?) OR (status='running' AND lease_until<?) "
                    "ORDER BY next_run_at LIMIT 1", (now, now)).fetchone()
                if row:
                    self.db.execute("UPDATE jobs SET status='running', lease_until=?, updated=? WHERE id=?",
                                    (now + self.lease_s, now, row[0]))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
        if not row:
            return None
        return Job(row[0], row[1], row[2], row[3], json.loads(row[4]), row[5], row[6])

    def complete(self, job: Job) -> None:
        with self.lock:
            self.db.execute("UPDATE jobs SET status='done', updated=?, last_error=NULL WHERE id=?", (time.time(), job.id))

    def backoff(self, attempts: int) -> float:
        delay = min(self.max_backoff_s, self.base_backoff_s * 2 ** (attempts - 1))
        return delay * (0.5 + random.random() / 2)  # "equal jitter"

    def fail(self, job: Job, error: str, permanent: bool = False) -> str:
        attempts = job.attempts + 1
        now = time.time()
        dead = permanent or attempts >= job.max_attempts
        with self.lock:
            self.db.execute("UPDATE jobs SET status=?, attempts=?, next_run_at=?, last_error=?, updated=? WHERE id=?",
                            ("dead" if dead else "queued", attempts, now + (0 if dead else self.backoff(attempts)),
                             error[:2000], now, job.id))
        return "dead" if dead else "retry"

    # ---------------------------------------------------------------- idempotent steps
    def step_result(self, event_id: str, step: str) -> tuple[bool, Any]:
        with self.lock:
            row = self.db.execute("SELECT result FROM step_runs WHERE event_id=? AND step=?",
                                  (event_id, step)).fetchone()
        return (True, json.loads(row[0])) if row else (False, None)

    def record_step(self, event_id: str, step: str, result: Any) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO step_runs VALUES (?,?,?,?)",
                            (event_id, step, json.dumps(result, default=str), time.time()))

    # ---------------------------------------------------------------- ops
    def dead_letters(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT id,event_id,type,attempts,last_error FROM jobs WHERE status='dead' "
                                   "ORDER BY updated").fetchall()
        return [dict(zip(("id", "event_id", "type", "attempts", "last_error"), r)) for r in rows]

    def replay(self, job_id: str) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE jobs SET status='queued', attempts=0, next_run_at=?, updated=? "
                                  "WHERE id=? AND status='dead'", (time.time(), time.time(), job_id))
        return cur.rowcount == 1

    def stats(self) -> dict:
        with self.lock:
            return {s: n for s, n in self.db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()}

    def jobs(self, limit: int = 50) -> list[dict]:
        cols = ("id", "event_id", "type", "status", "attempts", "last_error")
        with self.lock:
            rows = self.db.execute(f"SELECT {','.join(cols)} FROM jobs ORDER BY created DESC LIMIT ?",
                                   (limit,)).fetchall()
        return [dict(zip(cols, r)) for r in rows]

    def pending(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
