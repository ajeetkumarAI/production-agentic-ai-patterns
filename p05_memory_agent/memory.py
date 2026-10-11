"""Project 5 - memory layer.

* ShortTermBuffer - recent turns kept verbatim, bounded by a token budget;
  overflow is *compressed* into a running summary by the LLM.
* MemoryStore     - long-term vector memory in SQLite (per user) with
  semantic de-duplication and relevance scoring
  (similarity x recency decay x importance x usage).
* sync()          - cross-session / cross-device sync between two stores
  (last-write-wins on id, semantic merge of near-duplicates).
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.embeddings import get_embedder
from common.llm import BaseLLM, estimate_tokens


# ------------------------------------------------------------------ short-term
@dataclass
class ShortTermBuffer:
    max_tokens: int = 1500
    keep_last: int = 4          # never compress the most recent N messages
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    compressions: int = 0

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def tokens(self) -> int:
        return estimate_tokens(self.messages) + estimate_tokens(self.summary)

    def needs_compression(self) -> bool:
        return self.tokens() > self.max_tokens and len(self.messages) > self.keep_last

    def compress(self, llm: BaseLLM) -> None:
        old, self.messages = self.messages[:-self.keep_last], self.messages[-self.keep_last:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        self.summary = llm.chat([
            {"role": "system", "content": "Update the running conversation summary. Keep names, numbers, decisions, "
                                          "open questions and user preferences. Max 120 words."},
            {"role": "user", "content": f"Current summary:\n{self.summary or '(empty)'}\n\nNew turns:\n{transcript}"},
        ], tag="p05.compress").content or self.summary
        self.compressions += 1


# ------------------------------------------------------------------ long-term
@dataclass
class Memory:
    id: str
    user_id: str
    text: str
    category: str
    importance: float
    created_at: float
    updated_at: float
    last_accessed: float
    access_count: int
    vector: np.ndarray
    score: float = 0.0


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL, category TEXT, importance REAL,
  created_at REAL, updated_at REAL, last_accessed REAL, access_count INTEGER DEFAULT 0,
  vector TEXT NOT NULL, deleted INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_mem_user ON memories(user_id);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, user_id TEXT, summary TEXT, started_at REAL, ended_at REAL);
"""


class MemoryStore:
    def __init__(self, path: str | Path, embedder=None, dedup_threshold: float = 0.88,
                 half_life_days: float = 30.0, weights: tuple[float, float, float, float] = (0.6, 0.15, 0.2, 0.05)):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.embedder = embedder or get_embedder()
        self.dedup_threshold = dedup_threshold
        self.half_life_days = half_life_days
        self.w_sim, self.w_recency, self.w_importance, self.w_usage = weights

    # ---------------------------------------------------------------- rows
    def _row(self, r) -> Memory:
        return Memory(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], np.array(json.loads(r[9]), dtype=np.float32))

    def all(self, user_id: str | None = None, include_deleted: bool = False) -> list[Memory]:
        q = "SELECT id,user_id,text,category,importance,created_at,updated_at,last_accessed,access_count,vector FROM memories"
        conds, args = [], []
        if user_id:
            conds.append("user_id=?")
            args.append(user_id)
        if not include_deleted:
            conds.append("deleted=0")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        return [self._row(r) for r in self.db.execute(q, args)]

    def _upsert(self, m: Memory, deleted: int = 0) -> None:
        self.db.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET text=excluded.text, "
            "category=excluded.category, importance=excluded.importance, updated_at=excluded.updated_at, "
            "last_accessed=excluded.last_accessed, access_count=excluded.access_count, vector=excluded.vector, "
            "deleted=excluded.deleted",
            (m.id, m.user_id, m.text, m.category, m.importance, m.created_at, m.updated_at, m.last_accessed,
             m.access_count, json.dumps([round(float(x), 5) for x in m.vector]), deleted))
        self.db.commit()

    # ---------------------------------------------------------------- write
    def add(self, user_id: str, text: str, category: str = "fact", importance: float = 0.5,
            now: float | None = None) -> tuple[str, str]:
        """Returns (memory_id, 'created' | 'merged')."""
        now = now or time.time()
        vec = self.embedder.embed([text])[0]
        best, best_sim = None, 0.0
        for m in self.all(user_id):
            sim = float(m.vector @ vec)
            if sim > best_sim:
                best, best_sim = m, sim
        if best is not None and best_sim >= self.dedup_threshold:
            best.text, best.vector = text, vec  # newer phrasing wins (e.g. updated preference)
            best.importance = max(best.importance, importance)
            best.updated_at = now
            self._upsert(best)
            return best.id, "merged"
        m = Memory(uuid.uuid4().hex, user_id, text, category, importance, now, now, now, 0, vec)
        self._upsert(m)
        return m.id, "created"

    def forget(self, memory_id: str) -> None:
        # soft delete (tombstone) so the deletion also syncs to other devices
        self.db.execute("UPDATE memories SET deleted=1, updated_at=? WHERE id=?", (time.time(), memory_id))
        self.db.commit()

    # ---------------------------------------------------------------- read
    def relevance(self, m: Memory, qvec: np.ndarray, now: float) -> float:
        sim = max(0.0, float(m.vector @ qvec))
        age_days = max(0.0, (now - m.updated_at) / 86400)
        recency = math.exp(-math.log(2) * age_days / self.half_life_days)
        usage = 1 - math.exp(-m.access_count / 5)
        return self.w_sim * sim + self.w_recency * recency + self.w_importance * m.importance + self.w_usage * usage

    def recall(self, user_id: str, query: str, k: int = 5, min_similarity: float = 0.05,
               now: float | None = None) -> list[Memory]:
        now = now or time.time()
        qvec = self.embedder.embed([query])[0]
        scored = []
        for m in self.all(user_id):
            if float(m.vector @ qvec) < min_similarity:
                continue
            m.score = self.relevance(m, qvec, now)
            scored.append(m)
        top = sorted(scored, key=lambda m: -m.score)[:k]
        for m in top:
            self.db.execute("UPDATE memories SET last_accessed=?, access_count=access_count+1 WHERE id=?", (now, m.id))
        self.db.commit()
        return top

    # ---------------------------------------------------------------- sessions
    def save_session(self, session_id: str, user_id: str, summary: str, started_at: float) -> None:
        self.db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?)",
                        (session_id, user_id, summary, started_at, time.time()))
        self.db.commit()

    def last_session_summary(self, user_id: str, exclude: str | None = None) -> str:
        r = self.db.execute("SELECT summary FROM sessions WHERE user_id=? AND session_id!=? ORDER BY ended_at DESC LIMIT 1",
                            (user_id, exclude or "")).fetchone()
        return r[0] if r else ""


def sync(a: MemoryStore, b: MemoryStore) -> dict:
    """Bidirectional sync of memories + session summaries (last-write-wins by updated_at)."""
    stats = {"a_to_b": 0, "b_to_a": 0}

    def rows(s: MemoryStore):
        return {m.id: (m, d) for m, d in (
            (s._row(r[:10]), r[10]) for r in s.db.execute(
                "SELECT id,user_id,text,category,importance,created_at,updated_at,last_accessed,access_count,vector,deleted "
                "FROM memories"))}

    ra, rb = rows(a), rows(b)
    for mid in set(ra) | set(rb):
        ma, mb = ra.get(mid), rb.get(mid)
        if mb is None or (ma and ma[0].updated_at > mb[0].updated_at):
            b._upsert(ma[0], ma[1])
            stats["a_to_b"] += 1
        elif ma is None or mb[0].updated_at > ma[0].updated_at:
            a._upsert(mb[0], mb[1])
            stats["b_to_a"] += 1
    for src, dst in ((a, b), (b, a)):
        for row in src.db.execute("SELECT * FROM sessions").fetchall():
            dst.db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?)", row)
        dst.db.commit()
    return stats
