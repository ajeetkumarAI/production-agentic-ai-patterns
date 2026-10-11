"""Project 5 - Memory-Enabled Conversational Agent."""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from common.llm import BaseLLM
from common.structured import StructuredOutputError, generate_structured

from .memory import MemoryStore, ShortTermBuffer


class MemoryItem(BaseModel):
    text: str = Field(description="Self-contained fact about the user, third person, e.g. 'User prefers Python'")
    category: Literal["preference", "fact", "goal", "relationship", "other"]
    importance: float = Field(ge=0, le=1)


class MemoryExtraction(BaseModel):
    items: list[MemoryItem] = Field(default_factory=list, max_length=5)


EXTRACT_SYSTEM = """Extract durable facts worth remembering about the USER from their latest message
(preferences, goals, personal facts, projects). Ignore small talk, questions and one-off requests.
Return an empty list if nothing is worth remembering."""


class MemoryAgent:
    def __init__(self, llm: BaseLLM, store: MemoryStore, user_id: str, session_id: str | None = None,
                 buffer_tokens: int = 1500, recall_k: int = 5, extract: bool = True):
        self.llm = llm
        self.store = store
        self.user_id = user_id
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.started_at = time.time()
        self.buffer = ShortTermBuffer(max_tokens=buffer_tokens)
        self.recall_k = recall_k
        self.extract = extract
        self.previous_session = store.last_session_summary(user_id, exclude=self.session_id)
        self.last_recalled: list = []

    def _system_prompt(self, recalled) -> str:
        parts = ["You are a helpful assistant with long-term memory. Use remembered facts naturally when relevant; "
                 "don't recite them. If a remembered fact conflicts with what the user says now, trust the user."]
        if recalled:
            parts.append("Relevant memories about the user:\n" + "\n".join(f"- {m.text}" for m in recalled))
        if self.previous_session:
            parts.append(f"Summary of your previous session with this user:\n{self.previous_session}")
        if self.buffer.summary:
            parts.append(f"Summary of earlier parts of this conversation:\n{self.buffer.summary}")
        return "\n\n".join(parts)

    def _extract(self, message: str) -> list[tuple[str, str]]:
        try:
            items = generate_structured(self.llm, [
                {"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": message}],
                MemoryExtraction, max_retries=1, tag="p05.extract").value.items
        except StructuredOutputError:
            return []
        return [self.store.add(self.user_id, i.text, i.category, i.importance) for i in items]

    def chat(self, message: str) -> str:
        self.last_recalled = self.store.recall(self.user_id, message, k=self.recall_k)
        msgs = [{"role": "system", "content": self._system_prompt(self.last_recalled)}, *self.buffer.messages,
                {"role": "user", "content": message}]
        reply = self.llm.chat(msgs, tag="p05.reply").content or ""
        self.buffer.add("user", message)
        self.buffer.add("assistant", reply)
        if self.buffer.needs_compression():
            self.buffer.compress(self.llm)
        if self.extract:
            self._extract(message)
        return reply

    def end_session(self) -> str:
        """Persist a session summary so the next session (any device after sync) can pick up."""
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in self.buffer.messages)
        summary = self.llm.chat([
            {"role": "system", "content": "Summarize this session in <=80 words: topics, decisions, open items."},
            {"role": "user", "content": f"Earlier summary: {self.buffer.summary}\n\n{transcript}"},
        ], tag="p05.session_summary").content or ""
        self.store.save_session(self.session_id, self.user_id, summary, self.started_at)
        return summary
