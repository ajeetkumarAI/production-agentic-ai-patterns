"""Project 2 - RAG Agent with Citation Grounding.

Pipeline: retrieve -> generate answer with [S#] citations + verbatim quotes ->
verify every citation against the retrieved text -> score confidence ->
flag low confidence -> fall back to web search when the knowledge base can't answer.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from common import config
from common.embeddings import VectorStore
from common.llm import BaseLLM
from common.structured import StructuredOutputError, generate_structured

DOCS_DIR = Path(__file__).parent / "docs"


# ------------------------------------------------------------------ ingestion
@dataclass
class Chunk:
    id: str
    source: str
    text: str


def chunk_markdown(path: Path, max_chars: int = 300) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title = next((l.lstrip("# ").strip() for l in raw.splitlines() if l.startswith("#")), path.stem)
    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip() and not p.strip().startswith("#")]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n{p}".strip()
    if buf:
        chunks.append(buf)
    return [Chunk(id=f"{path.stem}#{i}", source=f"{path.name} ({title})", text=" ".join(c.split()))
            for i, c in enumerate(chunks)]


def build_index(docs_dir: Path = DOCS_DIR, store: VectorStore | None = None) -> VectorStore:
    store = store or VectorStore()
    chunks = [c for f in sorted(docs_dir.glob("*.md")) for c in chunk_markdown(f)]
    store.add([c.text for c in chunks], [{"id": c.id, "source": c.source} for c in chunks])
    return store


# ------------------------------------------------------------------ schemas
class Citation(BaseModel):
    source_id: str = Field(description="The S-number label, e.g. 'S1'")
    quote: str = Field(min_length=3, description="Exact short quote copied from that source")


class GroundedAnswer(BaseModel):
    answer: str = Field(description="Answer where every sentence ends with citation markers like [S1]")
    citations: list[Citation]
    confidence: float = Field(ge=0, le=1, description="Your confidence the sources fully answer the question")
    answerable: bool = Field(description="False if the sources do not contain the answer")


@dataclass
class RAGResult:
    question: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    low_confidence: bool = True
    source: str = "knowledge_base"          # knowledge_base | web_search | none
    grounding: dict = field(default_factory=dict)
    retrieved: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------ fallback search
class SearchFallback(Protocol):
    def search(self, question: str) -> tuple[str, list[str]] | None: ...


class OpenAIWebSearch:
    """Uses the OpenAI Responses API web_search tool. Returns (answer, urls)."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("WEB_SEARCH_MODEL", config.DEFAULT_MODEL)

    def search(self, question: str):
        if not config.use_real_llm():
            return None
        try:
            from openai import OpenAI

            r = OpenAI(api_key=config.OPENAI_API_KEY).responses.create(
                model=self.model, tools=[{"type": "web_search"}], input=question)
            urls = []
            for item in r.output:
                for c in getattr(item, "content", None) or []:
                    for a in getattr(c, "annotations", None) or []:
                        if getattr(a, "url", None):
                            urls.append(a.url)
            return r.output_text, sorted(set(urls))
        except Exception as e:  # noqa: BLE001
            print(f"[p02] web search fallback unavailable: {e}")
            return None


# ------------------------------------------------------------------ agent
SYSTEM = """Answer ONLY from the numbered sources. Rules:
- End every sentence with the citation marker(s) of the source(s) supporting it, e.g. [S2].
- For each citation give a short EXACT quote copied from that source.
- If the sources don't contain the answer, set answerable=false and say so. Never use outside knowledge."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


class RAGAgent:
    def __init__(self, llm: BaseLLM, store: VectorStore | None = None, k: int = 4,
                 min_retrieval_score: float = 0.10, good_retrieval_score: float = 0.35,
                 confidence_threshold: float = 0.6, fallback: SearchFallback | None = None):
        self.llm = llm
        self.store = store or build_index()
        self.k = k
        self.min_retrieval_score = min_retrieval_score
        self.good_retrieval_score = good_retrieval_score
        self.confidence_threshold = confidence_threshold
        self.fallback = fallback

    def retrieve(self, question: str) -> list[dict]:
        hits = self.store.search(question, self.k)
        return [{"label": f"S{i + 1}", "score": round(s, 3), "text": t, **m} for i, (s, t, m) in enumerate(hits)]

    def verify(self, ans: GroundedAnswer, retrieved: list[dict]) -> dict:
        by_label = {r["label"]: r for r in retrieved}
        valid, invalid = [], []
        for c in ans.citations:
            src = by_label.get(c.source_id.strip("[] "))
            if src is None:
                invalid.append({"source_id": c.source_id, "reason": "unknown source"})
                continue
            q, t = _norm(c.quote), _norm(src["text"])
            q_tokens = q.split()
            overlap = sum(w in t.split() for w in q_tokens) / max(len(q_tokens), 1)
            if q in t or overlap >= 0.8:
                valid.append(c.source_id)
            else:
                invalid.append({"source_id": c.source_id, "reason": "quote not found in source", "quote": c.quote})
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", ans.answer.strip()) if len(s) > 3]
        cited = [s for s in sentences if re.search(r"\[S\d+\]", s)]
        markers = set(re.findall(r"\[(S\d+)\]", ans.answer))
        unknown_markers = sorted(m for m in markers if m not in by_label)
        citation_precision = len(valid) / max(len(ans.citations), 1)
        sentence_coverage = len(cited) / max(len(sentences), 1)
        grounded = citation_precision * (0.5 + 0.5 * sentence_coverage)  # a bad citation hurts multiplicatively
        if unknown_markers:
            grounded *= 0.5
        return {"citation_precision": round(citation_precision, 2), "sentence_coverage": round(sentence_coverage, 2),
                "grounded_score": round(grounded, 2), "invalid_citations": invalid,
                "unknown_markers": unknown_markers}

    def _fallback(self, question: str, reason: str, retrieved: list[dict]) -> RAGResult:
        res = self.fallback.search(question) if self.fallback else None
        if res:
            text, urls = res
            return RAGResult(question, text, [{"url": u} for u in urls], confidence=0.5, low_confidence=True,
                             source="web_search", grounding={"reason": reason}, retrieved=retrieved)
        return RAGResult(question, "I couldn't find this in the knowledge base and no search fallback is available.",
                         confidence=0.0, low_confidence=True, source="none", grounding={"reason": reason},
                         retrieved=retrieved)

    def ask(self, question: str) -> RAGResult:
        retrieved = self.retrieve(question)
        top = retrieved[0]["score"] if retrieved else 0.0
        if top < self.min_retrieval_score:
            return self._fallback(question, f"weak retrieval (top score {top:.2f})", retrieved)

        context = "\n\n".join(f"[{r['label']}] ({r['source']})\n{r['text']}" for r in retrieved)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}]
        try:
            ans = generate_structured(self.llm, messages, GroundedAnswer, tag="p02.answer").value
        except StructuredOutputError as e:
            return self._fallback(question, f"generation failed: {e}", retrieved)

        if not ans.answerable:
            return self._fallback(question, "model reported sources insufficient", retrieved)

        g = self.verify(ans, retrieved)
        retrieval_norm = min(1.0, top / self.good_retrieval_score)
        confidence = round(0.4 * ans.confidence + 0.4 * g["grounded_score"] + 0.2 * retrieval_norm, 3)
        result = RAGResult(
            question=question, answer=ans.answer,
            citations=[{"source_id": c.source_id,
                        "source": next((r["source"] for r in retrieved if r["label"] == c.source_id), None),
                        "quote": c.quote} for c in ans.citations],
            confidence=confidence, grounding=g,
            low_confidence=confidence < self.confidence_threshold or bool(g["invalid_citations"] or g["unknown_markers"]),
            retrieved=[{k: r[k] for k in ("label", "id", "score")} for r in retrieved],
        )
        if result.low_confidence and self.fallback:
            fb = self._fallback(question, f"low confidence {confidence}", retrieved)
            if fb.source == "web_search":
                fb.grounding["kb_answer"] = result.answer
                return fb
        return result
