"""Embeddings + a tiny vector store.

``HashingEmbedder`` is offline and free (feature hashing of words and bigrams) -
good enough for demos and tests. Set EMBEDDINGS_BACKEND=openai in .env to use
text-embedding-3-small for real semantic recall.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import config

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in on for and or is are was were be been it this that with as at by from "
            "what which who how why when do does did i you we they he she my your our their me us".split())


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        return h % self.dim, 1.0 if (h >> 64) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            # light stemming so "refunds" ~ "refund"
            toks = [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in toks]
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                j, s = self._bucket(f)
                out[i, j] += s
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class OpenAIEmbedder:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = model or config.EMBEDDING_MODEL

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self.client.embeddings.create(model=self.model, input=texts)
        arr = np.array([d.embedding for d in r.data], dtype=np.float32)
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)


def get_embedder():
    if config.EMBEDDINGS_BACKEND == "openai" and config.use_real_llm():
        return OpenAIEmbedder()
    return HashingEmbedder()


@dataclass
class VectorStore:
    embedder: Any = field(default_factory=get_embedder)
    texts: list[str] = field(default_factory=list)
    metas: list[dict] = field(default_factory=list)
    vectors: np.ndarray | None = None

    def add(self, texts: list[str], metas: list[dict] | None = None) -> None:
        if not texts:
            return
        metas = metas or [{} for _ in texts]
        vecs = self.embedder.embed(texts)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, query: str, k: int = 4) -> list[tuple[float, str, dict]]:
        if self.vectors is None or not self.texts:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.vectors @ q
        idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.texts[i], self.metas[i]) for i in idx]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"texts": self.texts, "metas": self.metas,
                                          "vectors": self.vectors.tolist() if self.vectors is not None else None}))

    @classmethod
    def load(cls, path: str | Path, embedder=None) -> "VectorStore":
        d = json.loads(Path(path).read_text())
        vs = cls(embedder=embedder or get_embedder(), texts=d["texts"], metas=d["metas"])
        vs.vectors = np.array(d["vectors"], dtype=np.float32) if d["vectors"] else None
        return vs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0
