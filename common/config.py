"""Central configuration: loads .env, model names, pricing and data paths."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
# gpt-5 family supports: minimal | low | medium | high
DEFAULT_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# "local" (offline hashing embedder, no cost) or "openai"
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "local")
# "auto" -> real OpenAI when a key is present, otherwise mock. "mock" forces mock.
LLM_MODE = os.getenv("LLM_MODE", "auto").lower()

# USD per 1M tokens: (input, cached_input, output). Update if OpenAI changes prices.
PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5": (1.25, 0.125, 10.00),
    "text-embedding-3-small": (0.02, 0.02, 0.0),
    "mock": (0.0, 0.0, 0.0),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """Cost of one call. Unknown models are priced as gpt-5-nano."""
    inp, cached, out = PRICING.get(model, PRICING["gpt-5-nano"])
    uncached = max(prompt_tokens - cached_tokens, 0)
    return (uncached * inp + cached_tokens * cached + completion_tokens * out) / 1_000_000


def use_real_llm() -> bool:
    return LLM_MODE != "mock" and bool(OPENAI_API_KEY) and not OPENAI_API_KEY.startswith("sk-your")
