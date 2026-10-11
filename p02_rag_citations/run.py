"""Demo: python -m p02_rag_citations.run"""
from __future__ import annotations

import json

from common.llm import get_llm

from .agent import OpenAIWebSearch, RAGAgent

QUESTIONS = [
    "How many days does a customer have to raise a chargeback for an online purchase?",
    "When is enhanced due diligence mandatory?",
    "What happens if a new account tries to spend INR 40,000 online in one day?",
    "What is the capital of Australia?",  # not in KB -> fallback
]


def main() -> None:
    agent = RAGAgent(get_llm(), fallback=OpenAIWebSearch())
    for q in QUESTIONS:
        r = agent.ask(q)
        print(f"\nQ: {q}\nA: {r.answer}")
        print(f"source={r.source} confidence={r.confidence} low_confidence={r.low_confidence}")
        print("citations:", json.dumps(r.citations, indent=1)[:600])
        print("grounding:", r.grounding)


if __name__ == "__main__":
    main()
