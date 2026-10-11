"""Demo: python -m p03_react_planning.run"""
from __future__ import annotations

from common.llm import get_llm

from .agent import ReActAgent

QUESTION = ("A 10-day-old account made three online purchases today: USD 120, EUR 80 and GBP 50. "
            "What is the total in INR, and does it breach the daily card-not-present limit for new accounts?")


def main() -> None:
    agent = ReActAgent(get_llm(), max_iterations=8)
    res = agent.run(QUESTION)
    for s in res.steps:
        print(f"[{s.i}] {s.kind:8} {s.action}({s.action_input}) -> {s.observation[:140]}")
        if s.thought and s.kind == "act":
            print(f"      thought: {s.thought[:140]}")
    print(f"\nSTATUS: {res.status} {('(' + res.reason + ')') if res.reason else ''}")
    print(f"ANSWER: {res.answer}")
    print(f"iterations={res.iterations} tokens={res.tokens} cost=${res.cost_usd:.5f}")


if __name__ == "__main__":
    main()
