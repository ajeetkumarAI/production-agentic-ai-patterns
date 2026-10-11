"""Demo: python -m p09_multi_agent_debate.run"""
from __future__ import annotations

from common.llm import get_llm

from .debate import DebateSystem

QUESTION = ("Our UPI fraud model flags 2% of transactions but only 8% of flags are real fraud. Customers are "
            "complaining about blocked payments. What should we change in the next 30 days?")


def main() -> None:
    llm = get_llm()
    res = DebateSystem(llm, max_rounds=3).run(QUESTION)
    for r in res.rounds:
        print(f"\n-- round {r.round}: leader={r.leader} consensus={r.consensus}")
        for name, p in r.proposals.items():
            print(f"  {name:20} borda={r.borda.get(name)} combined={r.combined.get(name)} "
                  f"conf={p['confidence']} :: {p['solution'][:110]}")
    print(f"\nWINNER: {res.winner} | consensus reached: {res.consensus_reached} | confidence: {res.confidence}")
    print("FINAL ANSWER:\n", res.answer)
    if res.dissent:
        print("DISSENT:", res.dissent)
    print(f"\ncalls={len(llm.history)} cost=${llm.total_cost:.5f}")


if __name__ == "__main__":
    main()
