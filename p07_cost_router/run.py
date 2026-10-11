"""Demo: python -m p07_cost_router.run"""
from __future__ import annotations

import json

from common.llm import get_llm

from .router import CostAwareRouter, TaskBudget

TASKS = [
    "Translate 'payment declined' into Hindi.",
    "What is the capital of Maharashtra?",
    "Summarize in one line: the customer disputes a duplicate INR 450 charge on 3 Oct.",
    "Compare isolation forest vs autoencoders for card fraud detection and explain the trade-offs for "
    "highly imbalanced, concept-drifting data. Which would you deploy and why?",
    "Design a step-by-step algorithm to detect circular money-laundering rings in a transaction graph with "
    "50M edges, and estimate its time complexity. Cover edge cases.",
]


def main() -> None:
    router = CostAwareRouter(get_llm())
    for t in TASKS:
        d = router.run(t, TaskBudget(max_cost_usd=0.005))
        print(f"\nTASK: {t[:80]}")
        print(f"  complexity={d.complexity} start={d.start_tier} tried={d.tiers_tried} final={d.final_tier} "
              f"conf={d.confidence:.2f} early_exit={d.early_exit} cost=${d.cost_usd:.6f}")
        print(f"  answer: {d.answer[:160]}")
    print("\nANALYTICS:", json.dumps(router.report(), indent=2))


if __name__ == "__main__":
    main()
