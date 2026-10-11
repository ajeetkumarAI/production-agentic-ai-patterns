"""Demo: python -m p10_self_reflective.run"""
from __future__ import annotations

import json

from common.llm import get_llm

from .agent import SelfReflectiveAgent, improvement_report, max_words, min_bullets, must_include

TASKS = [
    "Explain how to handle class imbalance in a credit-card fraud dataset.",
    "Explain how to choose a decision threshold for a fraud classifier.",
    "Explain how to monitor a fraud model for concept drift in production.",
]


def main() -> None:
    llm = get_llm()
    agent = SelfReflectiveAgent(llm, threshold=0.8, max_iterations=3)
    checks = [max_words(220), min_bullets(3), must_include("precision", "recall")]
    for task in TASKS:
        res = agent.run(task, task_type="ml_explainer", checks=checks)
        print(f"\nTASK: {task}")
        for it in res.iterations:
            print(f"  iter {it.n}: score={it.score:.2f} passed={it.passed} criteria={it.criteria} "
                  f"failed_checks={it.failed_checks}")
            print(f"           constraints={it.constraints_used[:4]}")
        print(f"  best={res.best_score:.2f} improvement=+{res.improvement:.2f}")
        print("  answer:", res.best_answer[:250].replace("\n", " "))
    print("\nIMPROVEMENT REPORT:", json.dumps(improvement_report(), indent=2))
    print(f"cost=${llm.total_cost:.5f}")


if __name__ == "__main__":
    main()
