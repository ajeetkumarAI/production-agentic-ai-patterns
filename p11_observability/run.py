"""Demo: python -m p11_observability.run

1. Serves traffic on stable v1 (traces + metrics + alerts).
2. Starts a canary v2 (deliberately worse config: max 2 iterations, no critique).
3. Evaluates the canary -> automatic rollback (or promotion if it's healthy).
Then open the dashboard:  uvicorn p11_observability.dashboard:app --port 8011
"""
from __future__ import annotations

import json
import random

from common import config
from common.llm import get_llm

from .ops import DeploymentManager
from .service import AgentVersion, ProductionAgent

QUERIES = [
    "Convert USD 120 to INR.",
    "What is 18% GST on INR 2,450?",
    "A new account spent EUR 80 and GBP 50 online today. Total in INR and does it breach the CF-04 limit?",
    "What is the chargeback window for card-not-present purchases?",
]


def main(n_stable: int = 10, n_canary: int = 24) -> None:
    random.seed(7)
    (config.DATA_DIR / "p11_deployment.json").unlink(missing_ok=True)
    prod = ProductionAgent(get_llm(), {"v1": AgentVersion("v1", max_iterations=8),
                                       "v2": AgentVersion("v2", max_iterations=2, self_critique=False)},
                           deployment=DeploymentManager(stable="v1"))
    print("== stable traffic ==")
    for i in range(n_stable):
        r = prod.handle(f"user{i}", random.choice(QUERIES))
        print(f"  {r['version']} {r['status']:9} {r['latency_ms']:7.0f}ms ${r['cost']:.6f} steps={r['steps']}")

    print("\n== canary v2 at 50% ==")
    prod.deployment.start_canary("v2", percent=50)
    for i in range(n_canary):
        r = prod.handle(f"user{100 + i}", random.choice(QUERIES))
        print(f"  {r['version']} {r['status']:9} {r['latency_ms']:7.0f}ms ${r['cost']:.6f} steps={r['steps']}")
    decision = prod.deployment.evaluate(prod.metrics, min_requests=5, alerter=prod.alerter)
    print(f"\ncanary decision: {decision}")
    print("deployment state:", json.dumps({k: v for k, v in prod.deployment.state.items() if k != "history"}))
    print("\nper-version metrics:")
    for v in ("v1", "v2"):
        print(f"  {v}: {prod.metrics.summary(v)}")
    print(f"\ntraces -> {config.DATA_DIR / 'p11_traces.jsonl'}")
    print("dashboard -> uvicorn p11_observability.dashboard:app --port 8011  then open http://localhost:8011/")


if __name__ == "__main__":
    main()
