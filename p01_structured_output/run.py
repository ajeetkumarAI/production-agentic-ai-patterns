"""Demo: python -m p01_structured_output.run"""
from __future__ import annotations

import json

from common.llm import get_llm
from common.structured import StructuredOutputError

from .agent import FAILURE_LOG, StructuredOutputAgent, failure_report

TRANSACTIONS = [
    {"id": "T-1", "customer_id": "C1001", "merchant_id": "M-GROCERY", "amount": 1800, "currency": "INR",
     "ip_country": "IN"},
    {"id": "T-2", "customer_id": "C2002", "merchant_id": "M-CRYPTO", "amount": 95000, "currency": "INR",
     "ip_country": "NG"},
    {"id": "T-3", "customer_id": "C3003", "merchant_id": "M-GROCERY", "amount": 400, "currency": "INR",
     "ip_country": "IN"},  # corrupt customer record -> tool output validation failure gets logged
]


def main() -> None:
    llm = get_llm()
    agent = StructuredOutputAgent(llm)
    for txn in TRANSACTIONS:
        print(f"\n=== {txn['id']} ===")
        try:
            res = agent.assess(txn)
            print(json.dumps(res.value.model_dump(), indent=2))
            print(f"attempts={res.attempts} repaired_errors={len(res.errors)}")
        except StructuredOutputError as e:
            print("FAILED after retries:", e.errors)
    print("\nValidation failure report:", json.dumps(failure_report(), indent=2))
    print(f"Full log: {FAILURE_LOG}")
    print(f"Total cost: ${llm.total_cost:.5f} over {len(llm.history)} calls")


if __name__ == "__main__":
    main()
