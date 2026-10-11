"""Demo: python -m p06_hitl_approval.run   (interactive reviewer prompt in the terminal)
Use --auto to approve automatically (non-interactive)."""
from __future__ import annotations

import json
import sys

from pydantic import ValidationError

from common import config
from common.llm import get_llm

from .core import HITLAgent

REQUESTS = [
    {"customer_id": "C1001", "message": "I was charged twice for txn T-88 (INR 450). Please refund the duplicate.",
     "txn": {"id": "T-88", "amount": 450}},
    {"customer_id": "C2002", "message": "I don't recognise a INR 72,000 payment to an electronics store. "
                                        "My card ends 4421. Please help!", "txn": {"id": "T-91", "amount": 72000}},
]


def ask_human(run: dict, auto: bool) -> dict:
    print("  PAUSED for human review. Reasons:", run.get("uncertainty", {}).get("reasons") or run.get("reasons"))
    print("  Proposal:", json.dumps(run.get("proposal"), indent=1)[:500])
    if auto:
        return {"decision": "approve", "reviewer": "priya.sr_analyst", "comment": "auto-approved in demo"}
    while True:
        raw = input("  decision [approve/reject/modify] > ").strip() or "approve"
        reviewer = input("  reviewer id (priya.sr_analyst | rahul.fraud_lead) > ").strip() or "priya.sr_analyst"
        comment = input("  comment > ").strip()
        d = {"decision": raw, "reviewer": reviewer, "comment": comment}
        if raw == "modify":
            d["modified_action"] = input("  new action [refund/block_card/escalate_to_fraud_team/no_action] > ").strip() or None
            params = input("  params as JSON (e.g. {\"amount\": 200}) > ").strip()
            d["modified_params"] = json.loads(params) if params else None
        return d


def main() -> None:
    auto = "--auto" in sys.argv
    agent = HITLAgent(get_llm(), config.DATA_DIR / "p06_hitl.db")
    for req in REQUESTS:
        print(f"\n=== {req['message'][:70]}...")
        run = agent.start(req)
        print("  status:", run["status"])
        while run["status"] == "PENDING_APPROVAL":
            try:
                run = agent.resume(run["run_id"], ask_human(run, auto))
            except (ValidationError, ValueError) as e:
                print("  rejected human input (logged to audit):", str(e)[:200])
                if auto:
                    break
        print("  final status:", run["status"])
        print("  customer message:", (run.get("customer_message") or "")[:300])
        for e in agent.audit.trail(run["run_id"]):
            print(f"    #{e['seq']} {e['actor']:18} {e['event']}")
    print("\naudit chain intact:", agent.audit.verify())


if __name__ == "__main__":
    main()
