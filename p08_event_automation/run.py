"""Demo: python -m p08_event_automation.run

Sends signed webhooks to the FastAPI app in-process: normal events, a duplicate
delivery, a poison message, and a flaky downstream that recovers after retries.
"""
from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from . import api


def post(client, event: dict, key: str | None = None, bad_sig: bool = False):
    body = json.dumps(event).encode()
    headers = {"X-Signature": "sha256=forged" if bad_sig else api.sign(body), "Content-Type": "application/json"}
    if key:
        headers["Idempotency-Key"] = key
    return client.post("/webhooks/payments", content=body, headers=headers)


def main() -> None:
    api.queue.base_backoff_s = 0.3          # fast retries for the demo
    api.integrations.flaky_failures = 2     # SMS gateway fails twice, then recovers
    with TestClient(api.app) as client:
        big = {"id": f"evt-{int(time.time())}-1", "type": "transaction.created",
               "data": {"txn_id": "T-500", "customer_id": "C2002", "amount": 72000, "merchant": "GadgetHub",
                        "ip_country": "NG"}}
        small = {"id": f"evt-{int(time.time())}-2", "type": "transaction.created",
                 "data": {"txn_id": "T-501", "customer_id": "C1001", "amount": 350, "merchant": "Bazaar"}}
        poison = {"id": f"evt-{int(time.time())}-3", "type": "transaction.created", "data": {"txn_id": "T-502"}}
        unknown = {"id": f"evt-{int(time.time())}-4", "type": "refund.created", "data": {}}

        print("big      ->", post(client, big).json())
        print("duplicate->", post(client, big).json(), "(same event id, not re-processed)")
        print("small    ->", post(client, small).json())
        print("poison   ->", post(client, poison).json())
        print("unknown  ->", post(client, unknown).json())
        print("forged   ->", post(client, small, bad_sig=True).status_code, "(rejected)")

        deadline = time.time() + 20
        while time.time() < deadline and api.queue.pending():
            time.sleep(0.3)

        print("\nstats:", client.get("/stats").json())
        print("jobs:")
        for j in client.get("/jobs").json():
            print(f"  {j['event_id'][-6:]} {j['type']:20} {j['status']:6} attempts={j['attempts']} "
                  f"{(j['last_error'] or '')[:70]}")
        dlq = client.get("/dlq").json()
        print("dead letters:", [(d["event_id"][-6:], d["last_error"][:60]) for d in dlq])
        print("cases opened:", api.integrations.cases, "| SMS sent:", len(api.integrations.notifications))


if __name__ == "__main__":
    main()
