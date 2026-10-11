"""Demo: python -m p04_multi_tool_orchestrator.run

Runs a fixed, hand-written plan first (so you can see parallelism, failover,
conflict resolution and permission denial deterministically), then lets the
LLM plan the same goal itself.
"""
from __future__ import annotations

from common.llm import get_llm

from .orchestrator import Orchestrator, Principal, ToolRegistry
from .tools_demo import demo_tools

GOAL = ("Investigate customer C2002 who paid USD 500 from IP 102.89.1.7: get their profile, IP intelligence, "
        "the USD/INR rate to convert the amount, a sanctions check on their name, and alert fraud ops if risky.")

FIXED_PLAN = [
    {"id": "t1", "capability": "customer_profile", "input": {"customer_id": "C2002"}, "depends_on": []},
    {"id": "t2", "capability": "ip_intel", "input": {"ip": "102.89.1.7"}, "depends_on": []},
    {"id": "t3", "capability": "fx_rate", "input": {"base": "USD", "quote": "INR"}, "depends_on": [], "fanout": True},
    {"id": "t4", "capability": "sanctions_check", "input": {"name": "${t1.name}"}, "depends_on": ["t1"]},
    {"id": "t5", "capability": "notify", "input": {"message": "Risky txn by ${t1.name} from ${t2.country}",
                                                     "severity": "high"}, "depends_on": ["t1", "t2"]},
]


def show(res) -> None:
    for tid, o in res.outcomes.items():
        print(f"  {tid} {o.capability:16} {o.status:7} tool={o.tool} {o.duration_ms:6.0f}ms "
              f"{(str(o.output) if o.status == 'ok' else o.error)[:90]}")
    for c in res.conflicts:
        print(f"  CONFLICT {c.task_id}: {c.results} -> {c.strategy}: {c.chosen} ({c.note})")
    for d in res.denials:
        print(f"  DENIED  {d}")
    print(f"  wall time {res.wall_ms:.0f}ms\n  ANSWER: {res.answer[:600]}")


def main() -> None:
    registry = ToolRegistry()
    for t in demo_tools():
        registry.register(t)
    analyst = Principal("analyst_01", frozenset({"pii:read"}))  # no alerts:write -> notify is denied
    orch = Orchestrator(get_llm(), registry)

    print("== Fixed plan ==")
    show(orch.run(GOAL, analyst, tasks=FIXED_PLAN))
    print("\n== LLM-planned ==")
    res = orch.run(GOAL, analyst)
    print("  plan:", [(t["id"], t["capability"], t.get("depends_on")) for t in res.plan])
    show(res)


if __name__ == "__main__":
    main()
