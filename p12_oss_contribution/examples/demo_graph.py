"""LangGraph tool agent (gpt-5-nano) protected by LoopGuard.

    python -m p12_oss_contribution.examples.demo_graph

The ``check_settlement_status`` tool always answers "pending - try again", which
lures the agent into polling forever. Without the guard the graph dies with
GraphRecursionError; with it, the agent stops after 2 repeats and answers.
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from common.llm import get_llm
from common.tools import Tool
from p12_oss_contribution.loop_guard import FALLBACK, LoopGuard, empty_guard
from pydantic import BaseModel


class State(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    loop_guard: dict


class TxnArgs(BaseModel):
    txn_id: str


TOOLS = {t.name: t for t in [
    Tool("check_settlement_status", "Check whether a transaction has settled.",
         lambda txn_id: {"txn_id": txn_id, "status": "pending", "hint": "try again"}, TxnArgs),
    Tool("get_transaction", "Get transaction details.",
         lambda txn_id: {"txn_id": txn_id, "amount": 4999, "currency": "INR", "merchant": "BookNook"}, TxnArgs),
]}


def build(llm, guarded: bool = True):
    specs = [t.to_openai() for t in TOOLS.values()]
    usage_before = {"i": len(llm.history)}

    def agent(state: State) -> dict:
        resp = llm.chat(state["messages"], tools=specs, tag="p12.agent")
        return {"messages": [resp.to_message()]}

    def tools(state: State) -> dict:
        last = state["messages"][-1]
        out = []
        for tc in last.get("tool_calls", []):
            fn = tc["function"]
            try:
                res = TOOLS[fn["name"]](fn["arguments"])
            except Exception as e:  # noqa: BLE001
                res = {"error": str(e)}
            out.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res)})
        return {"messages": out}

    def router(state: State) -> str:
        return "tools" if state["messages"][-1].get("tool_calls") else END

    def usage(_update: dict) -> tuple[int, float]:
        new = llm.history[usage_before["i"]:]
        usage_before["i"] = len(llm.history)
        return sum(u.total_tokens for u in new), sum(u.cost_usd for u in new)

    def summarize(state: State, reason: str) -> dict:
        resp = llm.chat(state["messages"] + [{"role": "user", "content": (
            f"Stop calling tools ({reason}). Answer now with what you know and say what is still unknown.")}],
            tag="p12.fallback")
        return {"messages": [{"role": "assistant", "content": resp.content or ""}]}

    g = StateGraph(State)
    if guarded:
        guard = LoopGuard(max_steps=8, max_repeats=2, max_cost_usd=0.02, usage=usage)
        g.add_node("agent", guard.wrap(agent))
        g.add_node(FALLBACK, guard.fallback_node(summarize))
        g.add_conditional_edges("agent", guard.route(router), {"tools": "tools", END: END, FALLBACK: FALLBACK})
        g.add_edge(FALLBACK, END)
    else:
        g.add_node("agent", agent)
        g.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_edge("tools", "agent")
    return g.compile()


def main() -> None:
    llm = get_llm()
    question = ("Has transaction T-77 settled yet? Keep checking until it settles, then tell me the amount.")
    init: dict[str, Any] = {"messages": [{"role": "user", "content": question}], "loop_guard": empty_guard()}
    for guarded in (False, True):
        print(f"\n=== guarded={guarded} ===")
        start = len(llm.history)
        try:
            out = build(llm, guarded).invoke(init, {"recursion_limit": 25})
            print("answer:", out["messages"][-1]["content"][:400])
            print("guard:", {k: v for k, v in out.get("loop_guard", {}).items() if k != "history"})
        except GraphRecursionError as e:
            print("CRASHED:", type(e).__name__, str(e)[:120])
        calls = llm.history[start:]
        print(f"llm calls={len(calls)} tokens={sum(u.total_tokens for u in calls)} "
              f"cost=${sum(u.cost_usd for u in calls):.6f}")


if __name__ == "__main__":
    main()
