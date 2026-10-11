"""LoopGuard - a reusable termination-safety pattern for LangGraph agents.

Problem: tool-calling agents in LangGraph can cycle (same tool, same args) until
``recursion_limit`` raises ``GraphRecursionError`` - the user gets an exception,
not an answer, and every wasted step costs tokens.

LoopGuard adds three things without changing your nodes' logic:

    guard = LoopGuard(max_steps=8, max_repeats=2, max_cost_usd=0.02)
    builder.add_node("agent", guard.wrap(agent_node, signature=tool_call_signature))
    builder.add_node("fallback", guard.fallback_node(summarize))
    builder.add_conditional_edges("agent", guard.route(my_router), {...,"fallback": "fallback"})

* ``wrap``     - counts steps, detects repeated actions (by signature), accumulates
                 cost/tokens, and marks the state as *tripped* with a reason.
* ``route``    - wraps your router: if tripped, go to ``fallback``; else defer to you.
* ``fallback_node`` - turns whatever was gathered into a graceful final answer.

State contract: your graph state needs one extra key, ``loop_guard: dict``.
Framework-light: the guard is pure Python over dicts; LangGraph is only needed
to build graphs (see examples/).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

GUARD_KEY = "loop_guard"
FALLBACK = "fallback"

SignatureFn = Callable[[dict, dict], "str | None"]   # (state, node_update) -> action signature
UsageFn = Callable[[dict], "tuple[int, float]"]       # node_update -> (tokens, cost_usd)


def empty_guard() -> dict:
    return {"steps": 0, "signatures": {}, "tokens": 0, "cost_usd": 0.0, "tripped": False, "reason": None,
            "history": []}


def tool_call_signature(state: dict, update: dict) -> str | None:
    """Default signature: tool calls on the last message in ``update['messages']``
    (works for LangChain AIMessage objects and OpenAI-style dicts)."""
    msgs = update.get("messages") or []
    if not isinstance(msgs, list):
        msgs = [msgs]
    if not msgs:
        return None
    last = msgs[-1]
    calls = getattr(last, "tool_calls", None)
    if calls is None and isinstance(last, dict):
        calls = last.get("tool_calls")
    if not calls:
        return None
    norm = []
    for c in calls:
        if isinstance(c, dict):
            name = c.get("name") or (c.get("function") or {}).get("name")
            args = c.get("args") or (c.get("function") or {}).get("arguments")
        else:
            name, args = getattr(c, "name", None), getattr(c, "args", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        norm.append([name, args])
    blob = json.dumps(sorted(norm, key=lambda x: str(x[0])), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


@dataclass
class LoopGuard:
    max_steps: int = 10
    max_repeats: int = 2            # same action allowed this many times; one more trips the guard
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    usage: UsageFn | None = None

    # ---------------------------------------------------------------- core logic (pure)
    def check(self, guard: dict, signature: str | None, tokens: int = 0, cost: float = 0.0) -> dict:
        g = json.loads(json.dumps(guard or empty_guard()))  # copy; state updates must be new objects
        if g["tripped"]:
            return g
        g["steps"] += 1
        g["tokens"] += int(tokens)
        g["cost_usd"] = round(g["cost_usd"] + float(cost), 8)
        if signature:
            g["signatures"][signature] = g["signatures"].get(signature, 0) + 1
            g["history"].append(signature)
        reason = None
        if signature and g["signatures"][signature] > self.max_repeats:
            reason = f"repeated action {signature} x{g['signatures'][signature]}"
        elif g["steps"] >= self.max_steps:
            reason = f"step limit {self.max_steps} reached"
        elif self.max_tokens is not None and g["tokens"] > self.max_tokens:
            reason = f"token budget {self.max_tokens} exceeded ({g['tokens']})"
        elif self.max_cost_usd is not None and g["cost_usd"] > self.max_cost_usd:
            reason = f"cost budget ${self.max_cost_usd} exceeded (${g['cost_usd']:.5f})"
        elif _is_cycle(g["history"]):
            reason = "alternating action cycle detected"
        if reason:
            g["tripped"], g["reason"] = True, reason
        return g

    # ---------------------------------------------------------------- LangGraph adapters
    def wrap(self, node: Callable[[dict], dict], signature: SignatureFn = tool_call_signature) -> Callable[[dict], dict]:
        def guarded(state: dict) -> dict:
            guard = state.get(GUARD_KEY) or empty_guard()
            if guard.get("tripped"):
                return {}  # don't spend more tokens once tripped
            update = node(state) or {}
            tokens, cost = self.usage(update) if self.usage else (0, 0.0)
            update[GUARD_KEY] = self.check(guard, signature(state, update), tokens, cost)
            return update
        guarded.__name__ = getattr(node, "__name__", "guarded_node")
        return guarded

    def route(self, router: Callable[[dict], Any]) -> Callable[[dict], Any]:
        def guarded_router(state: dict):
            if (state.get(GUARD_KEY) or {}).get("tripped"):
                return FALLBACK
            return router(state)
        return guarded_router

    def fallback_node(self, summarize: Callable[[dict, str], dict] | None = None) -> Callable[[dict], dict]:
        def fallback(state: dict) -> dict:
            reason = (state.get(GUARD_KEY) or {}).get("reason") or "guard tripped"
            if summarize:
                return summarize(state, reason)
            return {"messages": [{"role": "assistant", "content": (
                f"I stopped early ({reason}). Here is what I found so far; some parts may be incomplete.")}]}
        return fallback


def _is_cycle(history: list[str], min_period: int = 2, repeats: int = 3) -> bool:
    """Detect A,B,A,B,A,B style cycles (period 2..3) that per-signature counting
    would only catch late."""
    for p in range(min_period, 4):
        need = p * repeats
        if len(history) >= need:
            tail = history[-need:]
            if len(set(tail[:p])) == p and all(tail[i] == tail[i % p] for i in range(need)):
                return True
    return False
