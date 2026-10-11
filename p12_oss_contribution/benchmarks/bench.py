"""Reproducible benchmark: LoopGuard vs plain recursion_limit.

    python -m p12_oss_contribution.benchmarks.bench --episodes 500

Uses a *scripted* stochastic agent (no API cost, fully reproducible with a seed):
each episode needs ``k`` useful tool calls; with probability ``p_stuck`` the agent
gets stuck repeating one call, and with ``p_cycle`` it alternates between two.
Each agent step is billed at a typical gpt-5-nano step size so the cost columns
reflect real money wasted by runaway loops.
"""
from __future__ import annotations

import argparse
import operator
import random
import statistics
import time
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from common.config import cost_usd
from p12_oss_contribution.loop_guard import FALLBACK, LoopGuard, empty_guard

PROMPT_TOKENS_PER_STEP, OUTPUT_TOKENS_PER_STEP = 1200, 250


class S(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    loop_guard: dict
    plan: list[str]


def make_episode(rng: random.Random, p_stuck: float, p_cycle: float) -> list[str]:
    k = rng.randint(2, 5)
    useful = [f"tool_{i}" for i in range(k)]
    r = rng.random()
    if r < p_stuck:
        cut = rng.randint(0, k - 1)
        return useful[:cut] + [useful[cut]] * 1000
    if r < p_stuck + p_cycle:
        cut = rng.randint(0, k - 2)
        return useful[:cut] + [useful[cut], useful[cut + 1]] * 500
    return useful + ["FINAL"]


def build(guarded: bool, guard: LoopGuard):
    def agent(state: S) -> dict:
        i = sum(1 for m in state["messages"] if m["role"] == "assistant")
        action = state["plan"][i]
        if action == "FINAL":
            return {"messages": [{"role": "assistant", "content": "done"}]}
        return {"messages": [{"role": "assistant", "content": "",
                              "tool_calls": [{"name": action, "args": {"q": action}}]}]}

    def tools(state: S) -> dict:
        return {"messages": [{"role": "tool", "content": "ok"}]}

    def router(state: S):
        return "tools" if state["messages"][-1].get("tool_calls") else END

    usage = lambda _u: (PROMPT_TOKENS_PER_STEP + OUTPUT_TOKENS_PER_STEP,  # noqa: E731
                        cost_usd("gpt-5-nano", PROMPT_TOKENS_PER_STEP, OUTPUT_TOKENS_PER_STEP))
    guard.usage = usage
    g = StateGraph(S)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_edge("tools", "agent")
    if guarded:
        g.add_node("agent", guard.wrap(agent))
        g.add_node(FALLBACK, guard.fallback_node())
        g.add_conditional_edges("agent", guard.route(router), {"tools": "tools", END: END, FALLBACK: FALLBACK})
        g.add_edge(FALLBACK, END)
    else:
        g.add_node("agent", agent)
        g.add_conditional_edges("agent", router, {"tools": "tools", END: END})
    return g.compile()


def run(episodes: int, p_stuck: float, p_cycle: float, recursion_limit: int, seed: int) -> dict:
    results = {}
    for guarded in (False, True):
        rng = random.Random(seed)
        graph = build(guarded, LoopGuard(max_steps=10, max_repeats=2))
        steps, answered, crashed, t0 = [], 0, 0, time.perf_counter()
        for _ in range(episodes):
            plan = make_episode(rng, p_stuck, p_cycle)
            try:
                out = graph.invoke({"messages": [{"role": "user", "content": "go"}], "plan": plan,
                                    "loop_guard": empty_guard()}, {"recursion_limit": recursion_limit})
                answered += 1
                steps.append(sum(1 for m in out["messages"] if m["role"] == "assistant"))
            except GraphRecursionError:
                crashed += 1
                steps.append(recursion_limit // 2)  # agent steps executed before the crash
        per_step = cost_usd("gpt-5-nano", PROMPT_TOKENS_PER_STEP, OUTPUT_TOKENS_PER_STEP)
        results["guarded" if guarded else "baseline"] = {
            "episodes": episodes, "answered_rate": round(answered / episodes, 3),
            "crash_rate": round(crashed / episodes, 3), "avg_agent_steps": round(statistics.mean(steps), 2),
            "p95_agent_steps": sorted(steps)[int(0.95 * len(steps)) - 1],
            "cost_per_1k_requests_usd": round(statistics.mean(steps) * per_step * 1000, 4),
            "wall_s": round(time.perf_counter() - t0, 2)}
    return results


def to_markdown(res: dict, args) -> str:
    cols = ["answered_rate", "crash_rate", "avg_agent_steps", "p95_agent_steps", "cost_per_1k_requests_usd"]
    lines = [f"# LoopGuard benchmark\n",
             f"episodes={args.episodes}, p_stuck={args.p_stuck}, p_cycle={args.p_cycle}, "
             f"recursion_limit={args.recursion_limit}, seed={args.seed}; step cost modelled at gpt-5-nano prices "
             f"({PROMPT_TOKENS_PER_STEP} in / {OUTPUT_TOKENS_PER_STEP} out tokens).\n",
             "| variant | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in res.items():
        lines.append(f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |")
    b, g = res["baseline"], res["guarded"]
    lines.append(f"\nGuard reduces average steps by {100 * (1 - g['avg_agent_steps'] / b['avg_agent_steps']):.0f}% "
                 f"and turns {b['crash_rate']:.0%} crashes into graceful answers.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--p-stuck", type=float, default=0.15)
    ap.add_argument("--p-cycle", type=float, default=0.10)
    ap.add_argument("--recursion-limit", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    res = run(args.episodes, args.p_stuck, args.p_cycle, args.recursion_limit, args.seed)
    md = to_markdown(res, args)
    out = Path(__file__).parent / "RESULTS.md"
    out.write_text(md)
    print(md)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
