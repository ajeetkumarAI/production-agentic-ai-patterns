# Project 12: LoopGuard, an open-source contribution for LangGraph

**Pattern:** termination safety for tool-calling agents. It detects repeated actions, alternating cycles, step limits, and token/cost budgets. When one trips, the agent routes to a graceful fallback answer. You no longer get a `GraphRecursionError` crash or wasted tokens.

## Why it matters

LangGraph's only built-in stop is `recursion_limit`. When a looping agent hits it, the run raises an exception:

- the user gets no answer
- all the steps so far are paid for

LoopGuard is about 150 lines of pure Python. It plugs into any graph through three adapters (`wrap`, `route`, `fallback_node`) and doesn't change your nodes.

## Tutorial (5 minutes)

```python
from langgraph.graph import StateGraph, START, END
from p12_oss_contribution.loop_guard import LoopGuard, FALLBACK, empty_guard

class State(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    loop_guard: dict                       # 1. add one key to your state

guard = LoopGuard(max_steps=8, max_repeats=2, max_cost_usd=0.02)

g = StateGraph(State)
g.add_node("agent", guard.wrap(agent_node))            # 2. wrap the node that decides actions
g.add_node("tools", tool_node)
g.add_node(FALLBACK, guard.fallback_node(summarize))   # 3. add a fallback (optional custom summary)
g.add_edge(START, "agent")
g.add_edge("tools", "agent")
g.add_conditional_edges("agent", guard.route(my_router),   # 4. wrap your router
                        {"tools": "tools", END: END, FALLBACK: FALLBACK})
g.add_edge(FALLBACK, END)
app = g.compile()
app.invoke({"messages": [...], "loop_guard": empty_guard()})
```

* **Action signature:** by default this is a hash of the tool names plus arguments on the last message. It works with LangChain `AIMessage.tool_calls` and with OpenAI-style dicts. You can pass `signature=` to customize it.
* **Cost/tokens:** pass `usage=lambda update: (tokens, cost_usd)` to enforce budgets.
* **State:** `state["loop_guard"]` keeps `steps`, `signatures`, `tokens`, `cost_usd`, `tripped`, and `reason`. You can log it or show it in your UI.

## Demo (real gpt-5-nano)

```bash
python -m p12_oss_contribution.examples.demo_graph
```

A "check settlement status" tool always answers *pending, try again*:

* **Unguarded run:** crashes with `GraphRecursionError`.
* **Guarded run:** stops after 2 repeats and explains what's known.

## Benchmarks (reproducible, no API cost)

```bash
python -m p12_oss_contribution.benchmarks.bench --episodes 300
```

This writes `benchmarks/RESULTS.md`. With the default settings (15% stuck agents, 10% cycling, `recursion_limit=25`):

* **Crashes:** about 26% of unguarded runs crash. The guard turns all of them into answers.
* **Steps:** average agent steps drop by about 24%.

## Path to an upstream contribution

1. **Publish standalone first:** `pyproject.toml` is included. The package name is `langgraph-loop-guard`; publish it with `python -m build && twine upload dist/*`.
2. **Open a discussion or issue on `langchain-ai/langgraph`.** Link the benchmark and ask whether maintainers would prefer:
   * a how-to guide in the docs ("Prevent infinite tool loops"), or
   * a prebuilt helper next to `create_react_agent`.
3. **Submit the PR.** `PR_DESCRIPTION.md` is a ready-to-edit description. It needs tests, the docs notebook, and benchmark numbers.
4. **Write the tutorial post.** This README is the draft for a blog or LinkedIn tutorial.

Tests: `pytest tests/test_p12_loop_guard.py`
