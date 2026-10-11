# docs/prebuilt: add LoopGuard pattern for terminating looping tool agents

## Summary

Adds a small, dependency-free helper and a how-to guide. Together they let graph authors stop runaway tool loops gracefully: the run routes to a fallback answer instead of raising `GraphRecursionError` at `recursion_limit`.

## Motivation

- Tool agents often repeat the same call, e.g. polling a "pending" status, or alternate between two tools.
- Today the only safety net is `recursion_limit`. It raises an exception after the tokens are already spent, and the end user receives no answer.
- Users have repeatedly asked how to cap steps or cost per run. Each team re-implements counters in its own state.

## What's included

- `LoopGuard` dataclass:
  - `wrap(node)` records steps, action signatures and token/cost usage in `state["loop_guard"]`.
  - `route(router)` diverts to `"fallback"` once tripped.
  - `fallback_node(summarize=None)` produces a graceful final message.
- `tool_call_signature`: the default signature for `AIMessage.tool_calls` and OpenAI-style dicts.
- How-to guide: "How to stop infinite tool loops".
- Unit tests covering repeat, step, budget and cycle detection, plus a full graph run.
- Benchmark script (scripted agents, seeded, zero API cost).

## Benchmark (300 episodes, 25% loop-prone agents, recursion_limit=25)

| variant | answered | crashed | avg steps | p95 steps |
|---|---|---|---|---|
| baseline | 74% | 26% | 6.4 | 12 |
| LoopGuard | 100% | 0% | 4.85 | 7 |

## Backwards compatibility

The change is purely additive and opt-in. Existing graphs are unaffected.

## Checklist

- [ ] Tests pass (`pytest`)
- [ ] Docs page builds
- [ ] Lint (`ruff`)
- [ ] Discussed with maintainers in issue #<link>
