# LoopGuard benchmark

episodes=300, p_stuck=0.15, p_cycle=0.1, recursion_limit=25, seed=42; step cost modelled at gpt-5-nano prices (1200 in / 250 out tokens).

| variant | answered_rate | crash_rate | avg_agent_steps | p95_agent_steps | cost_per_1k_requests_usd |
|---|---|---|---|---|---|
| baseline | 0.743 | 0.257 | 6.4 | 12 | 1.0245 |
| guarded | 1.0 | 0.0 | 4.85 | 7 | 0.7755 |

Guard reduces average steps by 24% and turns 26% crashes into graceful answers.
