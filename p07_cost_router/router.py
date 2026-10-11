"""Project 7 - Cost-Aware Agent Router.

* Complexity scoring (free heuristics, optional cheap LLM classifier for the grey zone)
* Model routing by complexity/cost across tiers (model x reasoning effort)
* Token budgeting per task (input truncation, output caps, hard USD ceiling)
* Early exit on confidence: cascade cheap -> expensive, stop as soon as the answer is confident
* Cost-per-decision analytics in SQLite with savings vs an "always use the big model" baseline
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from common import config
from common.llm import BaseLLM, estimate_tokens
from common.structured import StructuredOutputError, generate_structured


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    reasoning_effort: str
    max_output_tokens: int


def default_tiers() -> list[Tier]:
    tiers = [Tier("nano-minimal", "gpt-5-nano", "minimal", 1500),
             Tier("nano-low", "gpt-5-nano", "low", 4000),
             Tier("nano-high", "gpt-5-nano", "high", 12000)]
    if os.getenv("ROUTER_ENABLE_MINI") == "1":
        tiers.append(Tier("mini-medium", "gpt-5-mini", "medium", 12000))
    return tiers


@dataclass
class TaskBudget:
    max_cost_usd: float = 0.01
    max_input_tokens: int = 8000


class RoutedAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1, description="Calibrated probability the answer is fully correct")
    needs_more_reasoning: bool = Field(description="True if a stronger model should double-check")


class Complexity(BaseModel):
    level: int = Field(ge=0, le=2, description="0 simple lookup/rewrite, 1 moderate reasoning, 2 hard multi-step")


_HARD = re.compile(r"\b(prove|derive|optimi[sz]e|step[- ]by[- ]step|algorithm|trade-?offs?|architecture|debug|"
                   r"why|compare|design|strategy|calculate|estimate|multi-?step|edge cases?)\b", re.I)
_CODE = re.compile(r"```|def |class |SELECT |\{.*\}|=>", re.S)


def heuristic_complexity(task: str) -> float:
    words = len(task.split())
    score = min(words / 250, 0.35)
    score += 0.1 * min(len(_HARD.findall(task)), 6)
    score += 0.2 if _CODE.search(task) else 0
    score += 0.1 * min(task.count("?") - 1, 2) if task.count("?") > 1 else 0
    score += 0.1 if re.search(r"\d+\s*[-+*/^%]\s*\d+", task) else 0
    return round(min(score, 1.0), 3)


@dataclass
class Decision:
    id: str
    task: str
    complexity: float
    start_tier: str
    tiers_tried: list[str] = field(default_factory=list)
    final_tier: str = ""
    answer: str = ""
    confidence: float = 0.0
    early_exit: bool = False
    budget_exhausted: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    latency_ms: float = 0.0


class CostAwareRouter:
    def __init__(self, llm: BaseLLM, tiers: list[Tier] | None = None, confidence_threshold: float = 0.8,
                 use_llm_classifier: bool = True, baseline_model: str = "gpt-5",
                 db_path: str | Path = config.DATA_DIR / "p07_router.db"):
        self.llm = llm
        self.tiers = tiers or default_tiers()
        self.threshold = confidence_threshold
        self.use_llm_classifier = use_llm_classifier
        self.baseline_model = baseline_model
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, ts REAL, task TEXT, complexity REAL, "
                        "start_tier TEXT, tiers_tried TEXT, final_tier TEXT, confidence REAL, early_exit INT, "
                        "budget_exhausted INT, prompt_tokens INT, completion_tokens INT, cost REAL, baseline_cost REAL, "
                        "latency_ms REAL)")
        self.db.commit()

    # ---------------------------------------------------------------- routing
    def start_tier(self, task: str) -> tuple[int, float]:
        score = heuristic_complexity(task)
        if self.use_llm_classifier and 0.3 <= score <= 0.6:  # grey zone: ask the cheapest tier
            try:
                lvl = generate_structured(self.llm, [
                    {"role": "system", "content": "Classify task difficulty. 0=simple, 1=moderate, 2=hard."},
                    {"role": "user", "content": task[:2000]}], Complexity, max_retries=0,
                    model=self.tiers[0].model, reasoning_effort="minimal", tag="p07.classify").value.level
                score = round((score + lvl / 2) / 2, 3)
            except StructuredOutputError:
                pass
        idx = 0 if score < 0.35 else 1 if score < 0.7 else len(self.tiers) - 1
        return min(idx, len(self.tiers) - 1), score

    @staticmethod
    def _truncate(text: str, max_tokens: int) -> str:
        if estimate_tokens(text) <= max_tokens:
            return text
        keep = max_tokens * 4
        return text[: keep // 2] + "\n...[truncated to fit token budget]...\n" + text[-keep // 2:]

    def _estimate_cost(self, tier: Tier, prompt: str) -> float:
        # worst case: full output cap is used (reasoning tokens count as output)
        return config.cost_usd(tier.model, estimate_tokens(prompt) + 150, tier.max_output_tokens)

    # ---------------------------------------------------------------- main
    def run(self, task: str, budget: TaskBudget | None = None, context: str = "") -> Decision:
        budget = budget or TaskBudget()
        t0 = time.perf_counter()
        idx, score = self.start_tier(task)
        d = Decision(uuid.uuid4().hex[:10], task, score, self.tiers[idx].name)
        prompt = self._truncate(f"{context}\n\nTask: {task}".strip(), budget.max_input_tokens)
        spent_before = len(self.llm.history)
        best: RoutedAnswer | None = None

        for tier in self.tiers[idx:]:
            spent = sum(u.cost_usd for u in self.llm.history[spent_before:])
            if best is not None and spent + self._estimate_cost(tier, prompt) > budget.max_cost_usd:
                d.budget_exhausted = True
                break
            d.tiers_tried.append(tier.name)
            try:
                ans = generate_structured(self.llm, [
                    {"role": "system", "content": "Answer the task. Then rate your confidence honestly; set "
                                                  "needs_more_reasoning=true if you are unsure or the task is hard."},
                    {"role": "user", "content": prompt}], RoutedAnswer, max_retries=1, model=tier.model,
                    reasoning_effort=tier.reasoning_effort, max_completion_tokens=tier.max_output_tokens,
                    tag=f"p07.{tier.name}").value
            except StructuredOutputError:
                continue
            if best is None or ans.confidence >= best.confidence:
                best, d.final_tier = ans, tier.name
            if ans.confidence >= self.threshold and not ans.needs_more_reasoning:
                d.early_exit = tier is not self.tiers[-1]
                break

        calls = self.llm.history[spent_before:]
        d.prompt_tokens = sum(u.prompt_tokens for u in calls)
        d.completion_tokens = sum(u.completion_tokens for u in calls)
        d.cost_usd = sum(u.cost_usd for u in calls)
        # counterfactual: one call to the baseline model with the same prompt/answer size
        last = calls[-1] if calls else None
        d.baseline_cost_usd = config.cost_usd(self.baseline_model, last.prompt_tokens if last else 0,
                                              last.completion_tokens if last else 0)
        d.answer = best.answer if best else "No answer within budget."
        d.confidence = best.confidence if best else 0.0
        d.latency_ms = (time.perf_counter() - t0) * 1000
        self._log(d)
        return d

    def _log(self, d: Decision) -> None:
        self.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (d.id, time.time(), d.task[:500], d.complexity, d.start_tier, ",".join(d.tiers_tried),
                         d.final_tier, d.confidence, int(d.early_exit), int(d.budget_exhausted), d.prompt_tokens,
                         d.completion_tokens, d.cost_usd, d.baseline_cost_usd, d.latency_ms))
        self.db.commit()

    # ---------------------------------------------------------------- analytics
    def report(self) -> dict:
        rows = self.db.execute("SELECT final_tier, tiers_tried, confidence, early_exit, budget_exhausted, cost, "
                               "baseline_cost, latency_ms, prompt_tokens + completion_tokens FROM decisions").fetchall()
        if not rows:
            return {"decisions": 0}
        n = len(rows)
        cost = sum(r[5] for r in rows)
        baseline = sum(r[6] for r in rows)
        dist: dict[str, int] = {}
        for r in rows:
            dist[r[0] or "none"] = dist.get(r[0] or "none", 0) + 1
        lat = sorted(r[7] for r in rows)
        return {
            "decisions": n,
            "total_cost_usd": round(cost, 6),
            "cost_per_decision_usd": round(cost / n, 7),
            "baseline_cost_usd": round(baseline, 6),
            "estimated_savings_pct": round(100 * (1 - cost / baseline), 1) if baseline else None,
            "avg_tokens_per_decision": round(sum(r[8] for r in rows) / n),
            "final_tier_distribution": dist,
            "escalation_rate": round(sum(1 for r in rows if "," in (r[1] or "")) / n, 3),
            "early_exit_rate": round(sum(r[3] for r in rows) / n, 3),
            "budget_exhausted_rate": round(sum(r[4] for r in rows) / n, 3),
            "avg_confidence": round(sum(r[2] for r in rows) / n, 3),
            "p50_latency_ms": round(lat[n // 2]), "p95_latency_ms": round(lat[min(n - 1, int(n * 0.95))]),
        }
