"""Project 10 - Self-Reflective Agent with Auto-Eval.

execute -> evaluate (LLM-as-judge on a rubric + deterministic checks)
        -> critique -> regenerate under accumulated constraints -> ... (keep the best)
Every iteration's metrics are logged; constraints that produced improvements
are *learned* per task type and applied up-front next time, so first-attempt
quality rises over runs (see ``improvement_report``).
"""
from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from common import config
from common.llm import BaseLLM
from common.structured import StructuredOutputError, generate_structured
from common.utils import append_jsonl, read_jsonl

METRICS_LOG = config.DATA_DIR / "p10_metrics.jsonl"
LEARNED_PATH = config.DATA_DIR / "p10_learned_constraints.json"


# ------------------------------------------------------------------ rubric + checks
@dataclass(frozen=True)
class Criterion:
    name: str
    description: str
    weight: float = 1.0


DEFAULT_RUBRIC = [
    Criterion("correctness", "Factually and technically correct, no hallucinated claims", 2.0),
    Criterion("completeness", "Addresses every part of the task", 1.5),
    Criterion("clarity", "Well structured, concise, easy to follow", 1.0),
    Criterion("actionability", "Gives concrete, usable steps or examples", 1.0),
]


@dataclass
class Check:
    """Deterministic, hard constraint. fn(answer) -> (passed, message)."""
    name: str
    fn: Callable[[str], tuple[bool, str]]
    rule: str = ""   # instruction added as a constraint when the check fails


def max_words(n: int) -> Check:
    return Check(f"max_words_{n}", lambda a: (len(a.split()) <= n, f"answer has {len(a.split())} words; limit is {n}"),
                 f"Keep the answer under {n} words")


def must_include(*terms: str) -> Check:
    def fn(a: str):
        missing = [t for t in terms if t.lower() not in a.lower()]
        return (not missing, f"must mention: {missing}")
    return Check(f"include_{'_'.join(terms)[:30]}", fn, f"Explicitly mention: {', '.join(terms)}")


def min_bullets(n: int) -> Check:
    return Check(f"min_bullets_{n}", lambda a: (len(re.findall(r"^\s*(?:[-*•]|\d+\.)\s", a, re.M)) >= n,
                                                f"needs at least {n} bullet points"),
                 f"Use at least {n} bullet points")


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=5)
    issue: str = Field(default="", description="What is wrong / missing (empty if score is 5)")


class JudgeVerdict(BaseModel):
    scores: list[CriterionScore]
    critique: str
    suggested_constraints: list[str] = Field(default_factory=list, max_length=4,
                                             description="Short imperative rules that would fix the issues, "
                                                         "e.g. 'Include a code example'")


@dataclass
class Iteration:
    n: int
    answer: str
    score: float
    passed: bool
    criteria: dict[str, int]
    failed_checks: list[str]
    critique: str
    constraints_used: list[str]


@dataclass
class ReflectionResult:
    best_answer: str
    best_score: float
    passed: bool
    iterations: list[Iteration] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        return round(self.best_score - self.iterations[0].score, 3) if self.iterations else 0.0


# ------------------------------------------------------------------ learned constraints
class ConstraintMemory:
    def __init__(self, path: Path = LEARNED_PATH):
        self.path = path
        self.data: dict[str, dict[str, dict]] = json.loads(path.read_text()) if path.exists() else {}

    def top(self, task_type: str, k: int = 3, min_gain: float = 0.01) -> list[str]:
        items = self.data.get(task_type, {})
        ranked = sorted(items.items(), key=lambda kv: -kv[1]["total_gain"] / max(kv[1]["uses"], 1))
        return [c for c, s in ranked if s["total_gain"] / max(s["uses"], 1) >= min_gain][:k]

    def credit(self, task_type: str, constraints: list[str], gain: float) -> None:
        bucket = self.data.setdefault(task_type, {})
        for c in constraints:
            s = bucket.setdefault(c, {"uses": 0, "total_gain": 0.0})
            s["uses"] += 1
            s["total_gain"] = round(s["total_gain"] + gain, 4)
        self.path.write_text(json.dumps(self.data, indent=1))


# ------------------------------------------------------------------ agent
class SelfReflectiveAgent:
    def __init__(self, llm: BaseLLM, rubric: list[Criterion] | None = None, threshold: float = 0.8,
                 max_iterations: int = 3, memory: ConstraintMemory | None = None, metrics_log: Path = METRICS_LOG,
                 judge_model: str | None = None):
        self.llm = llm
        self.rubric = rubric or DEFAULT_RUBRIC
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.memory = memory or ConstraintMemory()
        self.metrics_log = metrics_log
        self.judge_model = judge_model  # optionally use a different (stronger) model as judge
        self._last_failed_rules: list[str] = []

    def _generate(self, task: str, constraints: list[str], previous: Iteration | None) -> str:
        msgs = [{"role": "system", "content": "You are an expert assistant. Produce the best possible answer."}]
        if constraints:
            msgs.append({"role": "system", "content": "Hard constraints:\n" + "\n".join(f"- {c}" for c in constraints)})
        msgs.append({"role": "user", "content": task})
        if previous:
            msgs.append({"role": "assistant", "content": previous.answer})
            msgs.append({"role": "user", "content": (
                f"An evaluator reviewed your answer (score {previous.score:.2f}).\nCritique: {previous.critique}\n"
                f"Failed checks: {previous.failed_checks}\nRewrite the answer from scratch fixing every issue "
                "while respecting all constraints.")})
        return self.llm.chat(msgs, tag="p10.generate").content or ""

    def evaluate(self, task: str, answer: str, checks: list[Check]) -> tuple[float, dict, list[str], JudgeVerdict]:
        rubric_txt = "\n".join(f"- {c.name}: {c.description}" for c in self.rubric)
        kw = {"model": self.judge_model} if self.judge_model else {}
        try:
            verdict = generate_structured(self.llm, [
                {"role": "system", "content": "You are a strict evaluator (LLM-as-judge). Score the answer 1-5 on each "
                                              f"criterion. Be critical; 5 means flawless.\nRubric:\n{rubric_txt}"},
                {"role": "user", "content": f"TASK:\n{task}\n\nANSWER:\n{answer}"}],
                JudgeVerdict, max_retries=1, tag="p10.judge", **kw).value
        except StructuredOutputError:
            verdict = JudgeVerdict(scores=[], critique="judge unavailable")
        by_name = {s.criterion.lower(): s.score for s in verdict.scores}
        total_w = sum(c.weight for c in self.rubric)
        judge_score = sum(c.weight * (by_name.get(c.name, 3) - 1) / 4 for c in self.rubric) / total_w
        failed_checks = [(ch, msg) for ch in checks for ok, msg in [ch.fn(answer)] if not ok]
        self._last_failed_rules = [ch.rule or ch.name for ch, _ in failed_checks]
        failed = [msg for _, msg in failed_checks]
        score = judge_score * (1 - 0.15 * len(failed))  # each failed hard check costs 15%
        return round(max(score, 0.0), 3), {c.name: by_name.get(c.name, 0) for c in self.rubric}, failed, verdict

    def run(self, task: str, task_type: str = "general", checks: list[Check] | None = None,
            constraints: list[str] | None = None) -> ReflectionResult:
        checks = checks or []
        run_id = uuid.uuid4().hex[:8]
        learned = self.memory.top(task_type)
        active = list(dict.fromkeys((constraints or []) + learned))
        result = ReflectionResult("", -1.0, False)
        prev: Iteration | None = None
        start_hist = len(self.llm.history)
        for n in range(1, self.max_iterations + 1):
            answer = self._generate(task, active, prev)
            score, criteria, failed, verdict = self.evaluate(task, answer, checks)
            passed = score >= self.threshold and not failed
            it = Iteration(n, answer, score, passed, criteria, failed, verdict.critique, list(active))
            result.iterations.append(it)
            append_jsonl(self.metrics_log, {"run_id": run_id, "task_type": task_type, "iteration": n, "score": score,
                                            "passed": passed, "criteria": criteria, "failed_checks": failed,
                                            "constraints": active, "learned_applied": learned,
                                            "tokens": sum(u.total_tokens for u in self.llm.history[start_hist:])})
            if score > result.best_score:
                result.best_answer, result.best_score = answer, score
            if prev is not None:  # credit constraints added since the previous iteration
                new = [c for c in active if c not in prev.constraints_used]
                if new:
                    self.memory.credit(task_type, new, score - prev.score)
            if passed:
                result.passed = True
                break
            # derive new constraints from the critique and failed checks
            for c in self._last_failed_rules + verdict.suggested_constraints:
                if c not in active:
                    active.append(c)
            prev = it
        append_jsonl(self.metrics_log, {"run_id": run_id, "task_type": task_type, "summary": True,
                                        "first_score": result.iterations[0].score, "best_score": result.best_score,
                                        "improvement": result.improvement, "iterations": len(result.iterations),
                                        "passed": result.passed})
        return result


def improvement_report(path: Path = METRICS_LOG) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in read_jsonl(path):
        if r.get("summary"):
            by_type[r["task_type"]].append(r)
    out = {}
    for t, runs in by_type.items():
        n = len(runs)
        half = max(1, n // 2)
        out[t] = {
            "runs": n,
            "pass_rate": round(sum(r["passed"] for r in runs) / n, 3),
            "avg_first_score": round(sum(r["first_score"] for r in runs) / n, 3),
            "avg_best_score": round(sum(r["best_score"] for r in runs) / n, 3),
            "avg_improvement_within_run": round(sum(r["improvement"] for r in runs) / n, 3),
            "avg_iterations": round(sum(r["iterations"] for r in runs) / n, 2),
            # learning across runs: does the FIRST attempt get better over time?
            "first_score_early_runs": round(sum(r["first_score"] for r in runs[:half]) / half, 3),
            "first_score_recent_runs": round(sum(r["first_score"] for r in runs[-half:]) / half, 3),
        }
    return out
