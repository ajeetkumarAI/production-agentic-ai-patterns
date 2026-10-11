"""Project 3 - ReAct Planning Agent.

Observe -> Think -> Act -> Reflect loop with hard guarantees that it terminates:
max iterations, token budget, wall-clock timeout, repeated-action (loop)
detection, consecutive-error limit, periodic self-reflection, a self-critique
gate before accepting the final answer, and graceful degradation (best-effort
answer from what was gathered) instead of crashing or spinning forever.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from common.llm import BaseLLM
from common.structured import StructuredOutputError, generate_structured
from common.tools import Tool, ToolError, to_text
from common.utils import safe_eval


class Critique(BaseModel):
    acceptable: bool = Field(description="True if the work fully and correctly answers the question")
    issues: list[str] = Field(default_factory=list)
    suggestion: str = Field(default="", description="What to do next to fix the issues")


@dataclass
class StepRecord:
    i: int
    thought: str
    action: str
    action_input: dict
    observation: str = ""
    kind: str = "act"   # act | reflect | critique | guard


@dataclass
class AgentResult:
    answer: str
    status: Literal["success", "degraded", "failed"]
    reason: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    iterations: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


# ------------------------------------------------------------------ demo tools
_FX_TO_INR = {"USD": 83.2, "EUR": 90.1, "GBP": 105.4, "INR": 1.0}
_KB = {
    "cf-04": "Rule CF-04: accounts younger than 30 days have a daily card-not-present limit of INR 25,000.",
    "cf-01": "Rule CF-01: block when IP country != home country AND amount > 5x average.",
    "chargeback window": "Card-not-present chargebacks: 120 days; card-present: 60 days.",
}


class CalcArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '120*83.2 + 5'")


class FxArgs(BaseModel):
    amount: float
    currency: str = Field(description="ISO code: USD, EUR, GBP, INR")


class SearchArgs(BaseModel):
    query: str


def _search_kb(query: str) -> str:
    q = query.lower()
    hits = [v for k, v in _KB.items() if k in q or any(w in v.lower() for w in q.split() if len(w) > 4)]
    return " | ".join(hits) or "no results"


def _fx(amount: float, currency: str) -> dict:
    cur = currency.upper()
    if cur not in _FX_TO_INR:
        raise ValueError(f"unsupported currency {cur}")
    return {"inr": round(amount * _FX_TO_INR[cur], 2), "rate": _FX_TO_INR[cur]}


def default_tools() -> list[Tool]:
    return [
        Tool("calculator", "Evaluate an arithmetic expression.", lambda expression: safe_eval(expression), CalcArgs),
        Tool("convert_to_inr", "Convert an amount in a currency to INR.", _fx, FxArgs),
        Tool("search_policy", "Search internal fraud policy rules.", _search_kb, SearchArgs),
        Tool("today", "Get today's date.", lambda: date.today().isoformat()),
    ]


# ------------------------------------------------------------------ agent
SYSTEM = """You are a careful ReAct agent. At each step output JSON with:
- thought: your reasoning about what you know and what is still missing
- action: one tool name, or "final_answer" when done
- action_input: arguments for the tool (an object). For final_answer use {{"answer": "..."}}
Use tools for facts and arithmetic; never guess numbers. Don't repeat an action that already succeeded.

Tools:
{tools}"""


class ReActAgent:
    def __init__(self, llm: BaseLLM, tools: list[Tool] | None = None, max_iterations: int = 8,
                 max_tokens: int = 40_000, timeout_s: float = 120.0, max_repeats: int = 2,
                 max_consecutive_errors: int = 3, reflect_every: int = 3, self_critique: bool = True,
                 max_critiques: int = 1):
        self.llm = llm
        self.tools = {t.name: t for t in (tools or default_tools())}
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_repeats = max_repeats
        self.max_consecutive_errors = max_consecutive_errors
        self.reflect_every = reflect_every
        self.self_critique = self_critique
        self.max_critiques = max_critiques
        names = tuple(self.tools) + ("final_answer",)
        # action is constrained to real tool names -> hallucinated tools fail validation and get repaired
        self.Step = create_model("ReActStep", thought=(str, ...), action=(Literal[names], ...),  # type: ignore
                                 action_input=(dict[str, Any], Field(default_factory=dict)))

    # -------------------------------------------------------------- helpers
    def _tool_doc(self) -> str:
        return "\n".join(f"- {t.name}: {t.description} args={json.dumps(t.to_openai()['function']['parameters'].get('properties', {}))}"
                         for t in self.tools.values())

    def _tokens_used(self, start_idx: int) -> int:
        return sum(u.total_tokens for u in self.llm.history[start_idx:])

    def _critique(self, question: str, answer: str, steps: list[StepRecord]) -> Critique:
        transcript = "\n".join(f"{s.action}({json.dumps(s.action_input)}) -> {s.observation[:300]}" for s in steps
                               if s.kind == "act")
        msgs = [{"role": "system", "content": "You are a strict reviewer. Check the answer against the evidence. "
                                              "Mark acceptable=false if it is unsupported, incomplete or has arithmetic errors."},
                {"role": "user", "content": f"Question: {question}\nEvidence:\n{transcript}\nProposed answer: {answer}"}]
        return generate_structured(self.llm, msgs, Critique, max_retries=1, tag="p03.critique").value

    def _degrade(self, question: str, steps: list[StepRecord], reason: str, start_idx: int, i: int) -> AgentResult:
        evidence = "\n".join(f"- {s.action}: {s.observation[:300]}" for s in steps if s.kind == "act" and s.observation)
        try:
            text = self.llm.chat([
                {"role": "system", "content": "The agent was stopped early. Give the best possible answer using ONLY "
                                              "the evidence, and clearly state what remains uncertain."},
                {"role": "user", "content": f"Question: {question}\nStopped because: {reason}\nEvidence:\n{evidence or '(none)'}"},
            ], tag="p03.degrade", max_completion_tokens=2000).content or ""
            status = "degraded"
        except Exception as e:  # noqa: BLE001
            text, status = f"Unable to complete ({reason}). Partial evidence:\n{evidence or 'none'} [{e}]", "failed"
        return self._result(text, status, reason, steps, i, start_idx)

    def _result(self, answer, status, reason, steps, i, start_idx) -> AgentResult:
        hist = self.llm.history[start_idx:]
        return AgentResult(answer=answer, status=status, reason=reason, steps=steps, iterations=i,
                           tokens=sum(u.total_tokens for u in hist), cost_usd=sum(u.cost_usd for u in hist))

    # -------------------------------------------------------------- main loop
    def run(self, question: str) -> AgentResult:
        start_idx, t0 = len(self.llm.history), time.monotonic()
        messages = [{"role": "system", "content": SYSTEM.format(tools=self._tool_doc())},
                    {"role": "user", "content": f"Question: {question}"}]
        steps: list[StepRecord] = []
        seen: Counter = Counter()
        consecutive_errors = critiques = 0

        for i in range(1, self.max_iterations + 1):
            if self._tokens_used(start_idx) > self.max_tokens:
                return self._degrade(question, steps, "token budget exhausted", start_idx, i)
            if time.monotonic() - t0 > self.timeout_s:
                return self._degrade(question, steps, "timeout", start_idx, i)

            try:
                step = generate_structured(self.llm, messages, self.Step, max_retries=1, tag="p03.step").value
            except StructuredOutputError as e:
                consecutive_errors += 1
                steps.append(StepRecord(i, "", "invalid_step", {}, str(e), kind="guard"))
                if consecutive_errors >= self.max_consecutive_errors:
                    return self._degrade(question, steps, "model kept producing invalid steps", start_idx, i)
                continue
            messages.append({"role": "assistant", "content": step.model_dump_json()})

            # ---------------- final answer (+ self-critique gate)
            if step.action == "final_answer":
                answer = str(step.action_input.get("answer", "")).strip() or step.thought
                rec = StepRecord(i, step.thought, "final_answer", step.action_input, kind="act")
                steps.append(rec)
                if self.self_critique and critiques < self.max_critiques:
                    critiques += 1
                    try:
                        crit = self._critique(question, answer, steps)
                    except StructuredOutputError:
                        crit = Critique(acceptable=True)
                    steps.append(StepRecord(i, "self-critique", "critique", {}, crit.model_dump_json(), kind="critique"))
                    if not crit.acceptable:
                        messages.append({"role": "user", "content": (
                            f"Self-critique rejected the answer. Issues: {crit.issues}. "
                            f"Suggestion: {crit.suggestion}. Fix this, then give a new final_answer.")})
                        continue
                return self._result(answer, "success", "", steps, i, start_idx)

            # ---------------- loop detection
            sig = (step.action, json.dumps(step.action_input, sort_keys=True))
            seen[sig] += 1
            if seen[sig] > self.max_repeats:
                steps.append(StepRecord(i, step.thought, step.action, step.action_input, "loop detected", kind="guard"))
                return self._degrade(question, steps, f"loop detected: {step.action} repeated {seen[sig]}x", start_idx, i)

            # ---------------- act + observe
            try:
                obs = to_text(self.tools[step.action](step.action_input), 2000)
                consecutive_errors = 0
            except ToolError as e:
                obs = f"ERROR: {e}"
                consecutive_errors += 1
            if seen[sig] == self.max_repeats:
                obs += "\nWARNING: you already ran this exact action. Use the result above or try something else."
            steps.append(StepRecord(i, step.thought, step.action, step.action_input, obs))
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            if consecutive_errors >= self.max_consecutive_errors:
                return self._degrade(question, steps, f"{consecutive_errors} consecutive tool errors", start_idx, i)

            # ---------------- periodic reflection
            if self.reflect_every and i % self.reflect_every == 0:
                try:
                    crit = self._critique(question, "(in progress - judge whether the plan is on track)", steps)
                    note = f"Reflection: on_track={crit.acceptable}; issues={crit.issues}; next={crit.suggestion}"
                except StructuredOutputError:
                    note = "Reflection: (unavailable)"
                steps.append(StepRecord(i, "reflection", "reflect", {}, note, kind="reflect"))
                messages.append({"role": "user", "content": note})

        return self._degrade(question, steps, f"max iterations ({self.max_iterations}) reached", start_idx,
                             self.max_iterations)
