"""Project 4 - Multi-Tool Orchestrator Agent.

* Dynamic tool registry - tools are registered/unregistered at runtime and
  advertise *capabilities*; the planner never sees tool names.
* Capability-based routing - each task names a capability; the router picks the
  best permitted tool (priority order) and fails over to the next on error.
* Permission scoping - a Principal's scopes are checked against each tool's
  required scopes; denials are recorded, never silently skipped.
* Parallel execution - the plan is a DAG; independent tasks run concurrently.
* Conflict resolution - fan-out tasks query every provider; disagreeing results
  are resolved (majority / numeric tolerance / priority) and the conflict is reported.
  Side-effecting tools share resource locks so concurrent writes never interleave.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model, model_validator

from common.llm import BaseLLM
from common.structured import StructuredOutputError, generate_structured
from common.tools import Tool, ToolError, to_text


@dataclass(frozen=True)
class Principal:
    user: str
    scopes: frozenset[str] = frozenset()


class PermissionDenied(Exception):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.RLock()

    def register(self, tool: Tool) -> None:
        with self._lock:
            self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def capabilities(self) -> dict[str, list[str]]:
        caps: dict[str, list[str]] = defaultdict(list)
        with self._lock:
            for t in self._tools.values():
                for c in t.capabilities:
                    caps[c].append(t.description)
        return dict(caps)

    def candidates(self, capability: str, principal: Principal) -> tuple[list[Tool], list[str]]:
        """Permitted tools for a capability (best first) + names of tools denied by scope."""
        with self._lock:
            tools = [t for t in self._tools.values() if capability in t.capabilities]
        allowed = [t for t in tools if t.required_scopes <= principal.scopes]
        denied = [t.name for t in tools if not t.required_scopes <= principal.scopes]
        return sorted(allowed, key=lambda t: -t.priority), denied


# ------------------------------------------------------------------ conflict resolution
@dataclass
class Conflict:
    task_id: str
    capability: str
    results: dict[str, Any]
    strategy: str
    chosen: Any
    note: str = ""


def _numbers(x: Any) -> dict[str, float]:
    if isinstance(x, (int, float)):
        return {"": float(x)}
    if isinstance(x, dict):
        return {k: float(v) for k, v in x.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {}


def resolve_conflict(task_id: str, capability: str, results: dict[str, Any], priorities: dict[str, int],
                     rel_tolerance: float = 0.02) -> tuple[Any, Conflict | None]:
    """results: tool_name -> output. Returns (chosen value, conflict record or None)."""
    if len(results) == 1:
        return next(iter(results.values())), None
    canon = {name: json.dumps(v, sort_keys=True, default=str) for name, v in results.items()}
    counts = Counter(canon.values())
    if len(counts) == 1:
        return next(iter(results.values())), None
    top, n = counts.most_common(1)[0]
    if n > len(results) / 2:  # strict majority agrees
        chosen = json.loads(top)
        return chosen, Conflict(task_id, capability, results, "majority", chosen, f"{n}/{len(results)} agreed")
    nums = {name: _numbers(v) for name, v in results.items()}
    keys = set.intersection(*(set(d) for d in nums.values())) if all(nums.values()) else set()
    if keys:
        spread_ok = all(
            (max(d[k] for d in nums.values()) - min(d[k] for d in nums.values()))
            <= rel_tolerance * max(abs(max(d[k] for d in nums.values())), 1e-9) for k in keys)
        if spread_ok:
            best = max(results, key=lambda n: priorities.get(n, 0))
            chosen = dict(results[best]) if isinstance(results[best], dict) else results[best]
            for k in keys:
                mean = round(sum(d[k] for d in nums.values()) / len(nums), 6)
                if isinstance(chosen, dict):
                    chosen[k] = mean
                else:
                    chosen = mean
            return chosen, Conflict(task_id, capability, results, "numeric_mean", chosen,
                                    f"values within {rel_tolerance:.0%}; averaged")
    best = max(results, key=lambda n: priorities.get(n, 0))
    return results[best], Conflict(task_id, capability, results, "priority", results[best],
                                   f"irreconcilable; trusted highest-priority provider '{best}'")


# ------------------------------------------------------------------ orchestrator
@dataclass
class TaskOutcome:
    task_id: str
    capability: str
    status: Literal["ok", "failed", "denied", "skipped"]
    output: Any = None
    tool: str | None = None
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class OrchestrationResult:
    answer: str
    plan: list[dict]
    outcomes: dict[str, TaskOutcome]
    conflicts: list[Conflict] = field(default_factory=list)
    denials: list[dict] = field(default_factory=list)
    wall_ms: float = 0.0


_REF = re.compile(r"\$\{(\w+)(?:\.([\w.]+))?\}")


def _resolve_refs(value: Any, outputs: dict[str, Any]) -> Any:
    """Replace "${t1.field}" placeholders with outputs of earlier tasks."""
    if isinstance(value, dict):
        return {k: _resolve_refs(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v, outputs) for v in value]
    if not isinstance(value, str):
        return value

    def lookup(tid: str, path: str | None):
        cur = outputs.get(tid)
        for part in (path.split(".") if path else []):
            cur = cur.get(part) if isinstance(cur, dict) else None
        return cur

    m = _REF.fullmatch(value.strip())
    if m:
        return lookup(m.group(1), m.group(2))
    return _REF.sub(lambda mm: str(lookup(mm.group(1), mm.group(2))), value)


class Orchestrator:
    def __init__(self, llm: BaseLLM, registry: ToolRegistry, max_workers: int = 8):
        self.llm = llm
        self.registry = registry
        self.max_workers = max_workers
        self._resource_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    # -------------------------------------------------------------- planning
    def _plan_model(self):
        caps = tuple(self.registry.capabilities()) or ("none",)
        Task = create_model("PlanTask", id=(str, Field(description="short id like t1")),
                            capability=(Literal[caps], ...),  # type: ignore[valid-type]
                            input=(dict[str, Any], Field(default_factory=dict)),
                            depends_on=(list[str], Field(default_factory=list)),
                            fanout=(bool, Field(False, description="query ALL providers and reconcile")))

        class Plan(BaseModel):
            tasks: list[Task] = Field(min_length=1, max_length=12)  # type: ignore[valid-type]

            @model_validator(mode="after")
            def dag(self):
                ids = [t.id for t in self.tasks]
                if len(set(ids)) != len(ids):
                    raise ValueError("task ids must be unique")
                for t in self.tasks:
                    unknown = set(t.depends_on) - set(ids)
                    if unknown:
                        raise ValueError(f"{t.id} depends on unknown tasks {unknown}")
                _layers([x.model_dump() for x in self.tasks])  # raises on cycles
                return self
        return Plan

    def plan(self, goal: str) -> list[dict]:
        caps = self.registry.capabilities()
        catalog = "\n".join(f"- {c}: {' / '.join(sorted(set(d)))}" for c, d in caps.items())
        msgs = [{"role": "system", "content": (
            "You are a planner. Break the goal into tasks, each using ONE capability from the catalog. "
            "Make tasks independent where possible so they run in parallel. Use depends_on only when a task "
            "needs another's output, and reference that output inside input as \"${t1}\" or \"${t1.field}\". "
            "Set fanout=true for factual lookups where several providers exist and accuracy matters.\n"
            f"Capability catalog:\n{catalog}")},
            {"role": "user", "content": goal}]
        return [t.model_dump() for t in generate_structured(self.llm, msgs, self._plan_model(), tag="p04.plan").value.tasks]

    # -------------------------------------------------------------- execution
    def _call_tool(self, tool: Tool, args: dict) -> Any:
        def invoke():
            if tool.side_effects:
                with self._resource_locks[tool.name]:
                    return tool(args)
            return tool(args)
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(invoke)
            try:
                return fut.result(timeout=tool.timeout_s)
            except FutTimeout as e:
                raise ToolError(f"{tool.name} timed out after {tool.timeout_s}s") from e

    def _run_task(self, task: dict, principal: Principal, outputs: dict, conflicts: list, denials: list) -> TaskOutcome:
        t0 = time.perf_counter()
        cap = task["capability"]
        tools, denied = self.registry.candidates(cap, principal)
        for name in denied:
            denials.append({"task": task["id"], "tool": name, "user": principal.user, "capability": cap})
        if not tools:
            status = "denied" if denied else "failed"
            return TaskOutcome(task["id"], cap, status, error=f"no permitted tool for capability '{cap}'"
                               + (f" (denied: {denied})" if denied else ""))
        args = _resolve_refs(task.get("input", {}), outputs)
        errors: list[str] = []
        if task.get("fanout") and len(tools) > 1:
            results: dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=len(tools)) as ex:
                futs = {ex.submit(self._call_tool, t, args): t for t in tools}
                for f, t in futs.items():
                    try:
                        results[t.name] = f.result()
                    except ToolError as e:
                        errors.append(str(e))
            if results:
                chosen, conflict = resolve_conflict(task["id"], cap, results, {t.name: t.priority for t in tools})
                if conflict:
                    conflicts.append(conflict)
                return TaskOutcome(task["id"], cap, "ok", chosen, ",".join(results),
                                   "; ".join(errors), (time.perf_counter() - t0) * 1000)
        else:
            for t in tools:  # failover in priority order
                try:
                    out = self._call_tool(t, args)
                    return TaskOutcome(task["id"], cap, "ok", out, t.name, "; ".join(errors),
                                       (time.perf_counter() - t0) * 1000)
                except ToolError as e:
                    errors.append(str(e))
        return TaskOutcome(task["id"], cap, "failed", error="; ".join(errors), duration_ms=(time.perf_counter() - t0) * 1000)

    def execute(self, tasks: list[dict], principal: Principal) -> tuple[dict[str, TaskOutcome], list, list]:
        outcomes: dict[str, TaskOutcome] = {}
        outputs: dict[str, Any] = {}
        conflicts: list[Conflict] = []
        denials: list[dict] = []
        by_id = {t["id"]: t for t in tasks}
        for layer in _layers(tasks):
            runnable, skipped = [], []
            for tid in layer:
                deps = by_id[tid].get("depends_on", [])
                (skipped if any(outcomes[d].status != "ok" for d in deps) else runnable).append(tid)
            for tid in skipped:
                outcomes[tid] = TaskOutcome(tid, by_id[tid]["capability"], "skipped", error="dependency failed")
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = {tid: ex.submit(self._run_task, by_id[tid], principal, outputs, conflicts, denials)
                        for tid in runnable}
                for tid, f in futs.items():
                    outcomes[tid] = f.result()
                    if outcomes[tid].status == "ok":
                        outputs[tid] = outcomes[tid].output
        return outcomes, conflicts, denials

    # -------------------------------------------------------------- end to end
    def run(self, goal: str, principal: Principal, tasks: list[dict] | None = None) -> OrchestrationResult:
        t0 = time.perf_counter()
        try:
            tasks = tasks or self.plan(goal)
        except StructuredOutputError as e:
            return OrchestrationResult(f"Planning failed: {e}", [], {})
        outcomes, conflicts, denials = self.execute(tasks, principal)
        report = {tid: {"capability": o.capability, "status": o.status, "tool": o.tool,
                        "output": o.output if o.status == "ok" else None, "error": o.error}
                  for tid, o in outcomes.items()}
        notes = [f"{c.task_id}: providers disagreed, resolved by {c.strategy} ({c.note})" for c in conflicts]
        answer = self.llm.chat([
            {"role": "system", "content": "Synthesize the task results into a concise answer for the goal. "
                                          "Explicitly mention failed/denied tasks and any provider conflicts."},
            {"role": "user", "content": f"Goal: {goal}\nResults: {to_text(report, 6000)}\nConflicts: {notes}"},
        ], tag="p04.synthesize").content or ""
        return OrchestrationResult(answer, tasks, outcomes, conflicts, denials, (time.perf_counter() - t0) * 1000)


def _layers(tasks: list[dict]) -> list[list[str]]:
    """Topological layers (Kahn). Raises ValueError on cycles."""
    deps = {t["id"]: set(t.get("depends_on", [])) for t in tasks}
    layers, done = [], set()
    while len(done) < len(deps):
        layer = [tid for tid, d in deps.items() if tid not in done and d <= done]
        if not layer:
            raise ValueError("plan has a dependency cycle")
        layers.append(layer)
        done.update(layer)
    return layers
