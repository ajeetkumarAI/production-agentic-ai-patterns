import json
import time

import pytest
from pydantic import ValidationError

from common.embeddings import HashingEmbedder
from common.llm import FakeLLM
from p03_react_planning.agent import ReActAgent
from p04_multi_tool_orchestrator.orchestrator import Orchestrator, Principal, ToolRegistry, resolve_conflict
from p04_multi_tool_orchestrator.tools_demo import ALERTS_SENT, demo_tools
from p05_memory_agent.agent import MemoryAgent
from p05_memory_agent.memory import MemoryStore, ShortTermBuffer, sync
from p06_hitl_approval.core import HITLAgent, Policy


def step(thought, action, **inp):
    return json.dumps({"thought": thought, "action": action, "action_input": inp})


OK_CRITIQUE = json.dumps({"acceptable": True, "issues": [], "suggestion": ""})


# ------------------------------------------------------------------ project 3
def test_p03_happy_path_uses_tools_and_critique():
    llm = FakeLLM([step("convert", "convert_to_inr", amount=120, currency="USD"),
                   step("done", "final_answer", answer="USD 120 = INR 9984"), OK_CRITIQUE])
    res = ReActAgent(llm, reflect_every=0).run("Convert USD 120 to INR")
    assert res.status == "success" and "9984" in res.answer
    assert res.steps[0].observation.startswith('{"inr": 9984.0')


def test_p03_loop_detection_degrades_gracefully():
    llm = FakeLLM(default=lambda m, kw: "Best effort: unknown" if not kw.get("response_format")
                  else step("again", "search_policy", query="cf-04"))
    res = ReActAgent(llm, max_repeats=2, reflect_every=0).run("loop please")
    assert res.status == "degraded" and "loop detected" in res.reason and res.iterations == 3


def test_p03_max_iterations_and_critique_rejection():
    llm = FakeLLM(default=lambda m, kw: "partial" if not kw.get("response_format") else (
        json.dumps({"acceptable": False, "issues": ["wrong"], "suggestion": "redo"})
        if "strict reviewer" in m[0]["content"] else step("t", "today")))
    res = ReActAgent(llm, max_iterations=4, max_repeats=10, reflect_every=0).run("q")
    assert res.status == "degraded" and "max iterations" in res.reason


def test_p03_hallucinated_tool_is_repaired():
    llm = FakeLLM([step("x", "launch_rockets"), step("ok", "final_answer", answer="fine")])
    res = ReActAgent(llm, self_critique=False, reflect_every=0).run("q")
    assert res.status == "success"
    assert "launch_rockets" in llm.calls[1]["messages"][-2]["content"]  # repair turn showed the bad output


# ------------------------------------------------------------------ project 4
@pytest.fixture
def orch():
    reg = ToolRegistry()
    for t in demo_tools():
        reg.register(t)
    return Orchestrator(FakeLLM(default="summary"), reg), reg


def test_p04_parallel_failover_conflict_and_permissions(orch):
    o, _ = orch
    plan = [
        {"id": "t1", "capability": "customer_profile", "input": {"customer_id": "C1"}},
        {"id": "t2", "capability": "ip_intel", "input": {"ip": "1.2.3.4"}},
        {"id": "t3", "capability": "fx_rate", "input": {}, "fanout": True},
        {"id": "t4", "capability": "sanctions_check", "input": {"name": "${t1.name}"}, "depends_on": ["t1"]},
        {"id": "t5", "capability": "notify", "input": {"message": "x"}, "depends_on": ["t2"]},
    ]
    t0 = time.perf_counter()
    res = o.run("goal", Principal("a", frozenset({"pii:read"})), tasks=plan)
    assert time.perf_counter() - t0 < 1.4  # t1,t2,t3 ran in parallel (0.3s each)
    assert res.outcomes["t4"].tool == "sanctions_slow" and res.outcomes["t4"].output["name"] == "A. Sharma"
    assert res.outcomes["t3"].output["rate"] == pytest.approx(83.275)
    assert res.conflicts[0].strategy == "numeric_mean"
    assert res.outcomes["t5"].status == "denied" and res.denials[0]["tool"] == "send_alert"


def test_p04_scoped_principal_can_notify_and_dependents_skip(orch):
    o, reg = orch
    n = len(ALERTS_SENT)
    res = o.run("g", Principal("lead", frozenset({"pii:read", "alerts:write"})), tasks=[
        {"id": "a", "capability": "notify", "input": {"message": "hi"}},
        {"id": "b", "capability": "customer_profile", "input": {}},             # invalid args -> failed
        {"id": "c", "capability": "ip_intel", "input": {"ip": "${b.x}"}, "depends_on": ["b"]}])
    assert res.outcomes["a"].status == "ok" and len(ALERTS_SENT) == n + 1
    assert res.outcomes["b"].status == "failed" and res.outcomes["c"].status == "skipped"
    reg.unregister("geo_ip")
    assert "ip_intel" not in reg.capabilities()


def test_p04_conflict_strategies():
    assert resolve_conflict("t", "c", {"a": 1, "b": 1, "c": 2}, {})[1].strategy == "majority"
    val, c = resolve_conflict("t", "c", {"a": {"r": 1.0}, "b": {"r": 5.0}}, {"a": 1, "b": 9})
    assert c.strategy == "priority" and val == {"r": 5.0}


def test_p04_plan_validation_rejects_cycles(orch):
    o, _ = orch
    cyclic = json.dumps({"tasks": [{"id": "a", "capability": "ip_intel", "depends_on": ["b"]},
                                   {"id": "b", "capability": "ip_intel", "depends_on": ["a"]}]})
    good = json.dumps({"tasks": [{"id": "a", "capability": "ip_intel", "input": {"ip": "1.1.1.1"}}]})
    o.llm = FakeLLM([cyclic, good])
    assert o.plan("x")[0]["id"] == "a"
    assert "cycle" in o.llm.calls[1]["messages"][-1]["content"]


# ------------------------------------------------------------------ project 5
def test_p05_dedup_recall_scoring_and_sync(tmp_path):
    a = MemoryStore(tmp_path / "a.db", embedder=HashingEmbedder())
    b = MemoryStore(tmp_path / "b.db", embedder=HashingEmbedder())
    now = time.time()
    a.add("u", "User prefers concise answers with Python code examples", "preference", 0.8, now=now - 90 * 86400)
    _, how = a.add("u", "User prefers concise answers with Python code examples", "preference", 0.9, now=now)
    assert how == "merged" and len(a.all("u")) == 1
    a.add("u", "User is writing a thesis on fraud detection with XGBoost", "goal", 0.9)
    a.add("u", "User's favourite cricket team is Mumbai", "fact", 0.2)
    top = a.recall("u", "which model does my fraud thesis use?", k=1)
    assert "XGBoost" in top[0].text
    assert sync(a, b)["a_to_b"] == 3 and len(b.all("u")) == 3
    a.forget(top[0].id)
    sync(a, b)
    assert len(b.all("u")) == 2  # deletion tombstone propagated


def test_p05_compression_and_cross_session(tmp_path):
    store = MemoryStore(tmp_path / "m.db", embedder=HashingEmbedder())
    extraction = json.dumps({"items": [{"text": "User is preparing for the AI-901 exam", "category": "goal",
                                        "importance": 0.9}]})
    empty = json.dumps({"items": []})

    def respond(messages, kw):
        if kw.get("response_format"):
            return extraction if "AI-901" in messages[-1]["content"] else empty
        sys = messages[0]["content"]
        if "running conversation summary" in sys:
            return "SUMMARY: user studies for AI-901"
        if "Summarize this session" in sys:
            return "Session: discussed AI-901 prep"
        return "reply " + "x" * 400
    llm = FakeLLM(default=respond)
    agent = MemoryAgent(llm, store, "u", buffer_tokens=200)
    for msg in ["I'm preparing for the AI-901 exam", "tell me more", "and more", "more please"]:
        agent.chat(msg)
    assert agent.buffer.compressions >= 1 and "SUMMARY" in agent.buffer.summary
    agent.end_session()
    nxt = MemoryAgent(llm, store, "u")
    nxt.chat("what exam am I preparing for?")
    system = llm.calls[-2]["messages"][0]["content"]
    assert "AI-901" in system and "Session: discussed" in system


def test_p05_short_term_buffer_bounds():
    buf = ShortTermBuffer(max_tokens=50, keep_last=2)
    for i in range(6):
        buf.add("user", "x" * 100)
    assert buf.needs_compression()
    buf.compress(FakeLLM(["sum"]))
    assert len(buf.messages) == 2 and buf.summary == "sum"


# ------------------------------------------------------------------ project 6
def proposal(action, params, conf=0.95, missing=()):
    return json.dumps({"action": action, "params": params, "rationale": "r", "confidence": conf,
                       "missing_info": list(missing)})


def test_p06_auto_executes_when_confident(tmp_path):
    llm = FakeLLM(default=lambda m, kw: proposal("escalate_to_fraud_team", {"summary": "duplicate charge check"})
                  if kw.get("response_format") else "Dear customer...")
    run = HITLAgent(llm, tmp_path / "h.db").start({"msg": "hi"})
    assert run["status"] == "EXECUTED"


def test_p06_pause_validate_resume_and_audit(tmp_path):
    llm = FakeLLM(default=lambda m, kw: proposal("refund", {"txn_id": "T-9", "amount": 72000}, conf=0.6)
                  if kw.get("response_format") else "We refunded part of it.")
    agent = HITLAgent(llm, tmp_path / "h.db", Policy(samples=3))
    run = agent.start({"msg": "refund please"}, requested_by="priya.sr_analyst")
    assert run["status"] == "PENDING_APPROVAL"
    reasons = " ".join(run["uncertainty"]["reasons"])
    assert "low model confidence" in reasons and "exceeds auto limit" in reasons
    with pytest.raises(ValueError, match="not authorized"):
        agent.resume(run["run_id"], {"decision": "approve", "reviewer": "mallory"})
    with pytest.raises(ValueError, match="four-eyes"):
        agent.resume(run["run_id"], {"decision": "approve", "reviewer": "priya.sr_analyst"})
    with pytest.raises(ValidationError):
        agent.resume(run["run_id"], {"decision": "modify", "reviewer": "rahul.fraud_lead",
                                     "comment": "partial refund", "modified_params": {"amount": -5}})
    done = agent.resume(run["run_id"], {"decision": "modify", "reviewer": "rahul.fraud_lead",
                                        "comment": "partial refund only", "modified_params": {"amount": 4000}})
    assert done["status"] == "EXECUTED" and done["execution"]["params"]["amount"] == 4000
    events = [e["event"] for e in agent.audit.trail(run["run_id"])]
    assert events[:3] == ["request_received", "action_proposed", "paused_for_human"]
    assert events.count("invalid_human_input") == 3 and events[-1] == "action_executed"
    assert agent.audit.verify() == (True, None)
    agent.db.execute("UPDATE audit SET data='{\"amount\": 1}' WHERE event='action_executed'")
    assert agent.audit.verify()[0] is False  # tampering detected


def test_p06_self_consistency_disagreement_pauses(tmp_path):
    answers = [proposal("no_action", {}), proposal("escalate_to_fraud_team", {"summary": "looks odd to me"}),
               proposal("no_action", {}, conf=0.9)]
    llm = FakeLLM(answers, default="msg")
    run = HITLAgent(llm, tmp_path / "h.db", Policy(samples=3, min_agreement=0.9)).start({"m": 1})
    assert run["status"] == "PENDING_APPROVAL"
    assert any("disagreement" in r for r in run["uncertainty"]["reasons"])
