import json
import time

import pytest

from common.llm import FakeLLM, LLMResponse, Usage
from p07_cost_router.router import CostAwareRouter, TaskBudget, Tier, heuristic_complexity
from p08_event_automation.queue_store import JobQueue
from p08_event_automation.workflows import Integrations, Worker, make_transaction_workflow
from p09_multi_agent_debate.debate import DebateSystem
from p10_self_reflective.agent import ConstraintMemory, SelfReflectiveAgent, improvement_report, max_words, must_include
from p11_observability.ops import Alerter, DeploymentManager, MetricsStore
from p11_observability.service import AgentVersion, ProductionAgent
from p11_observability.tracing import Tracer


# ------------------------------------------------------------------ project 7
TIERS = [Tier("t0", "gpt-5-nano", "minimal", 500), Tier("t1", "gpt-5-nano", "low", 1000),
         Tier("t2", "gpt-5-mini", "medium", 2000)]


def ans(conf, more=False):
    return json.dumps({"answer": f"a{conf}", "confidence": conf, "needs_more_reasoning": more})


def test_p07_complexity_heuristic_orders_tasks():
    assert heuristic_complexity("Translate hello") < 0.35
    assert heuristic_complexity("Design a step-by-step algorithm and compare trade-offs, estimate cost. "
                                "Why? Prove it handles edge cases.") > 0.6


def test_p07_early_exit_and_escalation(tmp_path):
    r = CostAwareRouter(FakeLLM([ans(0.95)]), TIERS, use_llm_classifier=False, db_path=tmp_path / "r.db")
    d = r.run("Translate hello")
    assert d.tiers_tried == ["t0"] and d.early_exit
    llm = FakeLLM([ans(0.4), ans(0.6, True), ans(0.9)])
    r2 = CostAwareRouter(llm, TIERS, use_llm_classifier=False, db_path=tmp_path / "r.db")
    d2 = r2.run("Translate bye")
    assert d2.tiers_tried == ["t0", "t1", "t2"] and d2.final_tier == "t2"
    assert [c["model"] for c in llm.calls] == ["gpt-5-nano", "gpt-5-nano", "gpt-5-mini"]
    assert [c["reasoning_effort"] for c in llm.calls] == ["minimal", "low", "medium"]
    rep = r2.report()
    assert rep["decisions"] == 2 and rep["escalation_rate"] == 0.5 and rep["estimated_savings_pct"] > 0


def test_p07_budget_stops_escalation(tmp_path):
    r = CostAwareRouter(FakeLLM([ans(0.3), ans(0.99)]), TIERS, use_llm_classifier=False, db_path=tmp_path / "b.db")
    d = r.run("hello", TaskBudget(max_cost_usd=0.0000001))
    assert d.budget_exhausted and d.tiers_tried == ["t0"] and d.answer == "a0.3"


def test_p07_truncates_long_context(tmp_path):
    llm = FakeLLM([ans(0.95)])
    CostAwareRouter(llm, TIERS, use_llm_classifier=False, db_path=tmp_path / "c.db").run(
        "hi", TaskBudget(max_input_tokens=100), context="x" * 10_000)
    assert "truncated" in llm.calls[0]["messages"][-1]["content"]


# ------------------------------------------------------------------ project 8
def wf_llm():
    return FakeLLM(default=json.dumps({"risk": "high", "reason": "foreign IP and large amount"}))


def test_p08_idempotent_retry_and_dead_letters(tmp_path):
    q = JobQueue(tmp_path / "q.db", base_backoff_s=0.01, max_attempts=4)
    integ = Integrations(flaky_failures=2)
    w = Worker(q, {"transaction.created": make_transaction_workflow(wf_llm(), integ)}, poll_s=0.01)
    ev = {"txn_id": "T1", "customer_id": "C1", "amount": 90000, "merchant": "M"}
    assert q.enqueue("e1", "transaction.created", ev)[1] is False
    assert q.enqueue("e1", "transaction.created", ev)[1] is True          # duplicate delivery ignored
    q.enqueue("e2", "transaction.created", {"txn_id": "bad"})              # poison
    q.enqueue("e3", "nope", {})                                            # unknown type
    w.run_until_idle(timeout_s=10)
    assert q.stats() == {"done": 1, "dead": 2}
    assert len(integ.cases) == 1 and len(integ.notifications) == 1        # side effects exactly once
    assert {d["event_id"] for d in q.dead_letters()} == {"e2", "e3"}


def test_p08_exhausted_retries_go_to_dlq_and_replay(tmp_path):
    q = JobQueue(tmp_path / "q.db", base_backoff_s=0.01, max_attempts=2)
    integ = Integrations(flaky_failures=3)
    w = Worker(q, {"transaction.created": make_transaction_workflow(wf_llm(), integ)}, poll_s=0.01)
    jid, _ = q.enqueue("e9", "transaction.created", {"txn_id": "T", "customer_id": "C", "amount": 60000, "merchant": "M"})
    w.run_until_idle(timeout_s=10)
    assert q.dead_letters()[0]["attempts"] == 2
    assert q.replay(jid)
    w.run_until_idle(timeout_s=10)
    assert q.stats() == {"done": 1} and len(integ.cases) == 1


def test_p08_webhook_signature_and_dedup():
    from fastapi.testclient import TestClient
    from p08_event_automation import api
    body = json.dumps({"id": "evt-x", "type": "transaction.created",
                       "data": {"txn_id": "T", "customer_id": "C", "amount": 10, "merchant": "M"}}).encode()
    with TestClient(api.app) as c:
        assert c.post("/webhooks/x", content=body, headers={"X-Signature": "sha256=bad"}).status_code == 401
        r1 = c.post("/webhooks/x", content=body, headers={"X-Signature": api.sign(body)}).json()
        r2 = c.post("/webhooks/x", content=body, headers={"X-Signature": api.sign(body)}).json()
        assert r1["duplicate"] is False and r2["duplicate"] is True and r1["job_id"] == r2["job_id"]


# ------------------------------------------------------------------ project 9
def test_p09_borda_and_consensus():
    b = DebateSystem.borda({"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}, ["a", "b", "c"])
    assert b["a"] == 1.0 and b["c"] == 0.0

    def respond(m, kw):
        sys = " ".join(x["content"] for x in m if x["role"] == "system")
        if "impartial expert critic" in sys:
            return json.dumps({"scores": [{"proposal_id": p, "correctness": 9 if p == "risk_analyst" else 5,
                                           "feasibility": 8, "risk_awareness": 8, "critique": "ok"}
                                          for p in ("risk_analyst", "customer_advocate", "compliance_officer")]})
        if "Rank the OTHER" in sys:
            me = sys.split("You are ")[1].split(".")[0]
            order = [p for p in ("risk_analyst", "compliance_officer", "customer_advocate") if p != me]
            return json.dumps({"ranking": order + [me], "rationale": "merit"})
        if "aggregator" in sys:
            return json.dumps({"answer": "Tiered friction + threshold tuning", "dissent": ["CX cost"]})
        return json.dumps({"solution": "s", "key_arguments": ["k"], "confidence": 0.8})
    res = DebateSystem(FakeLLM(default=respond), max_rounds=3).run("q")
    assert res.winner == "risk_analyst" and res.consensus_reached and len(res.rounds) == 1
    assert 0.6 < res.confidence <= 1 and res.dissent == ["CX cost"]


# ------------------------------------------------------------------ project 10
def judge(score, suggestions=()):
    return json.dumps({"scores": [{"criterion": c, "score": score, "issue": ""} for c in
                                  ("correctness", "completeness", "clarity", "actionability")],
                       "critique": "c", "suggested_constraints": list(suggestions)})


def test_p10_regenerates_until_pass_and_learns(tmp_path):
    mem = ConstraintMemory(tmp_path / "learned.json")
    log = tmp_path / "m.jsonl"
    llm = FakeLLM(["too long " * 50, judge(3, ["Give a code example"]),
                   "short answer about precision and recall", judge(5)])
    agent = SelfReflectiveAgent(llm, threshold=0.8, max_iterations=3, memory=mem, metrics_log=log)
    res = agent.run("task", task_type="t", checks=[max_words(30), must_include("precision")])
    assert res.passed and len(res.iterations) == 2 and res.improvement > 0.5
    assert "Keep the answer under 30 words" in res.iterations[1].constraints_used
    # learned constraints are applied up-front on the next run of the same task type
    assert set(mem.top("t")) == {"Keep the answer under 30 words", "Explicitly mention: precision",
                                 "Give a code example"}
    llm2 = FakeLLM(["precision short", judge(5)])
    SelfReflectiveAgent(llm2, memory=mem, metrics_log=log).run("task2", task_type="t")
    assert "Keep the answer under 30 words" in llm2.calls[0]["messages"][1]["content"]
    rep = improvement_report(log)["t"]
    assert rep["runs"] == 2 and rep["first_score_recent_runs"] > rep["first_score_early_runs"]


# ------------------------------------------------------------------ project 11
class ListExporter:
    def __init__(self):
        self.spans = []

    def export(self, s):
        self.spans.append(s)


def test_p11_tracing_rolls_up_llm_cost():
    exp = ListExporter()
    tracer = Tracer([exp])
    llm = FakeLLM([LLMResponse("x", usage=Usage("gpt-5-nano", 1000, 500)), "y"])
    with tracer.span("root") as root:
        with tracer.span("child"):
            llm.text("hi")
        llm.text("again")
    tracer.close()
    names = [s.name for s in exp.spans]
    assert names[-1] == "root" and sum(n.startswith("llm.") for n in names) == 2
    assert root.attrs["llm_calls"] == 2 and root.attrs["cost_usd"] > 0.00025
    assert all(s.trace_id == root.trace_id for s in exp.spans)


def test_p11_alerts_canary_rollback(tmp_path):
    react_ok = [json.dumps({"thought": "t", "action": "final_answer", "action_input": {"answer": "42"}}),
                json.dumps({"acceptable": True, "issues": [], "suggestion": ""})]

    def respond(m, kw):
        return "best effort" if not kw.get("response_format") else react_ok[0 if "strict reviewer" not in
                                                                             m[0]["content"] else 1]
    metrics = MetricsStore(tmp_path / "m.db")
    alerter = Alerter(log=tmp_path / "a.jsonl", echo=False)
    dep = DeploymentManager(tmp_path / "d.json", stable="v1")
    prod = ProductionAgent(FakeLLM(default=respond), {"v1": AgentVersion("v1"), "v2": AgentVersion("v2")},
                           tracer=Tracer([ListExporter()]), metrics=metrics, alerter=alerter, deployment=dep)
    dep.start_canary("v2", 50)
    users = [f"u{i}" for i in range(60)]
    assert {dep.route(u) for u in users} == {"v1", "v2"} and dep.route("u1") == dep.route("u1")

    def broken_v2(user_id, query, _orig=prod.handle):
        if dep.route(user_id) == "v2":
            metrics.record(trace_id="x", version="v2", user=user_id, status="failed", latency_ms=10, cost=0,
                           tokens=0, steps=1, quality=0.0)
            return None
        return _orig(user_id, query)
    for u in users[:30]:
        broken_v2(u, "q")
    assert metrics.summary("v1")["avg_quality"] == 1.0
    assert dep.evaluate(metrics, min_requests=5, alerter=alerter) == "rollback"
    assert dep.state["canary"] is None and dep.route("anyone") == "v1"
    assert any(a.rule == "canary_rollback" for a in alerter.fired)
    loop_alert = alerter.evaluate({"trace_id": "t", "steps": 3, "loop": True, "reason": "loop", "cost": 0}, {})
    assert loop_alert[0].rule == "agent_loop"


def test_p11_dashboard_renders():
    from fastapi.testclient import TestClient
    from p11_observability import dashboard
    c = TestClient(dashboard.app)
    assert c.post("/ask", json={"user": "u1", "query": "Convert USD 1 to INR"}).status_code == 200
    page = c.get("/")
    assert page.status_code == 200 and "Agent operations" in page.text
    assert c.get("/metrics").json()["overall"]["requests"] >= 1


# ------------------------------------------------------------------ project 12
def test_p12_guard_logic():
    from p12_oss_contribution.loop_guard import LoopGuard, empty_guard
    g = LoopGuard(max_steps=50, max_repeats=2)
    s = empty_guard()
    for _ in range(2):
        s = g.check(s, "A")
    assert not s["tripped"]
    assert g.check(s, "A")["reason"].startswith("repeated action")
    s = empty_guard()
    for sig in ["A", "B", "A", "B", "A", "B"]:
        s = LoopGuard(max_steps=50, max_repeats=5).check(s, sig)
    assert s["tripped"] and "cycle" in s["reason"]
    assert LoopGuard(max_cost_usd=0.01).check(empty_guard(), None, cost=0.02)["tripped"]


def test_p12_guarded_langgraph_answers_instead_of_crashing():
    pytest.importorskip("langgraph")
    from langgraph.errors import GraphRecursionError
    from common.llm import tool_call_response
    from p12_oss_contribution.examples.demo_graph import build
    from p12_oss_contribution.loop_guard import empty_guard

    def loopy(m, kw):
        if kw.get("tools"):
            return tool_call_response("check_settlement_status", {"txn_id": "T-77"})
        return "Still pending after several checks; amount is unknown."
    init = {"messages": [{"role": "user", "content": "poll"}], "loop_guard": empty_guard()}
    with pytest.raises(GraphRecursionError):
        build(FakeLLM(default=loopy), guarded=False).invoke(init, {"recursion_limit": 12})
    out = build(FakeLLM(default=loopy), guarded=True).invoke(init, {"recursion_limit": 25})
    assert out["loop_guard"]["tripped"] and "repeated" in out["loop_guard"]["reason"]
    assert out["messages"][-1]["content"].startswith("Still pending")
