import json

import pytest

from common.llm import FakeLLM, MockLLM, tool_call_response
from common.structured import StructuredOutputError, extract_json, generate_structured
from common.utils import read_jsonl, safe_eval
from p01_structured_output.agent import StructuredOutputAgent, failure_report
from p01_structured_output.schemas import TransactionAssessment
from p02_rag_citations.agent import RAGAgent

GOOD = {"transaction_id": "T-1", "risk_level": "high", "risk_score": 0.9,
        "reasons": ["IP country differs from home country", "Merchant fraud rate is 9%"],
        "recommended_action": "block"}


# ------------------------------------------------------------------ common
def test_extract_json_handles_fences_and_chatter():
    assert json.loads(extract_json('Sure!\n```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(extract_json('Result: {"a": {"b": "}"}} trailing')) == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_safe_eval_blocks_code():
    assert safe_eval("120*83.2 + 2^3") == pytest.approx(9992.0)
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")


def test_mock_llm_returns_schema_valid_json():
    res = generate_structured(MockLLM(), [{"role": "user", "content": "x"}], TransactionAssessment)
    assert res.value.risk_level in ("low", "medium", "high")


# ------------------------------------------------------------------ project 1
def test_p01_repairs_invalid_output_and_logs_failure(tmp_path):
    log = tmp_path / "fail.jsonl"
    bad = dict(GOOD, recommended_action="approve")  # violates business rule
    llm = FakeLLM(["no tools needed", "```json\n" + json.dumps(bad) + "\n```", json.dumps(GOOD)])
    res = StructuredOutputAgent(llm, failure_log=log).assess({"id": "T-1"})
    assert res.value.recommended_action == "block" and res.attempts == 2
    recs = list(read_jsonl(log))
    assert recs[0]["kind"] == "llm_output" and "must not be approved" in recs[0]["error"]
    # the repair prompt contained the validation error
    assert "must not be approved" in llm.calls[-1]["messages"][-1]["content"]


def test_p01_gives_up_after_retries(tmp_path):
    llm = FakeLLM(["done"], default="not json at all")
    with pytest.raises(StructuredOutputError):
        StructuredOutputAgent(llm, max_retries=1, failure_log=tmp_path / "f.jsonl").assess({"id": "x"})
    assert failure_report(tmp_path / "f.jsonl")["by_kind"] == {"llm_output": 2, "gave_up": 1}


def test_p01_validates_tool_output(tmp_path):
    log = tmp_path / "f.jsonl"
    llm = FakeLLM([tool_call_response("get_customer_profile", {"customer_id": "C3003"}), "ok", json.dumps(GOOD)])
    StructuredOutputAgent(llm, failure_log=log).assess({"id": "T-3"})
    kinds = [r["kind"] for r in read_jsonl(log)]
    assert "tool_output" in kinds  # corrupt upstream record was caught
    tool_msg = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert "invalid output" in tool_msg["content"]


def test_p01_rejects_bad_tool_args(tmp_path):
    agent = StructuredOutputAgent(FakeLLM(), failure_log=tmp_path / "f.jsonl")
    out = json.loads(agent.run_tool("get_customer_profile", {"customer_id": "bob"}))
    assert "invalid arguments" in out["error"]


# ------------------------------------------------------------------ project 2
def _answer(text, cites, conf=0.9, answerable=True):
    return json.dumps({"answer": text, "citations": cites, "confidence": conf, "answerable": answerable})


def test_p02_grounded_answer_high_confidence():
    agent = RAGAgent(FakeLLM())
    q = "How many days does a customer have to raise a chargeback for card-not-present purchases?"
    hits = agent.retrieve(q)
    label = next(h["label"] for h in hits if h["id"] == "chargeback_policy#0")
    agent.llm = FakeLLM([_answer(f"Customers have 120 days for card-not-present purchases [{label}].",
                                 [{"source_id": label, "quote": "within 120 days of the transaction date"}])])
    r = agent.ask(q)
    assert r.source == "knowledge_base" and not r.low_confidence
    assert r.grounding["citation_precision"] == 1.0 and r.grounding["sentence_coverage"] == 1.0


def test_p02_fabricated_quote_is_flagged():
    agent = RAGAgent(FakeLLM([_answer("Customers have 365 days [S1].",
                                      [{"source_id": "S1", "quote": "customers have 365 days to complain"}])]))
    r = agent.ask("How many days does a customer have to raise a chargeback?")
    assert r.low_confidence
    assert r.grounding["invalid_citations"][0]["reason"] == "quote not found in source"


def test_p02_falls_back_when_retrieval_is_weak():
    class FakeSearch:
        def search(self, q):
            return "Canberra", ["https://example.org/canberra"]
    llm = FakeLLM()
    r = RAGAgent(llm, fallback=FakeSearch()).ask("What is the capital of Australia?")
    assert r.source == "web_search" and r.answer == "Canberra" and not llm.calls
