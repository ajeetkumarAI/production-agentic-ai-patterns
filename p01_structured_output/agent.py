"""Project 1 - Structured Output Agent.

* Enforces Pydantic schemas on the final answer (incl. business-rule validators)
* Validates tool arguments AND tool responses against Pydantic contracts
* Retries with the exact validation error fed back to the model
* Logs every validation failure (LLM output, tool args, tool output) to JSONL
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from common import config
from common.llm import BaseLLM
from common.structured import StructuredOutputError, StructuredResult, generate_structured
from common.tools import Tool, ToolError, to_text
from common.utils import append_jsonl, read_jsonl

from .schemas import (CustomerLookupArgs, CustomerProfile, MerchantInfo, MerchantLookupArgs,
                      TransactionAssessment)

FAILURE_LOG = config.DATA_DIR / "p01_validation_failures.jsonl"

# --- demo data sources (swap for real services) -----------------------------------------
_CUSTOMERS = {
    "C1001": {"customer_id": "C1001", "home_country": "IN", "account_age_days": 1450, "avg_txn_amount": 2500.0,
              "prior_chargebacks": 0},
    "C2002": {"customer_id": "C2002", "home_country": "IN", "account_age_days": 12, "avg_txn_amount": 900.0,
              "prior_chargebacks": 2},
    # deliberately corrupt upstream record -> tool output validation must catch it
    "C3003": {"customer_id": "C3003", "home_country": "INDIA", "account_age_days": -5, "avg_txn_amount": 100.0,
              "prior_chargebacks": 0},
}
_MERCHANTS = {
    "M-GROCERY": {"merchant_id": "M-GROCERY", "category": "grocery", "country": "IN", "fraud_rate": 0.002},
    "M-CRYPTO": {"merchant_id": "M-CRYPTO", "category": "crypto_exchange", "country": "SC", "fraud_rate": 0.09},
}


def _get_customer(customer_id: str) -> dict:
    if customer_id not in _CUSTOMERS:
        raise KeyError(f"unknown customer {customer_id}")
    return _CUSTOMERS[customer_id]


def _get_merchant(merchant_id: str) -> dict:
    if merchant_id not in _MERCHANTS:
        raise KeyError(f"unknown merchant {merchant_id}")
    return _MERCHANTS[merchant_id]


def default_tools() -> list[Tool]:
    return [
        Tool("get_customer_profile", "Look up a customer's risk profile by id (format C123).", _get_customer,
             CustomerLookupArgs, CustomerProfile),
        Tool("get_merchant_info", "Look up merchant category, country and historical fraud rate.", _get_merchant,
             MerchantLookupArgs, MerchantInfo),
    ]


SYSTEM = """You are a payments fraud analyst. Use the tools to gather facts about the customer and the
merchant, then produce a risk assessment. Base every reason on evidence you actually retrieved. If a tool
fails, say so in the reasons and be more conservative (prefer 'review')."""


class StructuredOutputAgent:
    def __init__(self, llm: BaseLLM, tools: list[Tool] | None = None, max_retries: int = 2,
                 max_tool_rounds: int = 4, failure_log: Path = FAILURE_LOG):
        self.llm = llm
        self.tools = {t.name: t for t in (tools or default_tools())}
        self.max_retries = max_retries
        self.max_tool_rounds = max_tool_rounds
        self.failure_log = failure_log

    # ------------------------------------------------------------------ logging
    def _log(self, kind: str, **data) -> None:
        append_jsonl(self.failure_log, {"kind": kind, **data})

    # ------------------------------------------------------------------ tools
    def run_tool(self, name: str, arguments: str | dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            self._log("unknown_tool", tool=name, arguments=arguments)
            return json.dumps({"error": f"unknown tool '{name}'. Available: {list(self.tools)}"})
        try:
            return to_text(tool(arguments))
        except ToolError as e:
            kind = "tool_output" if "invalid output" in str(e) else "tool_args" if "invalid arguments" in str(
                e) or "not valid JSON" in str(e) else "tool_error"
            self._log(kind, tool=name, arguments=arguments, error=str(e))
            return json.dumps({"error": str(e)})

    def _gather(self, messages: list[dict]) -> list[dict]:
        specs = [t.to_openai() for t in self.tools.values()]
        for _ in range(self.max_tool_rounds):
            resp = self.llm.chat(messages, tools=specs, tag="p01.gather")
            if not resp.tool_calls:
                break
            messages.append(resp.to_message())
            for tc in resp.tool_calls:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": self.run_tool(tc.name, tc.arguments)})
        return messages

    # ------------------------------------------------------------------ main
    def assess(self, transaction: dict) -> StructuredResult[TransactionAssessment]:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Assess this transaction:\n" + json.dumps(transaction)},
        ]
        messages = self._gather(messages)
        messages.append({"role": "user", "content": "Now return the final TransactionAssessment as JSON."})
        try:
            return generate_structured(
                self.llm, messages, TransactionAssessment, max_retries=self.max_retries,
                on_failure=lambda rec: self._log("llm_output", transaction_id=transaction.get("id"), **rec),
                tag="p01.assess",
            )
        except StructuredOutputError as e:
            self._log("gave_up", transaction_id=transaction.get("id"), errors=e.errors)
            raise


def failure_report(path: Path = FAILURE_LOG) -> dict:
    """Aggregate the failure log: which kinds/tools/fields fail most often."""
    kinds, tools, fields = Counter(), Counter(), Counter()
    for rec in read_jsonl(path):
        kinds[rec["kind"]] += 1
        if rec.get("tool"):
            tools[rec["tool"]] += 1
        for line in str(rec.get("error", "")).splitlines():
            if line.startswith("- ") and ":" in line:
                fields[line[2:].split(":")[0]] += 1
    return {"by_kind": dict(kinds), "by_tool": dict(tools), "top_fields": fields.most_common(5)}
