"""Project 6 - Human-in-the-Loop Approval Agent.

Flow:  request -> agent proposes action -> uncertainty detection
         - confident & within policy  -> execute automatically
         - uncertain / risky          -> checkpoint + PAUSE (ticket for a human)
       human decision (approve | reject | modify) is *validated* -> RESUME from the
       checkpoint with the validated context -> execute -> notify.
Every step is written to a hash-chained, tamper-evident audit trail.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from common.llm import BaseLLM
from common.structured import StructuredOutputError, generate_structured

# ------------------------------------------------------------------ action contracts
ActionName = Literal["refund", "block_card", "escalate_to_fraud_team", "no_action"]


class RefundParams(BaseModel):
    txn_id: str
    amount: float = Field(gt=0)


class BlockCardParams(BaseModel):
    card_last4: str = Field(pattern=r"^\d{4}$")
    reason: str = Field(min_length=5)


class EscalateParams(BaseModel):
    summary: str = Field(min_length=10)


class NoParams(BaseModel):
    pass


PARAM_MODELS: dict[str, type[BaseModel]] = {"refund": RefundParams, "block_card": BlockCardParams,
                                            "escalate_to_fraud_team": EscalateParams, "no_action": NoParams}
IRREVERSIBLE = {"block_card", "refund"}


class ProposedAction(BaseModel):
    action: ActionName
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    confidence: float = Field(ge=0, le=1)
    missing_info: list[str] = Field(default_factory=list, description="Facts you needed but did not have")


class HumanDecision(BaseModel):
    decision: Literal["approve", "reject", "modify"]
    reviewer: str
    comment: str = ""
    modified_action: ActionName | None = None
    modified_params: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check(self):
        if self.decision in ("reject", "modify") and len(self.comment.strip()) < 5:
            raise ValueError("a comment (>=5 chars) explaining the decision is required for reject/modify")
        if self.decision == "modify" and not (self.modified_action or self.modified_params):
            raise ValueError("modify requires modified_action and/or modified_params")
        return self


# ------------------------------------------------------------------ audit trail
class AuditLog:
    """Append-only, hash-chained log: each entry stores the hash of the previous one,
    so editing or deleting any past entry breaks ``verify()``."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self._lock = threading.Lock()
        db.execute("CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id TEXT, "
                   "actor TEXT, event TEXT, data TEXT, prev_hash TEXT, hash TEXT)")
        db.commit()

    @staticmethod
    def _hash(prev: str, ts: float, run_id: str, actor: str, event: str, data: str) -> str:
        return hashlib.sha256(f"{prev}|{ts:.6f}|{run_id}|{actor}|{event}|{data}".encode()).hexdigest()

    def record(self, run_id: str, actor: str, event: str, data: Any = None) -> None:
        with self._lock:
            row = self.db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else "GENESIS"
            ts, payload = time.time(), json.dumps(data, default=str, sort_keys=True)
            h = self._hash(prev, ts, run_id, actor, event, payload)
            self.db.execute("INSERT INTO audit (ts,run_id,actor,event,data,prev_hash,hash) VALUES (?,?,?,?,?,?,?)",
                            (ts, run_id, actor, event, payload, prev, h))
            self.db.commit()

    def trail(self, run_id: str | None = None) -> list[dict]:
        q, args = "SELECT seq,ts,run_id,actor,event,data,hash FROM audit", ()
        if run_id:
            q, args = q + " WHERE run_id=?", (run_id,)
        return [{"seq": r[0], "ts": r[1], "run_id": r[2], "actor": r[3], "event": r[4], "data": json.loads(r[5]),
                 "hash": r[6][:12]} for r in self.db.execute(q + " ORDER BY seq", args)]

    def verify(self) -> tuple[bool, int | None]:
        prev = "GENESIS"
        for seq, ts, run_id, actor, event, data, p, h in self.db.execute(
                "SELECT seq,ts,run_id,actor,event,data,prev_hash,hash FROM audit ORDER BY seq"):
            if p != prev or self._hash(prev, ts, run_id, actor, event, data) != h:
                return False, seq
            prev = h
        return True, None


# ------------------------------------------------------------------ uncertainty
@dataclass
class UncertaintyReport:
    uncertain: bool
    reasons: list[str]
    agreement: float
    confidence: float


@dataclass
class Policy:
    min_confidence: float = 0.75
    min_agreement: float = 0.67
    auto_refund_limit: float = 5000.0
    irreversible_needs_human: bool = True
    samples: int = 3
    authorized_reviewers: frozenset[str] = frozenset({"priya.sr_analyst", "rahul.fraud_lead"})


def detect_uncertainty(proposals: list[ProposedAction], policy: Policy, request: dict) -> UncertaintyReport:
    chosen = proposals[0]
    votes = Counter(p.action for p in proposals)
    agreement = votes[chosen.action] / len(proposals)
    confidence = sum(p.confidence for p in proposals) / len(proposals)
    reasons = []
    if confidence < policy.min_confidence:
        reasons.append(f"low model confidence ({confidence:.2f} < {policy.min_confidence})")
    if agreement < policy.min_agreement:
        reasons.append(f"self-consistency disagreement: {dict(votes)}")
    if chosen.missing_info:
        reasons.append(f"missing information: {chosen.missing_info}")
    if chosen.action in IRREVERSIBLE and policy.irreversible_needs_human:
        reasons.append(f"'{chosen.action}' is irreversible")
    amount = float(chosen.params.get("amount", 0) or 0)
    if chosen.action == "refund" and amount > policy.auto_refund_limit:
        reasons.append(f"refund {amount} exceeds auto limit {policy.auto_refund_limit}")
    try:
        PARAM_MODELS[chosen.action].model_validate(chosen.params)
    except ValidationError as e:
        reasons.append(f"invalid params: {e.errors()[0]['msg']}")
    return UncertaintyReport(bool(reasons), reasons, agreement, confidence)


# ------------------------------------------------------------------ executors
EXECUTED: list[dict] = []


def execute_action(action: str, params: dict) -> dict:
    """Side-effecting integrations go here (core banking, card management...)."""
    clean = PARAM_MODELS[action].model_validate(params).model_dump()
    EXECUTED.append({"action": action, "params": clean})
    return {"ok": True, "action": action, "params": clean, "ref": uuid.uuid4().hex[:10]}


# ------------------------------------------------------------------ agent
SYSTEM = """You are a card-dispute agent. Decide the single best action for the customer's request.
Actions: refund(txn_id, amount) | block_card(card_last4, reason) | escalate_to_fraud_team(summary) | no_action().
Be honest about confidence and list any information you'd need but don't have."""


class HITLAgent:
    def __init__(self, llm: BaseLLM, db_path: str | Path, policy: Policy | None = None):
        self.llm = llm
        self.policy = policy or Policy()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, status TEXT, state TEXT, "
                        "created REAL, updated REAL)")
        self.db.commit()
        self.audit = AuditLog(self.db)

    # ---------------------------------------------------------------- checkpoints
    def _save(self, run_id: str, status: str, state: dict) -> None:
        now = time.time()
        self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET status=excluded.status,"
                        " state=excluded.state, updated=excluded.updated", (run_id, status, json.dumps(state), now, now))
        self.db.commit()

    def get(self, run_id: str) -> dict | None:
        r = self.db.execute("SELECT status, state FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return {"run_id": run_id, "status": r[0], **json.loads(r[1])} if r else None

    def pending(self) -> list[dict]:
        return [self.get(r[0]) for r in self.db.execute("SELECT run_id FROM runs WHERE status='PENDING_APPROVAL'")]

    # ---------------------------------------------------------------- propose
    def _propose(self, request: dict) -> list[ProposedAction]:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(request)}]
        with ThreadPoolExecutor(max_workers=self.policy.samples) as ex:  # self-consistency sampling
            futs = [ex.submit(generate_structured, self.llm, msgs, ProposedAction, max_retries=1, tag="p06.propose")
                    for _ in range(self.policy.samples)]
            out = []
            for f in futs:
                try:
                    out.append(f.result().value)
                except StructuredOutputError:
                    pass
        return out

    def start(self, request: dict, requested_by: str = "customer") -> dict:
        run_id = uuid.uuid4().hex[:10]
        self.audit.record(run_id, requested_by, "request_received", request)
        proposals = self._propose(request)
        if not proposals:
            state = {"request": request, "reasons": ["agent could not produce a valid proposal"]}
            self._save(run_id, "PENDING_APPROVAL", state)
            self.audit.record(run_id, "agent", "paused_for_human", state["reasons"])
            return self.get(run_id)
        report = detect_uncertainty(proposals, self.policy, request)
        state = {"request": request, "requested_by": requested_by, "proposal": proposals[0].model_dump(),
                 "alternatives": [p.action for p in proposals[1:]],
                 "uncertainty": {"reasons": report.reasons, "agreement": report.agreement,
                                 "confidence": round(report.confidence, 3)}}
        self.audit.record(run_id, "agent", "action_proposed", state["proposal"] | {"uncertainty": state["uncertainty"]})
        if report.uncertain:
            self._save(run_id, "PENDING_APPROVAL", state)
            self.audit.record(run_id, "agent", "paused_for_human", report.reasons)
            return self.get(run_id)
        return self._execute(run_id, state, proposals[0].action, proposals[0].params, actor="agent")

    # ---------------------------------------------------------------- resume
    def resume(self, run_id: str, decision: dict | HumanDecision) -> dict:
        run = self.get(run_id)
        if run is None:
            raise KeyError(f"unknown run {run_id}")
        if run["status"] != "PENDING_APPROVAL":
            raise ValueError(f"run {run_id} is {run['status']}, not awaiting approval")
        try:
            d = decision if isinstance(decision, HumanDecision) else HumanDecision.model_validate(decision)
            if d.reviewer not in self.policy.authorized_reviewers:
                raise ValueError(f"reviewer '{d.reviewer}' is not authorized")
            if d.reviewer == run.get("requested_by"):
                raise ValueError("four-eyes rule: requester cannot approve their own request")
            action = d.modified_action or run.get("proposal", {}).get("action", "no_action")
            params = {**(run.get("proposal", {}).get("params", {}) if not d.modified_action else {}),
                      **(d.modified_params or {})}
            if d.decision != "reject":
                params = PARAM_MODELS[action].model_validate(params).model_dump()  # validated context
        except (ValidationError, ValueError) as e:
            self.audit.record(run_id, getattr(decision, "reviewer", None) or (decision.get("reviewer") if isinstance(
                decision, dict) else "?"), "invalid_human_input", str(e)[:500])
            raise
        self.audit.record(run_id, d.reviewer, f"human_{d.decision}", d.model_dump())
        state = {k: v for k, v in run.items() if k not in ("run_id", "status")} | {"human": d.model_dump()}
        if d.decision == "reject":
            state["customer_message"] = self._message(state, None)
            self._save(run_id, "REJECTED", state)
            self.audit.record(run_id, "agent", "closed_rejected", {"comment": d.comment})
            return self.get(run_id)
        return self._execute(run_id, state, action, params, actor=d.reviewer)

    def _execute(self, run_id: str, state: dict, action: str, params: dict, actor: str) -> dict:
        try:
            result = execute_action(action, params)
            state["execution"] = result
            state["customer_message"] = self._message(state, result)
            self._save(run_id, "EXECUTED", state)
            self.audit.record(run_id, actor, "action_executed", result)
        except Exception as e:  # noqa: BLE001
            state["execution"] = {"ok": False, "error": str(e)}
            self._save(run_id, "FAILED", state)
            self.audit.record(run_id, actor, "action_failed", str(e))
        return self.get(run_id)

    def _message(self, state: dict, result: dict | None) -> str:
        human = state.get("human")
        return self.llm.chat([
            {"role": "system", "content": "Write a short, polite message to the customer about the outcome of their "
                                          "request. Incorporate the reviewer's note if present. No internal jargon."},
            {"role": "user", "content": json.dumps({"request": state["request"], "outcome": result,
                                                    "reviewer_note": human and human.get("comment")})},
        ], tag="p06.message").content or ""
