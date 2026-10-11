"""REST API for reviewers:  uvicorn p06_hitl_approval.api:app --reload

GET  /runs/pending            -> tickets waiting for a human
GET  /runs/{run_id}           -> checkpointed state
POST /requests                -> start a new request
POST /runs/{run_id}/decision  -> {"decision": "approve|reject|modify", "reviewer": "...", ...}
GET  /audit?run_id=...        -> audit trail;  GET /audit/verify -> chain integrity
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from common import config
from common.llm import get_llm

from .core import HITLAgent, HumanDecision

agent = HITLAgent(get_llm(), config.DATA_DIR / "p06_hitl.db")
app = FastAPI(title="HITL Approval Agent")


@app.post("/requests")
def create(request: dict, requested_by: str = "customer"):
    return agent.start(request, requested_by)


@app.get("/runs/pending")
def pending():
    return agent.pending()


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    run = agent.get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@app.post("/runs/{run_id}/decision")
def decide(run_id: str, decision: HumanDecision):
    try:
        return agent.resume(run_id, decision)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except (ValueError, ValidationError) as e:
        raise HTTPException(422, str(e)) from e


@app.get("/audit")
def audit(run_id: str | None = None):
    return agent.audit.trail(run_id)


@app.get("/audit/verify")
def verify():
    ok, bad_seq = agent.audit.verify()
    return {"intact": ok, "first_bad_seq": bad_seq}
