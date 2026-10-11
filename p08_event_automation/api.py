"""Webhook receiver + background workers:  uvicorn p08_event_automation.api:app

POST /webhooks/{source}   body: {"id": "...", "type": "transaction.created", "data": {...}}
     headers: X-Signature: sha256=<hmac of raw body with WEBHOOK_SECRET>, optional Idempotency-Key
GET  /jobs   GET /dlq   POST /dlq/{job_id}/replay   GET /stats
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request

from common import config
from common.llm import get_llm

from .queue_store import JobQueue
from .workflows import Integrations, Worker, event_id_for, make_transaction_workflow

SECRET = os.getenv("WEBHOOK_SECRET", "dev-secret-change-me").encode()
queue = JobQueue(config.DATA_DIR / "p08_queue.db")
integrations = Integrations()
worker = Worker(queue, {"transaction.created": make_transaction_workflow(get_llm(), integrations)})


def sign(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET, body, hashlib.sha256).hexdigest()


@asynccontextmanager
async def lifespan(_app):
    worker.start(n_threads=2)
    yield
    worker.stop()


app = FastAPI(title="Event-Triggered Automation Agent", lifespan=lifespan)


@app.post("/webhooks/{source}", status_code=202)
async def webhook(source: str, request: Request, x_signature: str = Header(default=""),
                  idempotency_key: str | None = Header(default=None)):
    body = await request.body()
    if not hmac.compare_digest(sign(body), x_signature):
        raise HTTPException(401, "bad signature")
    try:
        event = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(400, "body must be JSON") from e
    eid = event_id_for(event, idempotency_key)
    job_id, duplicate = queue.enqueue(eid, event.get("type", "unknown"), event.get("data", {}), source)
    return {"job_id": job_id, "event_id": eid, "duplicate": duplicate}


@app.get("/jobs")
def jobs():
    return queue.jobs()


@app.get("/dlq")
def dlq():
    return queue.dead_letters()


@app.post("/dlq/{job_id}/replay")
def replay(job_id: str):
    if not queue.replay(job_id):
        raise HTTPException(404, "not a dead-lettered job")
    return {"replayed": job_id}


@app.get("/stats")
def stats():
    return {"jobs": queue.stats(), "cases": integrations.cases, "notifications": len(integrations.notifications)}
