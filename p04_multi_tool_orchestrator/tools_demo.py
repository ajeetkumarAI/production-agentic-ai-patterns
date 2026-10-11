"""Demo tools for the orchestrator (replace with real integrations)."""
from __future__ import annotations

import time

from pydantic import BaseModel

from common.tools import Tool

ALERTS_SENT: list[dict] = []


class FxArgs(BaseModel):
    base: str = "USD"
    quote: str = "INR"


class CustomerArgs(BaseModel):
    customer_id: str


class IpArgs(BaseModel):
    ip: str


class NameArgs(BaseModel):
    name: str


class AlertArgs(BaseModel):
    message: str
    severity: str = "medium"


def fx_primary(base: str, quote: str) -> dict:
    time.sleep(0.3)
    return {"pair": f"{base}/{quote}", "rate": 83.20}


def fx_backup(base: str, quote: str) -> dict:
    time.sleep(0.3)
    return {"pair": f"{base}/{quote}", "rate": 83.35}   # disagrees slightly -> conflict resolution


def customer_lookup(customer_id: str) -> dict:
    time.sleep(0.3)
    return {"customer_id": customer_id, "name": "A. Sharma", "home_country": "IN", "account_age_days": 10}


def geo_ip(ip: str) -> dict:
    time.sleep(0.3)
    return {"ip": ip, "country": "NG", "is_vpn": True}


def sanctions_fast(name: str) -> dict:
    raise ConnectionError("upstream 503")                 # fails -> router fails over


def sanctions_slow(name: str) -> dict:
    time.sleep(0.5)
    return {"name": name, "match": False, "lists_checked": ["OFAC", "UN", "EU"]}


def send_alert(message: str, severity: str = "medium") -> dict:
    ALERTS_SENT.append({"message": message, "severity": severity})
    return {"sent": True, "id": len(ALERTS_SENT)}


def demo_tools() -> list[Tool]:
    return [
        Tool("fx_primary", "Live FX rate (primary feed)", fx_primary, FxArgs, capabilities={"fx_rate"}, priority=2),
        Tool("fx_backup", "Live FX rate (backup feed)", fx_backup, FxArgs, capabilities={"fx_rate"}, priority=1),
        Tool("customer_lookup", "Customer profile by id (PII)", customer_lookup, CustomerArgs,
             capabilities={"customer_profile"}, required_scopes={"pii:read"}),
        Tool("geo_ip", "Geolocate an IP address and detect VPNs", geo_ip, IpArgs, capabilities={"ip_intel"}),
        Tool("sanctions_fast", "Sanctions screening (fast cache)", sanctions_fast, NameArgs,
             capabilities={"sanctions_check"}, priority=2),
        Tool("sanctions_slow", "Sanctions screening (authoritative)", sanctions_slow, NameArgs,
             capabilities={"sanctions_check"}, priority=1),
        Tool("send_alert", "Send an alert to the fraud operations channel", send_alert, AlertArgs,
             capabilities={"notify"}, required_scopes={"alerts:write"}, side_effects=True),
    ]
