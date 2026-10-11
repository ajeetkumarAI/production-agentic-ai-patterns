from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TransactionAssessment(BaseModel):
    """Final output of the agent. Business rules live in validators so a
    model reply that is valid JSON but *inconsistent* is still rejected."""

    transaction_id: str = Field(min_length=1)
    risk_level: Literal["low", "medium", "high"]
    risk_score: float = Field(ge=0, le=1, description="0 = safe, 1 = certainly fraud")
    reasons: list[str] = Field(min_length=1, max_length=6, description="Concrete evidence-based reasons")
    recommended_action: Literal["approve", "review", "block"]

    @field_validator("reasons")
    @classmethod
    def reasons_not_blank(cls, v: list[str]) -> list[str]:
        if any(len(r.strip()) < 5 for r in v):
            raise ValueError("each reason must be a meaningful sentence (>= 5 chars)")
        return v

    @model_validator(mode="after")
    def consistent(self) -> "TransactionAssessment":
        bands = {"low": (0.0, 0.4), "medium": (0.3, 0.75), "high": (0.6, 1.0)}
        lo, hi = bands[self.risk_level]
        if not lo <= self.risk_score <= hi:
            raise ValueError(f"risk_score {self.risk_score} inconsistent with risk_level '{self.risk_level}' "
                             f"(expected {lo}-{hi})")
        if self.risk_level == "high" and self.recommended_action == "approve":
            raise ValueError("high risk transactions must not be approved")
        return self


# ---- tool argument / output contracts ------------------------------------------------
class CustomerLookupArgs(BaseModel):
    customer_id: str = Field(pattern=r"^C\d{3,}$")


class CustomerProfile(BaseModel):
    customer_id: str
    home_country: str = Field(min_length=2, max_length=2)
    account_age_days: int = Field(ge=0)
    avg_txn_amount: float = Field(ge=0)
    prior_chargebacks: int = Field(ge=0)


class MerchantLookupArgs(BaseModel):
    merchant_id: str = Field(min_length=1)


class MerchantInfo(BaseModel):
    merchant_id: str
    category: str
    country: str = Field(min_length=2, max_length=2)
    fraud_rate: float = Field(ge=0, le=1)
