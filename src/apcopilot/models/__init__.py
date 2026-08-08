from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def canonicalize_sku(text: str) -> str:
    """Uppercase, strip everything but letters/digits.

    The single source of truth for turning a raw item name ('WidgetA',
    'Widget A', 'widget-a') into the SKU form used as the `items` primary
    key. Ingestion and validation must both call this so a description
    canonicalizes to the same SKU on both sides of the lookup.
    """
    return re.sub(r"[^A-Z0-9]", "", text.upper())


class Severity(StrEnum):
    INFO = "INFO"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


_SEVERITY_ORDER = [Severity.INFO, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


class LineItem(BaseModel):
    description: str
    sku: str | None = None
    quantity: int
    unit_price: Decimal | None = None
    line_total: Decimal | None = None


class ExtractedInvoice(BaseModel):
    invoice_number: str | None = None
    revision: str | None = None
    vendor_name: str | None = None
    currency: str = "USD"
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    payment_terms: str | None = None
    notes: str | None = None
    extraction_confidence: float = 1.0
    extraction_warnings: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    invoice: ExtractedInvoice
    method: str  # "deterministic" | "llm"
    attempts: int = 1
    source_format: str
    document_path: str
    content_hash: str
    raw_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal = Decimal(0)


class ValidationFlag(BaseModel):
    rule_id: str
    code: str
    severity: Severity
    message: str
    sku: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    flags: list[ValidationFlag] = Field(default_factory=list)
    fraud_score: int = 0
    total_usd: Decimal | None = None
    fx_rate_used: Decimal | None = None
    duplicate_of_run_id: str | None = None
    revision_conflict: bool = False
    already_paid: bool = False
    vendor_known: bool = False
    vendor_matched_name: str | None = None
    vendor_auto_approve_limit: Decimal | None = None

    @property
    def max_severity(self) -> Severity | None:
        present = {f.severity for f in self.flags}
        for sev in reversed(_SEVERITY_ORDER):
            if sev in present:
                return sev
        return None

    @property
    def has_blocking(self) -> bool:
        return self.max_severity in (Severity.HIGH, Severity.CRITICAL)

    def flags_at_or_above(self, floor: Severity) -> list[ValidationFlag]:
        floor_idx = _SEVERITY_ORDER.index(floor)
        return [f for f in self.flags if _SEVERITY_ORDER.index(f.severity) >= floor_idx]


class ApprovalDecisionValue(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    NEEDS_HUMAN = "needs_human"


class ApprovalDecision(BaseModel):
    decision: ApprovalDecisionValue
    rationale: str
    cited_policy_ids: list[str] = Field(default_factory=list)
    addressed_flag_rule_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    rounds: int = 1
    decided_by: str = "rules"  # "rules" | "vp_agent" | "human"
    critique_notes: list[str] = Field(default_factory=list)


class PaymentResult(BaseModel):
    status: str
    idempotency_key: str
    duplicate_suppressed: bool = False


__all__ = [
    "ApprovalDecision",
    "ApprovalDecisionValue",
    "ExtractedInvoice",
    "ExtractionResult",
    "LineItem",
    "PaymentResult",
    "Severity",
    "ValidationFlag",
    "ValidationResult",
    "canonicalize_sku",
]
