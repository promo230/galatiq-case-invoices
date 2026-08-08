"""Approval-stage tests: the hard guardrails and the deterministic rules fallback.

With `llm_mode="off"` no proposer/critic round ever runs, so everything here is
reproducible and free. The `no_llm_calls` autouse fixture in conftest turns any
attempted LLM call into a test failure, which is what makes the "rejected without
an LLM call" assertions real rather than aspirational.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from uuid import uuid4

import pytest

from apcopilot.agents.approval import run_approval
from apcopilot.agents.validation import validate
from apcopilot.models import (
    ApprovalDecisionValue,
    ExtractedInvoice,
    Severity,
    ValidationFlag,
    ValidationResult,
)
from apcopilot.tools.policy import threshold


def _approve(invoice: ExtractedInvoice, validation: ValidationResult):
    return asyncio.run(run_approval(invoice, validation, run_id=str(uuid4())))


def _flag(severity: Severity, *, code: str = "TEST_FLAG", rule_id: str = "POL-ITEM-01"):
    return ValidationFlag(
        rule_id=rule_id, code=code, severity=severity, message=f"{code} ({severity})"
    )


# --- hard guardrails (run before any LLM branch) --------------------------------


@pytest.mark.parametrize(
    "code",
    ["MISSING_FIELD", "NEGATIVE_VALUE", "REVISION_CONFLICT", "FRAUD_SCORE_CRITICAL"],
)
def test_a_critical_flag_is_an_automatic_reject_without_an_llm_call(make_invoice, code):
    validation = ValidationResult(
        flags=[_flag(Severity.CRITICAL, code=code, rule_id="POL-DATA-01")],
        total_usd=Decimal("500.00"),
    )
    decision = _approve(make_invoice(), validation)

    assert decision.decision is ApprovalDecisionValue.REJECT
    assert decision.decided_by == "rules"
    assert decision.rounds == 0  # no reflection round happened
    assert decision.confidence == 1.0
    assert "POL-DATA-01" in decision.cited_policy_ids
    assert code in decision.rationale


def test_critical_beats_everything_else_including_a_clean_total(make_invoice):
    """A CRITICAL flag short-circuits before the amount is even considered."""
    validation = ValidationResult(
        flags=[_flag(Severity.CRITICAL), _flag(Severity.MEDIUM, code="OVER_VENDOR_LIMIT")],
        total_usd=Decimal("1.00"),
        vendor_auto_approve_limit=Decimal("10000"),
    )
    assert _approve(make_invoice(), validation).decision is ApprovalDecisionValue.REJECT


def test_unpriceable_invoice_is_routed_to_a_human(make_invoice):
    """No USD-equivalent total (missing price data, or no FX rate on file) means
    there is nothing to check a threshold against."""
    validation = ValidationResult(flags=[], total_usd=None)
    decision = _approve(make_invoice(total=None), validation)

    assert decision.decision is ApprovalDecisionValue.NEEDS_HUMAN
    assert decision.decided_by == "rules"
    assert decision.rounds == 0
    assert "USD-equivalent total" in decision.rationale


# --- deterministic fallback (llm_mode == "off") ----------------------------------


def test_clean_invoice_within_limits_is_approved(make_invoice):
    validation = ValidationResult(
        flags=[],
        total_usd=Decimal("500.00"),
        vendor_known=True,
        vendor_auto_approve_limit=Decimal("10000"),
    )
    decision = _approve(make_invoice(), validation)

    assert decision.decision is ApprovalDecisionValue.APPROVE
    assert decision.decided_by == "rules"
    assert "no HIGH/CRITICAL flags" in decision.rationale


def test_a_high_flag_blocks_auto_approval(make_invoice):
    validation = ValidationResult(
        flags=[_flag(Severity.HIGH, code="STOCK_SHORTFALL", rule_id="POL-STOCK-01")],
        total_usd=Decimal("500.00"),
        vendor_auto_approve_limit=Decimal("10000"),
    )
    decision = _approve(make_invoice(), validation)

    assert decision.decision is ApprovalDecisionValue.NEEDS_HUMAN
    assert "blocking flag present" in decision.rationale


@pytest.mark.parametrize("severity", [Severity.INFO, Severity.MEDIUM])
def test_non_blocking_flags_still_allow_approval(make_invoice, severity):
    validation = ValidationResult(
        flags=[_flag(severity)],
        total_usd=Decimal("500.00"),
        vendor_auto_approve_limit=Decimal("10000"),
    )
    assert _approve(make_invoice(), validation).decision is ApprovalDecisionValue.APPROVE


def test_over_the_vendor_limit_needs_a_human_even_with_no_flags(make_invoice):
    validation = ValidationResult(
        flags=[],
        total_usd=Decimal("9000.00"),
        vendor_known=True,
        vendor_auto_approve_limit=Decimal("7500"),
    )
    decision = _approve(make_invoice(), validation)

    assert decision.decision is ApprovalDecisionValue.NEEDS_HUMAN
    assert "exceeds vendor auto-approve limit" in decision.rationale


def test_over_the_high_value_threshold_needs_a_human(make_invoice):
    """The brief's ">$10K requires additional scrutiny", enforced independently of
    the vendor's own limit."""
    high_value = threshold("high_value")
    validation = ValidationResult(
        flags=[],
        total_usd=high_value + Decimal("0.01"),
        vendor_known=True,
        vendor_auto_approve_limit=Decimal("50000"),  # vendor would allow it
    )
    decision = _approve(make_invoice(), validation)

    assert decision.decision is ApprovalDecisionValue.NEEDS_HUMAN
    assert "exceeds high-value threshold" in decision.rationale


def test_exactly_at_the_high_value_threshold_is_still_approvable(make_invoice):
    validation = ValidationResult(
        flags=[],
        total_usd=threshold("high_value"),
        vendor_known=True,
        vendor_auto_approve_limit=Decimal("50000"),
    )
    assert _approve(make_invoice(), validation).decision is ApprovalDecisionValue.APPROVE


def test_decision_is_never_approve_for_an_unknown_vendor(make_invoice, db_path):
    """Wired against the real rules engine rather than a hand-built
    ValidationResult, so this covers the validate -> approve handoff."""
    invoice = make_invoice(vendor_name="Fraudster LLC")
    validation = validate(invoice, run_id=str(uuid4()), db_path=db_path)

    assert validation.vendor_known is False
    assert _approve(invoice, validation).decision is not ApprovalDecisionValue.APPROVE
