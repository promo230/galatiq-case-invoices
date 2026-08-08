"""Unit tests for the deterministic rules engine (`apcopilot.agents.validation`).

Each test starts from the clean `make_invoice()` baseline and perturbs exactly one
thing, so any flag produced can only have come from the rule under test. No LLM is
involved at any point -- validation is pure rule evaluation against the seeded
inventory/vendor/FX database.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from apcopilot.agents.rules.fraud import compute_fraud_score
from apcopilot.agents.validation import validate
from apcopilot.db.runs import create_run, update_run
from apcopilot.models import ExtractedInvoice, LineItem, Severity
from apcopilot.tools.ledger import record_payment
from apcopilot.tools.policy import fraud_config, threshold, tolerance
from apcopilot.tools.vendors import match_vendor
from conftest import codes  # pytest puts tests/ on sys.path


def _validate(invoice: ExtractedInvoice, db_path: Path, run_id: str | None = None):
    return validate(invoice, run_id=run_id or str(uuid4()), db_path=db_path)


def _flag(result, code: str):
    matches = [f for f in result.flags if f.code == code]
    assert matches, f"expected a {code} flag, got {codes(result.flags)}"
    return matches[0]


# --- baseline -------------------------------------------------------------------


def test_clean_invoice_produces_no_flags(make_invoice, db_path):
    result = _validate(make_invoice(), db_path)

    assert result.flags == []
    assert result.fraud_score == 0
    assert result.max_severity is None
    assert result.has_blocking is False
    assert result.total_usd == Decimal("500.00")
    assert result.fx_rate_used is None
    assert result.vendor_known is True
    assert result.vendor_matched_name == "Widgets Inc."
    assert result.vendor_auto_approve_limit == Decimal("10000")


# --- POL-STOCK-01 / POL-ITEM-01 (inventory) -------------------------------------


def test_stock_shortfall_flags_when_quantity_exceeds_inventory(make_invoice, line_item, db_path):
    # GadgetX has 5 units in stock.
    invoice = make_invoice(
        line_items=[line_item("GadgetX", 20, "750.00")],
        subtotal=Decimal("15000.00"),
        tax=Decimal("0.00"),
        total=Decimal("15000.00"),
    )
    result = _validate(invoice, db_path)

    flag = _flag(result, "STOCK_SHORTFALL")
    assert flag.rule_id == "POL-STOCK-01"
    assert flag.severity is Severity.HIGH
    assert flag.sku == "GADGETX"
    assert flag.evidence == {"sku": "GADGETX", "requested": 20, "available": 5}
    assert result.has_blocking is True


def test_stock_check_aggregates_quantity_across_duplicate_skus(make_invoice, line_item, db_path):
    # 10 + 8 = 18 WidgetA against 15 in stock: neither line alone exceeds stock,
    # so this only flags if quantities are summed per SKU first.
    invoice = make_invoice(
        line_items=[line_item("WidgetA", 10), line_item("WidgetA", 8)],
        subtotal=Decimal("4500.00"),
        tax=Decimal("0.00"),
        total=Decimal("4500.00"),
    )
    result = _validate(invoice, db_path)

    assert codes(result.flags) == ["STOCK_SHORTFALL"]
    assert _flag(result, "STOCK_SHORTFALL").evidence["requested"] == 18


def test_quantity_within_stock_does_not_flag(make_invoice, line_item, db_path):
    invoice = make_invoice(
        line_items=[line_item("WidgetA", 15)],
        subtotal=Decimal("3750.00"),
        tax=Decimal("0.00"),
        total=Decimal("3750.00"),
    )
    assert _validate(invoice, db_path).flags == []


def test_unknown_item_flags(make_invoice, line_item, db_path):
    invoice = make_invoice(
        line_items=[line_item("WidgetC", 3, "350.00")],
        subtotal=Decimal("1050.00"),
        tax=Decimal("0.00"),
        total=Decimal("1050.00"),
    )
    result = _validate(invoice, db_path)

    flag = _flag(result, "UNKNOWN_ITEM")
    assert flag.rule_id == "POL-ITEM-01"
    assert flag.severity is Severity.HIGH
    assert flag.sku == "WIDGETC"
    # An unknown SKU has no stock level to compare against, so POL-STOCK-01 stays quiet.
    assert "STOCK_SHORTFALL" not in codes(result.flags)


def test_blocked_item_flags_and_feeds_the_fraud_score(make_invoice, db_path):
    invoice = make_invoice(
        line_items=[
            LineItem(description="FakeItem", sku="FAKEITEM", quantity=1, unit_price=None)
        ],
        subtotal=None,
        tax=None,
    )
    result = _validate(invoice, db_path)

    flag = _flag(result, "BLOCKED_ITEM")
    assert flag.severity is Severity.HIGH
    assert flag.evidence["status"] == "blocked"
    assert result.fraud_score == fraud_config()["weights"]["blocked_item"]


def test_item_lookup_uses_canonical_sku_not_raw_text(make_invoice, db_path):
    """'widget-a' must resolve to the WIDGETA catalog row, not read as unknown."""
    invoice = make_invoice(
        line_items=[
            LineItem(
                description="widget-a",
                sku=None,
                quantity=2,
                unit_price=Decimal("250.00"),
                line_total=Decimal("500.00"),
            )
        ]
    )
    assert _validate(invoice, db_path).flags == []


# --- POL-VENDOR-01 / -02, POL-CUR-01 --------------------------------------------


def test_unknown_vendor_flags(make_invoice, db_path):
    result = _validate(make_invoice(vendor_name="Fraudster LLC"), db_path)

    flag = _flag(result, "UNKNOWN_VENDOR")
    assert flag.rule_id == "POL-VENDOR-01"
    assert flag.severity is Severity.HIGH
    assert result.vendor_known is False
    assert result.vendor_auto_approve_limit is None
    # Nothing further is asserted about the vendor once it is off-master: no
    # limit check, no currency check.
    assert codes(result.flags) == ["UNKNOWN_VENDOR"]


def test_vendor_fuzzy_match_tolerates_a_typo(make_invoice, db_path):
    """'Distributers' vs 'Distributors' must still resolve to the vendor of record."""
    result = _validate(make_invoice(vendor_name="QuickShip Distributers"), db_path)

    assert result.vendor_known is True
    assert result.vendor_matched_name == "QuickShip Distributors"
    assert "UNKNOWN_VENDOR" not in codes(result.flags)


def test_watchlist_vendor_is_medium_not_high(make_invoice, db_path):
    result = _validate(make_invoice(vendor_name="QuickShip Distributors"), db_path)

    flag = _flag(result, "VENDOR_WATCHLIST")
    assert flag.severity is Severity.MEDIUM
    assert result.has_blocking is False


def test_over_vendor_limit_flags(make_invoice, line_item, db_path):
    # Gadgets Co. auto-approves up to $7,500.
    invoice = make_invoice(
        vendor_name="Gadgets Co.",
        line_items=[line_item("WidgetA", 1, "8000.00")],
        subtotal=Decimal("8000.00"),
        tax=Decimal("0.00"),
        total=Decimal("8000.00"),
    )
    result = _validate(invoice, db_path)

    flag = _flag(result, "OVER_VENDOR_LIMIT")
    assert flag.rule_id == "POL-VENDOR-02"
    assert flag.severity is Severity.MEDIUM
    assert flag.evidence == {"total_usd": "8000.00", "auto_approve_limit": "7500"}


def test_total_exactly_at_vendor_limit_does_not_flag(make_invoice, line_item, db_path):
    invoice = make_invoice(
        vendor_name="Gadgets Co.",
        line_items=[line_item("WidgetA", 1, "7500.00")],
        subtotal=Decimal("7500.00"),
        tax=Decimal("0.00"),
        total=Decimal("7500.00"),
    )
    assert "OVER_VENDOR_LIMIT" not in codes(_validate(invoice, db_path).flags)


def test_currency_mismatch_is_relative_to_the_vendor_not_to_usd(make_invoice, db_path):
    """POL-CUR-01 compares against the vendor's currency of record.

    Widgets Inc. bills in USD, so a EUR invoice from them mismatches...
    """
    result = _validate(make_invoice(currency="EUR"), db_path)

    flag = _flag(result, "CURRENCY_MISMATCH")
    assert flag.rule_id == "POL-CUR-01"
    assert flag.severity is Severity.MEDIUM
    assert result.fx_rate_used == Decimal("1.0900")
    assert result.total_usd == Decimal("545.00")


def test_non_usd_invoice_from_a_non_usd_vendor_does_not_mismatch(make_invoice, db_path):
    """...but TechParts International's currency of record *is* EUR, so an EUR
    invoice from them is correct and must not be flagged."""
    result = _validate(
        make_invoice(vendor_name="TechParts International", currency="EUR"), db_path
    )

    assert result.flags == []
    assert result.fx_rate_used == Decimal("1.0900")
    assert result.total_usd == Decimal("545.00")


# --- POL-MATH-01 ------------------------------------------------------------------


def test_math_mismatch_when_total_does_not_equal_subtotal_plus_tax(make_invoice, db_path):
    invoice = make_invoice(
        subtotal=Decimal("500.00"), tax=Decimal("40.00"), total=Decimal("500.00")
    )
    result = _validate(invoice, db_path)

    flag = _flag(result, "MATH_MISMATCH")
    assert flag.rule_id == "POL-MATH-01"
    assert flag.severity is Severity.HIGH
    assert flag.evidence["residual"] == "-40.00"


def test_math_mismatch_when_line_items_do_not_sum_to_subtotal(make_invoice, line_item, db_path):
    invoice = make_invoice(
        line_items=[line_item("WidgetA", 2)],  # 2 x 250 = 500
        subtotal=Decimal("900.00"),
        tax=Decimal("0.00"),
        total=Decimal("900.00"),
    )
    result = _validate(invoice, db_path)

    flag = _flag(result, "MATH_MISMATCH")
    assert flag.evidence["line_items_sum"] == "500.00"
    assert flag.evidence["subtotal"] == "900.00"


def test_residual_within_tolerance_does_not_flag(make_invoice, db_path):
    # total_residual_abs is $1.00; a 50-cent rounding residual is tolerated.
    assert tolerance("total_residual_abs") == Decimal("1.00")
    invoice = make_invoice(
        subtotal=Decimal("500.00"), tax=Decimal("0.00"), total=Decimal("500.50")
    )
    assert "MATH_MISMATCH" not in codes(_validate(invoice, db_path).flags)


def test_math_is_skipped_when_the_inputs_are_missing(make_invoice, db_path):
    """No subtotal/tax means POL-MATH-01 stays silent rather than double-flagging
    on top of POL-DATA-01's missing-field flag."""
    invoice = make_invoice(subtotal=None, tax=None, total=None)
    assert "MATH_MISMATCH" not in codes(_validate(invoice, db_path).flags)


# --- POL-DATA-01 ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"vendor_name": None}, "vendor_name"),
        ({"vendor_name": "   "}, "vendor_name"),
        ({"invoice_number": None}, "invoice_number"),
        ({"total": None, "subtotal": None, "tax": None}, "total"),
    ],
)
def test_missing_critical_field_is_critical(make_invoice, db_path, overrides, field):
    result = _validate(make_invoice(**overrides), db_path)

    missing = [f for f in result.flags if f.code == "MISSING_FIELD"]
    assert [f.evidence["field"] for f in missing] == [field]
    assert missing[0].severity is Severity.CRITICAL
    assert missing[0].rule_id == "POL-DATA-01"


def test_negative_total_is_critical(make_invoice, db_path):
    invoice = make_invoice(
        subtotal=Decimal("-500.00"), tax=Decimal("0.00"), total=Decimal("-500.00")
    )
    result = _validate(invoice, db_path)

    flag = _flag(result, "NEGATIVE_VALUE")
    assert flag.severity is Severity.CRITICAL
    assert flag.evidence["field"] == "total"


def test_negative_quantity_is_critical(make_invoice, db_path):
    invoice = make_invoice(
        line_items=[
            LineItem(
                description="WidgetA",
                sku="WIDGETA",
                quantity=-5,
                unit_price=Decimal("250.00"),
                line_total=Decimal("-1250.00"),
            )
        ],
        subtotal=Decimal("-1250.00"),
        tax=Decimal("0.00"),
        total=Decimal("-1250.00"),
    )
    result = _validate(invoice, db_path)

    negatives = [f for f in result.flags if f.code == "NEGATIVE_VALUE"]
    assert {f.evidence["field"] for f in negatives} == {"quantity", "total"}
    assert all(f.severity is Severity.CRITICAL for f in negatives)


# --- POL-DUP-01 -------------------------------------------------------------------


def _seed_prior_run(db_path: Path, *, invoice_number: str, content_hash: str) -> str:
    run_id = str(uuid4())
    create_run(
        run_id=run_id,
        document_path=f"/tmp/{invoice_number}.json",
        source_format="json",
        db_path=db_path,
    )
    update_run(
        run_id=run_id,
        invoice_number=invoice_number,
        content_hash=content_hash,
        status="approved",
        db_path=db_path,
    )
    return run_id


def test_same_invoice_number_with_different_content_is_a_revision_conflict(
    make_invoice, db_path
):
    prior_run_id = _seed_prior_run(db_path, invoice_number="INV-9001", content_hash="hash-v1")
    current_run_id = _seed_prior_run(
        db_path, invoice_number="INV-9001", content_hash="hash-v2"
    )

    result = validate(make_invoice(), run_id=current_run_id, db_path=db_path)

    flag = _flag(result, "REVISION_CONFLICT")
    assert flag.rule_id == "POL-DUP-01"
    assert flag.severity is Severity.CRITICAL
    assert result.revision_conflict is True
    assert result.duplicate_of_run_id == prior_run_id


def test_same_invoice_number_with_identical_content_is_not_a_conflict(make_invoice, db_path):
    """The same document re-arriving (e.g. in another format) is a re-run, not a
    revision: same number, same hash."""
    _seed_prior_run(db_path, invoice_number="INV-9001", content_hash="hash-v1")
    current_run_id = _seed_prior_run(
        db_path, invoice_number="INV-9001", content_hash="hash-v1"
    )

    result = validate(make_invoice(), run_id=current_run_id, db_path=db_path)

    assert result.revision_conflict is False
    assert result.duplicate_of_run_id is None
    assert result.flags == []


def test_already_paid_invoice_number_is_blocked(make_invoice, db_path):
    paid_run_id = _seed_prior_run(db_path, invoice_number="INV-9001", content_hash="hash-v1")
    record_payment(
        run_id=paid_run_id,
        invoice_number="INV-9001",
        vendor="Widgets Inc.",
        amount=Decimal("500.00"),
        db_path=db_path,
    )
    current_run_id = _seed_prior_run(
        db_path, invoice_number="INV-9001", content_hash="hash-v1"
    )

    result = validate(make_invoice(), run_id=current_run_id, db_path=db_path)

    flag = _flag(result, "DUPLICATE_ALREADY_PAID")
    assert flag.severity is Severity.CRITICAL
    assert result.already_paid is True


# --- POL-FRAUD-01 -------------------------------------------------------------------


def test_policy_file_pins_the_fraud_weights_and_thresholds():
    """The weights are the contract the eval's fraud scores are reproducible
    against; changing one here should be a deliberate, visible edit."""
    config = fraud_config()

    assert config["weights"] == {
        "unknown_vendor": 30,
        "blocked_item": 25,
        "urgency_language": 15,
        "alternate_payment_method": 20,
        "bad_due_date": 10,
        "round_high_value_total": 10,
        "exceeds_vendor_history": 15,
    }
    assert config["high_threshold"] == 60
    assert config["critical_threshold"] == 80


def _score(invoice: ExtractedInvoice, db_path: Path, *, has_blocked_item: bool = False):
    match = match_vendor(invoice.vendor_name, db_path=db_path)
    total_usd = invoice.total
    return compute_fraud_score(
        invoice,
        match=match,
        total_usd=total_usd,
        has_blocked_item=has_blocked_item,
        db_path=db_path,
    )


def test_clean_invoice_scores_zero(make_invoice, db_path):
    score, flags, fired = _score(make_invoice(), db_path)

    assert (score, flags, fired) == (0, [], {})


def test_unknown_vendor_signal(make_invoice, db_path):
    score, _flags, fired = _score(make_invoice(vendor_name="Fraudster LLC"), db_path)

    assert fired == {"unknown_vendor": 30}
    assert score == 30


def test_blocked_item_signal(make_invoice, db_path):
    score, _flags, fired = _score(make_invoice(), db_path, has_blocked_item=True)

    assert fired == {"blocked_item": 25}
    assert score == 25


def test_urgency_language_signal(make_invoice, db_path):
    invoice = make_invoice(notes="URGENT - pay immediately to avoid penalties!!!")
    score, _flags, fired = _score(invoice, db_path)

    assert fired == {"urgency_language": 15}
    assert score == 15


def test_alternate_payment_method_signal(make_invoice, db_path):
    invoice = make_invoice(notes="Please remit by wire transfer to our new account.")
    score, _flags, fired = _score(invoice, db_path)

    assert fired == {"alternate_payment_method": 20}
    assert score == 20


@pytest.mark.parametrize(
    ("due_date", "invoice_date"),
    [
        (None, date(2026, 1, 15)),
        (date(2026, 1, 20), date(2026, 1, 15)),  # already past as_of_date 2026-02-01
        (date(2026, 1, 10), date(2026, 1, 15)),  # due before it was even issued
    ],
    ids=["missing", "already_past_due", "before_invoice_date"],
)
def test_bad_due_date_signal(make_invoice, db_path, due_date, invoice_date):
    invoice = make_invoice(due_date=due_date, invoice_date=invoice_date)
    score, _flags, fired = _score(invoice, db_path)

    assert fired == {"bad_due_date": 10}
    assert score == 10


def test_round_high_value_total_signal(make_invoice, line_item, db_path):
    invoice = make_invoice(
        line_items=[line_item("WidgetA", 1, "15000.00")],
        subtotal=Decimal("15000.00"),
        tax=Decimal("0.00"),
        total=Decimal("15000.00"),
    )
    score, _flags, fired = _score(invoice, db_path)

    assert fired == {"round_high_value_total": 10}
    assert score == 10


def test_round_total_below_the_high_value_threshold_does_not_fire(make_invoice, db_path):
    """$5,000 is just as round as $15,000; the signal is round *and* high-value."""
    assert threshold("high_value") == Decimal("10000.00")
    invoice = make_invoice(
        subtotal=Decimal("5000.00"), tax=Decimal("0.00"), total=Decimal("5000.00")
    )
    score, _flags, fired = _score(invoice, db_path)

    assert fired == {}
    assert score == 0


def test_exceeds_vendor_history_signal(make_invoice, db_path):
    for _ in range(3):
        run_id = str(uuid4())
        create_run(
            run_id=run_id,
            document_path="/tmp/prior.json",
            source_format="json",
            db_path=db_path,
        )
        update_run(
            run_id=run_id,
            vendor_name="Widgets Inc.",
            total_usd=Decimal("1000.00"),
            db_path=db_path,
        )

    # 3x the $1,000 running average is the trip point; $4,000 clears it.
    invoice = make_invoice(
        subtotal=Decimal("4000.00"), tax=Decimal("0.00"), total=Decimal("4000.00")
    )
    score, _flags, fired = _score(invoice, db_path)

    assert fired == {"exceeds_vendor_history": 15}
    assert score == 15


def test_signals_accumulate_into_a_critical_flag(make_invoice, db_path):
    """The invoice_1003 shape: unknown vendor + blocked item + urgency + wire
    transfer + past due + round high-value total = 110."""
    invoice = make_invoice(
        vendor_name="Fraudster LLC",
        line_items=[
            LineItem(
                description="FakeItem",
                sku="FAKEITEM",
                quantity=100,
                unit_price=Decimal("1000.00"),
            )
        ],
        subtotal=None,
        tax=None,
        total=Decimal("100000.00"),
        due_date=None,
        notes="URGENT - Pay immediately to avoid penalties!!! Wire transfer preferred.",
    )
    score, flags, fired = _score(invoice, db_path, has_blocked_item=True)

    assert fired == {
        "unknown_vendor": 30,
        "blocked_item": 25,
        "urgency_language": 15,
        "alternate_payment_method": 20,
        "bad_due_date": 10,
        "round_high_value_total": 10,
    }
    assert score == 110
    assert [f.code for f in flags] == ["FRAUD_SCORE_CRITICAL"]
    assert flags[0].severity is Severity.CRITICAL
    assert flags[0].evidence == {"score": 110, "signals": fired}


def test_score_between_the_thresholds_is_high_not_critical(make_invoice, db_path):
    # unknown_vendor (30) + urgency (15) + wire transfer (20) = 65: >= 60, < 80.
    invoice = make_invoice(
        vendor_name="Fraudster LLC",
        notes="Final notice - remit by wire transfer.",
    )
    score, flags, _fired = _score(invoice, db_path)

    assert score == 65
    assert [f.code for f in flags] == ["FRAUD_SCORE_HIGH"]
    assert flags[0].severity is Severity.HIGH
