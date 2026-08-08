"""End-to-end tests: every file in the sample corpus through the real LangGraph
pipeline (ingest -> validate -> approve -> settle).

These are the regression net for the whole system. They run with
`APCOPILOT_LLM_MODE=off`, so the outcomes below are fully deterministic and
reproducible on a machine with no API key: the deterministic parsers, the
heuristic .txt fallback, the rules engine, and the rules-only approval fallback
are the only things involved.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from apcopilot.batch import discover_invoices, run_batch
from apcopilot.db.connection import get_connection
from apcopilot.graph import run_invoice

# (filename, status, lane, expected flag codes with multiplicity)
CORPUS = [
    ("invoice_1001.txt", "approved", "auto_approve", {}),
    (
        "invoice_1002.txt",
        "needs_human",
        "auto_reject",
        {"STOCK_SHORTFALL": 1, "OVER_VENDOR_LIMIT": 1},
    ),
    (
        "invoice_1003.txt",
        "rejected",
        "auto_reject",
        {
            "BLOCKED_ITEM": 1,
            "STOCK_SHORTFALL": 1,
            "UNKNOWN_VENDOR": 1,
            "FRAUD_SCORE_CRITICAL": 1,
        },
    ),
    ("invoice_1004.json", "approved", "auto_approve", {}),
    ("invoice_1004_revised.json", "approved", "auto_approve", {}),
    ("invoice_1006.csv", "approved", "auto_approve", {}),
    (
        "invoice_1008.txt",
        "needs_human",
        "auto_reject",
        {"UNKNOWN_ITEM": 2, "UNKNOWN_VENDOR": 1},
    ),
    (
        "invoice_1009.json",
        "rejected",
        "auto_reject",
        {
            "MISSING_FIELD": 1,
            "NEGATIVE_VALUE": 2,
            "MATH_MISMATCH": 2,
            "UNKNOWN_VENDOR": 1,
        },
    ),
    ("invoice_1014.xml", "approved", "auto_approve", {}),
    ("invoice_1016.json", "needs_human", "auto_reject", {"UNKNOWN_ITEM": 1}),
]


def _run(path: Path, db_path: Path, batch_id: str | None = None) -> dict:
    return asyncio.run(run_invoice(path, batch_id=batch_id, db_path=db_path))


def _codes(result: dict) -> dict[str, int]:
    return dict(Counter(flag["code"] for flag in result["flags"]))


def _count(db_path: Path, table: str, run_id: str) -> int:
    with get_connection(db_path) as conn:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE run_id=?", (run_id,))
        return int(row.fetchone()["n"])


@pytest.mark.parametrize(
    ("filename", "status", "lane", "expected_codes"),
    CORPUS,
    ids=[row[0] for row in CORPUS],
)
def test_corpus_outcomes(invoice_dir, db_path, filename, status, lane, expected_codes):
    result = _run(invoice_dir / filename, db_path)
    run = result["run"]

    assert run is not None
    assert run["error"] is None
    assert run["status"] == status
    assert run["lane"] == lane
    assert _codes(result) == expected_codes


def test_every_corpus_file_is_covered(invoice_dir):
    """A new fixture invoice must be added to CORPUS (or deliberately excluded),
    not silently left untested."""
    covered = {row[0] for row in CORPUS}
    # .pdf duplicates of .txt fixtures and the extra fixtures used by other tests
    # are intentionally out of scope for the outcome table.
    present = {p.name for p in invoice_dir.iterdir() if p.is_file()}

    assert covered <= present, f"CORPUS references missing files: {covered - present}"


def test_approved_run_records_a_payment(invoice_dir, db_path):
    result = _run(invoice_dir / "invoice_1001.txt", db_path)
    run = result["run"]

    assert run["status"] == "approved"
    assert _count(db_path, "payments", run["run_id"]) == 1
    with get_connection(db_path) as conn:
        payment = conn.execute(
            "SELECT * FROM payments WHERE run_id=?", (run["run_id"],)
        ).fetchone()
    assert payment["invoice_number"] == "INV-1001"
    assert payment["vendor"] == "Widgets Inc."
    assert Decimal(str(payment["amount"])) == Decimal("5000")
    assert payment["status"] == "success"


def test_rejected_run_records_a_rejection_with_reasoning_and_no_payment(
    invoice_dir, db_path
):
    result = _run(invoice_dir / "invoice_1003.txt", db_path)
    run_id = result["run"]["run_id"]

    assert _count(db_path, "payments", run_id) == 0
    assert _count(db_path, "rejections", run_id) == 1
    with get_connection(db_path) as conn:
        rejection = conn.execute(
            "SELECT * FROM rejections WHERE run_id=?", (run_id,)
        ).fetchone()
    assert rejection["decided_by"] == "rules"
    assert "FRAUD_SCORE_CRITICAL" in rejection["reason"]


def test_needs_human_run_neither_pays_nor_rejects(invoice_dir, db_path):
    result = _run(invoice_dir / "invoice_1002.txt", db_path)
    run_id = result["run"]["run_id"]

    assert result["run"]["status"] == "needs_human"
    assert _count(db_path, "payments", run_id) == 0
    assert _count(db_path, "rejections", run_id) == 0


def test_run_result_carries_a_full_four_stage_trace(invoice_dir, db_path):
    result = _run(invoice_dir / "invoice_1001.txt", db_path)

    nodes = [event["node"] for event in result["trace"]]
    assert nodes == ["ingest", "ingest", "validate", "approve", "settle"]
    assert [event["status"] for event in result["trace"]][1:] == ["done"] * 4
    assert [event["seq"] for event in result["trace"]] == [1, 2, 3, 4, 5]


def test_run_row_exposes_parsed_extraction_and_decision_objects(invoice_dir, db_path):
    """`extraction_json`/`decision_json` are stored as TEXT but every consumer
    (CLI, API, UI) wants real objects back."""
    run = _run(invoice_dir / "invoice_1004.json", db_path)["run"]

    assert isinstance(run["extraction_json"], dict)
    assert isinstance(run["decision_json"], dict)
    assert run["extraction_json"]["vendor_name"] == "Precision Parts Ltd."
    assert run["extraction_json"]["line_items"][0]["sku"] == "WIDGETA"
    assert run["decision_json"]["decision"] == "approve"
    assert run["decision_json"]["decided_by"] == "rules"


def test_eur_invoice_is_converted_with_the_dated_reference_rate(invoice_dir, db_path):
    """invoice_1014 is EUR and TechParts International's currency of record is also
    EUR, so this approves *despite* being non-USD -- no CURRENCY_MISMATCH."""
    result = _run(invoice_dir / "invoice_1014.xml", db_path)
    run = result["run"]

    assert result["flags"] == []
    assert run["currency"] == "EUR"
    assert Decimal(str(run["total"])) == Decimal("4125")
    assert Decimal(str(run["total_usd"])) == Decimal("4496.25")  # 4125 x 1.09
    assert run["status"] == "approved"


def test_fraudulent_invoice_scores_and_is_rejected_without_an_llm(invoice_dir, db_path):
    result = _run(invoice_dir / "invoice_1003.txt", db_path)
    run = result["run"]

    assert run["fraud_score"] == 110
    assert run["decision_json"]["decided_by"] == "rules"
    assert run["decision_json"]["rounds"] == 0
    fraud_flag = next(f for f in result["flags"] if f["code"] == "FRAUD_SCORE_CRITICAL")
    assert fraud_flag["evidence"]["signals"] == {
        "unknown_vendor": 30,
        "blocked_item": 25,
        "urgency_language": 15,
        "alternate_payment_method": 20,
        "bad_due_date": 10,
        "round_high_value_total": 10,
    }


# --- cross-run duplicate control ----------------------------------------------------


def test_revision_of_an_already_paid_invoice_is_blocked(invoice_dir, db_path):
    """The most interesting cross-run interaction in the system.

    invoice_1004.json and invoice_1004_revised.json share invoice number INV-1004
    but differ in content. Run in isolation each one approves; run into the same
    database, the second must be caught as a revision conflict against an invoice
    number that has already been paid, and must never pay twice.
    """
    first = _run(invoice_dir / "invoice_1004.json", db_path)
    assert first["run"]["status"] == "approved"
    assert first["flags"] == []

    second = _run(invoice_dir / "invoice_1004_revised.json", db_path)
    codes = _codes(second)

    assert "REVISION_CONFLICT" in codes
    assert "DUPLICATE_ALREADY_PAID" in codes
    assert second["run"]["status"] == "rejected"

    conflict = next(f for f in second["flags"] if f["code"] == "REVISION_CONFLICT")
    assert conflict["severity"] == "CRITICAL"
    assert conflict["evidence"]["prior_run_id"] == first["run"]["run_id"]
    assert conflict["evidence"]["prior_content_hash"] != conflict["evidence"]["content_hash"]

    # Exactly one payment for INV-1004 across both runs.
    with get_connection(db_path) as conn:
        paid = conn.execute(
            "SELECT COUNT(*) AS n FROM payments WHERE invoice_number='INV-1004'"
        ).fetchone()
    assert paid["n"] == 1


def test_reprocessing_the_identical_document_is_not_a_revision_conflict(
    invoice_dir, db_path
):
    """Same number *and* same content is a re-run, not a revision. It is still
    blocked -- by DUPLICATE_ALREADY_PAID, not by REVISION_CONFLICT."""
    _run(invoice_dir / "invoice_1004.json", db_path)
    second = _run(invoice_dir / "invoice_1004.json", db_path)
    codes = _codes(second)

    assert "REVISION_CONFLICT" not in codes
    assert codes == {"DUPLICATE_ALREADY_PAID": 1}
    assert second["run"]["status"] == "rejected"


# --- failure handling ----------------------------------------------------------------


def test_an_unreadable_document_fails_the_run_without_raising(tmp_path, db_path):
    """Node failures are recorded on the run and stop the pipeline; run_invoice
    still returns normally so one bad file never kills a batch."""
    bad = tmp_path / "invoice.docx"
    bad.write_text("not an invoice", encoding="utf-8")

    result = _run(bad, db_path)
    run = result["run"]

    assert run["status"] == "failed"
    assert "unsupported invoice document extension" in run["error"]
    assert [event["status"] for event in result["trace"]][-1] == "error"
    assert result["flags"] == []


def test_batch_shares_one_batch_id_and_survives_a_bad_file(invoice_dir, tmp_path, db_path):
    paths = [
        invoice_dir / "invoice_1001.txt",
        tmp_path / "broken.docx",
        invoice_dir / "invoice_1016.json",
    ]
    (tmp_path / "broken.docx").write_text("nope", encoding="utf-8")

    results = asyncio.run(run_batch(paths, db_path=db_path))

    assert [r["run"]["status"] for r in results] == ["approved", "failed", "needs_human"]
    batch_ids = {r["run"]["batch_id"] for r in results}
    assert len(batch_ids) == 1 and None not in batch_ids


def test_discover_invoices_is_sorted_and_filterable(invoice_dir, tmp_path):
    found = discover_invoices(invoice_dir)

    assert found == sorted(found)
    assert all(p.is_file() for p in found)
    assert discover_invoices(invoice_dir, "*.xml") == [invoice_dir / "invoice_1014.xml"]
    assert discover_invoices(tmp_path / "does-not-exist") == []
