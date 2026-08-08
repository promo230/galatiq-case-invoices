from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from apcopilot.db.connection import get_connection, to_decimal


def _now() -> str:
    return datetime.now(UTC).isoformat()


def prior_runs_for_invoice(
    invoice_number: str, exclude_run_id: str | None = None, db_path: Path | None = None
) -> list[dict]:
    """Every earlier run for this invoice number, with its content hash.

    Same number + different hash is a revision conflict; same number + same hash
    is the same document arriving in another format.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT run_id, content_hash, revision, status, total, total_usd, "
            "document_path, source_format, created_at "
            "FROM invoice_runs WHERE invoice_number = ? ORDER BY created_at",
            (invoice_number,),
        ).fetchall()
    return [dict(r) for r in rows if r["run_id"] != exclude_run_id]


def get_payment_history(invoice_number: str, db_path: Path | None = None) -> dict:
    """LLM-facing: has this invoice number already been paid?"""
    with get_connection(db_path) as conn:
        payments = conn.execute(
            "SELECT run_id, vendor, amount, currency, status, paid_at "
            "FROM payments WHERE invoice_number = ? ORDER BY paid_at",
            (invoice_number,),
        ).fetchall()
        runs = conn.execute(
            "SELECT run_id, status, total, created_at FROM invoice_runs "
            "WHERE invoice_number = ? ORDER BY created_at",
            (invoice_number,),
        ).fetchall()
    return {
        "invoice_number": invoice_number,
        "already_paid": len(payments) > 0,
        "payments": [dict(p) for p in payments],
        "prior_runs": [dict(r) for r in runs],
    }


def vendor_history(vendor_name: str, db_path: Path | None = None) -> dict:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n, AVG(CAST(total_usd AS REAL)) AS avg_total "
            "FROM invoice_runs WHERE vendor_name = ? AND total_usd IS NOT NULL",
            (vendor_name,),
        ).fetchone()
    count = int(row["n"] or 0)
    return {
        "invoice_count": count,
        "avg_amount_usd": round(float(row["avg_total"]), 2) if row["avg_total"] else None,
    }


def already_paid(invoice_number: str | None, db_path: Path | None = None) -> bool:
    if not invoice_number:
        return False
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM payments WHERE invoice_number = ? LIMIT 1", (invoice_number,)
        ).fetchone()
    return row is not None


def record_payment(
    *,
    run_id: str,
    invoice_number: str,
    vendor: str,
    amount: Decimal,
    currency: str = "USD",
    status: str = "success",
    db_path: Path | None = None,
) -> dict:
    """Persist a payment under an idempotency key.

    Replaying a checkpoint past the pay node re-executes it; the unique key is
    what makes that a no-op rather than a double payment.
    """
    key = f"{invoice_number}|{amount}|{run_id}"
    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT id, paid_at FROM payments WHERE idempotency_key = ?", (key,)
        ).fetchone()
        if existing:
            return {"idempotency_key": key, "duplicate_suppressed": True, "status": status}
        conn.execute(
            "INSERT INTO payments (idempotency_key, run_id, invoice_number, vendor, "
            "amount, currency, status, paid_at) VALUES (?,?,?,?,?,?,?,?)",
            (key, run_id, invoice_number, vendor, amount, currency, status, _now()),
        )
    return {"idempotency_key": key, "duplicate_suppressed": False, "status": status}


def record_rejection(
    *,
    run_id: str,
    invoice_number: str | None,
    vendor: str | None,
    amount: Decimal | None,
    reason: str,
    decided_by: str,
    detail_json: str | None = None,
    db_path: Path | None = None,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO rejections (run_id, invoice_number, vendor, amount, reason, "
            "decided_by, detail_json, rejected_at) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, invoice_number, vendor, amount, reason, decided_by, detail_json, _now()),
        )


def record_human_action(
    *, run_id: str, actor: str, outcome: str, note: str | None = None, db_path: Path | None = None
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO human_actions (run_id, actor, outcome, note, acted_at) VALUES (?,?,?,?,?)",
            (run_id, actor, outcome, note, _now()),
        )


def get_human_actions(run_id: str, db_path: Path | None = None) -> list[dict]:
    """Audit trail of every human review action taken on a run, oldest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, run_id, actor, outcome, note, acted_at FROM human_actions "
            "WHERE run_id = ? ORDER BY acted_at, id",
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def committed_vs_available(db_path: Path | None = None) -> list[dict]:
    """Aggregate demand across all processed invoices against the stock snapshot.

    Stock is never decremented during validation (that would make results depend
    on processing order and break the eval), so this view is how the pressure on
    inventory surfaces instead.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT i.sku, i.name, i.stock, i.status, "
            "COALESCE(SUM(CAST(f.evidence_json ->> '$.requested' AS INTEGER)), 0) AS requested "
            "FROM items i LEFT JOIN invoice_flags f "
            "  ON f.sku = i.sku AND f.code = 'STOCK_SHORTFALL' "
            "GROUP BY i.sku ORDER BY i.sku"
        ).fetchall()
    return [
        {
            "sku": r["sku"],
            "name": r["name"],
            "stock": int(r["stock"]),
            "status": r["status"],
            "requested_on_flagged_invoices": int(r["requested"] or 0),
        }
        for r in rows
    ]


__all__ = [
    "already_paid",
    "committed_vs_available",
    "get_human_actions",
    "get_payment_history",
    "prior_runs_for_invoice",
    "record_human_action",
    "record_payment",
    "record_rejection",
    "to_decimal",
    "vendor_history",
]
