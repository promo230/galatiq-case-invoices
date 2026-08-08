from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, TypedDict

from apcopilot.db.runs import append_trace, insert_flags, update_run
from apcopilot.graph.payment import mock_payment
from apcopilot.models import (
    ApprovalDecision,
    ApprovalDecisionValue,
    ExtractionResult,
    ValidationResult,
)
from apcopilot.tools.ledger import record_payment, record_rejection


class InvoiceState(TypedDict, total=False):
    """Graph state threaded through ingest -> validate -> approve -> settle.

    Kept as a flat TypedDict (no Annotated reducers) since every key is written
    by exactly one node and never accumulated across nodes; last-value-wins is
    what we want. `db_path` travels as a str (not Path) so it round-trips
    cleanly through the checkpointer's serializer.
    """

    run_id: str
    document_path: str
    batch_id: str | None
    db_path: str | None
    extraction: ExtractionResult | None
    validation: ValidationResult | None
    decision: ApprovalDecision | None
    errors: list[str]
    failed: bool


def _db_path(state: InvoiceState) -> Path | None:
    raw = state.get("db_path")
    return Path(raw) if raw else None


def _fail(run_id: str, node: str, exc: Exception, db_path: Path | None) -> dict[str, Any]:
    """Shared failure path for every node: trace it, mark the run failed, and
    signal the graph's conditional edges to stop advancing."""
    append_trace(
        run_id=run_id, node=node, status="error", detail={"error": str(exc)}, db_path=db_path
    )
    update_run(run_id=run_id, status="failed", error=str(exc), db_path=db_path)
    return {"failed": True, "errors": [f"{node}: {exc}"]}


def route_after(state: InvoiceState) -> str:
    """Conditional-edge selector shared by every stage: stop the moment a node
    reports failure instead of running later stages against missing data."""
    return "stop" if state.get("failed") else "continue"


async def ingest_node(state: InvoiceState) -> dict[str, Any]:
    from apcopilot.ingestion import ingest_document

    run_id = state["run_id"]
    db_path = _db_path(state)
    document_path = Path(state["document_path"])
    try:
        extraction = await ingest_document(document_path, run_id=run_id)
        invoice = extraction.invoice
        append_trace(
            run_id=run_id,
            node="ingest",
            status="done",
            summary=f"{extraction.method}, confidence={invoice.extraction_confidence}",
            detail=invoice.model_dump(mode="json"),
            db_path=db_path,
        )
        update_run(
            run_id=run_id,
            invoice_number=invoice.invoice_number,
            revision=invoice.revision,
            vendor_name=invoice.vendor_name,
            currency=invoice.currency,
            total=invoice.total,
            due_date=invoice.due_date.isoformat() if invoice.due_date else None,
            content_hash=extraction.content_hash,
            confidence=invoice.extraction_confidence,
            extraction_json=invoice.model_dump(mode="json"),
            extraction_attempts=extraction.attempts,
            input_tokens=extraction.input_tokens,
            output_tokens=extraction.output_tokens,
            cost_usd=extraction.cost_usd,
            db_path=db_path,
        )
        return {"extraction": extraction}
    except Exception as exc:
        return _fail(run_id, "ingest", exc, db_path)


async def validate_node(state: InvoiceState) -> dict[str, Any]:
    from apcopilot.agents.validation import validate

    run_id = state["run_id"]
    db_path = _db_path(state)
    extraction: ExtractionResult = state["extraction"]  # type: ignore[assignment]
    try:
        result = validate(extraction.invoice, run_id=run_id, db_path=db_path)
        insert_flags(run_id, result.flags, db_path=db_path)
        append_trace(
            run_id=run_id,
            node="validate",
            status="done",
            summary=f"{len(result.flags)} flags, max={result.max_severity}, "
            f"fraud={result.fraud_score}",
            db_path=db_path,
        )
        if not result.flags:
            lane = "auto_approve"
        elif result.has_blocking:
            lane = "auto_reject"
        else:
            lane = "review"
        update_run(
            run_id=run_id,
            total_usd=result.total_usd,
            fraud_score=result.fraud_score,
            lane=lane,
            db_path=db_path,
        )
        return {"validation": result}
    except Exception as exc:
        return _fail(run_id, "validate", exc, db_path)


async def approve_node(state: InvoiceState) -> dict[str, Any]:
    from apcopilot.agents.approval import run_approval

    run_id = state["run_id"]
    db_path = _db_path(state)
    extraction: ExtractionResult = state["extraction"]  # type: ignore[assignment]
    validation: ValidationResult = state["validation"]  # type: ignore[assignment]
    try:
        decision = await run_approval(extraction.invoice, validation, run_id=run_id)
        append_trace(
            run_id=run_id,
            node="approve",
            status="done",
            summary=f"{decision.decision} by {decision.decided_by} in {decision.rounds} rounds",
            detail=decision.model_dump(mode="json"),
            db_path=db_path,
        )
        update_run(
            run_id=run_id,
            decision_json=decision.model_dump(mode="json"),
            approval_rounds=decision.rounds,
            db_path=db_path,
        )
        return {"decision": decision}
    except Exception as exc:
        return _fail(run_id, "approve", exc, db_path)


async def settle_node(state: InvoiceState) -> dict[str, Any]:
    run_id = state["run_id"]
    db_path = _db_path(state)
    extraction: ExtractionResult = state["extraction"]  # type: ignore[assignment]
    validation: ValidationResult = state["validation"]  # type: ignore[assignment]
    decision: ApprovalDecision = state["decision"]  # type: ignore[assignment]
    invoice = extraction.invoice
    amount = validation.total_usd if validation.total_usd is not None else invoice.total
    invoice_number = invoice.invoice_number or f"UNKNOWN-{run_id[:8]}"
    vendor = invoice.vendor_name or "UNKNOWN"

    try:
        if decision.decision == ApprovalDecisionValue.APPROVE:
            mock_payment(vendor, amount if amount is not None else Decimal(0))
            record_payment(
                run_id=run_id,
                invoice_number=invoice_number,
                vendor=vendor,
                amount=amount if amount is not None else Decimal(0),
                currency=invoice.currency,
                db_path=db_path,
            )
            update_run(run_id=run_id, status="approved", db_path=db_path)
            final_status = "approved"
        elif decision.decision == ApprovalDecisionValue.REJECT:
            record_rejection(
                run_id=run_id,
                invoice_number=invoice.invoice_number,
                vendor=invoice.vendor_name,
                amount=amount,
                reason=decision.rationale,
                decided_by=decision.decided_by,
                detail_json=json.dumps(decision.model_dump(mode="json"), default=str),
                db_path=db_path,
            )
            update_run(run_id=run_id, status="rejected", db_path=db_path)
            final_status = "rejected"
        else:
            update_run(run_id=run_id, status="needs_human", db_path=db_path)
            final_status = "needs_human"

        append_trace(
            run_id=run_id,
            node="settle",
            status="done",
            summary=f"final status={final_status}",
            db_path=db_path,
        )
        return {}
    except Exception as exc:
        return _fail(run_id, "settle", exc, db_path)


__all__ = [
    "InvoiceState",
    "approve_node",
    "ingest_node",
    "route_after",
    "settle_node",
    "validate_node",
]
