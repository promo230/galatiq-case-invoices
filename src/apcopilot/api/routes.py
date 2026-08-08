from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from apcopilot.batch import discover_invoices, run_batch
from apcopilot.db.runs import append_trace, get_flags, get_run, get_trace, list_runs, update_run
from apcopilot.db.seed import reset_database
from apcopilot.graph.payment import mock_payment
from apcopilot.logging import get_logger
from apcopilot.tools.ledger import (
    committed_vs_available,
    get_human_actions,
    record_human_action,
    record_payment,
    record_rejection,
)
from apcopilot.tools.policy import business_case, load_policies

logger = get_logger(__name__)
router = APIRouter(prefix="/api")

# The sample corpus doesn't imply a real-world annual volume, but the case
# brief's "$2M/year" framing needs one to be reproduced as a concrete number.
# 50,000 invoices/year (~200/business day) is a documented, conservative
# assumption for a mid-size PE-backed manufacturer's AP desk.
ASSUMED_ANNUAL_VOLUME = 50_000


def safe_json(data: Any) -> Any:
    """Round-trip through json.dumps(default=str) so Decimal/date fields come
    out as strings rather than floats (see fastapi.encoders.jsonable_encoder,
    which silently turns Decimal into float otherwise)."""
    return json.loads(json.dumps(data, default=str))


class RunRequest(BaseModel):
    document_path: str


class BatchRequest(BaseModel):
    dir: str = "data/invoices"
    pattern: str = "*"


class ActionRequest(BaseModel):
    outcome: Literal["approve", "reject"]
    note: str | None = None
    actor: str | None = None


@router.get("/runs")
async def api_list_runs(status: str | None = None, limit: int = 200) -> Any:
    return safe_json(list_runs(status=status, limit=limit))


def _run_detail(run: dict) -> dict:
    """The GET /runs/{id} response body, also returned by the action endpoint so
    the UI can re-render from the write's response instead of refetching."""
    run_id = run["run_id"]
    return {
        "run": run,
        "flags": get_flags(run_id),
        "trace": get_trace(run_id),
        "human_actions": get_human_actions(run_id),
    }


def _require_run(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return run


@router.get("/runs/{run_id}")
async def api_get_run(run_id: str) -> Any:
    return safe_json(_run_detail(_require_run(run_id)))


@router.post("/runs")
async def api_create_run(body: RunRequest) -> Any:
    from apcopilot.graph import run_invoice

    result = await run_invoice(Path(body.document_path))
    return safe_json(result)


@router.post("/runs/{run_id}/action")
async def api_run_action(run_id: str, body: ActionRequest) -> Any:
    """Close the human-review loop: approve (pay) or reject a needs_human run.

    Mirrors settle_node's approve/reject branches so a VP click produces exactly
    the same ledger writes as an agent decision would have.
    """
    run = _require_run(run_id)
    if run["status"] != "needs_human":
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {run_id!r} has status {run['status']!r}, not 'needs_human'; "
                "human actions only apply to runs awaiting review"
            ),
        )

    actor = body.actor or "vp@acme"
    # Row values come back from SQLite as TEXT/None; Decimal(str(...)) keeps the
    # exact figure (never float). Fallbacks mirror settle_node exactly.
    raw_amount = run.get("total_usd") if run.get("total_usd") is not None else run.get("total")
    amount = Decimal(str(raw_amount)) if raw_amount is not None else None
    invoice_number = run.get("invoice_number") or f"UNKNOWN-{run_id[:8]}"
    vendor = run.get("vendor_name") or "UNKNOWN"

    if body.outcome == "approve":
        mock_payment(vendor, amount if amount is not None else Decimal(0))
        record_payment(
            run_id=run_id,
            invoice_number=invoice_number,
            vendor=vendor,
            amount=amount if amount is not None else Decimal(0),
            currency=run.get("currency") or "USD",
        )
        update_run(run_id=run_id, status="approved")
    else:
        record_rejection(
            run_id=run_id,
            invoice_number=run.get("invoice_number"),
            vendor=run.get("vendor_name"),
            amount=amount,
            reason=body.note or "Rejected on human review",
            decided_by="human",
        )
        update_run(run_id=run_id, status="rejected")

    record_human_action(run_id=run_id, actor=actor, outcome=body.outcome, note=body.note)
    append_trace(
        run_id=run_id,
        node="human_review",
        status="done",
        summary=f"{body.outcome} by {actor}",
    )
    logger.info("human_review %s on run %s by %s", body.outcome, run_id, actor)
    return safe_json(_run_detail(_require_run(run_id)))


@router.post("/runs/batch")
async def api_create_batch(body: BatchRequest) -> Any:
    paths = discover_invoices(Path(body.dir), body.pattern)
    if not paths:
        raise HTTPException(
            status_code=404, detail=f"no files matching {body.pattern!r} in {body.dir}"
        )
    results = await run_batch(paths)
    return safe_json(results)


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@router.get("/dashboard")
async def api_dashboard() -> Any:
    runs = list_runs(limit=1000)
    status_counts = {"approved": 0, "rejected": 0, "needs_human": 0, "failed": 0, "other": 0}
    for r in runs:
        status = r.get("status")
        key = status if status in status_counts else "other"
        status_counts[key] += 1

    durations = [r["duration_ms"] for r in runs if r.get("duration_ms") is not None]
    costs = [float(r["cost_usd"]) for r in runs if r.get("cost_usd") is not None]
    avg_duration_ms = _avg(durations)
    avg_llm_cost_usd = Decimal(str(round(_avg(costs) or 0.0, 6)))
    total_llm_cost_usd = Decimal(str(round(sum(costs), 6)))

    bc = business_case()
    manual_minutes = Decimal(str(bc["manual_minutes_per_invoice"]))
    hourly_rate = Decimal(str(bc["fully_loaded_hourly_rate"]))
    cents = Decimal("0.01")
    baseline_cost_per_invoice_usd = ((manual_minutes / Decimal(60)) * hourly_rate).quantize(cents)
    savings_per_invoice_usd = (baseline_cost_per_invoice_usd - avg_llm_cost_usd).quantize(cents)
    estimated_annual_savings_usd = (savings_per_invoice_usd * ASSUMED_ANNUAL_VOLUME).quantize(
        cents
    )

    payload = {
        "runs_observed": len(runs),
        "status_counts": status_counts,
        "avg_duration_ms": avg_duration_ms,
        "business_case": {
            "manual_minutes_per_invoice": manual_minutes,
            "fully_loaded_hourly_rate_usd": hourly_rate,
            "baseline_error_rate": bc["baseline_error_rate"],
            "baseline_days_to_process": bc["baseline_days_to_process"],
            "baseline_cost_per_invoice_usd": baseline_cost_per_invoice_usd,
            "actual_llm_cost_per_invoice_usd": avg_llm_cost_usd,
            "total_llm_cost_usd_observed": total_llm_cost_usd,
            "savings_per_invoice_usd": savings_per_invoice_usd,
            "assumed_annual_volume": ASSUMED_ANNUAL_VOLUME,
            "assumed_annual_volume_note": (
                "Not derivable from the sample corpus; 50,000 invoices/year "
                "(~200/business day) is a documented assumption used only to "
                "translate per-invoice savings into the case brief's annual framing."
            ),
            "estimated_annual_savings_usd": estimated_annual_savings_usd,
        },
        "inventory_pressure": committed_vs_available(),
    }
    return safe_json(payload)


@router.get("/policies")
async def api_policies() -> Any:
    return safe_json(load_policies())


@router.post("/reset")
async def api_reset() -> Any:
    counts = reset_database()
    return safe_json({"status": "reset", "seeded": counts})


__all__ = ["router"]
