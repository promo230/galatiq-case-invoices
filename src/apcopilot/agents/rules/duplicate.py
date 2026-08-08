from __future__ import annotations

from pathlib import Path

from apcopilot.db.runs import get_run
from apcopilot.models import ExtractedInvoice, Severity, ValidationFlag
from apcopilot.tools.ledger import already_paid, prior_runs_for_invoice

RULE_ID = "POL-DUP-01"


def check_duplicate(
    extraction: ExtractedInvoice, *, run_id: str, db_path: Path | None = None
) -> tuple[list[ValidationFlag], str | None, bool, bool]:
    """POL-DUP-01: same invoice number, different content is a revision
    conflict; if the earlier version was already paid, this one is blocked.

    Returns (flags, duplicate_of_run_id, revision_conflict, already_paid).

    `validate()` only receives the ExtractedInvoice, not the ExtractionResult
    that carries `content_hash`. By the time validation runs the orchestrator
    has already created this run's row in `invoice_runs` (that's what makes
    `exclude_run_id=run_id` meaningful), so the current content hash is read
    back from that row rather than threaded through the function signature.
    """
    flags: list[ValidationFlag] = []
    duplicate_of_run_id: str | None = None
    revision_conflict = False

    invoice_number = extraction.invoice_number
    if invoice_number:
        current_run = get_run(run_id, db_path=db_path)
        content_hash = current_run.get("content_hash") if current_run else None

        if content_hash is not None:
            for prior in prior_runs_for_invoice(
                invoice_number, exclude_run_id=run_id, db_path=db_path
            ):
                if prior["content_hash"] and prior["content_hash"] != content_hash:
                    revision_conflict = True
                    duplicate_of_run_id = prior["run_id"]
                    flags.append(
                        ValidationFlag(
                            rule_id=RULE_ID,
                            code="REVISION_CONFLICT",
                            severity=Severity.CRITICAL,
                            message=(
                                f"Invoice {invoice_number} conflicts with prior run "
                                f"{prior['run_id']}: same invoice number, different content."
                            ),
                            evidence={
                                "prior_run_id": prior["run_id"],
                                "prior_content_hash": prior["content_hash"],
                                "content_hash": content_hash,
                            },
                        )
                    )
                    break

    paid = already_paid(invoice_number, db_path=db_path)
    if paid:
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID,
                code="DUPLICATE_ALREADY_PAID",
                severity=Severity.CRITICAL,
                message=f"Invoice {invoice_number} has already been paid.",
                evidence={"invoice_number": invoice_number},
            )
        )

    return flags, duplicate_of_run_id, revision_conflict, paid


__all__ = ["check_duplicate"]
