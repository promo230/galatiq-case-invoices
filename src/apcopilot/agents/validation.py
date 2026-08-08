from __future__ import annotations

from pathlib import Path

from apcopilot.agents.rules.data_integrity import check_data_integrity
from apcopilot.agents.rules.duplicate import check_duplicate
from apcopilot.agents.rules.fraud import compute_fraud_score
from apcopilot.agents.rules.inventory import check_items, check_stock
from apcopilot.agents.rules.math import check_math
from apcopilot.agents.rules.vendor import check_vendor
from apcopilot.models import ExtractedInvoice, ValidationResult
from apcopilot.tools.fx import to_usd
from apcopilot.tools.vendors import match_vendor

# Deterministic rules engine: no LLM calls, pure rule evaluation against the
# mock inventory/vendor DB. Each `apcopilot.agents.rules.*` module owns one
# rule family and returns `list[ValidationFlag]`; this module just runs them
# all and folds the results (plus a few derived fields) into a ValidationResult.


def validate(
    extraction: ExtractedInvoice, *, run_id: str, db_path: Path | None = None
) -> ValidationResult:
    flags = []

    flags.extend(check_data_integrity(extraction))
    flags.extend(check_math(extraction))

    item_flags = check_items(extraction, db_path=db_path)
    flags.extend(item_flags)
    flags.extend(check_stock(extraction, db_path=db_path))

    total_usd, fx_rate = to_usd(extraction.total, extraction.currency, db_path=db_path)

    match = match_vendor(extraction.vendor_name, db_path=db_path)
    flags.extend(check_vendor(extraction, match, total_usd=total_usd))

    dup_flags, duplicate_of_run_id, revision_conflict, paid = check_duplicate(
        extraction, run_id=run_id, db_path=db_path
    )
    flags.extend(dup_flags)

    has_blocked_item = any(f.code == "BLOCKED_ITEM" for f in item_flags)
    fraud_score, fraud_flags, _fired = compute_fraud_score(
        extraction,
        match=match,
        total_usd=total_usd,
        has_blocked_item=has_blocked_item,
        db_path=db_path,
    )
    flags.extend(fraud_flags)

    return ValidationResult(
        flags=flags,
        fraud_score=fraud_score,
        total_usd=total_usd,
        fx_rate_used=fx_rate.rate if fx_rate else None,
        duplicate_of_run_id=duplicate_of_run_id,
        revision_conflict=revision_conflict,
        already_paid=paid,
        vendor_known=match.matched is not None,
        vendor_matched_name=match.matched.name if match.matched else None,
        vendor_auto_approve_limit=match.matched.auto_approve_limit if match.matched else None,
    )


__all__ = ["validate"]
