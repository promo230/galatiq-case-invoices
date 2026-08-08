from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from apcopilot.config import get_settings
from apcopilot.models import ExtractedInvoice, Severity, ValidationFlag
from apcopilot.tools.ledger import vendor_history
from apcopilot.tools.policy import fraud_config, threshold
from apcopilot.tools.vendors import VendorMatch

RULE_ID = "POL-FRAUD-01"


def _lexicon_hits(text: str | None, phrases: list[str]) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    return [p for p in phrases if p.lower() in lowered]


def compute_fraud_score(
    extraction: ExtractedInvoice,
    *,
    match: VendorMatch,
    total_usd: Decimal | None,
    has_blocked_item: bool,
    db_path: Path | None = None,
) -> tuple[int, list[ValidationFlag], dict[str, int]]:
    """POL-FRAUD-01: deterministic weighted score. Every contributing signal
    is recorded (name -> weight) so the flag evidence is auditable.

    Returns (score, flags, fired_signals).
    """
    config = fraud_config()
    weights: dict[str, int] = config["weights"]
    fired: dict[str, int] = {}

    if match.matched is None:
        fired["unknown_vendor"] = weights["unknown_vendor"]

    if has_blocked_item:
        fired["blocked_item"] = weights["blocked_item"]

    if _lexicon_hits(extraction.notes, config["urgency_lexicon"]):
        fired["urgency_language"] = weights["urgency_language"]

    if _lexicon_hits(extraction.notes, config["payment_method_lexicon"]):
        fired["alternate_payment_method"] = weights["alternate_payment_method"]

    as_of = get_settings().as_of_date
    bad_due_date = (
        extraction.due_date is None
        or extraction.due_date < as_of
        or (
            extraction.invoice_date is not None
            and extraction.due_date < extraction.invoice_date
        )
    )
    if bad_due_date:
        fired["bad_due_date"] = weights["bad_due_date"]

    if extraction.total is not None:
        high_value = threshold("high_value")
        is_round = extraction.total % 1000 == 0 or extraction.total % 5000 == 0
        if is_round and extraction.total > high_value:
            fired["round_high_value_total"] = weights["round_high_value_total"]

    if match.matched is not None and total_usd is not None:
        history = vendor_history(match.matched.name, db_path=db_path)
        avg_amount = history.get("avg_amount_usd")
        if (
            history.get("invoice_count", 0) > 0
            and avg_amount
            and total_usd > Decimal(str(avg_amount)) * 3
        ):
            fired["exceeds_vendor_history"] = weights["exceeds_vendor_history"]

    score = sum(fired.values())

    flags: list[ValidationFlag] = []
    if score >= config["critical_threshold"]:
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID,
                code="FRAUD_SCORE_CRITICAL",
                severity=Severity.CRITICAL,
                message=(
                    f"Fraud score {score} meets or exceeds the critical threshold "
                    f"{config['critical_threshold']}."
                ),
                evidence={"score": score, "signals": fired},
            )
        )
    elif score >= config["high_threshold"]:
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID,
                code="FRAUD_SCORE_HIGH",
                severity=Severity.HIGH,
                message=(
                    f"Fraud score {score} meets or exceeds the high threshold "
                    f"{config['high_threshold']}."
                ),
                evidence={"score": score, "signals": fired},
            )
        )

    return score, flags, fired


__all__ = ["compute_fraud_score"]
