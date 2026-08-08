from __future__ import annotations

from apcopilot.models import ExtractedInvoice, Severity, ValidationFlag

RULE_ID = "POL-DATA-01"


def check_data_integrity(extraction: ExtractedInvoice) -> list[ValidationFlag]:
    """POL-DATA-01: missing vendor/invoice_number/total, negative quantities,
    negative total. Per policy these are "rejected without review" but this
    stage only flags CRITICAL — the approval stage decides rejection."""
    flags: list[ValidationFlag] = []

    if not extraction.vendor_name or not extraction.vendor_name.strip():
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID,
                code="MISSING_FIELD",
                severity=Severity.CRITICAL,
                message="Invoice is missing a vendor name.",
                evidence={"field": "vendor_name"},
            )
        )

    if not extraction.invoice_number or not extraction.invoice_number.strip():
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID,
                code="MISSING_FIELD",
                severity=Severity.CRITICAL,
                message="Invoice is missing an invoice number.",
                evidence={"field": "invoice_number"},
            )
        )

    if extraction.total is None:
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID,
                code="MISSING_FIELD",
                severity=Severity.CRITICAL,
                message="Invoice is missing a total amount.",
                evidence={"field": "total"},
            )
        )
    elif extraction.total < 0:
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID,
                code="NEGATIVE_VALUE",
                severity=Severity.CRITICAL,
                message=f"Invoice total is negative ({extraction.total}).",
                evidence={"field": "total", "value": str(extraction.total)},
            )
        )

    for item in extraction.line_items:
        if item.quantity < 0:
            flags.append(
                ValidationFlag(
                    rule_id=RULE_ID,
                    code="NEGATIVE_VALUE",
                    severity=Severity.CRITICAL,
                    message=(
                        f"Line item '{item.description}' has a negative quantity "
                        f"({item.quantity})."
                    ),
                    sku=item.sku,
                    evidence={
                        "field": "quantity",
                        "description": item.description,
                        "value": item.quantity,
                    },
                )
            )

    return flags


__all__ = ["check_data_integrity"]
