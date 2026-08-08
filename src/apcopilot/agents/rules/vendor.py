from __future__ import annotations

from decimal import Decimal

from apcopilot.models import ExtractedInvoice, Severity, ValidationFlag
from apcopilot.tools.vendors import VendorMatch

RULE_ID_VENDOR = "POL-VENDOR-01"
RULE_ID_LIMIT = "POL-VENDOR-02"
RULE_ID_CURRENCY = "POL-CUR-01"


def check_vendor(
    extraction: ExtractedInvoice, match: VendorMatch, *, total_usd: Decimal | None
) -> list[ValidationFlag]:
    """POL-VENDOR-01: unmatched or blocked vendors are not payable; watchlist
    vendors get a softer MEDIUM flag. POL-VENDOR-02: over the vendor's
    auto-approval limit. POL-CUR-01: invoice currency differs from that
    vendor's currency of record (not merely != USD)."""
    flags: list[ValidationFlag] = []

    if match.matched is None:
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID_VENDOR,
                code="UNKNOWN_VENDOR",
                severity=Severity.HIGH,
                message=f"Vendor '{extraction.vendor_name}' is not in the vendor master.",
                evidence={"query": match.query, "best_score": match.score},
            )
        )
        return flags

    vendor = match.matched

    if vendor.status == "blocked":
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID_VENDOR,
                code="UNKNOWN_VENDOR",
                severity=Severity.HIGH,
                message=f"Vendor '{vendor.name}' is blocked and not payable.",
                evidence={"matched_name": vendor.name, "status": vendor.status},
            )
        )
    elif vendor.status == "watchlist":
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID_VENDOR,
                code="VENDOR_WATCHLIST",
                severity=Severity.MEDIUM,
                message=f"Vendor '{vendor.name}' is on the watchlist.",
                evidence={"matched_name": vendor.name, "status": vendor.status},
            )
        )

    if total_usd is not None and total_usd > vendor.auto_approve_limit:
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID_LIMIT,
                code="OVER_VENDOR_LIMIT",
                severity=Severity.MEDIUM,
                message=(
                    f"Invoice total ${total_usd} exceeds vendor auto-approval limit "
                    f"${vendor.auto_approve_limit} for '{vendor.name}'."
                ),
                evidence={
                    "total_usd": str(total_usd),
                    "auto_approve_limit": str(vendor.auto_approve_limit),
                },
            )
        )

    if extraction.currency.upper() != vendor.currency.upper():
        flags.append(
            ValidationFlag(
                rule_id=RULE_ID_CURRENCY,
                code="CURRENCY_MISMATCH",
                severity=Severity.MEDIUM,
                message=(
                    f"Invoice currency {extraction.currency.upper()} differs from vendor "
                    f"currency of record {vendor.currency.upper()} for '{vendor.name}'."
                ),
                evidence={
                    "invoice_currency": extraction.currency.upper(),
                    "vendor_currency": vendor.currency.upper(),
                },
            )
        )

    return flags


__all__ = ["check_vendor"]
