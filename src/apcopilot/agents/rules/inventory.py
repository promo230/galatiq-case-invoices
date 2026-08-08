from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from apcopilot.models import ExtractedInvoice, Severity, ValidationFlag, canonicalize_sku
from apcopilot.tools.inventory import lookup_items

RULE_ID_ITEM = "POL-ITEM-01"
RULE_ID_STOCK = "POL-STOCK-01"

_NON_PAYABLE_STATUSES = {"blocked", "discontinued"}


def _canonical_sku(sku: str | None, description: str) -> str:
    """Re-derive the canonical SKU defensively rather than trusting whatever
    ingestion put on the line item — both sides must agree on the same key."""
    return canonicalize_sku(sku) if sku else canonicalize_sku(description)


def check_items(extraction: ExtractedInvoice, *, db_path: Path | None = None) -> list[ValidationFlag]:
    """POL-ITEM-01: line items absent from the catalog, or present but
    blocked/discontinued, cannot be paid."""
    flags: list[ValidationFlag] = []
    if not extraction.line_items:
        return flags

    skus = [_canonical_sku(item.sku, item.description) for item in extraction.line_items]
    records = lookup_items(sorted(set(skus)), db_path=db_path)

    for item, sku in zip(extraction.line_items, skus, strict=True):
        record = records.get(sku)
        if record is None:
            flags.append(
                ValidationFlag(
                    rule_id=RULE_ID_ITEM,
                    code="UNKNOWN_ITEM",
                    severity=Severity.HIGH,
                    message=f"Item '{item.description}' (SKU {sku}) is not in the catalog.",
                    sku=sku,
                    evidence={"sku": sku, "description": item.description},
                )
            )
        elif record.status in _NON_PAYABLE_STATUSES:
            flags.append(
                ValidationFlag(
                    rule_id=RULE_ID_ITEM,
                    code="BLOCKED_ITEM",
                    severity=Severity.HIGH,
                    message=(
                        f"Item '{item.description}' (SKU {sku}) has status "
                        f"'{record.status}' and is non-payable."
                    ),
                    sku=sku,
                    evidence={
                        "sku": sku,
                        "description": item.description,
                        "status": record.status,
                    },
                )
            )

    return flags


def check_stock(extraction: ExtractedInvoice, *, db_path: Path | None = None) -> list[ValidationFlag]:
    """POL-STOCK-01: aggregate requested quantity per SKU across all lines
    (a duplicate SKU on two lines is summed first) must not exceed stock.
    Read-only against `items` — never decrements stock."""
    flags: list[ValidationFlag] = []
    if not extraction.line_items:
        return flags

    requested: dict[str, int] = defaultdict(int)
    for item in extraction.line_items:
        sku = _canonical_sku(item.sku, item.description)
        requested[sku] += item.quantity

    records = lookup_items(sorted(requested), db_path=db_path)

    for sku, qty in requested.items():
        record = records.get(sku)
        if record is None:
            continue  # unknown item is POL-ITEM-01's concern, not stock's
        if qty > record.stock:
            flags.append(
                ValidationFlag(
                    rule_id=RULE_ID_STOCK,
                    code="STOCK_SHORTFALL",
                    severity=Severity.HIGH,
                    message=(
                        f"Requested quantity {qty} for SKU {sku} exceeds available "
                        f"stock {record.stock}."
                    ),
                    sku=sku,
                    evidence={"sku": sku, "requested": qty, "available": record.stock},
                )
            )

    return flags


__all__ = ["check_items", "check_stock"]
