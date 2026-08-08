from __future__ import annotations

import re

from apcopilot.models import ExtractedInvoice, LineItem, canonicalize_sku

from ._shared import none_if_blank, parse_date_value, parse_decimal, parse_int

# Zero-API-call fallback for .txt/.pdf, used when llm_mode == "off" or the LLM path
# raised LLMUnavailableError. Deliberately not exhaustive -- it exists so the system
# still produces *something* usable, not a perfect parser. Field order below is the
# priority in which a line is tested against each label; anchoring each pattern at the
# start of the line (rather than searching) is what keeps e.g. "Subtotal:" from being
# swallowed by the "total" label.
_LABEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "invoice_number": re.compile(
        r"^\s*(?:invoice\s*number|invoice\s*#|invoice|inv\s*#|inv\s*no\.?|inv)\s*:\s*(.+)$",
        re.IGNORECASE,
    ),
    "vendor": re.compile(r"^\s*(?:vendor|vndr|supplier)\s*:\s*(.+)$", re.IGNORECASE),
    "invoice_date": re.compile(r"^\s*(?:invoice\s*date|date|dt)\s*:\s*(.+)$", re.IGNORECASE),
    "due_date": re.compile(r"^\s*(?:due\s*date|due\s*dt|due)\s*:\s*(.+)$", re.IGNORECASE),
    "subtotal": re.compile(r"^\s*subtotal\s*:\s*(.+)$", re.IGNORECASE),
    "tax": re.compile(r"^\s*tax[^:]*:\s*(.+)$", re.IGNORECASE),
    "total": re.compile(
        r"^\s*(?:total\s*amount|grand\s*total|total|amt)\s*:\s*(.+)$", re.IGNORECASE
    ),
    "payment_terms": re.compile(
        r"^\s*(?:payment\s*terms|pymnt\s*terms|terms)\s*:\s*(.+)$", re.IGNORECASE
    ),
    "notes": re.compile(r"^\s*notes?\s*:\s*(.+)$", re.IGNORECASE),
}

# Ambiguous labels that also occur in an email wrapper around the invoice. These
# are only consulted once the whole document has been scanned and the strong
# label above never matched, so "From: billing@noproduct.biz" in a mail header
# can never outrank an explicit "Vendor: NoProd Industries" further down.
_WEAK_LABEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "vendor": re.compile(r"^\s*from\s*:\s*(.+)$", re.IGNORECASE),
}

_DESC = r"[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*)*?"
_ITEM_PATTERNS: list[re.Pattern[str]] = [
    # "  WidgetA    qty: 10    unit price: $250.00"
    re.compile(
        rf"^\s*(?P<desc>{_DESC})\s+qty:\s*(?P<qty>\d+)\s+unit price:\s*\$?(?P<price>[\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # "GadgetX  qty 20   @ $750 ea"
    re.compile(
        rf"^\s*(?P<desc>{_DESC})\s+qty\s*(?P<qty>\d+)\s+@\s*\$?(?P<price>[\d,]+\.?\d*)\s*ea",
        re.IGNORECASE,
    ),
    # "- SuperGizmo       x12     $400.00 each"
    re.compile(
        rf"^\s*-\s*(?P<desc>{_DESC})\s+x(?P<qty>\d+)\s+\$?(?P<price>[\d,]+\.?\d*)\s*each",
        re.IGNORECASE,
    ),
]

_FIELD_WARNING = {
    "invoice_number": "heuristic parser could not find an invoice number",
    "vendor": "heuristic parser could not find a vendor name",
    "invoice_date": "heuristic parser could not find an invoice date",
    "due_date": "heuristic parser could not find a due date",
    "subtotal": "heuristic parser could not find a subtotal",
    "tax": "heuristic parser could not find a tax amount",
    "total": "heuristic parser could not find a total",
    "payment_terms": "heuristic parser could not find payment terms",
}

_CRITICAL_FIELDS = ("vendor", "invoice_number", "total")


def heuristic_extract(text: str) -> tuple[ExtractedInvoice, list[str]]:
    fields: dict[str, str] = {}
    weak_fields: dict[str, str] = {}
    line_items: list[LineItem] = []

    for line in text.splitlines():
        matched_label = False
        for field, pattern in _LABEL_PATTERNS.items():
            if field in fields:
                continue
            match = pattern.match(line)
            if match:
                fields[field] = match.group(1).strip()
                matched_label = True
                break
        if matched_label:
            continue

        for field, pattern in _WEAK_LABEL_PATTERNS.items():
            if field in weak_fields:
                continue
            match = pattern.match(line)
            if match:
                weak_fields[field] = match.group(1).strip()
                matched_label = True
                break
        if matched_label:
            continue

        for pattern in _ITEM_PATTERNS:
            match = pattern.match(line)
            if match:
                desc = match.group("desc").strip()
                line_items.append(
                    LineItem(
                        description=desc,
                        sku=canonicalize_sku(desc),
                        quantity=parse_int(match.group("qty")) or 0,
                        unit_price=parse_decimal(match.group("price")),
                    )
                )
                break

    for field, value in weak_fields.items():
        fields.setdefault(field, value)

    warnings = [message for field, message in _FIELD_WARNING.items() if field not in fields]
    if not line_items:
        warnings.append("heuristic parser could not find any line items")

    missing_critical = sum(1 for field in _CRITICAL_FIELDS if field not in fields)
    confidence = max(0.1, min(0.5, 0.5 - 0.1 * missing_critical))

    invoice = ExtractedInvoice(
        invoice_number=none_if_blank(fields.get("invoice_number")),
        vendor_name=none_if_blank(fields.get("vendor")),
        line_items=line_items,
        subtotal=parse_decimal(fields.get("subtotal")),
        tax=parse_decimal(fields.get("tax")),
        total=parse_decimal(fields.get("total")),
        invoice_date=parse_date_value(fields.get("invoice_date")),
        due_date=parse_date_value(fields.get("due_date")),
        payment_terms=none_if_blank(fields.get("payment_terms")),
        notes=none_if_blank(fields.get("notes")),
        extraction_confidence=confidence,
        extraction_warnings=warnings,
    )
    return invoice, warnings


__all__ = ["heuristic_extract"]
