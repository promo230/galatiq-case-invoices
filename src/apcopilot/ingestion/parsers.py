from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as DefusedET

from apcopilot.models import ExtractedInvoice, LineItem, canonicalize_sku

from ._shared import none_if_blank, parse_date_value, parse_decimal, parse_int

# .json / .csv / .xml are well-formed enough that a deterministic structural parse beats
# regex/LLM guessing. Each parser returns (invoice, warnings); warnings are also copied
# onto invoice.extraction_warnings by the caller so both are available independently.


def _warn_missing_critical_fields(invoice: ExtractedInvoice, warnings: list[str]) -> None:
    if invoice.vendor_name is None:
        warnings.append("vendor name missing or empty")
    if invoice.invoice_number is None:
        warnings.append("invoice number missing or empty")
    if invoice.total is None:
        warnings.append("total missing or empty")


def _line_item_description(raw_item: dict[str, Any]) -> str:
    return (
        none_if_blank(raw_item.get("item"))
        or none_if_blank(raw_item.get("description"))
        or none_if_blank(raw_item.get("name"))
        or ""
    )


def parse_json_invoice(text: str) -> tuple[ExtractedInvoice, list[str]]:
    warnings: list[str] = []
    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        warnings.append(f"could not parse JSON: {exc}")
        data = {}

    vendor_raw = data.get("vendor")
    if isinstance(vendor_raw, dict):
        vendor_name = none_if_blank(vendor_raw.get("name"))
    elif isinstance(vendor_raw, str):
        vendor_name = none_if_blank(vendor_raw)
    else:
        vendor_name = None

    line_items: list[LineItem] = []
    for raw_item in data.get("line_items") or []:
        if not isinstance(raw_item, dict):
            continue
        desc = _line_item_description(raw_item)
        if not desc:
            warnings.append("a line item is missing a description")
        line_total = parse_decimal(raw_item.get("amount"))
        if line_total is None:
            line_total = parse_decimal(raw_item.get("line_total"))
        line_items.append(
            LineItem(
                description=desc,
                sku=canonicalize_sku(desc),
                quantity=parse_int(raw_item.get("quantity")) or 0,
                unit_price=parse_decimal(raw_item.get("unit_price")),
                line_total=line_total,
            )
        )

    invoice = ExtractedInvoice(
        invoice_number=none_if_blank(data.get("invoice_number")),
        revision=none_if_blank(data.get("revision")),
        vendor_name=vendor_name,
        currency=none_if_blank(data.get("currency")) or "USD",
        line_items=line_items,
        subtotal=parse_decimal(data.get("subtotal")),
        tax=parse_decimal(data.get("tax_amount")),
        total=parse_decimal(data.get("total")),
        invoice_date=parse_date_value(data.get("date")),
        due_date=parse_date_value(data.get("due_date")),
        payment_terms=none_if_blank(data.get("payment_terms")),
        notes=none_if_blank(data.get("notes")),
        extraction_confidence=1.0,
    )
    _warn_missing_critical_fields(invoice, warnings)
    invoice.extraction_warnings = warnings
    return invoice, warnings


# --- CSV -------------------------------------------------------------------------

_CSV_TABULAR_SYNONYMS: dict[str, set[str]] = {
    "invoice_number": {"invoice number", "invoice_number", "invoice #", "invoice"},
    "vendor": {"vendor", "vendor name"},
    "date": {"date", "invoice date"},
    "due_date": {"due date", "due_date"},
    "item": {"item", "description"},
    "quantity": {"qty", "quantity"},
    "unit_price": {"unit price", "unit_price", "price", "rate"},
    "line_total": {"line total", "amount"},
    "payment_terms": {"payment terms", "payment_terms", "terms"},
    "currency": {"currency"},
}


def _first_amount(cells: list[str]) -> Any:
    for cell in cells:
        amount = parse_decimal(cell)
        if amount is not None:
            return amount
    return None


def _parse_csv_tabular(header: list[str], rows: list[list[str]]) -> ExtractedInvoice:
    colidx: dict[str, int] = {}
    for canon, synonyms in _CSV_TABULAR_SYNONYMS.items():
        for i, name in enumerate(header):
            if name in synonyms:
                colidx[canon] = i
                break

    def cell(row: list[str], canon: str) -> str:
        idx = colidx.get(canon)
        if idx is None or idx >= len(row):
            return ""
        return row[idx].strip()

    header_vals: dict[str, str] = {}
    line_items: list[LineItem] = []
    subtotal = tax = total = None

    for row in rows:
        item_val = cell(row, "item")
        if item_val:
            for canon in ("invoice_number", "vendor", "date", "due_date", "payment_terms", "currency"):
                if canon not in header_vals:
                    value = cell(row, canon)
                    if value:
                        header_vals[canon] = value
            line_items.append(
                LineItem(
                    description=item_val,
                    sku=canonicalize_sku(item_val),
                    quantity=parse_int(cell(row, "quantity")) or 0,
                    unit_price=parse_decimal(cell(row, "unit_price")),
                    line_total=parse_decimal(cell(row, "line_total")),
                )
            )
            continue

        # A row with no item is a summary row (subtotal/tax/total) whose label and
        # amount may land in arbitrary columns depending on how ragged the sheet is;
        # scan for a recognizable label and take the first amount that follows it.
        for i, raw in enumerate(row):
            label = raw.strip().lower()
            if not label:
                continue
            amount = _first_amount(row[i + 1 :])
            if amount is None:
                continue
            if "subtotal" in label:
                subtotal = amount
            elif "tax" in label:
                tax = amount
            elif "total" in label:
                total = amount

    return ExtractedInvoice(
        invoice_number=none_if_blank(header_vals.get("invoice_number")),
        vendor_name=none_if_blank(header_vals.get("vendor")),
        currency=none_if_blank(header_vals.get("currency")) or "USD",
        line_items=line_items,
        subtotal=subtotal,
        tax=tax,
        total=total,
        invoice_date=parse_date_value(header_vals.get("date")),
        due_date=parse_date_value(header_vals.get("due_date")),
        payment_terms=none_if_blank(header_vals.get("payment_terms")),
        extraction_confidence=1.0,
    )


_CSV_KV_ITEM_FIELDS = {"item"}
_CSV_KV_QTY_FIELDS = {"quantity", "qty"}
_CSV_KV_PRICE_FIELDS = {"unit_price", "price"}
_CSV_KV_LINE_TOTAL_FIELDS = {"line_total", "amount"}
_CSV_KV_HEADER_FIELDS = {
    "invoice_number": "invoice_number",
    "vendor": "vendor_name",
    "date": "invoice_date",
    "due_date": "due_date",
    "subtotal": "subtotal",
    "tax": "tax",
    "total": "total",
    "payment_terms": "payment_terms",
    "terms": "payment_terms",
    "currency": "currency",
    "revision": "revision",
    "notes": "notes",
}


def _parse_csv_keyvalue(rows: list[list[str]]) -> ExtractedInvoice:
    fields: dict[str, str] = {}
    line_items: list[LineItem] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            desc = current.get("desc") or ""
            line_items.append(
                LineItem(
                    description=desc,
                    sku=canonicalize_sku(desc),
                    quantity=current.get("qty") or 0,
                    unit_price=current.get("price"),
                    line_total=current.get("line_total"),
                )
            )
        current = None

    for row in rows:
        if len(row) < 2:
            continue
        key = row[0].strip().lower()
        value = row[1].strip()
        if key in _CSV_KV_ITEM_FIELDS:
            flush()
            current = {"desc": value}
        elif key in _CSV_KV_QTY_FIELDS and current is not None:
            current["qty"] = parse_int(value) or 0
        elif key in _CSV_KV_PRICE_FIELDS and current is not None:
            current["price"] = parse_decimal(value)
        elif key in _CSV_KV_LINE_TOTAL_FIELDS and current is not None:
            current["line_total"] = parse_decimal(value)
        elif key in _CSV_KV_HEADER_FIELDS:
            fields[_CSV_KV_HEADER_FIELDS[key]] = value
    flush()

    return ExtractedInvoice(
        invoice_number=none_if_blank(fields.get("invoice_number")),
        revision=none_if_blank(fields.get("revision")),
        vendor_name=none_if_blank(fields.get("vendor_name")),
        currency=none_if_blank(fields.get("currency")) or "USD",
        line_items=line_items,
        subtotal=parse_decimal(fields.get("subtotal")),
        tax=parse_decimal(fields.get("tax")),
        total=parse_decimal(fields.get("total")),
        invoice_date=parse_date_value(fields.get("invoice_date")),
        due_date=parse_date_value(fields.get("due_date")),
        payment_terms=none_if_blank(fields.get("payment_terms")),
        notes=none_if_blank(fields.get("notes")),
        extraction_confidence=1.0,
    )


def parse_csv_invoice(text: str) -> tuple[ExtractedInvoice, list[str]]:
    warnings: list[str] = []
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if not rows:
        warnings.append("CSV file is empty")
        invoice = ExtractedInvoice(extraction_warnings=warnings)
        return invoice, warnings

    header = [cell.strip().lower() for cell in rows[0]]
    if header[:2] == ["field", "value"]:
        invoice = _parse_csv_keyvalue(rows[1:])
    else:
        invoice = _parse_csv_tabular(header, rows[1:])

    _warn_missing_critical_fields(invoice, warnings)
    invoice.extraction_warnings = warnings
    return invoice, warnings


# --- XML ---------------------------------------------------------------------------


def _xml_text(element: Any, *tags: str) -> str | None:
    for tag in tags:
        child = element.find(tag)
        if child is not None and child.text and child.text.strip():
            return child.text.strip()
    return None


def parse_xml_invoice(text: str) -> tuple[ExtractedInvoice, list[str]]:
    warnings: list[str] = []
    try:
        root = DefusedET.fromstring(text)
    except Exception as exc:  # malformed XML: never raise out of ingestion
        warnings.append(f"could not parse XML: {exc}")
        invoice = ExtractedInvoice(extraction_warnings=warnings)
        return invoice, warnings

    header_el = root.find("header")
    if header_el is None:
        header_el = root
    totals_el = root.find("totals")
    if totals_el is None:
        totals_el = root
    items_container = root.find("line_items")
    if items_container is None:
        items_container = root.find("items")
    if items_container is None:
        items_container = root

    invoice_number = _xml_text(header_el, "invoice_number", "invoice_no", "invoiceNumber", "number")

    vendor_el = header_el.find("vendor")
    vendor_name = None
    if vendor_el is not None:
        if vendor_el.text and vendor_el.text.strip():
            vendor_name = vendor_el.text.strip()
        else:
            vendor_name = _xml_text(vendor_el, "name")

    currency = _xml_text(header_el, "currency") or "USD"

    line_items: list[LineItem] = []
    for item_el in items_container.findall("item"):
        desc = _xml_text(item_el, "name", "description", "item") or ""
        if not desc:
            warnings.append("a line item is missing a description")
        line_items.append(
            LineItem(
                description=desc,
                sku=canonicalize_sku(desc),
                quantity=parse_int(_xml_text(item_el, "quantity", "qty")) or 0,
                unit_price=parse_decimal(_xml_text(item_el, "unit_price", "price")),
                line_total=parse_decimal(_xml_text(item_el, "amount", "line_total")),
            )
        )

    invoice = ExtractedInvoice(
        invoice_number=none_if_blank(invoice_number),
        vendor_name=none_if_blank(vendor_name),
        currency=currency,
        line_items=line_items,
        subtotal=parse_decimal(_xml_text(totals_el, "subtotal")),
        tax=parse_decimal(_xml_text(totals_el, "tax_amount", "tax")),
        total=parse_decimal(_xml_text(totals_el, "total")),
        invoice_date=parse_date_value(_xml_text(header_el, "date", "invoice_date")),
        due_date=parse_date_value(_xml_text(header_el, "due_date", "due")),
        payment_terms=none_if_blank(_xml_text(root, "payment_terms", "terms")),
        extraction_confidence=1.0,
    )
    _warn_missing_critical_fields(invoice, warnings)
    invoice.extraction_warnings = warnings
    return invoice, warnings


# --- PDF ----------------------------------------------------------------------------


def extract_pdf_text(path: Path) -> str:
    """Raw text via pdfplumber, then handed to the same messy-document path as .txt."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


__all__ = [
    "extract_pdf_text",
    "parse_csv_invoice",
    "parse_json_invoice",
    "parse_xml_invoice",
]
