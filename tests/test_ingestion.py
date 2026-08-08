"""Ingestion tests: deterministic .json/.csv/.xml parsers, the zero-API heuristic
fallback for .txt, and SKU canonicalization.

Expected values are read out of the real files in `data/invoices/` rather than
duplicated here, so a change to a fixture invoice cannot silently drift away from
what the test claims it contains.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest

from apcopilot.ingestion import ingest_document
from apcopilot.ingestion.heuristic import heuristic_extract
from apcopilot.ingestion.parsers import (
    parse_csv_invoice,
    parse_json_invoice,
    parse_xml_invoice,
)
from apcopilot.models import canonicalize_sku


def _read(invoice_dir: Path, name: str) -> str:
    return (invoice_dir / name).read_text(encoding="utf-8")


def _ingest(path: Path):
    """`ingest_document` is async; the suite has no async plugin, so every call
    goes through a fresh event loop here."""
    return asyncio.run(ingest_document(path))


# --- canonicalize_sku ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("WidgetA", "WIDGETA"),
        ("widget a", "WIDGETA"),
        ("Widget-A", "WIDGETA"),
        ("  widget_a  ", "WIDGETA"),
        ("GadgetX", "GADGETX"),
        ("Super Gizmo 2000", "SUPERGIZMO2000"),
        ("", ""),
    ],
)
def test_canonicalize_sku_normalizes_to_one_key(raw, expected):
    assert canonicalize_sku(raw) == expected


def test_parsers_canonicalize_every_line_item_sku(invoice_dir):
    """Ingestion and validation must agree on the lookup key, so the parsers
    populate `sku` from the description via the same canonicalizer."""
    invoice, _warnings = parse_json_invoice(_read(invoice_dir, "invoice_1016.json"))

    assert [item.sku for item in invoice.line_items] == ["WIDGETA", "WIDGETB", "WIDGETC"]
    for item in invoice.line_items:
        assert item.sku == canonicalize_sku(item.description)


# --- JSON ------------------------------------------------------------------------


def test_parse_json_invoice(invoice_dir):
    raw = _read(invoice_dir, "invoice_1004.json")
    source = json.loads(raw)
    invoice, warnings = parse_json_invoice(raw)

    assert warnings == []
    assert invoice.invoice_number == source["invoice_number"]
    assert invoice.vendor_name == source["vendor"]["name"]  # nested vendor object
    assert invoice.currency == source["currency"]
    assert invoice.total == Decimal(str(source["total"]))
    assert invoice.subtotal == Decimal(str(source["subtotal"]))
    assert invoice.tax == Decimal(str(source["tax_amount"]))  # tax_amount -> tax
    assert invoice.invoice_date.isoformat() == source["date"]
    assert invoice.due_date.isoformat() == source["due_date"]
    assert invoice.extraction_confidence == 1.0
    assert len(invoice.line_items) == len(source["line_items"])
    assert invoice.line_items[0].quantity == source["line_items"][0]["quantity"]


def test_parse_json_invoice_captures_revision_and_notes(invoice_dir):
    invoice, _warnings = parse_json_invoice(_read(invoice_dir, "invoice_1004_revised.json"))

    assert invoice.revision == "R1"
    assert invoice.notes is not None
    assert len(invoice.line_items) == 3


def test_parse_json_invoice_warns_on_blank_vendor_instead_of_raising(invoice_dir):
    """invoice_1009 has `"name": ""` -- a blank string must become None + a warning,
    not an empty vendor that silently fuzzy-matches nothing."""
    invoice, warnings = parse_json_invoice(_read(invoice_dir, "invoice_1009.json"))

    assert invoice.vendor_name is None
    assert "vendor name missing or empty" in warnings
    assert invoice.extraction_warnings == warnings
    assert invoice.line_items[0].quantity == -5  # negative data survives to validation
    assert invoice.total == Decimal("-250.00")


def test_parse_json_invoice_tolerates_malformed_json():
    invoice, warnings = parse_json_invoice("{not valid json")

    assert invoice.vendor_name is None
    assert invoice.total is None
    assert any("could not parse JSON" in w for w in warnings)


# --- CSV -------------------------------------------------------------------------


def test_parse_csv_invoice_key_value_layout(invoice_dir):
    """invoice_1006.csv is a field/value sheet where each item is a run of rows."""
    invoice, warnings = parse_csv_invoice(_read(invoice_dir, "invoice_1006.csv"))

    assert warnings == []
    assert invoice.invoice_number == "INV-1006"
    assert invoice.vendor_name == "Acme Industrial Supplies"
    assert invoice.total == Decimal("2750.00")
    assert [(i.description, i.quantity, i.unit_price) for i in invoice.line_items] == [
        ("WidgetA", 5, Decimal("250.00")),
        ("WidgetB", 3, Decimal("500.00")),
    ]


def test_parse_csv_invoice_tabular_layout_with_ragged_summary_rows(invoice_dir):
    """invoice_1007.csv repeats the header fields on every item row and puts
    subtotal/tax/total in trailing rows whose label lands in an arbitrary column."""
    invoice, warnings = parse_csv_invoice(_read(invoice_dir, "invoice_1007.csv"))

    assert warnings == []
    assert invoice.invoice_number == "INV-1007"
    assert invoice.vendor_name == "MegaWidgets Corp"
    assert len(invoice.line_items) == 3
    assert invoice.subtotal == Decimal("14750.00")
    assert invoice.tax == Decimal("885.00")
    assert invoice.total == Decimal("15525.00")
    assert invoice.invoice_date.isoformat() == "2026-01-28"  # 01/28/2026


def test_parse_csv_invoice_handles_an_empty_file():
    invoice, warnings = parse_csv_invoice("")

    assert warnings == ["CSV file is empty"]
    assert invoice.line_items == []


# --- XML -------------------------------------------------------------------------


def test_parse_xml_invoice(invoice_dir):
    invoice, warnings = parse_xml_invoice(_read(invoice_dir, "invoice_1014.xml"))

    assert warnings == []
    assert invoice.invoice_number == "INV-1014"
    assert invoice.vendor_name == "TechParts International"
    assert invoice.currency == "EUR"
    assert invoice.subtotal == Decimal("3750.00")
    assert invoice.tax == Decimal("375.00")
    assert invoice.total == Decimal("4125.00")
    assert [(i.description, i.quantity) for i in invoice.line_items] == [
        ("WidgetA", 4),
        ("WidgetB", 6),
    ]


def test_parse_xml_invoice_tolerates_malformed_xml():
    invoice, warnings = parse_xml_invoice("<invoice><header>")

    assert invoice.invoice_number is None
    assert any("could not parse XML" in w for w in warnings)


# --- heuristic .txt fallback -------------------------------------------------------


def test_heuristic_extract_clean_text_invoice(invoice_dir):
    invoice, warnings = heuristic_extract(_read(invoice_dir, "invoice_1001.txt"))

    assert warnings == []
    assert invoice.invoice_number == "INV-1001"
    assert invoice.vendor_name == "Widgets Inc."
    assert invoice.total == Decimal("5000.00")
    assert invoice.due_date.isoformat() == "2026-02-01"
    assert [(i.description, i.quantity, i.unit_price) for i in invoice.line_items] == [
        ("WidgetA", 10, Decimal("250.00")),
        ("WidgetB", 5, Decimal("500.00")),
    ]
    # Regex extraction is never presented as confidently as a structural parse.
    assert invoice.extraction_confidence < 1.0


def test_heuristic_extract_handles_abbreviated_labels(invoice_dir):
    """invoice_1002.txt uses 'Vndr:', 'Inv #:', 'Dt:', 'Due Dt:', 'Amt:'."""
    invoice, warnings = heuristic_extract(_read(invoice_dir, "invoice_1002.txt"))

    assert invoice.vendor_name == "Gadgets Co."
    assert invoice.invoice_number == "1002"
    assert invoice.total == Decimal("15000.00")
    assert invoice.invoice_date.isoformat() == "2026-01-30"  # 'Jan 30 2026'
    assert invoice.due_date.isoformat() == "2026-01-30"
    assert [(i.description, i.quantity) for i in invoice.line_items] == [("GadgetX", 20)]
    # Missing fields are reported, not invented.
    assert invoice.subtotal is None
    assert "heuristic parser could not find a subtotal" in warnings


def test_heuristic_extract_keeps_adversarial_notes_as_data(invoice_dir):
    """The urgency/wire-transfer text in invoice_1003 must reach the fraud rule as
    content, never be acted on."""
    invoice, _warnings = heuristic_extract(_read(invoice_dir, "invoice_1003.txt"))

    assert invoice.vendor_name == "Fraudster LLC"
    assert "URGENT" in invoice.notes
    assert "Wire transfer" in invoice.notes
    assert invoice.due_date is None  # 'Due Date: yesterday' is not a date
    assert invoice.total == Decimal("100000.00")


def test_heuristic_extract_prefers_an_explicit_vendor_label_over_an_email_header(
    invoice_dir,
):
    """invoice_1008.txt is an email wrapper: 'From: billing@noproduct.biz' appears
    above the real 'Vendor: NoProd Industries' line. The mail header must lose."""
    invoice, _warnings = heuristic_extract(_read(invoice_dir, "invoice_1008.txt"))

    assert invoice.vendor_name == "NoProd Industries"
    assert invoice.invoice_number == "INV-1008"
    assert [(i.description, i.quantity) for i in invoice.line_items] == [
        ("SuperGizmo", 12),
        ("MegaSprocket", 6),
    ]


def test_heuristic_confidence_drops_as_critical_fields_go_missing():
    complete, _ = heuristic_extract(
        "Vendor: Widgets Inc.\nInvoice Number: INV-1\nTotal: $10.00\n"
    )
    sparse, _ = heuristic_extract("Total: $10.00\n")

    assert sparse.extraction_confidence < complete.extraction_confidence
    assert sparse.vendor_name is None


# --- ingest_document dispatch --------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "source_format"),
    [
        ("invoice_1004.json", "json"),
        ("invoice_1006.csv", "csv"),
        ("invoice_1014.xml", "xml"),
    ],
)
def test_structured_formats_use_the_deterministic_parser(
    invoice_dir, filename, source_format
):
    result = _ingest(invoice_dir / filename)

    assert result.method == "deterministic"
    assert result.source_format == source_format
    assert result.attempts == 1
    assert result.cost_usd == Decimal(0)
    assert result.input_tokens == 0 and result.output_tokens == 0


def test_text_invoice_falls_back_to_the_heuristic_when_the_llm_is_off(invoice_dir):
    result = _ingest(invoice_dir / "invoice_1001.txt")

    assert result.method == "heuristic"
    assert result.source_format == "txt"
    assert result.cost_usd == Decimal(0)
    assert result.invoice.vendor_name == "Widgets Inc."


def test_content_hash_is_stable_for_the_same_document(invoice_dir):
    first = _ingest(invoice_dir / "invoice_1004.json")
    second = _ingest(invoice_dir / "invoice_1004.json")

    assert first.content_hash == second.content_hash


def test_content_hash_differs_between_an_invoice_and_its_revision(invoice_dir):
    """This is the signal POL-DUP-01 keys on to detect a revision conflict."""
    original = _ingest(invoice_dir / "invoice_1004.json")
    revised = _ingest(invoice_dir / "invoice_1004_revised.json")

    assert original.invoice.invoice_number == revised.invoice.invoice_number
    assert original.content_hash != revised.content_hash


def test_missing_document_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _ingest(tmp_path / "nope.json")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "invoice.docx"
    path.write_text("whatever", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported invoice document extension"):
        _ingest(path)


def test_ingestion_never_raises_on_a_malformed_document(tmp_path):
    """A garbage document becomes None fields plus warnings, so a bad file in a
    batch degrades to a flagged run instead of killing the batch."""
    path = tmp_path / "invoice.json"
    path.write_text("{{{ not json at all", encoding="utf-8")

    result = _ingest(path)

    assert result.invoice.vendor_name is None
    assert result.invoice.total is None
    assert result.invoice.extraction_warnings
