from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from apcopilot.config import get_settings
from apcopilot.logging import get_logger
from apcopilot.models import ExtractedInvoice, ExtractionResult

from .heuristic import heuristic_extract
from .llm_extract import LLMUnavailableError, llm_extract
from .parsers import extract_pdf_text, parse_csv_invoice, parse_json_invoice, parse_xml_invoice

logger = get_logger(__name__)

# .json/.csv/.xml are well-formed enough that a deterministic structural parse strictly
# beats LLM/regex guessing. .txt/.pdf are the messy-document path: typos, garbled OCR,
# email wrappers, fraud language -- handled by ingestion.llm_extract with a heuristic
# fallback (ingestion.heuristic) for llm_mode="off" or LLM unavailability.
_DETERMINISTIC_PARSERS = {
    ".json": parse_json_invoice,
    ".csv": parse_csv_invoice,
    ".xml": parse_xml_invoice,
}
_MESSY_DOCUMENT_SUFFIXES = {".txt", ".pdf"}


def _hash_default(value: Any) -> str:
    if isinstance(value, Decimal):
        # Canonicalize the exponent so numerically equal amounts hash equally.
        # A JSON `250.00` arrives as Decimal("250.0") (via float) while the same
        # figure in XML/CSV text arrives as Decimal("250.00"); without this, the
        # same invoice in two formats would false-positive as a REVISION_CONFLICT.
        return format(value.normalize(), "f")
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unhashable content value of type {type(value).__name__}")


def _content_hash(invoice: ExtractedInvoice) -> str:
    payload = json.dumps(invoice.model_dump(mode="python"), sort_keys=True, default=_hash_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pretty_render(document_path: Path, raw_text: str) -> str:
    if document_path.suffix.lower() != ".json":
        return raw_text
    try:
        return json.dumps(json.loads(raw_text), indent=2, sort_keys=True)
    except json.JSONDecodeError:
        return raw_text


async def ingest_document(document_path: Path, *, run_id: str | None = None) -> ExtractionResult:
    """Parse a single invoice document into an ExtractionResult.

    Never raises for a malformed or fraudulent document -- bad data becomes None
    fields plus extraction_warnings, not an exception. Only raises for genuinely
    unrecoverable I/O errors (the file doesn't exist, or has an unsupported extension).
    """
    if not document_path.exists():
        raise FileNotFoundError(f"invoice document not found: {document_path}")

    suffix = document_path.suffix.lower()
    settings = get_settings()

    if suffix == ".pdf":
        raw_text = extract_pdf_text(document_path)
    elif suffix in _DETERMINISTIC_PARSERS or suffix in _MESSY_DOCUMENT_SUFFIXES:
        raw_text = document_path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"unsupported invoice document extension: {suffix!r}")

    if suffix in _DETERMINISTIC_PARSERS:
        invoice, _warnings = _DETERMINISTIC_PARSERS[suffix](raw_text)
        return ExtractionResult(
            invoice=invoice,
            method="deterministic",
            attempts=1,
            source_format=suffix.lstrip("."),
            document_path=str(document_path),
            content_hash=_content_hash(invoice),
            raw_text=_pretty_render(document_path, raw_text),
        )

    # Messy-document path: .txt / .pdf.
    invoice: ExtractedInvoice | None = None
    method = "heuristic"
    attempts = 1
    input_tokens = output_tokens = 0
    cost_usd = Decimal(0)

    if settings.llm_mode != "off":
        try:
            invoice, attempts, input_tokens, output_tokens, cost_usd = await llm_extract(
                raw_text, run_id=run_id
            )
            method = "llm"
        except LLMUnavailableError:
            logger.info(
                "ingest_llm_unavailable_fallback_heuristic", document=str(document_path)
            )
        except Exception:
            # Ingestion must never hard-fail on a document: any other extraction
            # failure (persistent validation errors, network errors, ...) also falls
            # back to the heuristic parser rather than propagating.
            logger.warning(
                "ingest_llm_extract_failed_fallback_heuristic",
                document=str(document_path),
                exc_info=True,
            )

    if invoice is None:
        invoice, _warnings = heuristic_extract(raw_text)
        method = "heuristic"
        attempts = 1
        input_tokens = output_tokens = 0
        cost_usd = Decimal(0)

    return ExtractionResult(
        invoice=invoice,
        method=method,
        attempts=attempts,
        source_format=suffix.lstrip("."),
        document_path=str(document_path),
        content_hash=_content_hash(invoice),
        raw_text=raw_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


__all__ = ["ingest_document"]
