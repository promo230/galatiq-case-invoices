from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from dateutil import parser as dateutil_parser

_MONEY_RE = re.compile(r"-?[\d,]+\.?\d*")


def parse_decimal(value: Any) -> Decimal | None:
    """Best-effort money parser: tolerates '$', ',', surrounding whitespace, and raw
    numeric JSON/CSV values alike. Returns None (never raises) when nothing usable is
    found -- callers turn that into an extraction_warnings entry, not an exception."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int | float):
        return Decimal(str(value))
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("$", "").replace(",", "").strip()
    match = _MONEY_RE.search(text)
    if not match or match.group(0) in ("", "-"):
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def parse_int(value: Any) -> int | None:
    decimal_value = parse_decimal(value)
    if decimal_value is None:
        return None
    try:
        return int(decimal_value)
    except (ValueError, OverflowError):
        return None


def parse_date_value(value: Any) -> date | None:
    """Tolerant date parser: accepts already-parsed dates, ISO strings, and looser
    human forms ('January 27, 2026', '26-Jan-2O26'). Returns None on anything it can't
    make sense of rather than raising."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return dateutil_parser.parse(text, fuzzy=True).date()
    except (ValueError, OverflowError, TypeError, dateutil_parser.ParserError):
        return None


def none_if_blank(value: Any) -> str | None:
    """Missing/empty string fields should stay None rather than being coerced into ''
    (per spec: e.g. an empty vendor name is a data-quality problem, not a real value)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
