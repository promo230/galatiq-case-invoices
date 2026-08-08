from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from apcopilot.db.connection import get_connection, to_decimal


class FxRate(BaseModel):
    base: str
    quote: str
    rate: Decimal
    as_of: date


def get_rate(base: str, quote: str = "USD", db_path: Path | None = None) -> FxRate | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT base, quote, rate, as_of FROM fx_rates WHERE base = ? AND quote = ?",
            (base.upper(), quote.upper()),
        ).fetchone()
    if row is None:
        return None
    return FxRate(
        base=row["base"],
        quote=row["quote"],
        rate=to_decimal(row["rate"]) or Decimal(1),
        as_of=date.fromisoformat(row["as_of"]),
    )


def to_usd(
    amount: Decimal | None, currency: str, db_path: Path | None = None
) -> tuple[Decimal | None, FxRate | None]:
    """Convert to USD using the dated reference rate on file.

    Offline by design: the rate is seed data with an explicit as_of that the UI
    surfaces as an assumption rather than presenting as live market data.
    """
    if amount is None:
        return None, None
    if currency.upper() == "USD":
        return amount, None
    rate = get_rate(currency, "USD", db_path)
    if rate is None:
        return None, None
    return (amount * rate.rate).quantize(Decimal("0.01")), rate
