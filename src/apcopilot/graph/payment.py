from __future__ import annotations

from decimal import Decimal

# Mock "banking API" per the case brief. Deliberately just a print + a
# hardcoded success dict — this is the whole simulation, no HTTP client, no
# retries, no failure modes. Idempotency lives in tools.ledger.record_payment
# (unique key on invoice_number|amount|run_id), not here.


def mock_payment(vendor: str, amount: Decimal) -> dict:
    print(f"Paid {amount} to {vendor}")
    return {"status": "success"}


__all__ = ["mock_payment"]
