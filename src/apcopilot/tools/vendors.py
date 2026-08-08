from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel
from rapidfuzz import fuzz

from apcopilot.db.connection import get_connection, to_decimal
from apcopilot.tools.policy import tolerance


class VendorRecord(BaseModel):
    name: str
    aka: list[str] = []
    status: str
    risk_tier: str
    auto_approve_limit: Decimal
    currency: str
    first_seen: date | None = None
    name_changed_at: date | None = None


class VendorMatch(BaseModel):
    query: str
    matched: VendorRecord | None
    score: int
    matched_via_aka: bool = False

    @property
    def is_known(self) -> bool:
        return self.matched is not None


def _row_to_record(row) -> VendorRecord:
    return VendorRecord(
        name=row["name"],
        aka=[a.strip() for a in (row["aka"] or "").split("|") if a.strip()],
        status=row["status"],
        risk_tier=row["risk_tier"],
        auto_approve_limit=to_decimal(row["auto_approve_limit"]) or Decimal(0),
        currency=row["currency"],
        first_seen=date.fromisoformat(row["first_seen"]) if row["first_seen"] else None,
        name_changed_at=(
            date.fromisoformat(row["name_changed_at"]) if row["name_changed_at"] else None
        ),
    )


def all_vendors(db_path: Path | None = None) -> list[VendorRecord]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM vendors ORDER BY name").fetchall()
    return [_row_to_record(r) for r in rows]


def match_vendor(name: str | None, db_path: Path | None = None) -> VendorMatch:
    """Fuzzy-match against the vendor master, considering former names.

    Tolerates the typos the corpus contains ('QuickShip Distributers' for
    'QuickShip Distributors') without silently accepting a vendor that simply
    is not on file.
    """
    query = (name or "").strip()
    if not query:
        return VendorMatch(query="", matched=None, score=0)

    floor = int(tolerance("vendor_match_min"))
    best: VendorRecord | None = None
    best_score = 0
    best_via_aka = False

    for vendor in all_vendors(db_path):
        score = int(fuzz.token_set_ratio(query.lower(), vendor.name.lower()))
        via_aka = False
        for alias in vendor.aka:
            alias_score = int(fuzz.token_set_ratio(query.lower(), alias.lower()))
            if alias_score > score:
                score, via_aka = alias_score, True
        if score > best_score:
            best, best_score, best_via_aka = vendor, score, via_aka

    if best_score < floor:
        return VendorMatch(query=query, matched=None, score=best_score)
    return VendorMatch(query=query, matched=best, score=best_score, matched_via_aka=best_via_aka)


def get_vendor_profile(name: str, db_path: Path | None = None) -> dict:
    """LLM-facing vendor lookup, including spend history."""
    from apcopilot.tools.ledger import vendor_history

    match = match_vendor(name, db_path)
    if match.matched is None:
        return {
            "query": name,
            "found": False,
            "best_score": match.score,
            "note": "No vendor on file above the match threshold. Treat as unknown vendor.",
        }
    history = vendor_history(match.matched.name, db_path)
    return {
        "query": name,
        "found": True,
        "matched_name": match.matched.name,
        "match_score": match.score,
        "matched_via_former_name": match.matched_via_aka,
        "status": match.matched.status,
        "risk_tier": match.matched.risk_tier,
        "auto_approve_limit": str(match.matched.auto_approve_limit),
        "currency": match.matched.currency,
        "first_seen": match.matched.first_seen.isoformat() if match.matched.first_seen else None,
        "name_changed_at": (
            match.matched.name_changed_at.isoformat() if match.matched.name_changed_at else None
        ),
        **history,
    }
