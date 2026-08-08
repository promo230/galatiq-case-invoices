from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from apcopilot.db.connection import get_connection, to_decimal


class ItemRecord(BaseModel):
    sku: str
    name: str
    unit_price: Decimal | None
    stock: int
    category: str | None
    status: str

    @property
    def is_payable(self) -> bool:
        return self.status == "active"


def lookup_items(skus: list[str], db_path: Path | None = None) -> dict[str, ItemRecord | None]:
    """Look up canonicalized SKUs. Missing SKUs map to None so callers can tell
    'not in catalog' apart from 'in catalog with zero stock'."""
    result: dict[str, ItemRecord | None] = {sku: None for sku in skus}
    if not skus:
        return result
    placeholders = ",".join("?" * len(skus))
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT sku, name, unit_price, stock, category, status "
            f"FROM items WHERE sku IN ({placeholders})",
            skus,
        ).fetchall()
    for row in rows:
        result[row["sku"]] = ItemRecord(
            sku=row["sku"],
            name=row["name"],
            unit_price=to_decimal(row["unit_price"]),
            stock=int(row["stock"]),
            category=row["category"],
            status=row["status"],
        )
    return result


def get_inventory_status(skus: list[str], db_path: Path | None = None) -> list[dict]:
    """LLM-facing view of lookup_items."""
    found = lookup_items(skus, db_path)
    out = []
    for sku in skus:
        record = found[sku]
        if record is None:
            out.append({"sku": sku, "found": False})
        else:
            out.append(
                {
                    "sku": sku,
                    "found": True,
                    "name": record.name,
                    "stock": record.stock,
                    "unit_price": str(record.unit_price) if record.unit_price else None,
                    "category": record.category,
                    "status": record.status,
                }
            )
    return out


def all_items(db_path: Path | None = None) -> list[ItemRecord]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT sku, name, unit_price, stock, category, status FROM items ORDER BY sku"
        ).fetchall()
    return [
        ItemRecord(
            sku=r["sku"],
            name=r["name"],
            unit_price=to_decimal(r["unit_price"]),
            stock=int(r["stock"]),
            category=r["category"],
            status=r["status"],
        )
        for r in rows
    ]


def catalog_digest(db_path: Path | None = None) -> str:
    """Stable catalog rendering for the cached LLM system prefix."""
    return "\n".join(
        f"{i.sku} ({i.name}): stock={i.stock}, "
        f"unit_price={i.unit_price if i.unit_price is not None else 'n/a'}, status={i.status}"
        for i in all_items(db_path)
    )
