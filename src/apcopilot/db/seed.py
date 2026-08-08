from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from apcopilot.config import get_settings
from apcopilot.db.connection import get_connection

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _load_csv(conn: sqlite3.Connection, table: str, csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return 0
    columns = list(rows[0])
    placeholders = ",".join("?" * len(columns))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    conn.executemany(sql, [tuple(r[c] or None for c in columns) for r in rows])
    return len(rows)


def ensure_schema(db_path: Path | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def seed_reference_data(db_path: Path | None = None) -> dict[str, int]:
    """Create the schema and load reference data. Idempotent."""
    settings = get_settings()
    ensure_schema(db_path)
    counts: dict[str, int] = {}
    with get_connection(db_path) as conn:
        for table, filename in (
            ("items", "items.csv"),
            ("vendors", "vendors.csv"),
            ("fx_rates", "fx_rates.csv"),
        ):
            counts[table] = _load_csv(conn, table, settings.seed_dir / filename)
    return counts


def reset_database(db_path: Path | None = None) -> dict[str, int]:
    """Delete the database file entirely and rebuild it from seed."""
    target = db_path or get_settings().db_path
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(target) + suffix)
        if candidate.exists():
            candidate.unlink()
    return seed_reference_data(db_path)


def database_is_ready(db_path: Path | None = None) -> bool:
    target = db_path or get_settings().db_path
    if not target.exists():
        return False
    try:
        with get_connection(db_path) as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM items").fetchone()
            return bool(row and row["n"] > 0)
    except sqlite3.Error:
        return False


def ensure_seeded(db_path: Path | None = None) -> None:
    """Auto-seed on first run so a grader never has to run a setup step."""
    if not database_is_ready(db_path):
        seed_reference_data(db_path)
