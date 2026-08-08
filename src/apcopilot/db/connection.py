from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from apcopilot.config import get_settings

# Money is Decimal everywhere in Python and TEXT in SQLite. Binding a Decimal to
# a float would reintroduce exactly the rounding errors the math rules exist to
# catch.
sqlite3.register_adapter(Decimal, str)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_connection(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = _connect(path or get_settings().db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def to_decimal(value: object) -> Decimal | None:
    if value is None or value == "":
        return None
    return Decimal(str(value))
