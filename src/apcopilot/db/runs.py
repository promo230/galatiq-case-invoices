from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from apcopilot.db.connection import get_connection

# Writers for the operational tables (invoice_runs, invoice_flags, trace_events,
# llm_calls). Kept separate from tools/ledger.py, which is the LLM-facing /
# business-query surface (payments, rejections, vendor history) that agents
# call as tools. These are plumbing the graph and llm client use directly.


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_run(
    *,
    run_id: str,
    document_path: str,
    source_format: str,
    batch_id: str | None = None,
    db_path: Path | None = None,
) -> None:
    now = _now()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO invoice_runs (run_id, batch_id, document_path, source_format, "
            "status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (run_id, batch_id, document_path, source_format, "running", now, now),
        )


def update_run(*, run_id: str, db_path: Path | None = None, **fields: Any) -> None:
    """Patch arbitrary columns on invoice_runs. Decimal values are stringified;
    dict/list values are JSON-encoded (for extraction_json/decision_json)."""
    if not fields:
        return
    clean: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, Decimal):
            clean[key] = str(value)
        elif isinstance(value, dict | list):
            clean[key] = json.dumps(value, default=str)
        else:
            clean[key] = value
    clean["updated_at"] = _now()
    set_clause = ",".join(f"{k}=?" for k in clean)
    with get_connection(db_path) as conn:
        conn.execute(
            f"UPDATE invoice_runs SET {set_clause} WHERE run_id=?",
            (*clean.values(), run_id),
        )


_JSON_COLUMNS = ("extraction_json", "decision_json")


def _parse_json_columns(row: dict) -> dict:
    """`extraction_json`/`decision_json` are stored as TEXT (see update_run).
    Every consumer of a run row wants the parsed object, not a string a client
    would have to JSON.parse a second time, so parsing happens once here."""
    for key in _JSON_COLUMNS:
        value = row.get(key)
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                row[key] = json.loads(value)
    return row


def get_run(run_id: str, db_path: Path | None = None) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM invoice_runs WHERE run_id=?", (run_id,)).fetchone()
    return _parse_json_columns(dict(row)) if row else None


def list_runs(
    *, status: str | None = None, limit: int = 200, db_path: Path | None = None
) -> list[dict]:
    # flags_count is a correlated subquery rather than a join so a run with zero
    # flags still yields exactly one row (a join would need an extra GROUP BY to
    # avoid that, this is simpler at this table size). The UI's run-queue table
    # reads this key directly -- it never fetches per-row flag detail just to
    # show a count.
    query = "SELECT *, (SELECT COUNT(*) FROM invoice_flags f WHERE f.run_id = invoice_runs.run_id) AS flags_count FROM invoice_runs"
    params: tuple = ()
    if status:
        query += " WHERE status=?"
        params = (status,)
    query += " ORDER BY created_at DESC LIMIT ?"
    with get_connection(db_path) as conn:
        rows = conn.execute(query, (*params, limit)).fetchall()
    return [_parse_json_columns(dict(r)) for r in rows]


def insert_flags(run_id: str, flags: list[Any], db_path: Path | None = None) -> None:
    """`flags` is a list of ValidationFlag-like objects (rule_id, code, severity,
    message, sku, evidence)."""
    if not flags:
        return
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT INTO invoice_flags (run_id, rule_id, code, severity, message, sku, "
            "evidence_json) VALUES (?,?,?,?,?,?,?)",
            [
                (
                    run_id,
                    f.rule_id,
                    f.code,
                    str(f.severity),
                    f.message,
                    f.sku,
                    json.dumps(f.evidence, default=str),
                )
                for f in flags
            ],
        )


def get_flags(run_id: str, db_path: Path | None = None) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM invoice_flags WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["evidence"] = json.loads(d.pop("evidence_json") or "{}")
        out.append(d)
    return out


def append_trace(
    *,
    run_id: str,
    node: str,
    status: str,
    summary: str | None = None,
    duration_ms: int | None = None,
    detail: dict | None = None,
    db_path: Path | None = None,
) -> None:
    with get_connection(db_path) as conn:
        seq_row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM trace_events WHERE run_id=?", (run_id,)
        ).fetchone()
        seq = seq_row["n"]
        conn.execute(
            "INSERT INTO trace_events (run_id, seq, node, status, summary, duration_ms, "
            "detail_json, started_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id,
                seq,
                node,
                status,
                summary,
                duration_ms,
                json.dumps(detail, default=str) if detail else None,
                _now(),
            ),
        )


def get_trace(run_id: str, db_path: Path | None = None) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM trace_events WHERE run_id=? ORDER BY seq", (run_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["detail"] = json.loads(d.pop("detail_json")) if d.get("detail_json") else None
        out.append(d)
    return out


def log_llm_call(
    *,
    run_id: str | None,
    node: str,
    model: str,
    prompt_name: str | None = None,
    prompt_hash: str | None = None,
    attempt: int = 1,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cost_usd: Decimal = Decimal(0),
    latency_ms: int | None = None,
    stop_reason: str | None = None,
    request_preview: str | None = None,
    response_preview: str | None = None,
    db_path: Path | None = None,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO llm_calls (run_id, node, model, prompt_name, prompt_hash, attempt, "
            "input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, cost_usd, "
            "latency_ms, stop_reason, request_preview, response_preview, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                node,
                model,
                prompt_name,
                prompt_hash,
                attempt,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                str(cost_usd),
                latency_ms,
                stop_reason,
                request_preview,
                response_preview,
                _now(),
            ),
        )


def llm_cost_summary(run_id: str | None = None, db_path: Path | None = None) -> dict:
    query = (
        "SELECT COUNT(*) AS n, COALESCE(SUM(input_tokens),0) AS input_tokens, "
        "COALESCE(SUM(output_tokens),0) AS output_tokens, "
        "COALESCE(SUM(CAST(cost_usd AS REAL)),0) AS cost_usd FROM llm_calls"
    )
    params: tuple = ()
    if run_id:
        query += " WHERE run_id=?"
        params = (run_id,)
    with get_connection(db_path) as conn:
        row = conn.execute(query, params).fetchone()
    return dict(row)


__all__ = [
    "append_trace",
    "create_run",
    "get_flags",
    "get_run",
    "get_trace",
    "insert_flags",
    "list_runs",
    "llm_cost_summary",
    "log_llm_call",
    "update_run",
]
