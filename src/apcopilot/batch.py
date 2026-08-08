from __future__ import annotations

import uuid
from pathlib import Path

from apcopilot.graph import run_invoice
from apcopilot.logging import get_logger

logger = get_logger(__name__)


def discover_invoices(directory: Path, pattern: str = "*") -> list[Path]:
    """Every file in `directory` matching `pattern`, sorted for determinism."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob(pattern) if p.is_file())


async def run_batch(
    paths: list[Path], *, batch_id: str | None = None, db_path: Path | None = None
) -> list[dict]:
    """Run every path through run_invoice under a shared batch_id, sequentially.

    Sequential (not gathered) so batch runs don't hammer the LLM API concurrently
    and so a crash on one invoice doesn't orphan the others mid-flight.
    """
    shared_batch_id = batch_id or f"batch_{uuid.uuid4().hex[:12]}"
    results: list[dict] = []
    for path in paths:
        try:
            result = await run_invoice(path, batch_id=shared_batch_id, db_path=db_path)
        except Exception as exc:
            logger.error("batch_invoice_failed", document_path=str(path), error=str(exc))
            result = {
                "run": {"document_path": str(path), "status": "failed", "error": str(exc)},
                "flags": [],
                "trace": [],
            }
        results.append(result)
    return results


__all__ = ["discover_invoices", "run_batch"]
