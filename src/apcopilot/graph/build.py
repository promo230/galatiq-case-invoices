from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from apcopilot.config import get_settings
from apcopilot.db.runs import append_trace, create_run, get_flags, get_run, get_trace, update_run
from apcopilot.db.seed import ensure_seeded
from apcopilot.graph.nodes import (
    InvoiceState,
    approve_node,
    ingest_node,
    route_after,
    settle_node,
    validate_node,
)


def build_graph() -> StateGraph:
    """Wire the four-stage pipeline as a real LangGraph StateGraph.

    Linear today (ingest -> validate -> approve -> settle), but modeled as a
    graph rather than four plain function calls so that (a) each stage's state
    transition is checkpointed independently -- a crash between validate and
    approve resumes without re-running ingestion and re-billing the extraction
    LLM call -- and (b) it satisfies the case brief's "multi-agent
    orchestration framework" ask. The conditional edges after each stage are
    what implement the "stop on first failure, don't run downstream nodes
    against missing data" rule from nodes._fail / route_after.
    """
    graph: StateGraph = StateGraph(InvoiceState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("validate", validate_node)
    graph.add_node("approve", approve_node)
    graph.add_node("settle", settle_node)

    graph.add_edge(START, "ingest")
    graph.add_conditional_edges("ingest", route_after, {"continue": "validate", "stop": END})
    graph.add_conditional_edges("validate", route_after, {"continue": "approve", "stop": END})
    graph.add_conditional_edges("approve", route_after, {"continue": "settle", "stop": END})
    graph.add_edge("settle", END)
    return graph


async def run_invoice(
    document_path: Path, *, batch_id: str | None = None, db_path: Path | None = None
) -> dict:
    """Run one document through ingest -> validate -> approve -> settle.

    Never raises: any node failure is caught at the node boundary (see
    nodes._fail), recorded on the run, and the pipeline stops there. Always
    returns {"run", "flags", "trace"} for that run_id, whatever state it ended
    up in, so a bad document in a batch never kills the batch.
    """
    settings = get_settings()
    resolved_db_path = db_path or settings.db_path
    ensure_seeded(resolved_db_path)

    run_id = str(uuid4())
    start = time.monotonic()

    create_run(
        run_id=run_id,
        document_path=str(document_path),
        source_format=document_path.suffix.lstrip("."),
        batch_id=batch_id,
        db_path=resolved_db_path,
    )
    append_trace(run_id=run_id, node="ingest", status="started", db_path=resolved_db_path)

    checkpoint_path = settings.var_dir / "checkpoints.sqlite"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    initial_state: InvoiceState = {
        "run_id": run_id,
        "document_path": str(document_path),
        "batch_id": batch_id,
        "db_path": str(resolved_db_path),
        "errors": [],
        "failed": False,
    }

    try:
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            compiled = build_graph().compile(checkpointer=saver)
            await compiled.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": run_id}},
            )
    except Exception as exc:
        # Individual nodes already catch and record their own failures; this is
        # a belt-and-suspenders catch for wiring/checkpointer errors so
        # run_invoice still returns normally instead of raising into a batch.
        append_trace(
            run_id=run_id, node="graph", status="error", detail={"error": str(exc)},
            db_path=resolved_db_path,
        )
        current = get_run(run_id, db_path=resolved_db_path)
        if current is not None and current.get("status") == "running":
            update_run(run_id=run_id, status="failed", error=str(exc), db_path=resolved_db_path)

    duration_ms = int((time.monotonic() - start) * 1000)
    update_run(run_id=run_id, duration_ms=duration_ms, db_path=resolved_db_path)

    return {
        "run": get_run(run_id, db_path=resolved_db_path),
        "flags": get_flags(run_id, db_path=resolved_db_path),
        "trace": get_trace(run_id, db_path=resolved_db_path),
    }


__all__ = ["build_graph", "run_invoice"]
