from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table

# Shared between main.py and cli.py so the single-invoice and batch summaries
# look the same everywhere they're printed.

_STATUS_STYLE = {
    "approved": "bold green",
    "rejected": "bold red",
    "needs_human": "bold yellow",
    "failed": "bold red",
    "running": "cyan",
    "queued": "dim",
}

_SEVERITY_STYLE = {
    "critical": "bold red",
    "high": "red",
    "medium": "yellow",
    "low": "dim",
    "info": "dim",
}


def _status_text(status: str | None) -> str:
    style = _STATUS_STYLE.get(status or "", "white")
    return f"[{style}]{status or 'unknown'}[/{style}]"


def print_run_summary(console: Console, result: dict[str, Any]) -> None:
    """Human-readable summary of a single run_invoice() result."""
    run = result.get("run") or {}
    flags = result.get("flags") or []

    table = Table(title="Invoice Run Summary", show_header=False, box=None, padding=(0, 1))
    table.add_row("Run ID", str(run.get("run_id") or "-"))
    table.add_row("Document", str(run.get("document_path") or "-"))
    table.add_row("Vendor", str(run.get("vendor_name") or "-"))
    table.add_row("Invoice #", str(run.get("invoice_number") or "-"))
    total = run.get("total")
    currency = run.get("currency") or ""
    total_usd = run.get("total_usd")
    total_str = f"{total} {currency}".strip() if total is not None else "-"
    if total_usd is not None and str(total_usd) != str(total):
        total_str += f" (~{total_usd} USD)"
    table.add_row("Total", total_str)
    table.add_row("Due Date", str(run.get("due_date") or "-"))
    table.add_row("Status", _status_text(run.get("status")))
    table.add_row("Lane", str(run.get("lane") or "-"))
    table.add_row("Fraud Score", str(run.get("fraud_score") if run.get("fraud_score") is not None else "-"))
    table.add_row("Confidence", str(run.get("confidence") if run.get("confidence") is not None else "-"))
    table.add_row("Extraction Attempts", str(run.get("extraction_attempts") or 0))
    table.add_row("Approval Rounds", str(run.get("approval_rounds") or 0))
    table.add_row("Cost (USD)", str(run.get("cost_usd") or "0"))
    table.add_row("Duration (ms)", str(run.get("duration_ms") or "-"))
    if run.get("error"):
        table.add_row("Error", f"[bold red]{run['error']}[/bold red]")

    console.print(table)

    if flags:
        flag_table = Table(title=f"Flags ({len(flags)})")
        flag_table.add_column("Severity")
        flag_table.add_column("Code")
        flag_table.add_column("Message")
        for flag in sorted(
            flags, key=lambda f: list(_SEVERITY_STYLE).index(f.get("severity", "info"))
            if f.get("severity") in _SEVERITY_STYLE
            else len(_SEVERITY_STYLE)
        ):
            severity = flag.get("severity", "info")
            style = _SEVERITY_STYLE.get(severity, "white")
            flag_table.add_row(
                f"[{style}]{severity}[/{style}]",
                str(flag.get("code") or "-"),
                str(flag.get("message") or "-"),
            )
        console.print(flag_table)
    else:
        console.print("[dim]No flags raised.[/dim]")

    decision_json = run.get("decision_json")
    if decision_json:
        decision = decision_json if isinstance(decision_json, dict) else _try_json(decision_json)
        if decision:
            reasoning = decision.get("reasoning") or decision.get("rationale")
            if reasoning:
                console.print(f"\n[bold]Decision reasoning:[/bold] {reasoning}")

    payment_note = {
        "approved": "[bold green]Payment: issued via mock payment API.[/bold green]",
        "rejected": "[bold red]Payment: withheld — invoice rejected.[/bold red]",
        "needs_human": "[bold yellow]Payment: on hold — pending human review.[/bold yellow]",
        "failed": "[bold red]Payment: not attempted — run failed.[/bold red]",
    }.get(str(run.get("status") or ""))
    if payment_note:
        console.print(payment_note)


def _try_json(value: Any) -> dict | None:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def print_batch_table(console: Console, results: list[dict[str, Any]]) -> None:
    """Rich table summarizing a batch of run_invoice() results: one row per file."""
    table = Table(title=f"Batch Run ({len(results)} invoices)")
    table.add_column("File")
    table.add_column("Vendor")
    table.add_column("Status")
    table.add_column("Lane")
    table.add_column("Total")
    table.add_column("Flags")

    for result in results:
        run = result.get("run") or {}
        flags = result.get("flags") or []
        total = run.get("total")
        currency = run.get("currency") or ""
        total_str = f"{total} {currency}".strip() if total is not None else "-"
        table.add_row(
            str(run.get("document_path") or "-"),
            str(run.get("vendor_name") or "-"),
            _status_text(run.get("status")),
            str(run.get("lane") or "-"),
            total_str,
            str(len(flags)),
        )
    console.print(table)


def print_json_result(console: Console, data: Any) -> None:
    # Deliberately bypasses `console` (rich) here: Console.print() soft-wraps at
    # terminal width, which splits long JSON string values across lines and
    # inserts literal newlines into the output, corrupting it for anything
    # downstream that parses --json as machine-readable JSON.
    print(json.dumps(data, indent=2, default=str))


__all__ = ["print_batch_table", "print_json_result", "print_run_summary"]
