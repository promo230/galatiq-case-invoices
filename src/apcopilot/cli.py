from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from apcopilot.batch import discover_invoices, run_batch
from apcopilot.db.seed import ensure_seeded, reset_database
from apcopilot.logging import configure_logging, get_logger
from apcopilot.render import print_batch_table, print_json_result, print_run_summary

app = typer.Typer(
    name="apcopilot",
    help="AP Copilot: multi-agent invoice-processing automation.",
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)


def _bootstrap() -> None:
    configure_logging()
    ensure_seeded()


@app.command()
def run(
    invoice_path: Path = typer.Option(  # noqa: B008 - idiomatic typer usage
        ..., "--invoice-path", exists=True, help="Path to a single invoice file to process."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Print only the raw JSON result, skip the rich summary."
    ),
) -> None:
    """Run the pipeline on a single invoice."""
    from apcopilot.graph import run_invoice

    _bootstrap()
    result = asyncio.run(run_invoice(invoice_path))
    if not json_out:
        print_run_summary(console, result)
        console.rule("Full JSON result")
    print_json_result(console, result)


@app.command()
def batch(
    dir: Path = typer.Option(  # noqa: B008 - idiomatic typer usage
        Path("data/invoices"), "--dir", help="Directory of invoices to process."
    ),
    pattern: str = typer.Option("*", "--pattern", help="Glob pattern to filter files within DIR."),
    json_out: bool = typer.Option(
        False, "--json", help="Print only the raw JSON results, skip the rich table."
    ),
) -> None:
    """Run the pipeline on every invoice matching PATTERN inside DIR."""
    _bootstrap()
    paths = discover_invoices(dir, pattern)
    if not paths:
        console.print(f"[yellow]No files matching {pattern!r} found in {dir}[/yellow]")
        raise typer.Exit(code=0)

    results = asyncio.run(run_batch(paths))
    if not json_out:
        print_batch_table(console, results)
        console.rule("Full JSON results")
    print_json_result(console, results)


@app.command("reset-db")
def reset_db(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete and re-seed the local SQLite database."""
    if not yes:
        typer.confirm(
            "This will delete the local database (all runs, flags, payments) and re-seed "
            "reference data. Continue?",
            abort=True,
        )
    counts = reset_database()
    console.print("[bold green]Database reset.[/bold green]")
    for table, n in counts.items():
        console.print(f"  {table}: {n} rows seeded")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", help="Port to bind the API server to."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development."),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind the API server to."),
) -> None:
    """Start the FastAPI backend (and static frontend) with uvicorn."""
    import uvicorn

    uvicorn.run("apcopilot.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
