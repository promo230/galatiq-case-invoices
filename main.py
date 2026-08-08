"""Root CLI entrypoint required by the case brief:

    python main.py --invoice_path=data/invoices/invoice1.txt

Also supports a `--batch` mode for running every invoice in a directory, and
a `--json` flag to print raw JSON only. See `apcopilot.cli` for the richer
Typer-based CLI (installed as the `apcopilot` console script) which wraps the
same underlying `run_invoice` / `run_batch` calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src/apcopilot` importable even when the package hasn't been installed
# (e.g. a grader clones the repo and runs `python main.py` without `uv sync`).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import asyncio
import json

from rich.console import Console

from apcopilot.batch import discover_invoices, run_batch
from apcopilot.db.seed import ensure_seeded
from apcopilot.logging import configure_logging, get_logger
from apcopilot.render import print_batch_table, print_json_result, print_run_summary

logger = get_logger(__name__)

DEFAULT_BATCH_DIR = "data/invoices"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the AP Copilot invoice-processing pipeline on one invoice or a batch.",
    )
    parser.add_argument(
        "--invoice_path",
        dest="invoice_path",
        type=str,
        default=None,
        help="Path to a single invoice file to process.",
    )
    parser.add_argument(
        "--batch",
        dest="batch",
        nargs="?",
        const=DEFAULT_BATCH_DIR,
        default=None,
        metavar="DIR",
        help=f"Process every invoice in DIR (default: {DEFAULT_BATCH_DIR}) instead of a single file.",
    )
    parser.add_argument(
        "--json",
        dest="json_only",
        action="store_true",
        help="Print only the raw JSON result(s), skip the rich human-readable summary.",
    )
    return parser


async def _run_single(invoice_path: Path, *, json_only: bool, console: Console) -> dict:
    from apcopilot.graph import run_invoice

    result = await run_invoice(invoice_path)
    if not json_only:
        print_run_summary(console, result)
        console.rule("Full JSON result")
    print_json_result(console, result)
    return result


async def _run_batch(batch_dir: Path, *, json_only: bool, console: Console) -> list[dict]:
    paths = discover_invoices(batch_dir)
    if not paths:
        console.print(f"[yellow]No invoice files found in {batch_dir}[/yellow]")
        return []
    results = await run_batch(paths)
    if not json_only:
        print_batch_table(console, results)
        console.rule("Full JSON results")
    print_json_result(console, results)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.invoice_path and args.batch:
        parser.error("--invoice_path and --batch are mutually exclusive.")
    if not args.invoice_path and not args.batch:
        parser.error("one of --invoice_path or --batch is required.")

    console = Console()
    configure_logging()
    ensure_seeded()

    try:
        if args.batch:
            asyncio.run(_run_batch(Path(args.batch), json_only=args.json_only, console=console))
        else:
            invoice_path = Path(args.invoice_path)
            if not invoice_path.exists():
                parser.error(f"invoice file not found: {invoice_path}")
            asyncio.run(
                _run_single(invoice_path, json_only=args.json_only, console=console)
            )
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("main_crashed", error=str(exc), exc_info=True)
        console.print(f"[bold red]Fatal error:[/bold red] {exc}")
        if args.json_only:
            print(json.dumps({"error": str(exc)}))
        return 1

    # Exit 0 regardless of the invoice's own business outcome (approved /
    # rejected / needs_human are all valid results, not program failures).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
