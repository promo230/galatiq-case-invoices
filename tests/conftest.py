"""Shared fixtures.

The whole suite runs fully offline and free: `APCOPILOT_LLM_MODE=off` forces the
deterministic parsers, the heuristic .txt/.pdf fallback, and the rules-only
approval path, so no test needs an API key, opens a socket, or costs anything.

Two things have to happen before `apcopilot` is imported anywhere:

  * `APCOPILOT_LLM_MODE=off` -- `Settings` is a pydantic-settings BaseSettings
    with `env_prefix="APCOPILOT_"`, and a real environment variable outranks the
    repo's `.env` file, so this holds even for a developer whose `.env` sets
    `llm_mode=live`.
  * `APCOPILOT_VAR_DIR` -- everything derived from it (`db_path`, `log_path`, the
    LangGraph checkpoint file) then lands in a throwaway directory instead of the
    repo's `var/`, so a test that forgets to pass `db_path` still cannot touch
    the developer's real database.

`get_settings()` is `@lru_cache`d, so the session fixture below clears it once the
environment is in place.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INVOICE_DIR = REPO_ROOT / "data" / "invoices"

_TMP_VAR_DIR = Path(tempfile.mkdtemp(prefix="apcopilot-test-var-"))
os.environ["APCOPILOT_LLM_MODE"] = "off"
os.environ["APCOPILOT_VAR_DIR"] = str(_TMP_VAR_DIR)

from apcopilot.config import get_settings  # noqa: E402
from apcopilot.db.seed import ensure_seeded  # noqa: E402
from apcopilot.models import ExtractedInvoice, LineItem, ValidationFlag  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def offline_settings() -> Iterator[None]:
    """Drop the cached Settings so the env vars set above take effect, and assert
    the suite really is in offline mode before a single test runs."""
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_mode == "off", (
        "the test suite must run with APCOPILOT_LLM_MODE=off; "
        f"got {settings.llm_mode!r}"
    )
    assert settings.var_dir == _TMP_VAR_DIR
    yield
    get_settings.cache_clear()
    shutil.rmtree(_TMP_VAR_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def no_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard tripwire: any code path that reaches the Anthropic client fails the
    test loudly instead of silently trying to open a connection."""

    async def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an LLM call was attempted while APCOPILOT_LLM_MODE=off")

    import apcopilot.llm as llm_pkg
    import apcopilot.llm.client as llm_client

    monkeypatch.setattr(llm_client, "call_structured", _explode)
    monkeypatch.setattr(llm_pkg, "call_structured", _explode)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A freshly seeded SQLite database private to one test.

    Every function under test takes `db_path`; threading this through is what
    keeps tests from seeing each other's runs, payments, and flags.
    """
    path = tmp_path / "apcopilot-test.db"
    ensure_seeded(path)
    return path


@pytest.fixture
def invoice_dir() -> Path:
    return INVOICE_DIR


InvoiceFactory = Callable[..., ExtractedInvoice]


@pytest.fixture
def make_invoice() -> InvoiceFactory:
    """A clean, fully-valid invoice that passes every rule, plus keyword overrides.

    Rule tests perturb exactly one field so a resulting flag can only have come
    from the rule under test. Baseline: known active vendor (Widgets Inc., USD,
    $10k auto-approve limit), in-stock items, arithmetic that balances, and a due
    date after `Settings.as_of_date` (2026-02-01).
    """

    def _make(**overrides: Any) -> ExtractedInvoice:
        fields: dict[str, Any] = {
            "invoice_number": "INV-9001",
            "vendor_name": "Widgets Inc.",
            "currency": "USD",
            "line_items": [
                LineItem(
                    description="WidgetA",
                    sku="WIDGETA",
                    quantity=2,
                    unit_price=Decimal("250.00"),
                    line_total=Decimal("500.00"),
                ),
            ],
            "subtotal": Decimal("500.00"),
            "tax": Decimal("0.00"),
            "total": Decimal("500.00"),
            "invoice_date": date(2026, 1, 15),
            "due_date": date(2026, 3, 1),
        }
        fields.update(overrides)
        return ExtractedInvoice(**fields)

    return _make


@pytest.fixture
def line_item() -> Callable[..., LineItem]:
    def _make(description: str, quantity: int, unit_price: str = "250.00") -> LineItem:
        price = Decimal(unit_price)
        return LineItem(
            description=description,
            sku=description,
            quantity=quantity,
            unit_price=price,
            line_total=price * quantity,
        )

    return _make


def codes(flags: list[ValidationFlag]) -> list[str]:
    """Flag codes in order -- duplicates preserved so "x2" expectations are testable."""
    return [flag.code for flag in flags]
