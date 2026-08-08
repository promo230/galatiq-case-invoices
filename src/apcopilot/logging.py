from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

from apcopilot.config import get_settings

_configured = False


def configure_logging(*, verbose: bool = False, console: bool = True) -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    settings.ensure_dirs()

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    file_handler = logging.FileHandler(settings.log_path, encoding="utf-8")
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(processor=structlog.processors.JSONRenderer())
    )

    handlers: list[logging.Handler] = [file_handler]
    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
            )
        )
        handlers.append(stream)

    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "apcopilot") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


__all__ = ["bind_contextvars", "clear_contextvars", "configure_logging", "get_logger"]
