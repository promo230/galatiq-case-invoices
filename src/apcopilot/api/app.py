from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from apcopilot.api.routes import router
from apcopilot.db.seed import ensure_seeded
from apcopilot.logging import configure_logging, get_logger

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="AP Copilot",
        description="Multi-agent invoice-processing automation API.",
        version="0.1.0",
    )

    # Permissive CORS: this is a local demo/case-study API, not a hardened
    # multi-tenant deployment.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        ensure_seeded()
        logger.info("api_startup", static_dir=str(STATIC_DIR))

    app.include_router(router)

    # Mounted last: API routes above take precedence, everything else falls
    # through to the static frontend (owned by a parallel agent).
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()
