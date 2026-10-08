"""Northern Star FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI

from .api.routes import router

APP_VERSION = "0.3.0"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Northern Star",
        version=APP_VERSION,
        description=(
            "Software intelligence platform — v0.3: repository ingestion "
            "(clone → inspect files → detect languages/frameworks → metadata report), "
            "a SQLite evidence index with lexical (FTS5) retrieval, and "
            "evidence-grounded Q&A against a local Ollama model with validated "
            "file:line citations."
        ),
    )
    app.include_router(router, prefix="/api/v1")

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "service": "northern-star", "version": APP_VERSION}

    return app


app = create_app()