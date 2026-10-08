"""Northern Star FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router

APP_VERSION = "0.8.1"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Northern Star",
        version=APP_VERSION,
        description="Software intelligence platform",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "service": "northern-star", "version": APP_VERSION}

    return app


app = create_app()
