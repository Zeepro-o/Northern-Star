"""Tests for Settings: env parsing, especially the OLLAMA_THINK toggle."""

from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),  # unset → thinking OFF (fast path)
        ("false", False),
        ("0", False),
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
    ],
)
def test_ollama_think_env_parsing(monkeypatch, raw, expected):
    monkeypatch.delenv("OLLAMA_THINK", raising=False)
    if raw is not None:
        monkeypatch.setenv("OLLAMA_THINK", raw)
    assert get_settings().ollama_think is expected


def test_default_model_is_the_fast_cpu_option(monkeypatch):
    """On CPU-only dev boxes qwen3:4b is ~60-130s and its reasoning leaks into
    the answer; qwen2.5:3b answers clean in ~8s. Ride the env override."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert get_settings().ollama_model == "qwen2.5:3b"


def test_ollama_model_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    assert get_settings().ollama_model == "qwen3:4b"


def test_qa_passes_settings_think_into_the_client(monkeypatch):
    """Wire-level: qa must forward settings.ollama_think to OllamaClient."""
    monkeypatch.setenv("OLLAMA_THINK", "true")
    settings = get_settings()
    seen = {}

    class FakeClient:
        def __init__(self, *a, **kw):
            seen.update(kw)
            self._kw = kw

        def complete(self, messages):
            return '{"answer": "x [E1].", "citations": ["E1"], "confidence": "low", "evidence_sufficient": true}'

    monkeypatch.setattr("app.services.qa.OllamaClient", FakeClient)
    from app.services import qa as qa_service

    fake = type(
        "R",
        (),
        {"file_path": "x.py", "start_line": 1, "end_line": 2,
         "language": "py", "score": 1.0, "content": "body"},
    )()
    # retrieve is passed explicitly — its default is bound at def-time.
    qa_service.answer_question(
        "acme/x", "q", settings=settings, top_k=1,
        retrieve=lambda db, rid, q, **kw: [fake],
    )
    assert seen["think"] is True