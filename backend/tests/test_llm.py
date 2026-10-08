"""Tests for the Ollama HTTP client — mocked transport, never the network."""

from __future__ import annotations

import pytest
import httpx

from app.services.llm import (
    OllamaClient,
    OllamaError,
    OllamaModelNotInstalledError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


def _client(handler) -> OllamaClient:
    return OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="qwen3:4b",
        timeout_seconds=30,
        transport=httpx.MockTransport(handler),
    )


def _chat_json(content: str) -> dict:
    return {"model": "qwen3:4b", "message": {"role": "assistant", "content": content}}


class TestSuccess:
    def test_returns_content_from_message(self):
        client = _client(
            lambda req: httpx.Response(200, json=_chat_json("The answer."))
        )
        assert client.complete([{"role": "user", "content": "hi"}]) == "The answer."

    def test_posts_chat_endpoint_with_model(self):
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["url"] = str(req.url)
            captured["body"] = req.content
            return httpx.Response(200, json=_chat_json("ok"))

        client = _client(handler)
        client.complete([{"role": "user", "content": "hi"}])
        assert captured["url"].endswith("/api/chat")
        assert b'"model":"qwen3:4b"' in captured["body"]
        assert b'"stream":false' in captured["body"]
        # thinking is OFF by default — and must be sent explicitly, never
        # omitted, or Ollama's qwen3 models would reason by default (slow).
        assert b'"think":false' in captured["body"]

    def test_think_true_is_sent_when_enabled(self):
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["body"] = req.content
            return httpx.Response(200, json=_chat_json("ok"))

        client = OllamaClient(
            base_url="http://127.0.0.1:11434",
            model="qwen3:4b",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
            think=True,
        )
        client.complete([{"role": "user", "content": "hi"}])
        assert b'"think":true' in captured["body"]


class TestFailures:
    def test_connect_error_is_unavailable(self):
        def boom(req):
            raise httpx.ConnectError("connection refused", request=req)

        with pytest.raises(OllamaUnavailableError):
            _client(boom).complete([{"role": "user", "content": "hi"}])

    def test_timeout_is_timeout_error(self):
        def boom(req):
            raise httpx.TimeoutException("timed out", request=req)

        with pytest.raises(OllamaTimeoutError):
            _client(boom).complete([{"role": "user", "content": "hi"}])

    def test_model_missing_is_not_installed(self):
        client = _client(
            lambda req: httpx.Response(
                404, json={"error": "model 'qwen3:4b' not found"}
            )
        )
        with pytest.raises(OllamaModelNotInstalledError):
            client.complete([{"role": "user", "content": "hi"}])

    def test_generic_5xx_is_unavailable(self):
        client = _client(lambda req: httpx.Response(500, text="internal error"))
        with pytest.raises(OllamaUnavailableError):
            client.complete([{"role": "user", "content": "hi"}])

    def test_empty_content_is_response_error(self):
        client = _client(lambda req: httpx.Response(200, json=_chat_json("   ")))
        with pytest.raises(OllamaResponseError):
            client.complete([{"role": "user", "content": "hi"}])

    def test_missing_message_key_is_response_error(self):
        client = _client(
            lambda req: httpx.Response(200, json={"done": True})
        )
        with pytest.raises(OllamaResponseError):
            client.complete([{"role": "user", "content": "hi"}])

    def test_non_json_body_is_response_error(self):
        client = _client(lambda req: httpx.Response(200, text="<html>not json</html>"))
        with pytest.raises(OllamaResponseError):
            client.complete([{"role": "user", "content": "hi"}])

    def test_all_failures_are_typed_ollama_errors(self):
        """Every failure path must surface as an ol' OllamaError family member."""
        errors = [
            OllamaUnavailableError("x"),
            OllamaModelNotInstalledError("x"),
            OllamaTimeoutError("x"),
            OllamaResponseError("x"),
        ]
        assert all(isinstance(e, OllamaError) for e in errors)