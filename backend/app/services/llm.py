"""Thin Ollama HTTP client for the local LLM.

Talks to Ollama's ``/api/chat`` endpoint over plain HTTP (httpx). No agent
framework, no SDK — just a request, a parsed response, and typed errors so
the API layer can map failures to clean HTTP codes. ``transport`` is
injectable so tests exercise every failure path with ``httpx.MockTransport``
and never touch the network.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

OLLAMA_CHAT_PATH = "/api/chat"


class OllamaError(Exception):
    """Base class for every failure talking to Ollama."""


class OllamaUnavailableError(OllamaError):
    """Ollama is down, unreachable, or returned a non-2xx status."""


class OllamaModelNotInstalledError(OllamaError):
    """The requested model does not exist on this Ollama instance."""


class OllamaTimeoutError(OllamaError):
    """The request exceeded the configured timeout."""


class OllamaResponseError(OllamaError):
    """Ollama returned 2xx but the payload had no usable message content."""


def _classify_error(exc: httpx.HTTPError) -> OllamaError:
    """Map an httpx failure onto Northern Star's typed Ollama errors."""
    if isinstance(exc, httpx.TimeoutException):
        return OllamaTimeoutError("Ollama request timed out.")
    return OllamaUnavailableError(f"Could not reach Ollama: {exc}")


class OllamaClient:
    """A minimal, mockable client for ``POST /api/chat``."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 180,
        transport: Optional[Any] = None,
        think: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._timeout_seconds = timeout_seconds
        # transport is injected by tests (httpx.MockTransport); None uses the
        # real network. Forwarded type is Any to keep 3.14 friendly.
        self._transport = transport
        # qwen3-style hidden "thinking". Sent explicitly (never omitted) so the
        # default really disables it on thinking models; no-op elsewhere.
        self.think = think

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send a chat request and return the assistant's text content.

        ``messages`` follows the OpenAI-style [{"role", "content"}] shape that
        Ollama's ``/api/chat`` accepts directly.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {"temperature": 0},  # deterministic answers
        }
        with httpx.Client(
            base_url=self.base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            try:
                response = client.post(
                    OLLAMA_CHAT_PATH,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.HTTPError as exc:
                raise _classify_error(exc) from exc

        if response.status_code == 404:
            raise OllamaModelNotInstalledError(
                f"Ollama model '{self.model}' is not installed on "
                f"{self.base_url}. Run: ollama pull {self.model}"
            )
        if response.status_code != 200:
            raise OllamaUnavailableError(
                f"Ollama returned HTTP {response.status_code}: "
                f"{_truncate(response.text)}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                "Ollama returned a non-JSON response body."
            ) from exc
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise OllamaResponseError(
                "Ollama response was missing message.content."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseError("Ollama returned an empty answer.")
        return content


def _truncate(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"