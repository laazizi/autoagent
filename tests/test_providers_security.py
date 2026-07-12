"""Red tests for credential leakage in provider HTTP calls.

CURRENT BUG (autoagent/providers/gemini.py:18):
    The API key is embedded directly in the request URL as `?key=...`.
    URLs leak into:
      - HTTP server logs (corporate proxies, ELB, nginx access logs)
      - Browser history / referer headers
      - Crash dumps / stack traces
      - Telemetry of any HTTP middleware

The Gemini API accepts the key in the `x-goog-api-key` request header,
which is the secure-by-default mode used by Google's official SDKs.

These tests are RED until gemini.py sends the key via header and never
embeds it in the URL.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from autoagent.providers.gemini import GeminiProvider
from autoagent.schema import LLMRequest, Message, ModelConfig


def _build_provider() -> GeminiProvider:
    config = ModelConfig(
        provider="gemini",
        model="gemini-2.0-flash",
        api_key="SUPER-SECRET-API-KEY-1234567890",
    )
    return GeminiProvider(config)


def _make_request() -> LLMRequest:
    return LLMRequest(
        messages=[Message(role="user", content="hi")],
        tools=[],
    )


class TestGeminiAPIKeyHandling:
    def test_api_key_is_not_in_url(self) -> None:
        provider = _build_provider()
        captured: dict[str, Any] = {}

        def fake_post_json(url: str, payload: dict, headers=None, timeout=None) -> dict:
            captured["url"] = url
            captured["headers"] = headers or {}
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

        with patch("autoagent.providers.gemini.post_json", fake_post_json):
            provider.complete(_make_request())

        assert "SUPER-SECRET" not in captured["url"], f"API key leaked into URL: {captured['url']}"
        assert "key=" not in captured["url"], f"`?key=` query param still present: {captured['url']}"

    def test_api_key_is_in_header(self) -> None:
        provider = _build_provider()
        captured: dict[str, Any] = {}

        def fake_post_json(url: str, payload: dict, headers=None, timeout=None) -> dict:
            captured["headers"] = dict(headers or {})
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

        with patch("autoagent.providers.gemini.post_json", fake_post_json):
            provider.complete(_make_request())

        header_values = {k.lower(): v for k, v in captured["headers"].items()}
        assert (
            "x-goog-api-key" in header_values
        ), f"Expected `x-goog-api-key` header, got: {list(header_values)}"
        assert header_values["x-goog-api-key"] == "SUPER-SECRET-API-KEY-1234567890"
