"""Tests for the HTTP transport (autoagent/http.py).

This module is the single network surface of the library. Every provider
funnels its requests through `post_json`, so its behaviour under network
failure, bad upstream responses, and header propagation must be pinned
explicitly.

We mock `urllib.request.urlopen` rather than spin up a real server so
the tests stay deterministic and offline.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest.mock import patch

import pytest

from autoagent.errors import ProviderError
from autoagent.http import post_json


class _FakeResponse:
    """Mimic the context-manager response object from urllib."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class TestPostJsonSuccess:
    def test_returns_parsed_json(self) -> None:
        with patch(
            "autoagent.http.urllib.request.urlopen",
            return_value=_FakeResponse(b'{"answer": 42}'),
        ):
            result = post_json("http://example/test", {"x": 1})
        assert result == {"answer": 42}

    def test_sends_payload_as_utf8_json_body(self) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
            captured["body"] = request.data
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["timeout"] = timeout
            return _FakeResponse(b'{"ok": true}')

        with patch("autoagent.http.urllib.request.urlopen", fake_urlopen):
            post_json("http://example/x", {"hello": "wörld"}, timeout=12.5)

        assert captured["method"] == "POST"
        assert captured["url"] == "http://example/x"
        assert captured["timeout"] == 12.5
        # Body must be UTF-8 encoded JSON, not str.
        assert isinstance(captured["body"], bytes)
        assert json.loads(captured["body"].decode("utf-8")) == {"hello": "wörld"}

    def test_default_content_type_set(self) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
            captured["headers"] = dict(request.headers)
            return _FakeResponse(b"{}")

        with patch("autoagent.http.urllib.request.urlopen", fake_urlopen):
            post_json("http://example/x", {})
        # urllib title-cases headers internally.
        normalized = {k.lower(): v for k, v in captured["headers"].items()}
        assert normalized["content-type"] == "application/json"

    def test_extra_headers_propagated(self) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeResponse:
            captured["headers"] = dict(request.headers)
            return _FakeResponse(b"{}")

        with patch("autoagent.http.urllib.request.urlopen", fake_urlopen):
            post_json(
                "http://example/x",
                {},
                headers={"x-api-key": "secret", "x-trace-id": "abc"},
            )
        normalized = {k.lower(): v for k, v in captured["headers"].items()}
        assert normalized["x-api-key"] == "secret"
        assert normalized["x-trace-id"] == "abc"


def _http_error(code: int, body: bytes = b"{}", msg: str = "err") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://example/x", code=code, msg=msg,
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body),
    )


class TestPostJsonFailures:
    def test_http_500_retried_then_raises_with_metadata(self) -> None:
        # 5xx is transient upstream: retried (with backoff, patched out here),
        # and the final ProviderError carries programmatic metadata.
        calls = {"n": 0}

        def fake_urlopen(request, timeout=None):
            calls["n"] += 1
            raise _http_error(500, b'{"error": "internal"}', "Server Error")

        with (
            patch("autoagent.http.urllib.request.urlopen", fake_urlopen),
            patch("autoagent.http.time.sleep"),
            pytest.raises(ProviderError, match="HTTP 500") as exc_info,
        ):
            post_json("http://example/x", {}, retries=2)
        assert calls["n"] == 3  # initial + 2 retries
        assert exc_info.value.status_code == 500
        assert exc_info.value.retryable is True

    def test_http_429_retried_then_succeeds(self) -> None:
        # Rate limit then recovery: the caller never sees the 429.
        calls = {"n": 0}

        def fake_urlopen(request, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(429, b'{"error": "rate limited"}', "Too Many Requests")
            return _FakeResponse(b'{"ok": true}')

        with (
            patch("autoagent.http.urllib.request.urlopen", fake_urlopen),
            patch("autoagent.http.time.sleep"),
        ):
            result = post_json("http://example/x", {})
        assert result == {"ok": True}
        assert calls["n"] == 2

    def test_http_400_not_retried(self) -> None:
        # Caller errors fail fast: exactly one attempt, retryable=False.
        calls = {"n": 0}

        def fake_urlopen(request, timeout=None):
            calls["n"] += 1
            raise _http_error(400, b'{"reason": "bad"}', "Bad Request")

        with (
            patch("autoagent.http.urllib.request.urlopen", fake_urlopen),
            pytest.raises(ProviderError, match="HTTP 400") as exc_info,
        ):
            post_json("http://example/x", {})
        assert calls["n"] == 1
        assert exc_info.value.status_code == 400
        assert exc_info.value.retryable is False

    def test_http_error_message_includes_body(self) -> None:
        err = _http_error(400, b'{"reason": "invalid-model"}', "Bad Request")
        with (
            patch("autoagent.http.urllib.request.urlopen", side_effect=err),
            pytest.raises(ProviderError, match="invalid-model"),
        ):
            post_json("http://example/x", {})

    def test_url_error_raises_provider_error(self) -> None:
        err = urllib.error.URLError("name resolution failed")
        with (
            patch("autoagent.http.urllib.request.urlopen", side_effect=err),
            pytest.raises(ProviderError, match="Request failed"),
        ):
            post_json("http://example/x", {})

    def test_invalid_json_response_raises_provider_error(self) -> None:
        with (
            patch(
                "autoagent.http.urllib.request.urlopen",
                return_value=_FakeResponse(b"<html>not json</html>"),
            ),
            pytest.raises(ProviderError, match="invalid JSON"),
        ):
            post_json("http://example/x", {})
