"""Tests for the HTTP transport (autoagent/http.py).

This module is the single network surface of the library. Every provider
funnels its requests through `post_json` / `post_sse`, so its behaviour under
network failure, bad upstream responses, header propagation — and, since
0.21.0, CONNECTION REUSE — must be pinned explicitly.

We mock the connection factory (`autoagent.http._connection`) rather than spin
up a real server so the tests stay deterministic and offline. The fake
connection records exactly what `http.client` would receive (method, path,
body bytes, headers) so every assertion of the previous urllib-based version
is preserved, and the persistent-connection semantics are testable: the SAME
fake connection object must be handed back for the same host.
"""

from __future__ import annotations

import http.client
import json
from typing import Any
from unittest.mock import patch

import pytest

from autoagent.errors import ProviderError
from autoagent.http import close_connections, post_json, post_sse


class _FakeResponse:
    """Mimic `http.client.HTTPResponse`: status, headers, read(), readline()."""

    def __init__(self, body: bytes, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self._lines = body.splitlines(keepends=True)
        self._body = body
        self.status = status
        self._headers = headers or {}
        self.closed = False

    def read(self) -> bytes:
        return self._body

    def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""

    def getheaders(self) -> list[tuple[str, str]]:
        return list(self._headers.items())

    def close(self) -> None:
        self.closed = True


class _FakeConn:
    """Records requests; serves scripted responses (or raises scripted errors)."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.timeout: float | None = None
        self.closed = False

    def request(self, method: str, path: str, body: bytes | None = None,
                headers: dict[str, str] | None = None) -> None:
        self.requests.append({"method": method, "path": path, "body": body,
                              "headers": dict(headers or {})})

    def getresponse(self) -> _FakeResponse:
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


def _mock(script: list[Any]) -> tuple[Any, list[_FakeConn]]:
    """Patch the connection factory with a fake POOL: the same host gets the same
    fake connection back (like the real one), and all connections consume ONE
    shared script in order — so retries are observable on `conns[0].requests`."""
    conns: list[_FakeConn] = []
    pool: dict[tuple[str, str, int | None], _FakeConn] = {}
    partage = script                       # une seule liste, consommée dans l'ordre

    def factory(scheme: str, host: str, port: int | None, timeout: float) -> _FakeConn:
        conn = pool.get((scheme, host, port))
        if conn is None:
            conn = _FakeConn([])
            conn.script = partage
            pool[(scheme, host, port)] = conn
            conns.append(conn)
        conn.timeout = timeout
        return conn

    def discard(scheme: str, host: str, port: int | None) -> None:
        conn = pool.pop((scheme, host, port), None)
        if conn is not None:
            conn.closed = True

    class _Both:
        def __enter__(self) -> None:
            self._a = patch("autoagent.http._connection", factory)
            self._b = patch("autoagent.http._discard", discard)
            self._a.__enter__()
            self._b.__enter__()

        def __exit__(self, *exc: Any) -> None:
            self._b.__exit__(*exc)
            self._a.__exit__(*exc)

    return _Both(), conns


def _resp(status: int, body: bytes, headers: dict[str, str] | None = None) -> _FakeResponse:
    return _FakeResponse(body, status=status, headers=headers)


@pytest.fixture(autouse=True)
def _reset_pool() -> None:
    close_connections()


class TestPostJsonSuccess:
    def test_returns_parsed_json(self) -> None:
        p, _ = _mock([_resp(200, b'{"answer": 42}')])
        with p:
            assert post_json("https://example/test", {"x": 1}) == {"answer": 42}

    def test_sends_payload_as_utf8_json_body(self) -> None:
        p, conns = _mock([_resp(200, b'{"ok": true}')])
        with p:
            post_json("https://example/x?y=1", {"hello": "wörld"}, timeout=12.5)
        req = conns[0].requests[0]
        assert req["method"] == "POST"
        assert req["path"] == "/x?y=1"
        assert conns[0].timeout == 12.5
        assert isinstance(req["body"], bytes)
        assert json.loads(req["body"].decode("utf-8")) == {"hello": "wörld"}

    def test_default_content_type_set(self) -> None:
        p, conns = _mock([_resp(200, b"{}")])
        with p:
            post_json("https://example/x", {})
        normalized = {k.lower(): v for k, v in conns[0].requests[0]["headers"].items()}
        assert normalized["content-type"] == "application/json"
        assert normalized["content-length"] == "2"

    def test_extra_headers_propagated(self) -> None:
        p, conns = _mock([_resp(200, b"{}")])
        with p:
            post_json("https://example/x", {}, headers={"x-api-key": "secret", "x-trace-id": "abc"})
        normalized = {k.lower(): v for k, v in conns[0].requests[0]["headers"].items()}
        assert normalized["x-api-key"] == "secret"
        assert normalized["x-trace-id"] == "abc"

    def test_http_scheme_still_supported(self) -> None:
        """Ollama, vLLM, LM Studio listen in clear text locally; urllib served
        them — the persistent layer must too."""
        p, conns = _mock([_resp(200, b'{"ok": true}')])
        with p:
            assert post_json("http://localhost:11434/v1/chat", {}) == {"ok": True}
        assert conns[0].requests[0]["path"] == "/v1/chat"


class TestPostJsonFailures:
    def test_http_500_retried_then_raises_with_metadata(self) -> None:
        # 5xx is transient upstream: retried (with backoff, patched out here),
        # and the final ProviderError carries programmatic metadata.
        p, conns = _mock([_resp(500, b'{"error": "internal"}')] * 3)
        with p, patch("autoagent.http.time.sleep"), pytest.raises(ProviderError, match="HTTP 500") as exc_info:
            post_json("https://example/x", {}, retries=2)
        assert len(conns[0].requests) == 3  # initial + 2 retries, same connection
        assert exc_info.value.status_code == 500
        assert exc_info.value.retryable is True

    def test_http_429_retried_then_succeeds_and_honours_retry_after(self) -> None:
        p, conns = _mock([_resp(429, b'{"error": "rate limited"}', {"retry-after": "3"}),
                          _resp(200, b'{"ok": true}')])
        waits: list[float] = []
        with p, patch("autoagent.http.time.sleep", waits.append):
            assert post_json("https://example/x", {}) == {"ok": True}
        assert len(conns[0].requests) == 2
        assert waits == [3.0]

    def test_http_400_not_retried(self) -> None:
        p, conns = _mock([_resp(400, b'{"reason": "bad"}')])
        with p, pytest.raises(ProviderError, match="HTTP 400") as exc_info:
            post_json("https://example/x", {})
        assert len(conns[0].requests) == 1
        assert exc_info.value.status_code == 400
        assert exc_info.value.retryable is False

    def test_http_error_message_includes_body(self) -> None:
        p, _ = _mock([_resp(400, b'{"reason": "invalid-model"}')])
        with p, pytest.raises(ProviderError, match="invalid-model"):
            post_json("https://example/x", {})

    def test_network_error_raises_provider_error(self) -> None:
        p, _ = _mock([OSError("name resolution failed")])
        with p, pytest.raises(ProviderError, match="Request failed"):
            post_json("https://example/x", {})

    def test_invalid_json_response_raises_provider_error(self) -> None:
        p, _ = _mock([_resp(200, b"<html>not json</html>")])
        with p, pytest.raises(ProviderError, match="invalid JSON"):
            post_json("https://example/x", {})

    def test_unsupported_scheme(self) -> None:
        with pytest.raises(ProviderError, match="Unsupported URL scheme"):
            post_json("ftp://example/x", {})


class TestConnexionPersistante:
    """Ce pour quoi la couche a été réécrite : une poignée de main TLS par HÔTE,
    pas par appel (~120 ms économisés par appel, mesuré)."""

    def test_deux_appels_meme_hote_meme_connexion(self) -> None:
        # Sans mock de la fabrique : on inspecte le vrai pool via un faux HTTPSConnection.
        created: list[Any] = []

        class FakeHTTPS:
            def __init__(self, host, port=None, timeout=None, context=None) -> None:  # type: ignore[no-untyped-def]
                created.append(self)
                self.timeout = timeout
                self._script = [_resp(200, b'{"n": 1}'), _resp(200, b'{"n": 2}')]

            def request(self, *a, **k) -> None:  # type: ignore[no-untyped-def]
                pass

            def getresponse(self) -> _FakeResponse:
                return self._script.pop(0)

            def close(self) -> None:
                pass

        with patch("autoagent.http.http.client.HTTPSConnection", FakeHTTPS):
            assert post_json("https://api.example/a", {}) == {"n": 1}
            assert post_json("https://api.example/b", {}, timeout=99) == {"n": 2}
        assert len(created) == 1, "une deuxième connexion a été ouverte vers le même hôte"
        assert created[0].timeout == 99, "le timeout de l'appelant doit s'appliquer à la connexion réutilisée"

    def test_connexion_morte_jetee_et_reessai_immediat(self) -> None:
        """Le serveur ferme une connexion inactive : la première requête échoue
        (RemoteDisconnected). On jette la connexion et on réessaie TOUT DE SUITE
        sur une neuve — sans attendre un backoff."""
        script = [http.client.RemoteDisconnected("closed by peer"), _resp(200, b'{"ok": true}')]
        p, conns = _mock(script)
        waits: list[float] = []
        with p, patch("autoagent.http.time.sleep", waits.append):
            assert post_json("https://example/x", {}) == {"ok": True}
        assert len(conns) == 2, "la connexion morte devait être remplacée par une neuve"
        assert conns[0].closed is True, "la connexion morte devait être fermée et retirée du pool"
        assert len(conns[1].requests) == 1
        assert waits == [], "le premier réessai après une connexion morte est immédiat"


class TestPostSse:
    def test_yields_data_events_and_swallows_done(self) -> None:
        body = (b"event: ping\n"
                b"data: {\"a\": 1}\n"
                b"\n"
                b": commentaire\n"
                b"data: not json\n"
                b"data: {\"b\": 2}\n"
                b"data: [DONE]\n")
        p, conns = _mock([_resp(200, body)])
        with p:
            events = list(post_sse("https://example/stream", {"q": 1}))
        assert events == [{"a": 1}, {"b": 2}]
        headers = {k.lower(): v for k, v in conns[0].requests[0]["headers"].items()}
        assert headers["accept"] == "text/event-stream"

    def test_http_error_before_stream_raises(self) -> None:
        p, _ = _mock([_resp(401, b'{"error": "unauthorized"}')])
        with p, pytest.raises(ProviderError, match="HTTP 401"):
            list(post_sse("https://example/stream", {}))

    def test_5xx_before_stream_retried(self) -> None:
        p, conns = _mock([_resp(503, b"busy"), _resp(200, b'data: {"ok": true}\n')])
        with p, patch("autoagent.http.time.sleep"):
            assert list(post_sse("https://example/stream", {})) == [{"ok": True}]
        assert len(conns[0].requests) == 2

    def test_stream_read_to_end_closes_response(self) -> None:
        resp = _resp(200, b'data: {"x": 1}\n')
        p, _ = _mock([resp])
        with p:
            list(post_sse("https://example/stream", {}))
        assert resp.closed, "la réponse doit être fermée pour que la connexion soit réutilisable"
