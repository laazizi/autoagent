"""Tests for the logging utilities.

We must guarantee:
  * `get_logger` returns a namespaced `autoagent.*` logger.
  * The redaction filter is attached and ACTUALLY scrubs known secret
    patterns (Bearer tokens, api keys in headers / query strings, keys
    in JSON bodies).
  * The filter runs on the producer thread before any handler sees the
    record, so a misconfigured remote handler cannot leak the raw
    secret.
"""

from __future__ import annotations

import logging

import pytest

from autoagent.logging import SecretRedactingFilter, get_logger, redact


class TestGetLogger:
    def test_root_logger_namespace(self) -> None:
        log = get_logger()
        assert log.name == "autoagent"

    def test_sublogger_namespace(self) -> None:
        log = get_logger("registry")
        assert log.name == "autoagent.registry"

    def test_redacting_filter_attached(self) -> None:
        log = get_logger("test-attach")
        assert any(isinstance(f, SecretRedactingFilter) for f in log.filters)

    def test_filter_attached_only_once(self) -> None:
        log1 = get_logger("test-once")
        log2 = get_logger("test-once")
        # Same logger instance returned (Python logger registry).
        assert log1 is log2
        filters = [f for f in log1.filters if isinstance(f, SecretRedactingFilter)]
        assert len(filters) == 1


class TestRedaction:
    @pytest.mark.parametrize(
        "raw",
        [
            "Authorization: Bearer sk-1234567890abcdef",
            "Headers: {x-api-key: sk-veryverysecret}",
            "x-goog-api-key=AIzaSyDeFaKeKeY_ForTesting",
            "POST https://x/v1?key=AIzaSyDeFaKeKey1234",
            '{"api_key": "sk-livesecret"}',
        ],
    )
    def test_secret_pattern_is_redacted(self, raw: str) -> None:
        cleaned = redact(raw)
        # The literal secret value must not appear; the redaction
        # placeholder must.
        assert "REDACTED" in cleaned
        # And the original secret string must NOT survive.
        for secret in (
            "sk-1234567890abcdef",
            "sk-veryverysecret",
            "AIzaSyDeFaKeKeY_ForTesting",
            "AIzaSyDeFaKeKey1234",
            "sk-livesecret",
        ):
            if secret in raw:
                assert secret not in cleaned, f"Secret leaked: {secret!r} in {cleaned!r}"


class TestFilterEndToEnd:
    def test_filter_redacts_msg_in_record(self, caplog: pytest.LogCaptureFixture) -> None:
        log = get_logger("e2e-msg")
        with caplog.at_level(logging.INFO, logger="autoagent.e2e-msg"):
            log.info("Calling with Bearer sk-end2endsecretXYZ")
        assert "sk-end2endsecretXYZ" not in caplog.text
        assert "REDACTED" in caplog.text

    def test_filter_redacts_args(self, caplog: pytest.LogCaptureFixture) -> None:
        log = get_logger("e2e-args")
        with caplog.at_level(logging.INFO, logger="autoagent.e2e-args"):
            log.info("upstream %s", "Bearer sk-from-arg-secret")
        assert "sk-from-arg-secret" not in caplog.text
