"""Centralized logging for autoagent.

Use `get_logger(name)` inside the library — never call `logging.getLogger`
directly. Every logger returned here has a `SecretRedactingFilter`
attached, which scrubs API keys and bearer tokens out of every log
record before they leave the process.

The library does NOT add a handler. Hosts decide where logs go. The
namespace is `autoagent.*` so a host can selectively configure verbosity
with `logging.getLogger("autoagent").setLevel(logging.DEBUG)`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

_ROOT_NAME = "autoagent"

# Regex that matches common ways an API key shows up in a log message:
#   Authorization: Bearer sk-abc123...
#   x-api-key: sk-abc123...
#   x-goog-api-key=sk-abc123...
#   "api_key": "sk-abc123..."
#   key=sk-abc123  (legacy URL form, redact even though we no longer emit it)
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", re.IGNORECASE),
    re.compile(
        r"((?:x-api-key|x-goog-api-key|api[_-]?key)['\"]?\s*[:=]\s*['\"]?)[A-Za-z0-9._\-]{8,}",
        re.IGNORECASE,
    ),
    re.compile(r"(\bkey=)[A-Za-z0-9._\-]{8,}"),
)
_REDACTED = "***REDACTED***"


def _redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: m.group(1) + _REDACTED, text)
    return text


class SecretRedactingFilter(logging.Filter):
    """Strip API keys / bearer tokens from log messages and arguments.

    Filters run synchronously on the producer thread, before any handler
    sees the record. That means a misconfigured handler (e.g. a remote
    logging endpoint) cannot leak the raw secret.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _redact(record.msg)
        if record.args:
            # Only redact STRING args. Coercing every arg to str would break
            # numeric format specs downstream ("%d" % "5" -> TypeError) —
            # secrets are strings anyway, numbers can't leak a key.
            if isinstance(record.args, dict):
                record.args = {
                    k: _redact(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    _redact(a) if isinstance(a, str) else a for a in record.args
                )
        return True


_FILTER = SecretRedactingFilter()
_INSTALLED: set[str] = set()


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger with secret redaction installed.

    Pass a sub-module name (e.g. ``"registry"``); it is automatically
    placed under the ``autoagent`` namespace. Passing ``None`` returns
    the root ``autoagent`` logger.
    """
    full_name = _ROOT_NAME if not name else f"{_ROOT_NAME}.{name}"
    logger = logging.getLogger(full_name)
    if full_name not in _INSTALLED:
        logger.addFilter(_FILTER)
        _INSTALLED.add(full_name)
    return logger


def redact(value: Any) -> str:
    """Public helper for one-off redaction (e.g. when building an error
    message that may embed a header value)."""
    return _redact(str(value))


__all__ = ["SecretRedactingFilter", "get_logger", "redact"]
