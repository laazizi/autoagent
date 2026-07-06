"""Structured event tracing for autoagent (added in 0.5.0).

A `TraceEmitter` lets a host observe an agent's execution as a stream of
typed events instead of parsing free-form log text. Each event has a
stable shape — `type`, `span_id`, `parent_id`, `ts`, `payload` — so a
host can:

* Render tool calls live in a UI (consume `tool_call_start` /
  `tool_call_end` events as they arrive instead of waiting for the full
  turn to complete).
* Persist a JSONL trace file and replay it later.
* Forward events to an external observability tool (Langfuse, Phoenix,
  OpenTelemetry, ...) by wrapping the `on_event` callback.

The emitter is OPTIONAL. An `Agent` constructed without `trace=...`
emits nothing and pays no overhead — every emit site in the agent loop
is guarded.

Event types currently emitted by the agent loop, with their payload
schemas:

    run_start              {max_steps, model, message_count, tool_count}
    run_end                {status: ok|cancelled|max_steps|error, steps,
                            output_preview}
    llm_request            {step, message_count, tool_count}
    llm_response           {step, content_preview, tool_call_count,
                            has_reasoning}
    tool_call_start        {name, call_id, arguments_preview}
    tool_call_end          {name, call_id, status: ok|error,
                            duration_ms, content_preview}
    post_turn_hook_invoked {correction_count}
    post_turn_hook_correction {content_preview}
    cancelled              {step}
    max_steps_exceeded     {max_steps}

`*_preview` fields are truncated to keep individual events small;
hosts that need the full payload should plug their own callback that
captures the actual messages from the agent's return value, or wrap
the provider/registry.

Thread-safety: `emit()` is serialized by an internal RLock so two
threads cannot interleave a JSONL line or call the user callback
concurrently. Span ids and timestamps are also generated under the
lock, so events appear in the trace file in true generation order.
Keep the callback fast — heavy work should be pushed to a queue.

Secret redaction: every ``*_preview`` payload field produced via
:func:`truncate_preview` is filtered through the same
``SecretRedactingFilter`` patterns used by ``autoagent.logging``
(Bearer tokens, ``x-api-key`` / ``x-goog-api-key`` headers,
``api_key`` JSON / dict fields, legacy ``?key=`` URL form). This
matters when the trace is forwarded to an external observability
backend or committed to a repo — the agent never has a reliable way
to tell which arguments are secret, so we redact eagerly. Hosts that
want stricter scrubbing (PII, customer ids, ...) should layer their
own filter on top of ``on_event``.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable

from .logging import get_logger, redact

_log = get_logger("trace")

_PREVIEW_LIMIT = 200


@dataclass
class TraceEvent:
    """One structured event in the agent's execution trace.

    Attributes:
        type: Event kind (e.g. ``"tool_call_start"``). See the module
            docstring for the catalog of types emitted by the agent.
        span_id: Unique id of THIS event. Use it as the ``parent_id``
            of events that conceptually nest inside it.
        parent_id: Span id of the enclosing event, or ``None`` for the
            root event of a run.
        ts: Unix timestamp (seconds, float) at emit time.
        payload: Event-specific data. Keys are stable per event type.
    """

    type: str
    span_id: str
    parent_id: str | None
    ts: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "ts": self.ts,
            "payload": self.payload,
        }


OnEvent = Callable[[TraceEvent], None]


class TraceEmitter:
    """Receives lifecycle events from an `Agent` and dispatches them.

    Pass ``file`` to append each event as a JSON line; pass ``on_event``
    to receive each event as a Python object; pass both. Pass neither
    and the emitter is a structural no-op (useful for tests).

    Args:
        file: Path or already-open text file-like object (``.write(str)``
            supported). When a path is given, the emitter opens it in
            append mode and closes it on ``close()`` / context-manager
            exit. When an open file is given, the host retains
            ownership (we never close it).
        on_event: Synchronous callback invoked for every event.
            Exceptions raised by the callback are caught and logged
            via ``autoagent.trace`` — they cannot break the agent loop.
        clock: Optional clock function for tests. Defaults to
            ``time.time``.
    """

    def __init__(
        self,
        *,
        file: str | Path | IO[str] | None = None,
        on_event: OnEvent | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._on_event = on_event
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._file: IO[str] | None
        self._owns_file: bool
        if file is None:
            self._file = None
            self._owns_file = False
        elif hasattr(file, "write"):
            self._file = file  # type: ignore[assignment]
            self._owns_file = False
        else:
            # The handle stays open for the emitter's lifetime — we
            # cannot wrap this in `with` because every emit() reuses it.
            # `close()` (or the context-manager exit) releases it.
            self._file = open(file, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
            self._owns_file = True

    def emit(
        self,
        type_: str,
        payload: dict[str, Any] | None = None,
        *,
        parent_id: str | None = None,
    ) -> str:
        """Emit one event and return its ``span_id``.

        The span_id is a short hex token (16 hex chars). Use it as the
        ``parent_id`` of nested events so a consumer can reconstruct a
        tree later.

        File-write errors, callback exceptions, and even failures in
        the user-supplied ``clock`` are caught and logged; this method
        never raises. A best-effort fallback span id is returned if
        the secrets module itself fails (extremely unlikely).

        Span id and timestamp are computed INSIDE the lock so that
        events appear in the JSONL trace in their true generation
        order, even under concurrent emits.
        """
        with self._lock:
            try:
                span_id = secrets.token_hex(8)
            except Exception:
                _log.exception("secrets.token_hex failed; using fallback id")
                span_id = "0" * 16
            try:
                ts = self._clock()
            except Exception:
                _log.exception("trace clock raised; falling back to time.time()")
                ts = time.time()
            event = TraceEvent(
                type=type_,
                span_id=span_id,
                parent_id=parent_id,
                ts=ts,
                payload=payload or {},
            )
            if self._file is not None:
                try:
                    self._file.write(json.dumps(event.to_dict(), default=str) + "\n")
                except Exception:
                    _log.exception("trace file write failed")
            if self._on_event is not None:
                try:
                    self._on_event(event)
                except Exception:
                    _log.exception("trace on_event callback failed")
        return span_id

    def close(self) -> None:
        """Close the underlying file if we opened it. Idempotent."""
        with self._lock:
            if self._owns_file and self._file is not None:
                try:
                    self._file.close()
                finally:
                    self._file = None
                    self._owns_file = False

    def __enter__(self) -> TraceEmitter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def truncate_preview(value: Any, limit: int = _PREVIEW_LIMIT) -> str:
    """Render any value as a short, secret-redacted string for a trace payload.

    Used internally by the agent to keep tool-call / message previews
    bounded and free of obvious credentials. The same redaction rules
    that protect log records via ``SecretRedactingFilter`` are applied
    here:

    * ``Bearer <token>`` in any header-shaped string
    * ``x-api-key`` / ``x-goog-api-key`` / ``api_key`` / ``api-key``
      in JSON-, dict-, or header-style serialisations
    * legacy URL form ``?key=...`` (and ``key=...`` more generally)

    Exposed publicly so host callbacks can apply the same rule when
    they augment events with custom payload fields.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=repr)
        except Exception:
            text = repr(value)
    text = redact(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


__all__ = ["OnEvent", "TraceEmitter", "TraceEvent", "truncate_preview"]
