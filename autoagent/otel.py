"""OpenTelemetry exporter for autoagent traces — optional dependency.

``OTelTraceExporter`` is an ``on_event`` callback for ``TraceEmitter``
that rebuilds the agent's span tree as real OpenTelemetry spans, so a
run shows up in Jaeger / Grafana Tempo / Langfuse / Phoenix — anything
that speaks OTLP — with proper nesting and durations:

    agent.run ─ llm (step 1) ─ tool.lire_fichier
              └ llm (step 2) ─ tool.ecrire_rapport

The mapping mirrors how the agent emits events (see ``trace.py``):

* ``run_start`` / ``llm_request`` / ``tool_call_start`` OPEN a span;
  the closing event (``run_end`` / ``llm_response`` / ``tool_call_end``)
  points at the opener via its ``parent_id``, which is how they pair.
* Everything else (``cancelled``, ``max_steps_exceeded``, post-turn-hook
  events, custom host events…) becomes a point-in-time span *event* on
  the nearest open span.
* Payload fields become span attributes under the ``autoagent.`` prefix.
  Previews are already secret-redacted by ``TraceEmitter``.

The ``opentelemetry-api`` package is imported lazily at construction —
the core library keeps its zero-dependency contract. Install the SDK to
actually export somewhere::

    pip install opentelemetry-sdk opentelemetry-exporter-otlp

Typical use (the host owns the OTel setup, as usual with OTel)::

    from opentelemetry import trace as ot
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    ot.set_tracer_provider(TracerProvider())
    ot.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    from autoagent import TraceEmitter
    from autoagent.otel import OTelTraceExporter

    with OTelTraceExporter() as exporter:
        trace = TraceEmitter(on_event=exporter)      # JSONL file optional, both work
        agent = Agent.from_model("gemini", "gemini-3.5-flash", trace=trace)
        agent.run("...")

Share ONE ``TraceEmitter`` between an agent and its ``as_tool``
sub-agents and the whole swarm lands in a single trace tree — same rule
as the JSONL file.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .errors import AutoAgentError
from .logging import get_logger
from .trace import TraceEvent

__all__ = ["OTelTraceExporter"]

_log = get_logger(__name__)

# Events that OPEN a span, and the span name they produce. The name for
# tool_call_start is refined with the tool name from the payload.
_OPENERS = {
    "run_start": "agent.run",
    "llm_request": "llm",
    "tool_call_start": "tool",
}
# Events that CLOSE the span their parent_id points at.
_CLOSERS = {"run_end", "llm_response", "tool_call_end"}
# Payload values that mean the span ended badly.
_ERROR_STATUSES = {"error", "cancelled", "max_steps"}

_MAX_OPEN_SPANS = 10_000  # leak guard if an end event never arrives


class _OTelAPI:
    """The thin slice of the OpenTelemetry API the exporter needs.

    Kept behind one object so tests can inject a stand-in and the real
    import stays lazy (constructor time, not module import time).
    """

    def __init__(self) -> None:
        try:
            from opentelemetry import trace as ot_trace
            from opentelemetry.trace import Status, StatusCode, set_span_in_context
        except ImportError as exc:
            raise AutoAgentError(
                "OTelTraceExporter requires the 'opentelemetry-api' package "
                "(pip install opentelemetry-sdk opentelemetry-exporter-otlp). "
                "The autoagent core stays dependency-free — OpenTelemetry is "
                "only needed by this exporter."
            ) from exc
        self.get_tracer = ot_trace.get_tracer
        self.set_span_in_context = set_span_in_context
        self.Status = Status
        self.StatusCode = StatusCode


class OTelTraceExporter:
    """``TraceEmitter`` callback that mirrors agent events as OTel spans.

    Args:
        tracer: An already-built OTel tracer. Default: the global one
            (``opentelemetry.trace.get_tracer(tracer_name)``) — a no-op
            tracer when the host has not configured a provider, so the
            exporter is always safe to attach.
        tracer_name: Instrumentation name for the default tracer.

    Use it as the ``on_event`` of a ``TraceEmitter`` (it is callable),
    alone or alongside the JSONL file. ``close()`` (or the context
    manager exit) ends any span left open by an interrupted run.

    Thread-safety: calls arrive serialized by the emitter's lock, but
    the exporter keeps its own lock so several emitters can share one
    instance.
    """

    def __init__(
        self,
        tracer: Any | None = None,
        *,
        tracer_name: str = "autoagent",
        _api: Any | None = None,
    ) -> None:
        self._api = _api if _api is not None else _OTelAPI()
        self._tracer = tracer if tracer is not None else self._api.get_tracer(tracer_name)
        self._open: dict[str, Any] = {}
        self._lock = threading.Lock()

    def __call__(self, event: TraceEvent) -> None:
        try:
            self._handle(event)
        except Exception:
            # Same contract as TraceEmitter callbacks: observability can
            # never break the agent loop.
            _log.exception("otel export failed for event %r", event.type)

    def _handle(self, event: TraceEvent) -> None:
        ts_ns = int(event.ts * 1_000_000_000)
        with self._lock:
            if event.type in _OPENERS:
                if len(self._open) >= _MAX_OPEN_SPANS:
                    _log.warning("otel: too many open spans; dropping %r", event.type)
                    return
                name = _OPENERS[event.type]
                if event.type == "tool_call_start" and event.payload.get("name"):
                    name = f"tool.{event.payload['name']}"
                parent = self._open.get(event.parent_id) if event.parent_id else None
                context = self._api.set_span_in_context(parent) if parent is not None else None
                span = self._tracer.start_span(
                    name,
                    context=context,
                    start_time=ts_ns,
                    attributes=_attributes(event),
                )
                self._open[event.span_id] = span
                return

            if event.type in _CLOSERS:
                span = self._open.pop(event.parent_id, None) if event.parent_id else None
                if span is None:
                    _log.debug("otel: closer %r without open span", event.type)
                    return
                for key, value in _attributes(event).items():
                    span.set_attribute(key, value)
                status = event.payload.get("status")
                if status in _ERROR_STATUSES:
                    span.set_status(self._api.Status(self._api.StatusCode.ERROR, str(status)))
                elif event.type in ("run_end", "tool_call_end"):
                    span.set_status(self._api.Status(self._api.StatusCode.OK))
                span.end(end_time=ts_ns)
                return

            # Point-in-time event: attach to the nearest open span.
            target = self._open.get(event.parent_id) if event.parent_id else None
            if target is not None:
                target.add_event(event.type, attributes=_attributes(event), timestamp=ts_ns)
            else:
                _log.debug("otel: event %r has no open span; dropped", event.type)

    def close(self) -> None:
        """End every span still open (interrupted runs). Idempotent."""
        now_ns = time.time_ns()
        with self._lock:
            spans = list(self._open.values())
            self._open.clear()
        for span in spans:
            try:
                span.end(end_time=now_ns)
            except Exception:
                _log.exception("otel: failed to end leftover span")

    def __enter__(self) -> "OTelTraceExporter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _attributes(event: TraceEvent) -> dict[str, Any]:
    """Flatten a payload into OTel-legal attributes (autoagent.* keys)."""
    attributes: dict[str, Any] = {}
    for key, value in event.payload.items():
        if value is None:
            continue
        if not isinstance(value, (str, bool, int, float)):
            value = str(value)
        attributes[f"autoagent.{key}"] = value
    return attributes
