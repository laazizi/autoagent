"""OTelTraceExporter — event→span mapping, statuses, leak guard.

The OpenTelemetry API is injected as a fake (`_api` seam) so the suite
runs without the optional dependency installed.
"""

from __future__ import annotations

import sys

import pytest

from autoagent.errors import AutoAgentError
from autoagent.otel import OTelTraceExporter
from autoagent.trace import TraceEmitter


class FakeStatusCode:
    ERROR = "ERROR"
    OK = "OK"


class FakeStatus:
    def __init__(self, code, description=""):
        self.code = code
        self.description = description


class FakeSpan:
    def __init__(self, name, context, start_time, attributes):
        self.name = name
        self.context = context
        self.start_time = start_time
        self.attributes = dict(attributes or {})
        self.events = []
        self.status = None
        self.end_time = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def add_event(self, name, attributes=None, timestamp=None):
        self.events.append((name, dict(attributes or {}), timestamp))

    def end(self, end_time=None):
        self.end_time = end_time


class FakeTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name, context=None, start_time=None, attributes=None):
        span = FakeSpan(name, context, start_time, attributes)
        self.spans.append(span)
        return span


class FakeAPI:
    Status = FakeStatus
    StatusCode = FakeStatusCode

    def get_tracer(self, name):
        return FakeTracer()

    def set_span_in_context(self, span):
        return ("ctx", span)


@pytest.fixture()
def exporter():
    tracer = FakeTracer()
    return OTelTraceExporter(tracer, _api=FakeAPI()), tracer


def _drive_full_run(exporter, clock=None):
    """Replay the event tree exactly as the agent loop emits it."""
    emitter = TraceEmitter(on_event=exporter, clock=clock)
    run = emitter.emit("run_start", {"model": "fake", "max_steps": 8})
    req = emitter.emit("llm_request", {"step": 1}, parent_id=run)
    tool = emitter.emit("tool_call_start", {"name": "lire_fichier", "call_id": "c1"}, parent_id=req)
    emitter.emit("tool_call_end", {"name": "lire_fichier", "status": "ok", "duration_ms": 12},
                 parent_id=tool)
    emitter.emit("llm_response", {"step": 1, "tool_call_count": 1}, parent_id=req)
    emitter.emit("run_end", {"status": "ok", "steps": 1}, parent_id=run)
    return emitter


def test_full_run_builds_nested_closed_spans(exporter):
    export, tracer = exporter
    ticks = iter(range(100, 200))
    _drive_full_run(export, clock=lambda: float(next(ticks)))

    assert [s.name for s in tracer.spans] == ["agent.run", "llm", "tool.lire_fichier"]
    run, llm, tool = tracer.spans

    # parenting: llm under run, tool under llm
    assert run.context is None
    assert llm.context == ("ctx", run)
    assert tool.context == ("ctx", llm)

    # all closed, chronologically consistent (ns timestamps)
    assert run.end_time and llm.end_time and tool.end_time
    assert run.start_time == 100 * 10**9
    assert tool.start_time < tool.end_time < llm.end_time < run.end_time

    # payload → prefixed attributes; end-event payload merged in
    assert run.attributes["autoagent.model"] == "fake"
    assert run.attributes["autoagent.steps"] == 1
    assert tool.attributes["autoagent.duration_ms"] == 12

    # statuses
    assert run.status.code == "OK"
    assert tool.status.code == "OK"


def test_error_statuses_mark_spans_as_error(exporter):
    export, tracer = exporter
    emitter = TraceEmitter(on_event=export)
    run = emitter.emit("run_start", {})
    tool = emitter.emit("tool_call_start", {"name": "boom"}, parent_id=run)
    emitter.emit("tool_call_end", {"name": "boom", "status": "error"}, parent_id=tool)
    emitter.emit("run_end", {"status": "max_steps"}, parent_id=run)

    run_span, tool_span = tracer.spans
    assert tool_span.status.code == "ERROR"
    assert run_span.status.code == "ERROR"
    assert run_span.status.description == "max_steps"


def test_point_events_attach_to_open_span(exporter):
    export, tracer = exporter
    emitter = TraceEmitter(on_event=export)
    run = emitter.emit("run_start", {})
    emitter.emit("cancelled", {"step": 3}, parent_id=run)
    emitter.emit("run_end", {"status": "cancelled"}, parent_id=run)

    (run_span,) = tracer.spans
    assert run_span.events == [("cancelled", {"autoagent.step": 3}, run_span.events[0][2])]
    assert run_span.status.code == "ERROR"


def test_orphan_events_are_dropped_not_raised(exporter):
    export, tracer = exporter
    emitter = TraceEmitter(on_event=export)
    emitter.emit("tool_call_end", {"status": "ok"}, parent_id="nope")   # closer sans opener
    emitter.emit("cancelled", {})                                        # point event sans parent
    assert tracer.spans == []


def test_close_ends_leftover_spans(exporter):
    export, tracer = exporter
    emitter = TraceEmitter(on_event=export)
    emitter.emit("run_start", {})
    with export:
        pass
    (run_span,) = tracer.spans
    assert run_span.end_time is not None
    assert run_span.status is None  # interrupted, not judged


def test_broken_tracer_never_breaks_the_agent_loop():
    class BrokenTracer:
        def start_span(self, *a, **k):
            raise RuntimeError("otel backend down")

    export = OTelTraceExporter(BrokenTracer(), _api=FakeAPI())
    emitter = TraceEmitter(on_event=export)
    emitter.emit("run_start", {})  # must not raise


def test_missing_opentelemetry_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)
    with pytest.raises(AutoAgentError, match="opentelemetry"):
        OTelTraceExporter()
