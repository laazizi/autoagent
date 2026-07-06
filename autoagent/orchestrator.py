"""Host-driven orchestration — deterministic flow, LLM as executor (0.9.0).

``Agent`` (agent.py) hands control to the model : the LLM decides which
tool to call and when the task is done. That is the right shape for
open-ended work (explore, code, search). It is the WRONG shape for a
certified process — a survey, a wizard, a checkout, an intake form —
where every step, transition and recorded value must follow rules
exactly. However strict the prompt, an autonomous agent keeps residual
freedom to drift : ask two questions, skip a step, claim it recorded
something it didn't.

``Orchestrator`` inverts control. The HOST owns the state machine :

    ┌───────────────────────────────────────────────────────────┐
    │ HOST (your code — deterministic)                          │
    │   current_steps()  → the one current step (+ horizon)     │
    │   record(id, val)  → validate + store ; advancing or not  │
    │                      is a CONSEQUENCE of what you store   │
    └──────┬───────────────────────────────────┬────────────────┘
           │ LLM micro-task 1                  │ LLM micro-task 2
           ▼                                   ▼
    INTERPRET the user's reply          PHRASE the current step
    into typed values (strict JSON)     in natural language (streamed)

The LLM cannot advance, skip, or invent : it never even sees the steps
the host didn't expose. A garbage LLM output degrades ONE turn's
wording — never the flow. This module owns the turn mechanics (LLM
calls, JSON-parse robustness, faithful acknowledgments, anti-loop
escalation, event protocol) ; everything domain-specific (step
selection, validation, prompt language) is injected by the host.

Minimal example::

    fields = ["name", "email"]
    answers: dict[str, str] = {}

    orch = Orchestrator(
        provider,
        current_steps=lambda: [
            Step(id=f, payload={"ask": f})
            for f in fields if f not in answers
        ][:1],
        record=lambda sid, val: answers.__setitem__(sid, str(val)),
    )
    for ev in orch.turn("my name is Ana"):
        if ev.type == "text":
            print(ev.text, end="")

Persistence note : ``turn()`` is synchronous and stateless apart from
the anti-loop counters (``stuck_slot`` / ``stuck_count``). Hosts that
rebuild the Orchestrator per HTTP request should save those two fields
with their session state and restore them on construction.

Proven in ``examples/cati_chat`` (a 100+ question certified mobility
survey with conditional filters, nested loops and typed validation).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .logging import get_logger
from .providers.base import LLMProvider
from .schema import LLMRequest, Message

__all__ = [
    "DEFAULT_INTERPRET_SYSTEM",
    "DEFAULT_PHRASE_SYSTEM",
    "InterpretOutcome",
    "Orchestrator",
    "PhraseSignals",
    "Step",
    "TurnEvent",
]

_log = get_logger("orchestrator")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One step of a host-driven flow.

    ``id`` is the host's stable identifier for the slot (e.g.
    ``"menage.INT1#0"`` or ``"email"``). ``payload`` is what the LLM
    needs to interpret replies / phrase the step : question text,
    options, expected type, bounds — any JSON-safe dict.
    """

    id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnEvent:
    """Event emitted by ``Orchestrator.turn``.

    types :
      * ``text`` — a chunk of the reply to show/stream to the user.
      * ``recorded`` — a value was validated + stored (``step_id``,
        ``value``). Observability ; the host's ``record`` already ran.
      * ``done`` — turn finished. ``flow_complete`` is True when
        ``current_steps()`` returned empty (the whole flow is over).
    """

    type: Literal["text", "recorded", "done"]
    text: str = ""
    step_id: str = ""
    value: Any = None
    flow_complete: bool = False


@dataclass
class PhraseSignals:
    """What happened this turn — input for the host's phrase context.

    The host maps these to its own context keys (and language) for the
    phrasing prompt. ``status`` is ``answered`` / ``unclear`` /
    ``offtopic``. ``ack`` is a faithful, human-readable list of what was
    ACTUALLY recorded (never claim more). ``stuck_count`` counts
    consecutive non-answers on the same step — at 2+ the phrasing
    should change strategy entirely instead of repeating itself.
    """

    status: str = "answered"
    ack: str | None = None
    offtopic_note: str | None = None
    validation_error: str | None = None
    user_verbatim: str | None = None
    stuck_count: int = 0


@dataclass
class InterpretOutcome:
    status: str  # answered | unclear | offtopic
    values: list[tuple[str, Any]] = field(default_factory=list)  # (step_id, value)
    note: str = ""


# ---------------------------------------------------------------------------
# Default prompts (English, domain-neutral). Hosts usually override.
# ---------------------------------------------------------------------------

DEFAULT_INTERPRET_SYSTEM = """You are an extraction module for a step-driven flow.
You receive the CURRENT step (with its expected answer shape), possibly UPCOMING steps, and the user's raw reply.
Convert the reply into typed values. Respond with STRICT JSON only :

{"status": "answered", "values": [{"id": "<step id>", "value": <typed>}, ...]}
or {"status": "unclear"}
or {"status": "offtopic", "note": "<short summary of what the user said>"}
or {"status": "refused"}

Rules :
- The FIRST entry of values must answer the current step.
- If the reply also answers some UPCOMING steps, add entries for them.
- "unclear" when you cannot map the reply to the current step.
- "offtopic" when the user changed subject or asked something.
- "refused" when the user EXPLICITLY declines to answer this step.
STRICT JSON only."""

DEFAULT_PHRASE_SYSTEM = """You are the voice of a step-driven assistant.
You receive the official current step and context signals. Reword the step naturally —
short, clear, ONE question or instruction, then stop.
- Never answer in the user's place ; never invent steps.
- If `ack` lists what was just recorded, briefly acknowledge it (never claim more).
- If `validation_error` is present, explain simply what is expected, then re-ask.
- If `user_verbatim` is present, respond to what the user said before re-asking.
- If `escalation` is present, change strategy completely : do not repeat your previous wording."""


def _default_interpret_payload(steps: Sequence[Step], user_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "current_step": {"id": steps[0].id, **steps[0].payload},
        "user_reply": user_text,
    }
    if len(steps) > 1:
        out["upcoming_steps"] = [{"id": s.id, **s.payload} for s in steps[1:]]
    return out


def _default_parse_values(data: dict[str, Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for entry in data.get("values") or []:
        if isinstance(entry, dict) and "id" in entry:
            out.append((str(entry["id"]), entry.get("value")))
    return out


def _default_phrase_context(step: Step, signals: PhraseSignals) -> dict[str, Any]:
    ctx: dict[str, Any] = {"step": {"id": step.id, **step.payload}}
    if signals.ack:
        ctx["ack"] = signals.ack
    if signals.status == "refused":
        ctx["refusal_note"] = (
            "the user declined the previous step; briefly confirm it is "
            "skipped, then ask this step"
        )
    elif signals.status == "offtopic":
        ctx["refocus_note"] = signals.offtopic_note or "the user changed subject"
    elif signals.status == "unclear" and not signals.validation_error:
        ctx["refocus_note"] = "the reply was unclear; re-ask kindly"
    if signals.validation_error:
        ctx["validation_error"] = signals.validation_error
    if signals.user_verbatim:
        ctx["user_verbatim"] = signals.user_verbatim
    if signals.stuck_count >= 2:
        ctx["escalation"] = (
            f"Attempt #{signals.stuck_count} on this same step. Do NOT repeat "
            "your previous wording : address what the user said, then present "
            "the possible answers one by one."
        )
    return ctx


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Deterministic flow driver ; the LLM interprets and phrases, only.

    Args:
        provider: Any ``LLMProvider`` (use ``create_provider``). Both
            micro-tasks run on it ; pass a cheap/fast model.
        current_steps: Zero-arg callable returning the CURRENT step
            first, optionally followed by a small horizon of upcoming
            steps the interpreter may fill from compound replies
            (« Julie 55 and me, 54 »). Empty sequence = flow complete.
            Called again after recording — advancement is purely a
            consequence of what the host stored.
        record: ``(step_id, value) -> error_message | None``. Validate
            and store. Return a human-readable error string to REJECT
            the value (the step stays current and the error is phrased
            back) ; return None on success. Only ids exposed by
            ``current_steps`` are ever passed.
        describe: Optional ``(step_id, value) -> str`` for the faithful
            acknowledgment line. Default : ``"id = value"``. Resolve IDs
            to human labels here — phrasing models misread raw codes.
        phrase_context: Optional ``(step, signals) -> dict`` building
            the full context for the phrasing prompt (host controls key
            names and language). Default : English keys.
        interpret_payload: Optional ``(steps, user_text) -> dict``
            building the interpreter input. Default : English keys.
        parse_values: Optional ``(json_dict) -> [(step_id, value)]``
            mapping the interpreter's JSON to step ids. Override when
            your prompt uses a different value shape.
        interpret_system / phrase_system: System prompts for the two
            micro-tasks (defaults are English + domain-neutral).
        closing_text: Emitted as the final reply when the flow is over.
        on_offtopic: Optional callback invoked with the off-topic note
            (host-side counters, audit).
        on_refused: Optional ``(step_id) -> None`` invoked when the user
            EXPLICITLY refuses to answer. Mark the slot skipped / store a
            refusal code so ``current_steps()`` advances — refusing is a
            right, not an obstacle to argue with. Without the hook the
            step stays current (for flows with mandatory steps).
        interpret_temperature / phrase_temperature: Per-task sampling.
        stuck_slot / stuck_count: Restore anti-loop state persisted by
            the host (read the attributes back after each turn).
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        current_steps: Callable[[], Sequence[Step]],
        record: Callable[[str, Any], str | None],
        describe: Callable[[str, Any], str] | None = None,
        phrase_context: Callable[[Step, PhraseSignals], dict[str, Any]] | None = None,
        interpret_payload: Callable[[Sequence[Step], str], dict[str, Any]] | None = None,
        parse_values: Callable[[dict[str, Any]], list[tuple[str, Any]]] | None = None,
        interpret_system: str = DEFAULT_INTERPRET_SYSTEM,
        phrase_system: str = DEFAULT_PHRASE_SYSTEM,
        closing_text: str = "That was the last step — thank you!",
        on_offtopic: Callable[[str], None] | None = None,
        on_refused: Callable[[str], None] | None = None,
        accept_extra: Callable[[str], bool] | None = None,
        interpret_temperature: float = 0.0,
        phrase_temperature: float = 0.6,
        stuck_slot: str | None = None,
        stuck_count: int = 0,
    ) -> None:
        self.provider = provider
        self.current_steps = current_steps
        self.record = record
        self.describe = describe or (lambda sid, val: f"{sid} = {val}")
        self.phrase_context = phrase_context or _default_phrase_context
        self.interpret_payload = interpret_payload or _default_interpret_payload
        self.parse_values = parse_values or _default_parse_values
        self.interpret_system = interpret_system
        self.phrase_system = phrase_system
        self.closing_text = closing_text
        self.on_offtopic = on_offtopic
        self.on_refused = on_refused
        # Optional escape hatch to the « only slots the host exposed »
        # rule : when provided, a value whose step_id is OUTSIDE the
        # current steps is still recorded if accept_extra(step_id) is
        # True. Hosts use it for CORRECTIONS of already-answered slots
        # (« actually we have 2 cars ») — never for forward drift.
        self.accept_extra = accept_extra
        self.interpret_temperature = interpret_temperature
        self.phrase_temperature = phrase_temperature
        self.stuck_slot = stuck_slot
        self.stuck_count = stuck_count

    # -- micro-task 1 : interpret ----------------------------------------

    def interpret(self, steps: Sequence[Step], user_text: str) -> InterpretOutcome:
        """Run the interpreter LLM call. Failure-safe : any malformed
        output collapses to ``unclear`` (the flow stays put)."""
        request = LLMRequest(
            messages=[
                Message(role="system", content=self.interpret_system),
                Message(
                    role="user",
                    content=json.dumps(
                        self.interpret_payload(steps, user_text), ensure_ascii=False
                    ),
                ),
            ],
            temperature=self.interpret_temperature,
        )
        try:
            raw = (self.provider.complete(request).content or "").strip()
            raw = _FENCE_RE.sub("", raw)
            data = json.loads(raw)
        except Exception:
            _log.warning("interpret call failed or returned non-JSON; treating as unclear")
            return InterpretOutcome(status="unclear")
        status = data.get("status")
        if status == "answered":
            values = self.parse_values(data)
            if not values:
                return InterpretOutcome(status="unclear")
            return InterpretOutcome(status="answered", values=values)
        if status == "offtopic":
            return InterpretOutcome(status="offtopic", note=str(data.get("note") or ""))
        if status == "refused":
            return InterpretOutcome(status="refused", note=str(data.get("note") or ""))
        return InterpretOutcome(status="unclear")

    # -- micro-task 2 : phrase (streamed) --------------------------------

    def phrase_stream(self, step: Step, signals: PhraseSignals) -> Iterator[str]:
        request = LLMRequest(
            messages=[
                Message(role="system", content=self.phrase_system),
                Message(
                    role="user",
                    content=json.dumps(
                        self.phrase_context(step, signals), ensure_ascii=False
                    ),
                ),
            ],
            temperature=self.phrase_temperature,
        )
        for chunk in self.provider.stream(request):
            if chunk.type == "text" and chunk.text:
                yield chunk.text

    # -- the turn ---------------------------------------------------------

    def turn(self, user_text: str) -> Iterator[TurnEvent]:
        """Process one user message. The host's state machine decides
        everything ; see module docstring for the contract."""
        steps = list(self.current_steps())
        if not steps:
            yield TurnEvent(type="text", text=self.closing_text)
            yield TurnEvent(type="done", flow_complete=True)
            return

        outcome = self.interpret(steps, user_text)
        signals = PhraseSignals(status=outcome.status)

        if outcome.status == "answered":
            exposed = {s.id for s in steps}
            current_id = steps[0].id
            current_recorded = False
            described: list[str] = []
            for step_id, value in outcome.values:
                if step_id not in exposed and not (
                    self.accept_extra is not None and self.accept_extra(step_id)
                ):
                    continue  # the host only accepts slots IT exposed
                error = None
                try:
                    error = self.record(step_id, value)
                except Exception as exc:
                    _log.exception("record(%s) raised", step_id)
                    error = str(exc)
                if error:
                    if step_id == current_id:
                        signals.validation_error = error
                    continue  # horizon extras that fail are dropped
                if step_id == current_id:
                    current_recorded = True
                described.append(self.describe(step_id, value))
                yield TurnEvent(type="recorded", step_id=step_id, value=value)
            if current_recorded:
                signals.ack = "; ".join(described[:10])
            elif described:
                # Nothing for the current slot but extras WERE recorded
                # (e.g. a correction of a past answer) : acknowledge them
                # and keep status answered — the flow re-walks from the
                # host's state, which may now re-present invalidated steps.
                signals.ack = "; ".join(described[:10])
            elif signals.validation_error is None:
                # Answered, but not the current slot → effectively unclear.
                signals.status = "unclear"

        if signals.status == "offtopic":
            signals.offtopic_note = outcome.note
            if self.on_offtopic is not None:
                try:
                    self.on_offtopic(outcome.note)
                except Exception:
                    _log.exception("on_offtopic hook raised; continuing")

        # An explicit refusal is an ANSWER, not an obstacle : the
        # respondent has the right to decline. Re-asking (observed in
        # production : 3 polite-but-insistent re-asks) is both bad UX
        # and a compliance issue. The host decides what « skipping »
        # means (mark the slot, store a refusal code, ...) via
        # ``on_refused`` ; without the hook the step stays current
        # (some flows have steps that CANNOT be skipped).
        if signals.status == "refused" and self.on_refused is not None:
            try:
                self.on_refused(steps[0].id)
            except Exception:
                _log.exception("on_refused hook raised; continuing")

        # Anti-loop : consecutive non-answers on the SAME step escalate
        # the phrasing strategy instead of repeating a canned re-ask.
        # A handled refusal advances the flow, so it does NOT count.
        if signals.status == "refused" and self.on_refused is not None:
            self.stuck_slot = None
            self.stuck_count = 0
        elif signals.status in ("unclear", "offtopic") or signals.validation_error:
            if self.stuck_slot == steps[0].id:
                self.stuck_count += 1
            else:
                self.stuck_slot = steps[0].id
                self.stuck_count = 1
            signals.user_verbatim = user_text[:300]
        else:
            self.stuck_slot = None
            self.stuck_count = 0
        signals.stuck_count = self.stuck_count

        next_steps = list(self.current_steps())
        if not next_steps:
            yield TurnEvent(type="text", text=self.closing_text)
            yield TurnEvent(type="done", flow_complete=True)
            return

        for chunk in self.phrase_stream(next_steps[0], signals):
            yield TurnEvent(type="text", text=chunk)
        yield TurnEvent(type="done", flow_complete=False)
