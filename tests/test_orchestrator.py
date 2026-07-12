"""Tests for `autoagent.orchestrator` (0.9.0).

Public contract — the determinism guarantees :

* The HOST decides the flow : `current_steps()` is the only source of
  position ; advancement is a consequence of what `record` stores.
* A valid reply records + advances ; `unclear` / `offtopic` / garbage
  LLM output leaves the flow EXACTLY where it was.
* The interpreter may fill several exposed slots from one compound
  reply, but ids the host didn't expose are silently refused.
* `record` returning an error string rejects the value (the step stays
  current and the error reaches the phrasing context).
* Consecutive non-answers on the same step raise `stuck_count` and add
  an escalation note to the phrasing context ; a real answer resets.
* Empty `current_steps()` → closing text + `flow_complete=True`.
"""

from __future__ import annotations

import json
from typing import Any

from autoagent.orchestrator import Orchestrator, Step
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, StreamChunk


class FakeFlowProvider:
    """Routes on the system prompt : scripted interpreter, echo phraser."""

    def __init__(self) -> None:
        self.config = ModelConfig(provider="fake", model="fake")
        self.next_interpret = '{"status":"unclear"}'
        self.phrase_contexts: list[dict[str, Any]] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=self.next_interpret)

    def stream(self, request: LLMRequest):
        ctx = json.loads(request.messages[1].content)
        self.phrase_contexts.append(ctx)
        text = f"[ask:{ctx['step']['id']}]"
        yield StreamChunk(type="text", text=text)
        yield StreamChunk(type="final", response=LLMResponse(content=text))


def make_flow():
    """A 3-field linear flow with one validating field."""
    fields = ["name", "age", "city"]
    answers: dict[str, Any] = {}

    def current_steps():
        remaining = [f for f in fields if f not in answers]
        return [Step(id=f, payload={"ask": f}) for f in remaining[:2]]

    def record(step_id: str, value: Any):
        if step_id == "age":
            try:
                v = int(value)
            except (TypeError, ValueError):
                return "age must be a number"
            if not 0 <= v <= 140:
                return "age out of range"
            answers[step_id] = v
            return None
        answers[step_id] = value
        return None

    return answers, current_steps, record


def drive(orch: Orchestrator, text: str):
    events = list(orch.turn(text))
    out_text = "".join(e.text for e in events if e.type == "text")
    recorded = [(e.step_id, e.value) for e in events if e.type == "recorded"]
    done = [e for e in events if e.type == "done"][-1]
    return out_text, recorded, done


class TestDeterminism:
    def test_valid_answer_records_and_advances(self) -> None:
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        orch = Orchestrator(p, current_steps=steps, record=record)
        p.next_interpret = '{"status":"answered","values":[{"id":"name","value":"Ana"}]}'
        text, recorded, done = drive(orch, "my name is Ana")
        assert answers == {"name": "Ana"}
        assert recorded == [("name", "Ana")]
        assert "[ask:age]" in text  # advanced to the next step
        assert done.flow_complete is False

    def test_unclear_stays_put(self) -> None:
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        orch = Orchestrator(p, current_steps=steps, record=record)
        p.next_interpret = '{"status":"unclear"}'
        text, recorded, _ = drive(orch, "uh?")
        assert answers == {}
        assert recorded == []
        assert "[ask:name]" in text  # same step re-asked

    def test_garbage_llm_output_is_safe(self) -> None:
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        orch = Orchestrator(p, current_steps=steps, record=record)
        p.next_interpret = "NOT JSON {{{"
        text, recorded, _ = drive(orch, "whatever")
        assert answers == {}
        assert "[ask:name]" in text

    def test_unexposed_slot_refused(self) -> None:
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        orch = Orchestrator(p, current_steps=steps, record=record)
        # 'city' is NOT in the first 2-step horizon (name, age).
        p.next_interpret = (
            '{"status":"answered","values":'
            '[{"id":"name","value":"Ana"},{"id":"city","value":"Lyon"}]}'
        )
        drive(orch, "Ana, and I live in Lyon")
        assert answers == {"name": "Ana"}  # city refused

    def test_compound_reply_fills_exposed_horizon(self) -> None:
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        orch = Orchestrator(p, current_steps=steps, record=record)
        p.next_interpret = (
            '{"status":"answered","values":'
            '[{"id":"name","value":"Ana"},{"id":"age","value":33}]}'
        )
        text, recorded, _ = drive(orch, "Ana, 33 years old")
        assert answers == {"name": "Ana", "age": 33}
        assert len(recorded) == 2
        assert "[ask:city]" in text  # skipped straight to the third field

    def test_validation_error_keeps_step_and_reaches_phrasing(self) -> None:
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        orch = Orchestrator(p, current_steps=steps, record=record)
        p.next_interpret = '{"status":"answered","values":[{"id":"name","value":"Ana"}]}'
        drive(orch, "Ana")
        p.next_interpret = '{"status":"answered","values":[{"id":"age","value":999}]}'
        text, recorded, _ = drive(orch, "999")
        assert "age" not in answers
        assert recorded == []
        assert "[ask:age]" in text
        assert p.phrase_contexts[-1].get("validation_error") == "age out of range"

    def test_flow_complete_emits_closing(self) -> None:
        answers, steps, record = make_flow()
        answers.update({"name": "x", "age": 1, "city": "y"})
        p = FakeFlowProvider()
        orch = Orchestrator(p, current_steps=steps, record=record,
                            closing_text="All done, thanks!")
        text, _, done = drive(orch, "anything")
        assert text == "All done, thanks!"
        assert done.flow_complete is True


class TestAntiLoop:
    def test_stuck_counter_and_escalation(self) -> None:
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        orch = Orchestrator(p, current_steps=steps, record=record)
        p.next_interpret = '{"status":"unclear"}'
        drive(orch, "blah")
        assert orch.stuck_count == 1
        assert "escalation" not in p.phrase_contexts[-1]
        assert p.phrase_contexts[-1].get("user_verbatim") == "blah"
        drive(orch, "blah again")
        assert orch.stuck_count == 2
        assert "escalation" in p.phrase_contexts[-1]
        # A real answer resets the counter.
        p.next_interpret = '{"status":"answered","values":[{"id":"name","value":"Ana"}]}'
        drive(orch, "Ana")
        assert orch.stuck_count == 0

    def test_stuck_state_restorable(self) -> None:
        """Hosts rebuild the orchestrator per request — counters restore."""
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        orch1 = Orchestrator(p, current_steps=steps, record=record)
        p.next_interpret = '{"status":"unclear"}'
        drive(orch1, "huh")
        orch2 = Orchestrator(
            p, current_steps=steps, record=record,
            stuck_slot=orch1.stuck_slot, stuck_count=orch1.stuck_count,
        )
        drive(orch2, "huh again")
        assert orch2.stuck_count == 2


class TestHooks:
    def test_offtopic_hook_and_note(self) -> None:
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        notes: list[str] = []
        orch = Orchestrator(p, current_steps=steps, record=record,
                            on_offtopic=notes.append)
        p.next_interpret = '{"status":"offtopic","note":"asked about pizza"}'
        text, _, _ = drive(orch, "do you sell pizza?")
        assert notes == ["asked about pizza"]
        assert p.phrase_contexts[-1].get("refocus_note") == "asked about pizza"
        assert "[ask:name]" in text  # still on the first step

    def test_describe_shapes_the_ack(self) -> None:
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        orch = Orchestrator(
            p, current_steps=steps, record=record,
            describe=lambda sid, val: f"<{sid} is {val}>",
        )
        p.next_interpret = '{"status":"answered","values":[{"id":"name","value":"Ana"}]}'
        drive(orch, "Ana")
        assert p.phrase_contexts[-1].get("ack") == "<name is Ana>"

    def test_record_raising_is_contained(self) -> None:
        p = FakeFlowProvider()

        def bad_record(sid, val):
            raise RuntimeError("host bug")

        orch = Orchestrator(
            p,
            current_steps=lambda: [Step(id="x", payload={})],
            record=bad_record,
        )
        p.next_interpret = '{"status":"answered","values":[{"id":"x","value":1}]}'
        text, recorded, done = drive(orch, "1")
        assert recorded == []
        assert done.flow_complete is False  # turn completed despite host bug


class TestRefusal:
    def test_refusal_skips_and_advances(self) -> None:
        answers, steps, record = make_flow()
        skipped: list[str] = []
        fields_skipped: set[str] = set()

        def steps_with_skip():
            remaining = [f for f in ["name", "age", "city"]
                         if f not in answers and f not in fields_skipped]
            return [Step(id=f, payload={"ask": f}) for f in remaining[:2]]

        def on_refused(step_id: str) -> None:
            skipped.append(step_id)
            fields_skipped.add(step_id)

        p = FakeFlowProvider()
        orch = Orchestrator(p, current_steps=steps_with_skip, record=record,
                            on_refused=on_refused)
        p.next_interpret = '{"status":"refused"}'
        text, recorded, done = drive(orch, "I don't want to answer")
        assert skipped == ["name"]
        assert recorded == []          # nothing stored for a refusal
        assert "name" not in answers
        assert "[ask:age]" in text     # flow ADVANCED past the refused step
        assert "refusal_note" in p.phrase_contexts[-1]
        assert orch.stuck_count == 0   # a handled refusal is not « stuck »

    def test_refusal_without_hook_stays_put(self) -> None:
        answers, steps, record = make_flow()
        p = FakeFlowProvider()
        orch = Orchestrator(p, current_steps=steps, record=record)  # no hook
        p.next_interpret = '{"status":"refused"}'
        text, _, _ = drive(orch, "no way")
        assert "[ask:name]" in text    # mandatory step stays current
