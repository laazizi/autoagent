"""Tests for Message / ToolCall / ImageAttachment serialisation (0.7.0).

Public contract:

* ``Message.to_dict()``, ``ToolCall.to_dict()``, ``ImageAttachment.to_dict()``
  return plain JSON-safe dicts (no custom types, no None for required fields).
* ``Message.from_dict(d)`` (etc.) rebuild a value equal to the original.
* Round-trip through ``json.dumps`` / ``json.loads`` is lossless.
* Missing optional fields in the input dict are tolerated (forward-compat).
* Snapshots are compact: empty optional fields are omitted on the way out.

These guarantees enable the FastAPI chat-session pattern: load a stored
history, append a user message, run the agent, save the new history.
"""

from __future__ import annotations

import json

from autoagent.schema import ImageAttachment, Message, ToolCall


class TestToolCallSerialisation:
    def test_minimal_round_trip(self) -> None:
        tc = ToolCall(id="call_1", name="search", arguments={"q": "Lyon"})
        restored = ToolCall.from_dict(tc.to_dict())
        assert restored == tc

    def test_with_thought_signature(self) -> None:
        tc = ToolCall(id="x", name="y", arguments={"a": 1}, thought_signature="opaque")
        d = tc.to_dict()
        assert d["thought_signature"] == "opaque"
        assert ToolCall.from_dict(d) == tc

    def test_json_round_trip(self) -> None:
        tc = ToolCall(id="x", name="y", arguments={"nested": {"k": [1, 2]}})
        as_json = json.dumps(tc.to_dict())
        restored = ToolCall.from_dict(json.loads(as_json))
        assert restored == tc

    def test_missing_optional_field_tolerated(self) -> None:
        # Older snapshots predating thought_signature still load.
        restored = ToolCall.from_dict({"id": "x", "name": "y", "arguments": {}})
        assert restored.thought_signature is None

    def test_missing_arguments_defaults_to_empty(self) -> None:
        restored = ToolCall.from_dict({"id": "x", "name": "y"})
        assert restored.arguments == {}


class TestImageAttachmentSerialisation:
    def test_data_url_round_trip(self) -> None:
        att = ImageAttachment(data="data:image/png;base64,abc=")
        assert ImageAttachment.from_dict(att.to_dict()) == att

    def test_raw_base64_with_mime_round_trip(self) -> None:
        att = ImageAttachment(data="abc=", mime_type="image/png")
        assert ImageAttachment.from_dict(att.to_dict()) == att

    def test_missing_mime_tolerated(self) -> None:
        restored = ImageAttachment.from_dict({"data": "data:image/jpeg;base64,xyz="})
        assert restored.data == "data:image/jpeg;base64,xyz="
        assert restored.mime_type is None


class TestMessageSerialisation:
    def test_simple_text_message(self) -> None:
        msg = Message(role="user", content="bonjour")
        d = msg.to_dict()
        # Compact: no empty optional fields.
        assert d == {"role": "user", "content": "bonjour"}
        assert Message.from_dict(d) == msg

    def test_assistant_with_tool_calls(self) -> None:
        msg = Message(
            role="assistant",
            content="searching",
            tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "Lyon"})],
        )
        d = msg.to_dict()
        assert "tool_calls" in d
        assert d["tool_calls"][0]["name"] == "search"
        assert Message.from_dict(d) == msg

    def test_tool_message_with_id(self) -> None:
        msg = Message(role="tool", content="result", tool_call_id="c1", name="search")
        d = msg.to_dict()
        assert d["tool_call_id"] == "c1"
        assert d["name"] == "search"
        assert Message.from_dict(d) == msg

    def test_user_with_attachments(self) -> None:
        att = ImageAttachment(data="data:image/png;base64,abc=")
        msg = Message(role="user", content="see this", attachments=[att])
        d = msg.to_dict()
        assert "attachments" in d
        assert d["attachments"][0]["data"].startswith("data:image/png")
        assert Message.from_dict(d) == msg

    def test_with_reasoning_content(self) -> None:
        msg = Message(role="assistant", content="ok", reasoning_content="thinking step")
        d = msg.to_dict()
        assert d["reasoning_content"] == "thinking step"
        assert Message.from_dict(d) == msg

    def test_empty_fields_omitted_from_output(self) -> None:
        """Compact snapshot: empty optional fields don't appear in to_dict()."""
        msg = Message(role="user", content="hi")
        d = msg.to_dict()
        assert "tool_calls" not in d  # empty list omitted
        assert "attachments" not in d
        assert "tool_call_id" not in d
        assert "name" not in d
        assert "reasoning_content" not in d

    def test_minimal_dict_loads(self) -> None:
        """Forward-compat: a dict with only role still loads."""
        restored = Message.from_dict({"role": "user"})
        assert restored.role == "user"
        assert restored.content == ""
        assert restored.tool_calls == []

    def test_full_conversation_round_trip_through_json(self) -> None:
        """The canonical FastAPI pattern: serialise a full chat history."""
        history = [
            Message(role="system", content="You are a helpful agent."),
            Message(role="user", content="bonjour"),
            Message(
                role="assistant",
                content="checking address",
                tool_calls=[ToolCall(id="c1", name="geocode", arguments={"q": "12 rue X"})],
            ),
            Message(role="tool", content='{"lat":48.8,"lon":2.3}', tool_call_id="c1", name="geocode"),
            Message(role="assistant", content="trouvé"),
        ]
        serialised = json.dumps([m.to_dict() for m in history])
        restored = [Message.from_dict(d) for d in json.loads(serialised)]
        assert restored == history
