"""Tests for `RoutingProvider`.

The routing provider sits in front of two (or more) sub-providers and
picks one per `LLMRequest`. These tests pin the default routing policy
(images go to vision, text goes to default with attachment stripping)
plus the custom-router escape hatch, and verify that nothing else
(content, tool_calls, system messages, request-level fields) is mutated
on the way through.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from autoagent.providers import RoutingProvider
from autoagent.schema import (
    ImageAttachment,
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    ToolCall,
    ToolSpec,
)

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk" "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class _Recorder:
    """Mini stand-in for an `LLMProvider`. Records every call.

    We intentionally don't subclass `LLMProvider` so the test stays
    decoupled from any base-class machinery — `RoutingProvider` only
    needs the duck-typed `config` + `complete(request)` shape.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.config = ModelConfig(provider=label, model=f"{label}-model")
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(content=self.label, model=self.config.model)


def _img() -> ImageAttachment:
    return ImageAttachment(data=_TINY_PNG_B64, mime_type="image/png")


# ---------------------------------------------------------------------------
# Default routing policy
# ---------------------------------------------------------------------------


class TestDefaultPolicy:
    def test_no_images_no_vision_routes_to_default(self) -> None:
        default = _Recorder("default")
        provider = RoutingProvider(default=default)

        provider.complete(LLMRequest(messages=[Message(role="user", content="hi")]))

        assert len(default.calls) == 1
        # No attachments to strip — nothing changes.
        assert default.calls[0].messages[0].content == "hi"
        assert default.calls[0].messages[0].attachments == []

    def test_latest_user_has_image_routes_to_vision(self) -> None:
        default = _Recorder("default")
        vision = _Recorder("vision")
        provider = RoutingProvider(default=default, vision=vision)

        msg = Message(role="user", content="what is this?", attachments=[_img()])
        provider.complete(LLMRequest(messages=[msg]))

        assert default.calls == []
        assert len(vision.calls) == 1
        # Attachments must be preserved when routing to vision.
        assert len(vision.calls[0].messages[0].attachments) == 1
        assert vision.calls[0].messages[0].attachments[0].mime_type == "image/png"

    def test_history_has_image_latest_user_does_not_strips_history(self) -> None:
        default = _Recorder("default")
        vision = _Recorder("vision")
        provider = RoutingProvider(default=default, vision=vision)

        history = [
            Message(role="user", content="here is a picture", attachments=[_img()]),
            Message(role="assistant", content="that's a tiny PNG"),
            Message(role="user", content="now translate it to spanish"),  # no image
        ]
        provider.complete(LLMRequest(messages=history))

        assert vision.calls == []
        assert len(default.calls) == 1
        wire = default.calls[0].messages
        # Every message's attachments must be wiped.
        assert all(m.attachments == [] for m in wire)
        # ... but content is preserved.
        assert wire[0].content == "here is a picture"
        assert wire[2].content == "now translate it to spanish"

    def test_vision_none_with_image_falls_back_to_default_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        default = _Recorder("default")
        provider = RoutingProvider(default=default, vision=None)

        msg = Message(role="user", content="?", attachments=[_img()])
        with caplog.at_level(logging.WARNING, logger="autoagent.providers.routing"):
            provider.complete(LLMRequest(messages=[msg]))

        assert len(default.calls) == 1
        # Stripping applies on the fallback path.
        assert default.calls[0].messages[0].attachments == []
        assert any("no vision provider" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Custom router
# ---------------------------------------------------------------------------


class TestCustomRouter:
    def test_router_overrides_default_policy(self) -> None:
        """A custom router based on message count instead of images."""
        default = _Recorder("default")
        vision = _Recorder("vision")

        def router(req: LLMRequest) -> Any:
            return vision if len(req.messages) >= 3 else default

        provider = RoutingProvider(default=default, vision=vision, router=router)

        # 2 messages — custom router → default.
        provider.complete(LLMRequest(messages=[Message(role="user", content="a")]))
        # 3+ messages — custom router → vision (even though no image present).
        provider.complete(
            LLMRequest(
                messages=[
                    Message(role="user", content="a"),
                    Message(role="assistant", content="b"),
                    Message(role="user", content="c"),
                ]
            )
        )

        assert len(default.calls) == 1
        assert len(vision.calls) == 1

    def test_custom_router_returning_vision_skips_stripping(self) -> None:
        default = _Recorder("default")
        vision = _Recorder("vision")

        provider = RoutingProvider(
            default=default,
            vision=vision,
            router=lambda req: vision,
        )

        msg = Message(role="user", content="x", attachments=[_img()])
        provider.complete(LLMRequest(messages=[msg]))

        assert len(vision.calls) == 1
        assert len(vision.calls[0].messages[0].attachments) == 1

    def test_custom_router_returning_default_strips_attachments(self) -> None:
        default = _Recorder("default")
        vision = _Recorder("vision")

        provider = RoutingProvider(
            default=default,
            vision=vision,
            router=lambda req: default,
        )

        msg = Message(role="user", content="x", attachments=[_img()])
        provider.complete(LLMRequest(messages=[msg]))

        assert len(default.calls) == 1
        assert default.calls[0].messages[0].attachments == []


# ---------------------------------------------------------------------------
# Configuration knobs + invariants
# ---------------------------------------------------------------------------


class TestKnobsAndInvariants:
    def test_strip_attachments_for_default_false_keeps_attachments(self) -> None:
        default = _Recorder("default")
        provider = RoutingProvider(default=default, strip_attachments_for_default=False)

        msg = Message(role="user", content="x", attachments=[_img()])
        provider.complete(LLMRequest(messages=[msg]))

        assert len(default.calls) == 1
        # Stripping disabled — attachments still there.
        assert len(default.calls[0].messages[0].attachments) == 1

    def test_config_proxies_to_default(self) -> None:
        default = _Recorder("default")
        vision = _Recorder("vision")
        provider = RoutingProvider(default=default, vision=vision)

        assert provider.config is default.config

    def test_other_message_fields_preserved_when_stripping(self) -> None:
        """Stripping must only touch `attachments`, not tool_calls /
        reasoning_content / tool_call_id / name."""
        default = _Recorder("default")
        provider = RoutingProvider(default=default)

        history = [
            Message(role="system", content="be terse"),
            Message(role="user", content="add 1+2", attachments=[_img()]),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})],
                reasoning_content="thinking…",
            ),
            Message(role="tool", content="3", tool_call_id="c1", name="add"),
            Message(role="user", content="thanks"),
        ]
        provider.complete(LLMRequest(messages=history, temperature=0.7))

        wire = default.calls[0].messages
        # System message preserved as-is.
        assert wire[0].role == "system"
        assert wire[0].content == "be terse"
        # User message: content kept, attachments stripped.
        assert wire[1].content == "add 1+2"
        assert wire[1].attachments == []
        # Assistant message: tool_calls and reasoning_content preserved.
        assert wire[2].role == "assistant"
        assert wire[2].tool_calls[0].id == "c1"
        assert wire[2].tool_calls[0].arguments == {"a": 1, "b": 2}
        assert wire[2].reasoning_content == "thinking…"
        # Tool message: tool_call_id and name preserved.
        assert wire[3].role == "tool"
        assert wire[3].tool_call_id == "c1"
        assert wire[3].name == "add"
        assert wire[3].content == "3"
        # Last user message round-trips.
        assert wire[4].content == "thanks"

    def test_request_level_fields_preserved(self) -> None:
        """`tools`, `temperature`, `max_tokens`, `tool_choice` must survive
        the stripping rewrite intact."""
        default = _Recorder("default")
        provider = RoutingProvider(default=default)

        tools = [ToolSpec(name="add", description="add two numbers")]
        req = LLMRequest(
            messages=[Message(role="user", content="x", attachments=[_img()])],
            tools=tools,
            temperature=0.3,
            max_tokens=128,
            tool_choice="auto",
        )
        provider.complete(req)

        forwarded = default.calls[0]
        assert forwarded.tools is tools or forwarded.tools == tools
        assert forwarded.temperature == 0.3
        assert forwarded.max_tokens == 128
        assert forwarded.tool_choice == "auto"
