"""Tests for provider wire-format conversion.

Each provider has a strict wire format. These tests pin the conversion
in both directions (request payload + response parsing) using a mocked
`post_json`. They guard against regressions that would silently break
production calls.

Coverage targets:
  * OpenAI/DeepSeek: assistant tool_calls in payload, tool results as
    `role=tool` with tool_call_id, response parsing extracts content +
    tool_calls.
  * Anthropic: system extracted into top-level `system`, tool_use blocks
    in assistant content, tool_result blocks wrapped in user role,
    response parser handles content blocks.
  * Gemini: api key in header (regression test for security fix),
    function calls roundtrip.
  * `create_provider` dispatch.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from autoagent.providers import (
    AnthropicProvider,
    DeepSeekProvider,
    GeminiProvider,
    OpenAIProvider,
    create_provider,
)
from autoagent.schema import (
    ImageAttachment,
    LLMRequest,
    Message,
    ModelConfig,
    ToolCall,
    ToolSpec,
)

# A tiny 1x1 transparent PNG, base64 encoded — enough to assert serialization
# without depending on real image bytes.
_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


def _basic_request(messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMRequest:
    return LLMRequest(messages=messages, tools=tools or [])


# ---------------------------------------------------------------------------
# create_provider dispatch
# ---------------------------------------------------------------------------


class TestCreateProvider:
    @pytest.mark.parametrize(
        "provider_name, expected_cls",
        [
            ("openai", OpenAIProvider),
            ("anthropic", AnthropicProvider),
            ("deepseek", DeepSeekProvider),
            ("gemini", GeminiProvider),
            ("google", GeminiProvider),
        ],
    )
    def test_known_provider_returns_instance(self, provider_name: str, expected_cls: type) -> None:
        provider = create_provider(ModelConfig(provider=provider_name, model="m"))
        assert isinstance(provider, expected_cls)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_provider(ModelConfig(provider="cohere-not-supported", model="m"))


# ---------------------------------------------------------------------------
# OpenAI-compatible providers (OpenAI, DeepSeek)
# ---------------------------------------------------------------------------


class TestOpenAIWire:
    def _capture(self, provider, messages, tools=None):
        captured: dict[str, Any] = {}

        def fake_post_json(url, payload, headers=None, timeout=None):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers or {}
            return {
                "model": "fake",
                "choices": [{"message": {"content": "hello", "tool_calls": []}}],
            }

        with patch("autoagent.providers.openai.post_json", fake_post_json):
            provider.complete(_basic_request(messages, tools))
        return captured

    def test_user_message_serialized(self) -> None:
        provider = OpenAIProvider(ModelConfig(provider="openai", model="gpt-4", api_key="k"))
        cap = self._capture(provider, [Message(role="user", content="hi")])
        assert cap["payload"]["messages"] == [{"role": "user", "content": "hi"}]
        assert cap["headers"]["authorization"] == "Bearer k"

    def test_assistant_tool_call_serialized(self) -> None:
        provider = OpenAIProvider(ModelConfig(provider="openai", model="gpt-4", api_key="k"))
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})],
        )
        cap = self._capture(provider, [msg])
        out = cap["payload"]["messages"][0]
        assert out["role"] == "assistant"
        assert out["tool_calls"][0]["id"] == "c1"
        assert out["tool_calls"][0]["function"]["name"] == "add"
        # arguments must be JSON-encoded as a string per OpenAI spec.
        assert json.loads(out["tool_calls"][0]["function"]["arguments"]) == {"a": 1, "b": 2}

    def test_tool_message_uses_tool_call_id(self) -> None:
        provider = OpenAIProvider(ModelConfig(provider="openai", model="gpt-4", api_key="k"))
        msg = Message(role="tool", content='{"result": 3}', tool_call_id="c1", name="add")
        cap = self._capture(provider, [msg])
        out = cap["payload"]["messages"][0]
        assert out == {"role": "tool", "tool_call_id": "c1", "content": '{"result": 3}'}

    def test_response_tool_call_parsed(self) -> None:
        provider = OpenAIProvider(ModelConfig(provider="openai", model="gpt-4", api_key="k"))

        def fake(url, payload, headers=None, timeout=None):
            return {
                "model": "gpt-4",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c42",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q": "claude"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            }

        with patch("autoagent.providers.openai.post_json", fake):
            response = provider.complete(_basic_request([Message(role="user", content="x")]))
        assert response.content == ""
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "c42"
        assert response.tool_calls[0].arguments == {"q": "claude"}

    def test_malformed_tool_arguments_fallback(self) -> None:
        provider = OpenAIProvider(ModelConfig(provider="openai", model="gpt-4", api_key="k"))

        def fake(url, payload, headers=None, timeout=None):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [{"id": "c1", "function": {"name": "x", "arguments": "NOT-JSON"}}],
                        }
                    }
                ]
            }

        with patch("autoagent.providers.openai.post_json", fake):
            response = provider.complete(_basic_request([Message(role="user", content="x")]))
        # Arguments fall back to a wrapped payload rather than crashing.
        assert response.tool_calls[0].arguments == {"_raw": "NOT-JSON"}

    def test_user_message_with_image_uses_content_parts(self) -> None:
        """OpenAI multimodal: when a user message has an ImageAttachment, the
        wire payload must be a list of {type: text|image_url, ...} parts."""
        provider = OpenAIProvider(ModelConfig(provider="openai", model="gpt-4o", api_key="k"))
        captured: dict[str, Any] = {}

        def fake(url, payload, headers=None, timeout=None):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}

        msg = Message(
            role="user",
            content="What is this?",
            attachments=[ImageAttachment(data=_TINY_PNG_B64, mime_type="image/png")],
        )
        with patch("autoagent.providers.openai.post_json", fake):
            provider.complete(LLMRequest(messages=[msg]))
        wire = captured["payload"]["messages"][0]
        assert wire["role"] == "user"
        assert isinstance(wire["content"], list)
        types = [p["type"] for p in wire["content"]]
        assert types == ["text", "image_url"]
        assert wire["content"][0]["text"] == "What is this?"
        assert wire["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_user_message_without_image_stays_string(self) -> None:
        """No attachments → content remains a plain string (legacy format)."""
        provider = OpenAIProvider(ModelConfig(provider="openai", model="gpt-4o", api_key="k"))
        captured: dict[str, Any] = {}

        def fake(url, payload, headers=None, timeout=None):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}

        with patch("autoagent.providers.openai.post_json", fake):
            provider.complete(LLMRequest(messages=[Message(role="user", content="hi")]))
        wire = captured["payload"]["messages"][0]
        assert wire["content"] == "hi"  # not a list

    def test_deepseek_uses_deepseek_base_url(self) -> None:
        provider = DeepSeekProvider(ModelConfig(provider="deepseek", model="deepseek-chat", api_key="k"))
        assert provider.base_url.startswith("https://api.deepseek.com")

    @pytest.mark.parametrize("model", ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"])
    def test_legacy_models_use_max_tokens(self, model: str) -> None:
        """Older OpenAI models (gpt-4*, gpt-3.5*) accept `max_tokens`."""
        provider = OpenAIProvider(ModelConfig(provider="openai", model=model, api_key="k"))
        captured: dict[str, Any] = {}

        def fake(url, payload, headers=None, timeout=None):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}

        with patch("autoagent.providers.openai.post_json", fake):
            provider.complete(LLMRequest(messages=[Message(role="user", content="x")], max_tokens=500))
        assert captured["payload"]["max_tokens"] == 500
        assert "max_completion_tokens" not in captured["payload"]

    def test_reasoning_content_parsed_from_response(self) -> None:
        """Reasoning-mode providers (DeepSeek thinking, OpenAI o-series)
        return a `reasoning_content` field. The lib must capture it."""
        provider = OpenAIProvider(ModelConfig(provider="openai", model="o1", api_key="k"))

        def fake(url, payload, headers=None, timeout=None):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                            "reasoning_content": "Let me think step by step…",
                            "tool_calls": [],
                        }
                    }
                ]
            }

        with patch("autoagent.providers.openai.post_json", fake):
            response = provider.complete(LLMRequest(messages=[Message(role="user", content="x")]))
        assert response.reasoning_content == "Let me think step by step…"

    def test_reasoning_content_echoed_in_next_request(self) -> None:
        """When an assistant message carries `reasoning_content`, the next
        request payload must include it. Required by DeepSeek v4 thinking
        mode (otherwise the API rejects with 400 invalid_request_error)."""
        provider = OpenAIProvider(ModelConfig(provider="openai", model="deepseek-v4-pro", api_key="k"))
        captured: dict[str, Any] = {}

        def fake(url, payload, headers=None, timeout=None):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}

        history = [
            Message(role="user", content="prev question"),
            Message(
                role="assistant",
                content="prev answer",
                reasoning_content="hidden CoT trace",
            ),
            Message(role="user", content="follow-up"),
        ]
        with patch("autoagent.providers.openai.post_json", fake):
            provider.complete(LLMRequest(messages=history))
        wire_messages = captured["payload"]["messages"]
        assistant_wire = next(m for m in wire_messages if m["role"] == "assistant")
        assert assistant_wire.get("reasoning_content") == "hidden CoT trace"

    def test_reasoning_content_absent_when_none(self) -> None:
        """Plain (non-reasoning) assistant messages must NOT carry an empty
        `reasoning_content` field — only when the model actually emitted one."""
        provider = OpenAIProvider(ModelConfig(provider="openai", model="gpt-4o-mini", api_key="k"))
        captured: dict[str, Any] = {}

        def fake(url, payload, headers=None, timeout=None):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}

        history = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
            Message(role="user", content="again"),
        ]
        with patch("autoagent.providers.openai.post_json", fake):
            provider.complete(LLMRequest(messages=history))
        wire_messages = captured["payload"]["messages"]
        assistant_wire = next(m for m in wire_messages if m["role"] == "assistant")
        assert "reasoning_content" not in assistant_wire

    @pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "o1", "o1-mini", "o3-mini", "o4"])
    def test_new_models_use_max_completion_tokens(self, model: str) -> None:
        """Newer OpenAI models (gpt-5*, o1*, o3*, o4*) reject `max_tokens`
        and need `max_completion_tokens` instead. Regression test for the
        API breaking change applied by OpenAI in 2025."""
        provider = OpenAIProvider(ModelConfig(provider="openai", model=model, api_key="k"))
        captured: dict[str, Any] = {}

        def fake(url, payload, headers=None, timeout=None):
            captured["payload"] = payload
            return {"choices": [{"message": {"content": "ok", "tool_calls": []}}]}

        with patch("autoagent.providers.openai.post_json", fake):
            provider.complete(LLMRequest(messages=[Message(role="user", content="x")], max_tokens=500))
        assert captured["payload"]["max_completion_tokens"] == 500
        assert "max_tokens" not in captured["payload"]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class TestAnthropicWire:
    def _capture(self, provider, messages, tools=None):
        captured: dict[str, Any] = {}

        def fake_post_json(url, payload, headers=None, timeout=None):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers or {}
            return {"model": "claude", "content": [{"type": "text", "text": "ok"}]}

        with patch("autoagent.providers.anthropic.post_json", fake_post_json):
            provider.complete(_basic_request(messages, tools))
        return captured

    def test_system_extracted_to_top_level(self) -> None:
        provider = AnthropicProvider(ModelConfig(provider="anthropic", model="claude", api_key="k"))
        cap = self._capture(
            provider,
            [
                Message(role="system", content="be terse"),
                Message(role="user", content="hi"),
            ],
        )
        assert cap["payload"]["system"] == "be terse"
        # System message must NOT appear in the messages array.
        assert all(m["role"] != "system" for m in cap["payload"]["messages"])

    def test_api_key_in_x_api_key_header(self) -> None:
        provider = AnthropicProvider(ModelConfig(provider="anthropic", model="claude", api_key="secret"))
        cap = self._capture(provider, [Message(role="user", content="hi")])
        assert cap["headers"]["x-api-key"] == "secret"
        assert cap["headers"]["anthropic-version"]

    def test_assistant_tool_use_block(self) -> None:
        provider = AnthropicProvider(ModelConfig(provider="anthropic", model="claude", api_key="k"))
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="lookup", arguments={"q": "x"})],
        )
        cap = self._capture(provider, [msg])
        wire = cap["payload"]["messages"][0]
        assert wire["role"] == "assistant"
        tool_use = next(block for block in wire["content"] if block["type"] == "tool_use")
        assert tool_use["id"] == "c1"
        assert tool_use["input"] == {"q": "x"}

    def test_tool_result_wrapped_in_user_message(self) -> None:
        provider = AnthropicProvider(ModelConfig(provider="anthropic", model="claude", api_key="k"))
        msg = Message(role="tool", content="42", tool_call_id="c1", name="add")
        cap = self._capture(provider, [msg])
        wire = cap["payload"]["messages"][0]
        assert wire["role"] == "user"
        block = wire["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "c1"
        assert block["content"] == "42"

    def test_response_tool_use_parsed(self) -> None:
        provider = AnthropicProvider(ModelConfig(provider="anthropic", model="claude", api_key="k"))

        def fake(url, payload, headers=None, timeout=None):
            return {
                "model": "claude",
                "content": [
                    {"type": "text", "text": "let me check"},
                    {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "a"}},
                ],
            }

        with patch("autoagent.providers.anthropic.post_json", fake):
            response = provider.complete(_basic_request([Message(role="user", content="x")]))
        assert response.content == "let me check"
        assert response.tool_calls[0].id == "tu1"
        assert response.tool_calls[0].arguments == {"q": "a"}

    def test_user_message_with_image(self) -> None:
        """Anthropic multimodal: image block with `source.type=base64`."""
        provider = AnthropicProvider(
            ModelConfig(provider="anthropic", model="claude-sonnet-4-5", api_key="k")
        )
        msg = Message(
            role="user",
            content="What is this?",
            attachments=[ImageAttachment(data=_TINY_PNG_B64, mime_type="image/png")],
        )
        cap = self._capture(provider, [msg])
        wire = cap["payload"]["messages"][0]
        assert wire["role"] == "user"
        types = [p["type"] for p in wire["content"]]
        assert types == ["text", "image"]
        image_block = wire["content"][1]
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["media_type"] == "image/png"
        assert image_block["source"]["data"] == _TINY_PNG_B64


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class TestGeminiWire:
    def _capture(self, provider, messages, tools=None):
        captured: dict[str, Any] = {}

        def fake_post_json(url, payload, headers=None, timeout=None):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers or {}
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

        with patch("autoagent.providers.gemini.post_json", fake_post_json):
            provider.complete(_basic_request(messages, tools))
        return captured

    def test_function_call_in_response_parsed(self) -> None:
        provider = GeminiProvider(ModelConfig(provider="gemini", model="gemini-2", api_key="k"))

        def fake(url, payload, headers=None, timeout=None):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "let me search"},
                                {"functionCall": {"name": "search", "args": {"q": "x"}}},
                            ]
                        }
                    }
                ]
            }

        with patch("autoagent.providers.gemini.post_json", fake):
            response = provider.complete(_basic_request([Message(role="user", content="hi")]))
        assert response.content == "let me search"
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[0].arguments == {"q": "x"}

    def test_system_message_extracted_into_systemInstruction(self) -> None:
        provider = GeminiProvider(ModelConfig(provider="gemini", model="gemini-2", api_key="k"))
        cap = self._capture(
            provider,
            [
                Message(role="system", content="be terse"),
                Message(role="user", content="hi"),
            ],
        )
        assert cap["payload"]["systemInstruction"]["parts"][0]["text"] == "be terse"

    def test_extra_headers_merged_with_api_key_header(self) -> None:
        provider = GeminiProvider(
            ModelConfig(
                provider="gemini",
                model="gemini-2",
                api_key="k",
                extra_headers={"x-trace-id": "abc"},
            )
        )
        cap = self._capture(provider, [Message(role="user", content="hi")])
        assert cap["headers"]["x-goog-api-key"] == "k"
        assert cap["headers"]["x-trace-id"] == "abc"

    def test_user_message_with_image(self) -> None:
        """Gemini multimodal: `inline_data` part with mime_type + data."""
        provider = GeminiProvider(ModelConfig(provider="gemini", model="gemini-2.0-flash", api_key="k"))
        msg = Message(
            role="user",
            content="What is this?",
            attachments=[ImageAttachment(data=_TINY_PNG_B64, mime_type="image/png")],
        )
        cap = self._capture(provider, [msg])
        parts = cap["payload"]["contents"][0]["parts"]
        # Expect [text, inline_data]
        assert parts[0]["text"] == "What is this?"
        assert parts[1]["inline_data"]["mime_type"] == "image/png"
        assert parts[1]["inline_data"]["data"] == _TINY_PNG_B64

    def test_thought_signature_captured_from_response(self) -> None:
        """Gemini 3+ thinking models return `thoughtSignature` on each
        functionCall. The lib must capture it so we can echo it back."""
        provider = GeminiProvider(ModelConfig(provider="gemini", model="gemini-3.5-flash", api_key="k"))

        def fake(url, payload, headers=None, timeout=None):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "search",
                                        "args": {"q": "x"},
                                        "thoughtSignature": "SIG-AAA-encrypted-blob",
                                    }
                                },
                            ]
                        }
                    }
                ]
            }

        with patch("autoagent.providers.gemini.post_json", fake):
            response = provider.complete(_basic_request([Message(role="user", content="hi")]))
        assert response.tool_calls[0].thought_signature == "SIG-AAA-encrypted-blob"

    def test_thought_signature_echoed_back_at_part_level(self) -> None:
        """When the host sends back an assistant message that previously
        had a function call, the thoughtSignature MUST be re-attached
        AT THE PART LEVEL (sibling of `functionCall`, NOT nested inside
        it). Gemini's REQUEST format is asymmetric vs the RESPONSE
        format where the signature is nested inside `functionCall`.
        Putting it inside `functionCall` yields a 400 'Unknown name
        thoughtSignature ... Cannot find field'."""
        provider = GeminiProvider(ModelConfig(provider="gemini", model="gemini-3.5-flash", api_key="k"))
        msgs = [
            Message(role="user", content="search for foo"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="gemini_tool_call_0",
                        name="search",
                        arguments={"q": "foo"},
                        thought_signature="SIG-BBB",
                    )
                ],
            ),
            Message(
                role="tool",
                name="search",
                tool_call_id="gemini_tool_call_0",
                content='{"hits": []}',
            ),
        ]
        cap = self._capture(provider, msgs)
        model_block = next(c for c in cap["payload"]["contents"] if c.get("role") == "model")
        function_call_part = next(p for p in model_block["parts"] if "functionCall" in p)
        # Signature at the PART level (sibling of functionCall).
        assert function_call_part["thoughtSignature"] == "SIG-BBB"
        # And NOT nested inside functionCall (would yield a 400).
        assert "thoughtSignature" not in function_call_part["functionCall"]
        assert function_call_part["functionCall"]["name"] == "search"

    def test_no_thought_signature_omits_field(self) -> None:
        """When the ToolCall has no thought_signature (older Gemini
        versions or non-thinking models), neither the part nor the
        functionCall must carry a `thoughtSignature` key."""
        provider = GeminiProvider(ModelConfig(provider="gemini", model="gemini-2.0-flash", api_key="k"))
        msgs = [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="x", name="echo", arguments={"v": 1}),
                ],
            ),
            Message(role="tool", name="echo", tool_call_id="x", content='{"v": 1}'),
        ]
        cap = self._capture(provider, msgs)
        model_block = next(c for c in cap["payload"]["contents"] if c.get("role") == "model")
        part = next(p for p in model_block["parts"] if "functionCall" in p)
        assert "thoughtSignature" not in part
        assert "thoughtSignature" not in part["functionCall"]
