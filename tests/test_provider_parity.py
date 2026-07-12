"""Parity fixes across providers (0.10.0 hardening).

Pins the behaviours added by the library audit:
  * `tool_choice` is now honoured by Anthropic and Gemini (it was
    OpenAI-only — accepted but silently ignored elsewhere);
  * Gemini recovers the functionResponse `name` from the assistant
    tool_calls via `tool_call_id` when the host forgot `Message.name`;
  * token usage is extracted into `LLMResponse.usage` for all three
    wire formats;
  * OpenAI-compatible providers stream natively (SSE deltas), and
    `RoutingProvider.stream` preserves the chosen provider's streaming.
"""

from __future__ import annotations

from unittest.mock import patch

from autoagent.providers.anthropic import AnthropicProvider
from autoagent.providers.gemini import GeminiProvider
from autoagent.providers.openai import OpenAIProvider
from autoagent.providers.routing import RoutingProvider
from autoagent.schema import (
    ImageAttachment,
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    ToolCall,
    ToolSpec,
)

_TOOL = ToolSpec(
    name="lookup",
    description="d",
    input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
)


def _anthropic() -> AnthropicProvider:
    return AnthropicProvider(ModelConfig(provider="anthropic", model="claude-x", api_key="k"))


def _gemini() -> GeminiProvider:
    return GeminiProvider(ModelConfig(provider="gemini", model="gemini-x", api_key="k"))


def _openai() -> OpenAIProvider:
    return OpenAIProvider(ModelConfig(provider="openai", model="gpt-x", api_key="k"))


def _req(tool_choice: str | None = "auto") -> LLMRequest:
    return LLMRequest(
        messages=[Message(role="user", content="hi")],
        tools=[_TOOL],
        tool_choice=tool_choice,
    )


class TestToolChoiceParity:
    def test_anthropic_required_maps_to_any(self) -> None:
        payload = _anthropic()._build_payload(_req("required"))
        assert payload["tool_choice"] == {"type": "any"}

    def test_anthropic_specific_tool_name(self) -> None:
        payload = _anthropic()._build_payload(_req("lookup"))
        assert payload["tool_choice"] == {"type": "tool", "name": "lookup"}

    def test_anthropic_none_drops_tools(self) -> None:
        payload = _anthropic()._build_payload(_req("none"))
        assert "tools" not in payload and "tool_choice" not in payload

    def test_anthropic_auto_stays_implicit(self) -> None:
        payload = _anthropic()._build_payload(_req("auto"))
        assert "tools" in payload and "tool_choice" not in payload

    def test_gemini_required_maps_to_mode_any(self) -> None:
        payload = _gemini()._build_payload(_req("required"))
        assert payload["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}

    def test_gemini_specific_tool_restricts_names(self) -> None:
        payload = _gemini()._build_payload(_req("lookup"))
        cfg = payload["toolConfig"]["functionCallingConfig"]
        assert cfg == {"mode": "ANY", "allowedFunctionNames": ["lookup"]}

    def test_gemini_none_maps_to_mode_none(self) -> None:
        payload = _gemini()._build_payload(_req("none"))
        assert payload["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}

    def test_gemini_auto_sends_no_tool_config(self) -> None:
        assert "toolConfig" not in _gemini()._build_payload(_req("auto"))


class TestGeminiToolResultNaming:
    def test_function_response_name_recovered_from_call_id(self) -> None:
        # Host forgot Message.name on the tool message: Gemini used to send
        # the default "tool", breaking call/response matching.
        messages = [
            Message(role="user", content="go"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", name="lookup", arguments={"q": "x"})],
            ),
            Message(role="tool", tool_call_id="c1", content='{"ok": true}'),
        ]
        contents = _gemini()._messages_to_contents(messages)
        fr = contents[-1]["parts"][0]["functionResponse"]
        assert fr["name"] == "lookup"


class TestUsageExtraction:
    def test_openai_usage(self) -> None:
        raw = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        }
        with patch("autoagent.providers.openai.post_json", return_value=raw):
            resp = _openai().complete(_req())
        assert resp.usage is not None
        assert (resp.usage.input_tokens, resp.usage.output_tokens) == (11, 4)
        assert resp.usage.total_tokens == 15

    def test_anthropic_usage(self) -> None:
        raw = {
            "content": [{"type": "text", "text": "hi"}],
            "model": "claude-x",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        with patch("autoagent.providers.anthropic.post_json", return_value=raw):
            resp = _anthropic().complete(_req())
        assert resp.usage is not None
        assert resp.usage.total_tokens == 10  # derived when provider omits it

    def test_gemini_usage(self) -> None:
        raw = {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
            },
        }
        with patch("autoagent.providers.gemini.post_json", return_value=raw):
            resp = _gemini().complete(_req())
        assert resp.usage is not None
        assert resp.usage.total_tokens == 7


def _openai_sse_events() -> list[dict]:
    return [
        {"model": "gpt-x", "choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "lookup", "arguments": '{"q": '},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]}}
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 6}},
    ]


class TestOpenAINativeStreaming:
    def test_stream_yields_deltas_and_assembles_tool_call(self) -> None:
        with patch(
            "autoagent.providers.openai.post_sse", return_value=iter(_openai_sse_events())
        ):
            chunks = list(_openai().stream(_req()))
        texts = [c.text for c in chunks if c.type == "text"]
        assert texts == ["Hel", "lo"]
        final = chunks[-1]
        assert final.type == "final" and final.response is not None
        assert final.response.content == "Hello"
        assert final.response.tool_calls == [
            ToolCall(id="call_1", name="lookup", arguments={"q": "x"})
        ]
        assert final.response.usage is not None
        assert final.response.usage.input_tokens == 9
        assert final.response.raw is not None  # parity with complete()


class TestResponseFormat:
    def test_openai_passthrough(self) -> None:
        req = _req()
        req.response_format = {"type": "json_object"}
        payload = _openai()._build_payload(req)
        assert payload["response_format"] == {"type": "json_object"}

    def test_gemini_maps_to_response_mime_type(self) -> None:
        req = _req()
        req.response_format = {"type": "json_object"}
        payload = _gemini()._build_payload(req)
        assert payload["generationConfig"]["responseMimeType"] == "application/json"

    def test_anthropic_appends_strict_json_instruction(self) -> None:
        req = LLMRequest(
            messages=[
                Message(role="system", content="Tu es un assistant."),
                Message(role="user", content="hi"),
            ],
            response_format={"type": "json_object"},
        )
        payload = _anthropic()._build_payload(req)
        assert payload["system"].startswith("Tu es un assistant.")
        assert "JSON object only" in payload["system"]

    def test_absent_by_default(self) -> None:
        assert "response_format" not in _openai()._build_payload(_req())
        assert "responseMimeType" not in _gemini()._build_payload(_req()).get(
            "generationConfig", {}
        )


class TestRoutingStream:
    def test_stream_routes_to_vision_and_keeps_native_streaming(self) -> None:
        class _Fake(LLMResponse):
            pass

        class _Recorder(OpenAIProvider):
            def __init__(self, tag: str) -> None:
                super().__init__(ModelConfig(provider="openai", model=tag, api_key="k"))
                self.tag = tag
                self.streamed_with: LLMRequest | None = None

            def stream(self, request: LLMRequest):
                self.streamed_with = request
                yield from ()

        text_p = _Recorder("text")
        vision_p = _Recorder("vision")
        router = RoutingProvider(default=text_p, vision=vision_p)

        image_req = LLMRequest(
            messages=[
                Message(
                    role="user",
                    content="look",
                    attachments=[ImageAttachment(data="data:image/png;base64,AAAA")],
                )
            ]
        )
        list(router.stream(image_req))
        assert vision_p.streamed_with is not None  # image -> vision provider
        assert text_p.streamed_with is None

        list(router.stream(LLMRequest(messages=[Message(role="user", content="hi")])))
        assert text_p.streamed_with is not None  # text -> default provider
