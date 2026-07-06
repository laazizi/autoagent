from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from autoagent.http import post_json, post_sse
from autoagent.schema import (
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    StreamChunk,
    TokenUsage,
    ToolCall,
)

from .base import LLMProvider


def _uses_max_completion_tokens(model: str) -> bool:
    """OpenAI's newer model families (o-series reasoning models, GPT-5+)
    reject `max_tokens` and require `max_completion_tokens` instead."""
    m = model.lower()
    return m.startswith(("o1", "o3", "o4", "gpt-5"))


def _usage_from(u: Any) -> TokenUsage | None:
    if not isinstance(u, dict):
        return None
    return TokenUsage(
        input_tokens=u.get("prompt_tokens"),
        output_tokens=u.get("completion_tokens"),
        total_tokens=u.get("total_tokens"),
    )


class OpenAICompatibleProvider(LLMProvider):
    default_base_url = ""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.base_url = (config.base_url or self.default_base_url).rstrip("/")

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [self._message_to_wire(message) for message in request.messages],
        }
        if request.tools:
            payload["tools"] = [tool.as_openai_tool() for tool in request.tools]
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            if _uses_max_completion_tokens(self.config.model):
                payload["max_completion_tokens"] = request.max_tokens
            else:
                payload["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.config.resolved_api_key()}",
            **self.config.extra_headers,
        }

    def complete(self, request: LLMRequest) -> LLMResponse:
        raw = post_json(
            f"{self.base_url}/chat/completions",
            self._build_payload(request),
            headers=self._headers(),
            timeout=self.config.timeout,
        )
        message = raw["choices"][0]["message"]
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=self._parse_tool_calls(message.get("tool_calls") or []),
            raw=raw,
            model=raw.get("model"),
            reasoning_content=message.get("reasoning_content"),
            usage=_usage_from(raw.get("usage")),
        )

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        """Native SSE streaming via ``stream: true`` (OpenAI wire format,
        also spoken by DeepSeek, Groq, vLLM, OpenRouter...).

        Deltas arrive as partial ``choices[0].delta`` objects: ``content``
        fragments (yielded live), ``reasoning_content`` fragments
        (DeepSeek), and ``tool_calls`` deltas keyed by ``index`` whose
        ``function.arguments`` accumulate as JSON text. We deliberately do
        NOT send ``stream_options`` (not universally supported by
        compatible backends); providers like DeepSeek attach ``usage`` to
        the last chunk anyway — captured when present.
        """
        payload = self._build_payload(request)
        payload["stream"] = True

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # index -> {"id", "name", "arguments": "<accumulated json text>"}
        tool_deltas: dict[int, dict[str, Any]] = {}
        model: str | None = None
        usage_raw: dict[str, Any] | None = None

        for event in post_sse(
            f"{self.base_url}/chat/completions",
            payload,
            headers=self._headers(),
            timeout=self.config.timeout,
        ):
            model = event.get("model") or model
            if isinstance(event.get("usage"), dict):
                usage_raw = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            fragment = delta.get("content")
            if fragment:
                text_parts.append(fragment)
                yield StreamChunk(type="text", text=fragment)
            reasoning = delta.get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)
            for tc in delta.get("tool_calls") or []:
                slot = tool_deltas.setdefault(
                    tc.get("index", 0), {"id": None, "name": "", "arguments": ""}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                function = tc.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                if function.get("arguments"):
                    slot["arguments"] += function["arguments"]

        tool_calls: list[ToolCall] = []
        for index in sorted(tool_deltas):
            slot = tool_deltas[index]
            raw_args = slot["arguments"].strip()
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            tool_calls.append(
                ToolCall(
                    id=slot["id"] or f"tool_call_{index}",
                    name=slot["name"],
                    arguments=args,
                )
            )

        yield StreamChunk(
            type="final",
            response=LLMResponse(
                content="".join(text_parts),
                tool_calls=tool_calls,
                model=model or self.config.model,
                reasoning_content="".join(reasoning_parts) or None,
                raw={"stream": True, "model": model, "usage": usage_raw},
                usage=_usage_from(usage_raw),
            ),
        )

    def _message_to_wire(self, message: Message) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }

        # User messages with image attachments need the multi-part content
        # form: a list of {type: text | image_url, ...}.
        if message.role == "user" and message.attachments:
            parts: list[dict[str, Any]] = []
            if message.content:
                parts.append({"type": "text", "text": message.content})
            for att in message.attachments:
                parts.append({"type": "image_url", "image_url": {"url": att.as_data_url()}})
            return {"role": "user", "content": parts}

        data: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.role == "assistant" and message.tool_calls:
            data["content"] = message.content or None
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        if message.role == "assistant" and message.reasoning_content:
            data["reasoning_content"] = message.reasoning_content
        return data

    def _parse_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolCall]:
        parsed: list[ToolCall] = []
        for index, call in enumerate(tool_calls):
            function = call.get("function") or {}
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            parsed.append(
                ToolCall(
                    id=call.get("id") or f"tool_call_{index}",
                    name=function.get("name") or "",
                    arguments=args,
                )
            )
        return parsed


class OpenAIProvider(OpenAICompatibleProvider):
    default_base_url = "https://api.openai.com/v1"


class DeepSeekProvider(OpenAICompatibleProvider):
    default_base_url = "https://api.deepseek.com"


class GroqProvider(OpenAICompatibleProvider):
    # Groq expose une API compatible OpenAI (inférence LPU ultra-rapide).
    default_base_url = "https://api.groq.com/openai/v1"
