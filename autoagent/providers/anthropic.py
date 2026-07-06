from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from autoagent.http import post_json, post_sse
from autoagent.schema import LLMRequest, LLMResponse, Message, StreamChunk, TokenUsage, ToolCall

from .base import LLMProvider


def _usage_from(u: Any) -> TokenUsage | None:
    if not isinstance(u, dict):
        return None
    return TokenUsage(input_tokens=u.get("input_tokens"), output_tokens=u.get("output_tokens"))


class AnthropicProvider(LLMProvider):
    default_base_url = "https://api.anthropic.com"
    default_version = "2023-06-01"

    _JSON_ONLY_INSTRUCTION = (
        "Respond with a single valid JSON object only — no markdown fences, "
        "no prose before or after."
    )

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        system_text = self._system_text(request.messages)
        if request.response_format is not None:
            # Anthropic has no native JSON mode — enforce via a strict system
            # instruction (best effort; callers keep a tolerant parser).
            system_text = (
                f"{system_text}\n\n{self._JSON_ONLY_INSTRUCTION}"
                if system_text
                else self._JSON_ONLY_INSTRUCTION
            )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": request.max_tokens if request.max_tokens is not None else 2048,
            "messages": self._messages_to_wire(request.messages),
        }
        if system_text:
            payload["system"] = system_text
        if request.tools and request.tool_choice == "none":
            # Anthropic has no portable "none": simply don't offer the tools.
            pass
        elif request.tools:
            payload["tools"] = [tool.as_anthropic_tool() for tool in request.tools]
            choice = request.tool_choice
            if choice in ("required", "any"):
                payload["tool_choice"] = {"type": "any"}
            elif choice and choice != "auto":  # a specific tool name
                payload["tool_choice"] = {"type": "tool", "name": choice}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.resolved_api_key(),
            "anthropic-version": self.default_version,
            **self.config.extra_headers,
        }

    def _url(self) -> str:
        return f"{(self.config.base_url or self.default_base_url).rstrip('/')}/v1/messages"

    def complete(self, request: LLMRequest) -> LLMResponse:
        raw = post_json(
            self._url(),
            self._build_payload(request),
            headers=self._headers(),
            timeout=self.config.timeout,
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for index, block in enumerate(raw.get("content", [])):
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id") or f"tool_call_{index}",
                        name=block.get("name") or "",
                        arguments=block.get("input") or {},
                    )
                )
        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            raw=raw,
            model=raw.get("model"),
            usage=_usage_from(raw.get("usage")),
        )

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        """Native SSE streaming via Anthropic's ``stream: true``.

        Anthropic emits a sequence of events. We care about:
          * ``content_block_start`` — opens a text or tool_use block.
            For tool_use we capture id + name (input arrives as deltas).
          * ``content_block_delta`` — ``text_delta`` (user-visible text)
            or ``input_json_delta`` (a fragment of the tool's JSON args).
          * ``content_block_stop`` — closes a block; we parse the
            accumulated tool args JSON here.
        At the end we assemble the same ``LLMResponse`` that
        ``complete()`` would have returned, so the agent loop is
        identical whether streaming or not.
        """
        payload = self._build_payload(request)
        payload["stream"] = True

        text_parts: list[str] = []
        # index -> {"id","name","json": "<accumulated partial json>"}
        tool_blocks: dict[int, dict[str, Any]] = {}
        model: str | None = None
        usage_in: int | None = None
        usage_out: int | None = None

        for event in post_sse(
            self._url(),
            payload,
            headers=self._headers(),
            timeout=self.config.timeout,
        ):
            etype = event.get("type")
            if etype == "message_start":
                message = event.get("message") or {}
                model = message.get("model")
                usage_in = (message.get("usage") or {}).get("input_tokens")
            elif etype == "message_delta":
                # The closing delta carries the final output token count.
                out = (event.get("usage") or {}).get("output_tokens")
                if out is not None:
                    usage_out = out
            elif etype == "content_block_start":
                index = event.get("index", 0)
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    tool_blocks[index] = {
                        "id": block.get("id") or f"tool_call_{index}",
                        "name": block.get("name") or "",
                        "json": "",
                    }
            elif etype == "content_block_delta":
                index = event.get("index", 0)
                delta = event.get("delta") or {}
                dtype = delta.get("type")
                if dtype == "text_delta":
                    chunk_text = delta.get("text") or ""
                    if chunk_text:
                        text_parts.append(chunk_text)
                        yield StreamChunk(type="text", text=chunk_text)
                elif dtype == "input_json_delta" and index in tool_blocks:
                    tool_blocks[index]["json"] += delta.get("partial_json") or ""
            # content_block_stop / message_delta / message_stop need no action;
            # tool args are parsed below once the stream completes.

        tool_calls: list[ToolCall] = []
        for index in sorted(tool_blocks):
            block = tool_blocks[index]
            raw_json = block["json"].strip()
            try:
                args = json.loads(raw_json) if raw_json else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(id=block["id"], name=block["name"], arguments=args)
            )

        usage = (
            TokenUsage(input_tokens=usage_in, output_tokens=usage_out)
            if usage_in is not None or usage_out is not None
            else None
        )
        yield StreamChunk(
            type="final",
            response=LLMResponse(
                content="".join(text_parts),
                tool_calls=tool_calls,
                model=model or self.config.model,
                # No single raw dict exists for a stream; provide a summary so
                # `response.raw` is not None only in the non-streaming path.
                raw={"stream": True, "model": model, "usage": {
                    "input_tokens": usage_in, "output_tokens": usage_out}},
                usage=usage,
            ),
        )

    def _system_text(self, messages: list[Message]) -> str:
        return "\n\n".join(message.content for message in messages if message.role == "system")

    def _messages_to_wire(self, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "assistant":
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                wire.append({"role": "assistant", "content": content or ""})
            elif message.role == "tool":
                wire.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": message.content,
                            }
                        ],
                    }
                )
            else:
                # User message — may include image attachments. Anthropic
                # expects a list of content blocks when there are images:
                # [{type: text, text: ...}, {type: image, source: {...}}, ...]
                if message.attachments:
                    user_parts: list[dict[str, Any]] = []
                    if message.content:
                        user_parts.append({"type": "text", "text": message.content})
                    for att in message.attachments:
                        mime, b64 = att.as_base64()
                        user_parts.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": b64,
                                },
                            }
                        )
                    wire.append({"role": "user", "content": user_parts})
                else:
                    wire.append({"role": "user", "content": message.content})
        return wire
