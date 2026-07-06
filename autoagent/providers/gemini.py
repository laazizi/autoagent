from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

from autoagent.http import post_json, post_sse
from autoagent.schema import LLMRequest, LLMResponse, Message, StreamChunk, TokenUsage, ToolCall

from .base import LLMProvider


def _usage_from(meta: Any) -> TokenUsage | None:
    if not isinstance(meta, dict):
        return None
    return TokenUsage(
        input_tokens=meta.get("promptTokenCount"),
        output_tokens=meta.get("candidatesTokenCount"),
        total_tokens=meta.get("totalTokenCount"),
    )


def _gemini_fix_arrays(node: Any) -> Any:
    """Gemini REFUSE (HTTP 400) tout schéma `{"type": "array"}` sans `items`,
    là où OpenAI/Anthropic/DeepSeek le tolèrent. On injecte un `items` par défaut
    (string) partout où il manque, récursivement — y compris dans les schémas des
    outils générés dynamiquement. Mutation en place du dict de déclaration."""
    if isinstance(node, dict):
        if node.get("type") == "array" and "items" not in node:
            node["items"] = {"type": "string"}
        for value in node.values():
            _gemini_fix_arrays(value)
    elif isinstance(node, list):
        for item in node:
            _gemini_fix_arrays(item)
    return node


class GeminiProvider(LLMProvider):
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"x-goog-api-key": self.config.resolved_api_key()}
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def _url(self, method: str) -> str:
        model = quote(self.config.model, safe="")
        base = (self.config.base_url or self.default_base_url).rstrip("/")
        return f"{base}/models/{model}:{method}"

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": self._messages_to_contents(request.messages),
        }
        system_text = self._system_text(request.messages)
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        if request.tools:
            payload["tools"] = [
                {"functionDeclarations": [
                    _gemini_fix_arrays(tool.as_gemini_declaration()) for tool in request.tools
                ]}
            ]
            choice = request.tool_choice
            if choice and choice != "auto":
                if choice == "none":
                    mode_config: dict[str, Any] = {"mode": "NONE"}
                elif choice in ("required", "any"):
                    mode_config = {"mode": "ANY"}
                else:  # a specific tool name: force a call to that tool
                    mode_config = {"mode": "ANY", "allowedFunctionNames": [choice]}
                payload["toolConfig"] = {"functionCallingConfig": mode_config}
        generation_config: dict[str, Any] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.response_format is not None:
            # Gemini's JSON mode: the response is a single JSON document
            # (no markdown fences). Schema enforcement (responseSchema) is
            # deliberately not mapped — its OpenAPI dialect diverges from
            # JSON Schema; validate on the caller side instead.
            generation_config["responseMimeType"] = "application/json"
        if generation_config:
            payload["generationConfig"] = generation_config
        return payload

    def _parse_parts(
        self, parts: list[dict[str, Any]], tool_calls: list[ToolCall], text_parts: list[str]
    ) -> Iterator[str]:
        """Walk a list of Gemini content parts, appending text + tool calls.

        Yields each text fragment so the streaming path can emit deltas;
        the non-streaming path just drains the generator.
        """
        for part in parts:
            if "text" in part:
                # Thinking models (Gemini 3.5+) interleave thought-summary
                # parts flagged ``"thought": true`` with the real answer.
                # Those are internal reasoning — leaking them into content
                # shows markdown debris ("**Considering the greeting**…")
                # to the end user. Skip them; only emit answer text.
                if part.get("thought"):
                    continue
                fragment = part.get("text") or ""
                if fragment:
                    text_parts.append(fragment)
                    yield fragment
            if "functionCall" in part:
                call = part["functionCall"]
                signature = call.get("thoughtSignature") or part.get("thoughtSignature")
                tool_calls.append(
                    ToolCall(
                        id=f"gemini_tool_call_{len(tool_calls)}",
                        name=call.get("name") or "",
                        arguments=call.get("args") or {},
                        thought_signature=signature,
                    )
                )

    def complete(self, request: LLMRequest) -> LLMResponse:
        raw = post_json(
            self._url("generateContent"),
            self._build_payload(request),
            headers=self._headers(),
            timeout=self.config.timeout,
        )
        candidate = (raw.get("candidates") or [{}])[0]
        content = candidate.get("content") or {}
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        # Drain the parser (non-streaming: we don't need the yielded deltas).
        for _ in self._parse_parts(content.get("parts") or [], tool_calls, text_parts):
            pass
        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            raw=raw,
            model=self.config.model,
            usage=_usage_from(raw.get("usageMetadata")),
        )

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        """Native SSE streaming via Gemini's ``streamGenerateContent``.

        Each SSE event is a partial ``GenerateContentResponse``. Text
        parts arrive incrementally (emitted as ``text`` chunks);
        ``functionCall`` parts arrive complete and are collected. The
        final assembled ``LLMResponse`` matches what ``complete()``
        would return.
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage_meta: dict[str, Any] | None = None

        for event in post_sse(
            f"{self._url('streamGenerateContent')}?alt=sse",
            self._build_payload(request),
            headers=self._headers(),
            timeout=self.config.timeout,
        ):
            candidate = (event.get("candidates") or [{}])[0]
            content = candidate.get("content") or {}
            if isinstance(event.get("usageMetadata"), dict):
                usage_meta = event["usageMetadata"]  # cumulative; last one wins
            for fragment in self._parse_parts(content.get("parts") or [], tool_calls, text_parts):
                yield StreamChunk(type="text", text=fragment)

        yield StreamChunk(
            type="final",
            response=LLMResponse(
                content="".join(text_parts),
                tool_calls=tool_calls,
                model=self.config.model,
                # No single raw dict exists for a stream; keep parity with the
                # non-streaming path by providing at least a summary.
                raw={"stream": True, "model": self.config.model, "usageMetadata": usage_meta},
                usage=_usage_from(usage_meta),
            ),
        )

    def _system_text(self, messages: list[Message]) -> str:
        return "\n\n".join(message.content for message in messages if message.role == "system")

    def _messages_to_contents(self, messages: list[Message]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        # Gemini matches a functionResponse to its call by NAME (OpenAI and
        # Anthropic match by id). If the host forgot to set `Message.name` on
        # a tool message, recover it from the assistant tool_calls via
        # tool_call_id — otherwise the default "tool" mismatches every
        # function as soon as two tools exist.
        call_names = {
            call.id: call.name
            for message in messages
            for call in (message.tool_calls or [])
            if call.id and call.name
        }
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "assistant":
                parts: list[dict[str, Any]] = []
                if message.content:
                    parts.append({"text": message.content})
                for call in message.tool_calls:
                    # ASYMMETRIC API: in RESPONSES Gemini nests
                    # `thoughtSignature` INSIDE `functionCall`. In
                    # REQUESTS it must be at the PART level (sibling of
                    # `functionCall`). Putting it inside `functionCall`
                    # in a request yields a 400 "Unknown name
                    # thoughtSignature ... Cannot find field".
                    part: dict[str, Any] = {
                        "functionCall": {
                            "name": call.name,
                            "args": call.arguments,
                        },
                    }
                    if call.thought_signature:
                        part["thoughtSignature"] = call.thought_signature
                    parts.append(part)
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            elif message.role == "tool":
                contents.append(
                    {
                        "role": "tool",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message.name
                                    or call_names.get(message.tool_call_id or "", "tool"),
                                    "response": {"result": message.content},
                                }
                            }
                        ],
                    }
                )
            else:
                # User message — may include image attachments via Gemini's
                # `inline_data: {mime_type, data}` part format.
                user_parts: list[dict[str, Any]] = []
                if message.content:
                    user_parts.append({"text": message.content})
                for att in message.attachments:
                    mime, b64 = att.as_base64()
                    user_parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                if not user_parts:
                    user_parts = [{"text": ""}]
                contents.append({"role": "user", "parts": user_parts})
        return contents
