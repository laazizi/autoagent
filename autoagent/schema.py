from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "DEFAULT_API_KEY_ENVS",
    "ImageAttachment",
    "JsonDict",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "ModelConfig",
    "StreamChunk",
    "StreamEvent",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
]

JsonDict = dict[str, Any]


DEFAULT_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}


@dataclass
class ModelConfig:
    provider: str
    model: str
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    timeout: float = 60.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_name = self.api_key_env or DEFAULT_API_KEY_ENVS.get(self.provider.lower())
        if env_name:
            value = os.getenv(env_name)
            if value:
                return value
        raise ValueError(
            f"Missing API key for provider '{self.provider}'. "
            f"Set api_key or environment variable {env_name!r}."
        )


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: JsonDict = field(default_factory=lambda: {"type": "object", "properties": {}})
    permissions: list[str] = field(default_factory=list)

    def as_openai_tool(self) -> JsonDict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def as_anthropic_tool(self) -> JsonDict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def as_gemini_declaration(self) -> JsonDict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _sanitize_schema_for_gemini(self.input_schema),
        }


# Gemini's function-declaration parameter schema is a subset of OpenAPI 3.0
# Schema — NOT full JSON Schema. Specifically, fields like
# `additionalProperties` are unknown and cause:
#   "Unknown name 'additionalProperties' ... Cannot find field"
# We strip the known offenders recursively before sending. This is a
# stricter-than-needed safety net: we keep `properties` / `items` /
# `enum` / `required` / `description` / `type` / `format` etc. that
# Gemini does accept.
_GEMINI_SCHEMA_BLOCKLIST = frozenset(
    {
        "additionalProperties",
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "patternProperties",
        "unevaluatedProperties",
    }
)


def _sanitize_schema_for_gemini(node: Any) -> Any:
    """Recursively strip JSON Schema fields that Gemini's tool API rejects.

    Keeps every node otherwise intact (description, type, properties,
    items, enum, required, ...). Walks dicts and lists so nested schemas
    inside `properties.<name>` or `items` are also sanitised.
    """
    if isinstance(node, dict):
        return {
            k: _sanitize_schema_for_gemini(v) for k, v in node.items() if k not in _GEMINI_SCHEMA_BLOCKLIST
        }
    if isinstance(node, list):
        return [_sanitize_schema_for_gemini(item) for item in node]
    return node


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: JsonDict = field(default_factory=dict)
    # `thought_signature` is Gemini-specific (Gemini 3+ thinking models).
    # When a thinking model emits a function call it returns an encrypted
    # `thoughtSignature` that the host MUST echo back on the assistant
    # message of the next request, otherwise Gemini 3 rejects with a
    # hard 400 ("Function call is missing a thought_signature").
    # Other providers ignore this field silently. Stored on the ToolCall
    # so it survives history serialisation and provider switching
    # (RoutingProvider).
    thought_signature: str | None = None

    def to_dict(self) -> JsonDict:
        """Serialise to a plain JSON-safe dict (added in 0.7.0).

        Round-trips losslessly through ``ToolCall.from_dict``. Use this
        to persist conversations across HTTP requests / processes (chat
        sessions, queue workers, audit logs).
        """
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "thought_signature": self.thought_signature,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ToolCall":
        """Rebuild a ``ToolCall`` from a dict produced by ``to_dict``.

        Tolerant of missing optional fields so an older snapshot still
        loads after the schema gains new optional fields. Required
        fields (``id``, ``name``) raise ``KeyError`` if absent.
        """
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data.get("arguments") or {},
            thought_signature=data.get("thought_signature"),
        )


@dataclass
class ImageAttachment:
    """An image attached to a user message for multimodal LLM calls.

    `data` accepts:
      - a base64 payload alone (`mime_type` then required), or
      - a full data URL (`data:image/jpeg;base64,...`), or
      - a public HTTPS URL.
    Each provider serializes it into its own wire format (OpenAI
    `image_url`, Anthropic `image`, Gemini `inline_data`); hosts don't
    care about the differences.
    """

    data: str
    mime_type: str | None = None

    def as_data_url(self) -> str:
        if not self.data:
            raise ValueError("ImageAttachment.data is empty")
        if self.data.startswith(("data:", "http://", "https://")):
            return self.data
        if not self.mime_type:
            raise ValueError("mime_type is required for raw base64 data")
        return f"data:{self.mime_type};base64,{self.data}"

    def as_base64(self) -> tuple[str, str]:
        """Return ``(mime_type, raw_base64)``. Useful for providers that
        don't accept data URLs (Anthropic, Gemini).

        Accepted shapes for ``self.data``:
          * canonical base64 data URL: ``data:image/png;base64,<b64>``
          * raw base64 payload (then ``self.mime_type`` MUST be set)

        Raises ``ValueError`` for remote URLs, missing MIME, or
        non-base64 data URLs (the older ``data:image/png,<urlencoded>``
        form is rejected because Anthropic/Gemini expect raw base64).
        """
        if self.data.startswith("data:"):
            header, _, payload = self.data.partition(",")
            # `header` is "data:image/png;base64" in the canonical case.
            # Reject the rare URL-encoded variant (data:image/png,...)
            # explicitly — silently returning URL-encoded bytes as if
            # they were base64 would corrupt the request to the LLM.
            if ";base64" not in header:
                raise ValueError(
                    "Data URL must be base64-encoded (got: data:<mime>,...). "
                    "Re-encode the payload before constructing ImageAttachment."
                )
            media_part = header[len("data:") :]
            mime = media_part.split(";", 1)[0] or self.mime_type or ""
            if not mime:
                raise ValueError("Could not determine MIME type from data URL")
            return mime, payload
        if self.data.startswith(("http://", "https://")):
            raise ValueError("Remote URLs cannot be re-encoded as base64")
        if not self.mime_type:
            raise ValueError("mime_type is required for raw base64 data")
        return self.mime_type, self.data

    def to_dict(self) -> JsonDict:
        """Serialise to a plain JSON-safe dict (added in 0.7.0)."""
        return {"data": self.data, "mime_type": self.mime_type}

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ImageAttachment":
        """Rebuild from a dict produced by ``to_dict``."""
        return cls(data=data["data"], mime_type=data.get("mime_type"))


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Optional image attachments. Providers serialize each into their own
    # multimodal wire format. Only meaningful on user messages.
    attachments: list[ImageAttachment] = field(default_factory=list)
    # `reasoning_content` carries the "thinking" trace emitted by reasoning
    # models (DeepSeek thinking mode, OpenAI o-series with reveal). Some
    # providers REQUIRE the host to echo it back in the next request.
    reasoning_content: str | None = None

    def to_dict(self) -> JsonDict:
        """Serialise to a plain JSON-safe dict (added in 0.7.0).

        Round-trips losslessly through ``Message.from_dict``. Use this
        to persist conversation history across HTTP requests / processes.
        Empty optional fields are omitted to keep snapshots compact.
        """
        out: JsonDict = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            out["name"] = self.name
        if self.tool_calls:
            out["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.attachments:
            out["attachments"] = [att.to_dict() for att in self.attachments]
        if self.reasoning_content is not None:
            out["reasoning_content"] = self.reasoning_content
        return out

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Message":
        """Rebuild a ``Message`` from a dict produced by ``to_dict``.

        Tolerant of missing optional fields so older snapshots still
        load after the schema gains new optional fields. ``role`` is
        the only strictly required key.
        """
        return cls(
            role=data["role"],  # type: ignore[arg-type]
            content=data.get("content", ""),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls") or []],
            attachments=[ImageAttachment.from_dict(a) for a in data.get("attachments") or []],
            reasoning_content=data.get("reasoning_content"),
        )


@dataclass
class LLMRequest:
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    tool_choice: str | None = "auto"
    # Structured output (0.10.0). ``{"type": "json_object"}`` asks the model
    # for a single valid JSON object. Provider mapping:
    #   * OpenAI-compatible — passed through verbatim (also accepts the
    #     richer ``{"type": "json_schema", "json_schema": {...}}`` form);
    #   * Gemini — ``generationConfig.responseMimeType = application/json``;
    #   * Anthropic — no native JSON mode: a strict "JSON only, no fences"
    #     system instruction is appended (best effort — keep a tolerant
    #     parser on the caller side).
    response_format: dict[str, Any] | None = None


@dataclass
class TokenUsage:
    """Token accounting for one provider call (added in 0.10.0).

    ``input_tokens``/``output_tokens`` are ``None`` when the provider did
    not report them (never invented). ``total_tokens`` falls back to the
    sum of the two when the provider omits an explicit total.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.total_tokens is None and (
            self.input_tokens is not None or self.output_tokens is not None
        ):
            self.total_tokens = (self.input_tokens or 0) + (self.output_tokens or 0)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    model: str | None = None
    reasoning_content: str | None = None
    usage: TokenUsage | None = None


@dataclass
class StreamChunk:
    """A piece of a streaming provider response (added in 0.8.0).

    A provider's ``stream()`` yields zero or more ``"text"`` chunks as
    the model emits text, then EXACTLY ONE ``"final"`` chunk carrying
    the fully-assembled ``LLMResponse`` (content + tool_calls +
    reasoning). The agent loop consumes text chunks to emit live
    deltas and uses the final chunk to drive tool execution — exactly
    like the non-streaming ``complete()`` path.

    Providers without native streaming fall back (in ``LLMProvider.
    stream``) to calling ``complete()`` and emitting the whole content
    as one ``"text"`` chunk followed by the ``"final"`` chunk, so the
    streaming API is uniform across every provider.
    """

    type: Literal["text", "final"]
    text: str = ""
    response: "LLMResponse | None" = None


@dataclass
class StreamEvent:
    """A high-level event emitted by ``Agent.run_stream`` (added 0.8.0).

    Event types:
      * ``text`` — an incremental chunk of assistant text. Append it to
        the current bubble as it arrives.
      * ``tool_start`` — a tool call is about to execute (``tool_name``).
      * ``tool_end`` — a tool finished (``tool_name`` + ``tool_status``
        = ``"ok"`` | ``"error"``).
      * ``correction`` — the post_turn_hook injected a correction
        (``text`` carries it); another iteration follows.
      * ``done`` — the run finished. ``output`` is the final assistant
        text, ``messages`` the full conversation (persist this),
        ``steps`` the iteration count.
      * ``error`` — the run aborted (``error`` carries the reason:
        ``"cancelled"``, ``"max_steps"``, or an exception string).
    """

    type: Literal["text", "tool_start", "tool_end", "correction", "done", "error"]
    text: str = ""
    tool_name: str | None = None
    tool_status: str | None = None
    output: str = ""
    messages: list[Message] = field(default_factory=list)
    steps: int = 0
    error: str = ""
    usage: "TokenUsage | None" = None  # sur l'événement done (0.10.0)
    # Sur l'événement ``error`` "approval_required: ..." (0.11.0) : le
    # snapshot RunState à passer à Agent.resume() une fois l'humain décidé.
    state: Any = None
