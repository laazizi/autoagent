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
    # NORMALISATION À LA FRONTIÈRE — le point délicat de tout ce fichier.
    #
    # OpenAI et Gemini rapportent un `input_tokens` qui INCLUT déjà la part
    # servie par le cache. Anthropic non : il rend `input_tokens` (l'entrée NON
    # mise en cache) et met à côté `cache_read_input_tokens` (lue depuis le
    # cache) et `cache_creation_input_tokens` (écrite dans le cache). Recopier
    # tel quel ferait dire à `TokenUsage` deux choses différentes selon le
    # fournisseur — et sous-évaluerait l'entrée dès que le cache mord, donc
    # fausserait `token_budget` en silence.
    #
    # On ramène donc tout le monde à la convention majoritaire : `input_tokens`
    # = TOUT ce qui est entré, `cached_tokens` = la part qui en venait du cache.
    entree = u.get("input_tokens")
    lu = u.get("cache_read_input_tokens")
    ecrit = u.get("cache_creation_input_tokens")
    if entree is not None and (lu is not None or ecrit is not None):
        entree = entree + (lu or 0) + (ecrit or 0)
    return TokenUsage(
        input_tokens=entree,
        output_tokens=u.get("output_tokens"),
        cached_tokens=lu,
    )


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
            # Anthropic est le seul des trois à exiger un marqueur EXPLICITE : le
            # cache couvre le préfixe `tools` + `system`, dans cet ordre, jusqu'au
            # dernier bloc marqué. Marquer le bloc système met donc les schémas
            # d'outils dans le cache par la même occasion — d'où un seul marqueur.
            # Sans `cache_prompt`, on garde la chaîne nue : payload plus court, et
            # aucune écriture de cache facturée pour un préfixe qui ne sert qu'une
            # fois.
            if self.config.cache_prompt:
                payload["system"] = [{
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
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
        return self._with_extra_body(payload)

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
        usage_start: dict[str, Any] | None = None   # bloc `usage` du message_start
        usage_out: int | None = None
        emis: set[int] = set()

        def _assemble(block: dict[str, Any]) -> ToolCall:
            raw_json = block["json"].strip()
            try:
                args = json.loads(raw_json) if raw_json else {}
            except json.JSONDecodeError:
                args = {}
            return ToolCall(id=block["id"], name=block["name"], arguments=args)

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
                # On garde le bloc ENTIER : il porte aussi les compteurs de
                # cache, et c'est `_usage_from` qui sait les normaliser. Ne
                # prélever que `input_tokens` ici sous-évaluerait l'entrée dès
                # que le cache mord — le bug que la version non streamée évite.
                usage_start = message.get("usage") or {}
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
            elif etype == "content_block_stop":
                # Le bloc tool_use est COMPLET : on l'émet tout de suite, sans
                # attendre message_stop — la boucle peut lancer un outil
                # idempotent pendant que le modèle émet les blocs suivants.
                index = event.get("index", 0)
                if index in tool_blocks and index not in emis:
                    emis.add(index)
                    yield StreamChunk(type="tool_call", tool_call=_assemble(tool_blocks[index]))
            # message_delta / message_stop need no action here.

        tool_calls: list[ToolCall] = [_assemble(tool_blocks[i]) for i in sorted(tool_blocks)]

        # Le compte d'entrée (cache compris) arrive dans `message_start`, celui
        # de sortie dans le dernier `message_delta` : on recompose le bloc que
        # `_usage_from` attend, pour que streamé et non streamé comptent pareil.
        usage = None
        if usage_start is not None or usage_out is not None:
            usage = _usage_from({**(usage_start or {}), "output_tokens": usage_out})
        yield StreamChunk(
            type="final",
            response=LLMResponse(
                content="".join(text_parts),
                tool_calls=tool_calls,
                model=model or self.config.model,
                # No single raw dict exists for a stream; provide a summary so
                # `response.raw` is not None only in the non-streaming path.
                raw={"stream": True, "model": model,
                     "usage": {**(usage_start or {}), "output_tokens": usage_out}},
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
