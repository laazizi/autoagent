"""Per-request dispatch across multiple LLM providers.

Use when the same agent loop should send text-only turns to a cheap text
provider (e.g. DeepSeek) but route any turn carrying an image to a
vision-capable provider (e.g. Gemini). The choice is made on each
`complete()` call, based on the content of the request.

Why this lives in the lib rather than in each host: text-only providers
crash hard on `image_url` parts ("unknown variant image_url"). Hosts
need a single object that exposes the `LLMProvider` contract and hides
the routing decision from `Agent`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace

from autoagent.logging import get_logger
from autoagent.schema import LLMRequest, LLMResponse, StreamChunk

from .base import LLMProvider

_log = get_logger("providers.routing")


class RoutingProvider(LLMProvider):
    """Dispatch each `LLMRequest` to a sub-provider based on its content.

    Default behaviour: if the latest user message has any image
    attachment, route to ``vision``; otherwise route to ``default`` AND
    strip attachments from the message history so text-only providers
    don't choke on past ``image_url`` parts.

    Custom routing: pass a ``router`` callable for any other policy
    (e.g. "small model for prompts < 1000 tokens, large for longer").
    The router returns the chosen ``LLMProvider``; stripping still
    applies — if the returned provider is the configured ``vision``,
    attachments are preserved; otherwise they are stripped when
    ``strip_attachments_for_default`` is true.

    ``self.config`` proxies to ``default.config`` so callers that read
    ``agent.provider.config.model`` keep working transparently.
    """

    def __init__(
        self,
        *,
        default: LLMProvider,
        vision: LLMProvider | None = None,
        router: Callable[[LLMRequest], LLMProvider] | None = None,
        strip_attachments_for_default: bool = True,
    ) -> None:
        # Do NOT call super().__init__ — we proxy config instead of owning one.
        self._default = default
        self._vision = vision
        self._router = router
        self._strip = strip_attachments_for_default
        self.config = default.config

    def complete(self, request: LLMRequest) -> LLMResponse:
        chosen = self._choose(request)
        if chosen is not self._vision and self._strip:
            request = self._with_stripped_attachments(request)
        return chosen.complete(request)

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        # Same routing + stripping policy as complete(). Without this
        # override, the base-class fallback would call self.complete() and
        # silently lose the chosen provider's NATIVE streaming.
        chosen = self._choose(request)
        if chosen is not self._vision and self._strip:
            request = self._with_stripped_attachments(request)
        yield from chosen.stream(request)

    def _choose(self, request: LLMRequest) -> LLMProvider:
        if self._router is not None:
            return self._router(request)
        return self._default_route(request)

    def _default_route(self, request: LLMRequest) -> LLMProvider:
        last_user_idx = max(
            (i for i, m in enumerate(request.messages) if m.role == "user"),
            default=None,
        )
        if last_user_idx is None:
            return self._default
        if not request.messages[last_user_idx].attachments:
            return self._default
        if self._vision is None:
            count = len(request.messages[last_user_idx].attachments)
            _log.warning(
                "RoutingProvider: latest user message has %s attachment(s) "
                "but no vision provider is configured; falling back to default "
                "with attachment stripping.",
                count,
            )
            return self._default
        return self._vision

    @staticmethod
    def _with_stripped_attachments(request: LLMRequest) -> LLMRequest:
        """Return a NEW request whose every message has empty attachments.

        Only ``attachments`` is touched; ``content``, ``tool_calls``,
        ``reasoning_content``, ``tool_call_id``, ``name`` and the
        request-level ``tools``/``temperature``/``max_tokens``/
        ``tool_choice`` fields are preserved exactly.
        """
        new_messages = [replace(m, attachments=[]) if m.attachments else m for m in request.messages]
        return replace(request, messages=new_messages)
