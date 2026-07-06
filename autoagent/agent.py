from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .dynamic import DynamicToolBuilder, ToolBuildRequest
from .errors import (
    AgentCancelled,
    AutoAgentError,
    MaxStepsExceeded,
    TokenBudgetExceeded,
    ToolError,
)
from .logging import get_logger
from .memory import Memory
from .providers import create_provider
from .providers.base import LLMProvider
from .registry import ToolRegistry
from .schema import (
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from .trace import TraceEmitter, truncate_preview

__all__ = ["Agent", "AgentResult", "AgentTurnContext", "PostTurnHook"]

_log = get_logger("agent")

DEFAULT_SYSTEM_PROMPT = """You are an AI agent with tools.
Use tools when they are useful. If a required capability is missing and the
create_python_tool tool is available, create a small focused tool first, then use it.
Keep final answers concise and grounded in tool results."""


@dataclass
class AgentResult:
    output: str
    messages: list[Message]
    steps: int
    # Total tokens du run (somme des usages rapportés par le provider).
    # None quand aucun appel n'a rapporté d'usage. Added in 0.10.0.
    usage: TokenUsage | None = None


@dataclass
class AgentTurnContext:
    """Snapshot passed to a `post_turn_hook` when the agent would naturally
    end a turn (LLM emitted a text-only response).

    The hook receives this context and decides whether to confirm the turn
    (return `None`) or inject a correction (`Message(role="user", ...)`)
    that triggers another agent iteration.

    Attributes:
        messages: Full conversation up to this point, including the just-
            emitted assistant message.
        new_messages: Messages produced since the most recent user/system
            input — the assistant + any tool messages from this user turn.
        tool_calls: Flat list of every tool call made during the messages
            in `new_messages`. Convenient for hosts that want to verify
            specific actions (e.g. "did the agent write any file?").
        correction_count: 0 on the first hook invocation for this run;
            incremented each time the hook injects a correction.
    """

    messages: list[Message]
    new_messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    correction_count: int = 0


PostTurnHook = Callable[["AgentTurnContext"], "Message | None"]


def _tool_message(call: ToolCall, tool_result: Any) -> Message:
    """The transcript message carrying one tool result back to the LLM."""
    return Message(
        role="tool",
        name=call.name,
        tool_call_id=call.id,
        content=tool_result.to_message_content(),
    )


class Agent:
    """LLM agent that runs a tool-use loop until the model emits a final answer.

    Lifecycle:
        1. `run(prompt)` or `run_messages(messages)` is called.
        2. The agent sends the conversation + registered tool specs to
           the provider.
        3. If the response contains tool calls, each is validated against
           its `input_schema`, executed via the registry, and its result
           is appended to the conversation. The loop repeats.
        4. The loop stops on the first response with no tool calls
           (returned as `AgentResult.output`) or when `max_steps` is
           exceeded (raises `MaxStepsExceeded`).

    Thread-safety:
        A single `Agent` instance should be driven by ONE caller at a
        time — running `agent.run` concurrently from two threads will
        interleave conversations unpredictably. Tool execution itself
        is safe because `ToolRegistry` is internally locked.

    Args:
        provider: Concrete `LLMProvider` (use `create_provider` for a
            quick start, or pass a custom subclass).
        registry: Optional pre-populated `ToolRegistry`. A fresh empty
            one is created when omitted.
        system_prompt: Top-level instruction prepended to every run.
            Accepts a string (static) or a zero-arg callable that returns
            a string (re-evaluated at the start of every run). The
            callable form lets hosts inject live state — form progress,
            current step in a workflow, remaining questions — that the
            LLM should see fresh on each turn. Added in 0.7.0.
        max_steps: Hard cap on tool-call iterations per run.
        max_dynamic_tools_per_run: How many tools the agent may generate
            via `create_python_tool` in a single run (when
            `enable_dynamic_tools` has been called).
        temperature / max_tokens: Forwarded to the provider when set.
        post_turn_hook: Optional callback invoked when the LLM emits a
            text-only response (would normally end the run). The hook
            receives an `AgentTurnContext` and may return a `Message`
            to inject as a correction. Added in 0.2.0.
        max_corrections_per_run: Hard cap on the number of corrections
            `post_turn_hook` may inject in a single run. Defaults to 1
            to prevent loops. Added in 0.2.0.
        trace: Optional `TraceEmitter` that receives typed lifecycle
            events (run_start, llm_request, tool_call_start, ...). When
            ``None`` the agent emits nothing. Added in 0.5.0.
        parallel_tool_calls: When ``True`` and the model requests SEVERAL
            tools in one turn, they execute concurrently (thread pool)
            instead of one after the other — a direct latency win when
            tools are I/O-bound (HTTP, DB). OPT-IN because your tool
            handlers must then be thread-safe and they share the same
            ``context`` dict. Results are appended to the conversation
            in the model's call order regardless of completion order,
            so the transcript is deterministic. Added in 0.10.0.
        token_budget: Hard cap on the run's cumulative token usage
            (input + output, as reported by the provider). Checked
            BEFORE each provider call: once reached, the run raises
            ``TokenBudgetExceeded`` (streaming: a terminal ``error``
            event) instead of issuing another call. Only enforceable
            when the provider reports usage — unreported calls count
            as zero. Added in 0.10.0.
        memory: Optional `Memory` instance that shapes the conversation
            before each run. `Agent.run_messages` calls `compact()`
            ONCE before the loop. Errors raised by `compact()` are
            isolated and logged. Added in 0.6.0.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        registry: ToolRegistry | None = None,
        system_prompt: str | Callable[[], str] = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 8,
        max_dynamic_tools_per_run: int = 3,
        temperature: float | None = None,
        max_tokens: int | None = None,
        post_turn_hook: PostTurnHook | None = None,
        max_corrections_per_run: int = 1,
        trace: TraceEmitter | None = None,
        memory: Memory | None = None,
        parallel_tool_calls: bool = False,
        token_budget: int | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_dynamic_tools_per_run = max_dynamic_tools_per_run
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.post_turn_hook = post_turn_hook
        self.max_corrections_per_run = max_corrections_per_run
        self.trace = trace
        self.memory = memory
        self.parallel_tool_calls = parallel_tool_calls
        self.token_budget = token_budget
        self.dynamic_builder: DynamicToolBuilder | None = None
        self._dynamic_tools_built_this_run = 0

    @classmethod
    def from_model(
        cls,
        provider: str,
        model: str,
        **kwargs: Any,
    ) -> "Agent":
        return cls(create_provider(ModelConfig(provider=provider, model=model)), **kwargs)

    @classmethod
    def from_model_config(cls, config: ModelConfig, **kwargs: Any) -> "Agent":
        return cls(create_provider(config), **kwargs)

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        permissions: list[str] | None = None,
    ):
        return self.registry.register(
            func,
            name=name,
            description=description,
            input_schema=input_schema,
            permissions=permissions,
        )

    def add_tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        return self.registry.add_function(func)

    def enable_dynamic_tools(self, builder: DynamicToolBuilder) -> None:
        self.dynamic_builder = builder

        def create_python_tool(
            capability: str,
            tool_name: str | None = None,
            input_schema: dict[str, Any] | None = None,
            permissions: list[str] | None = None,
        ) -> dict[str, Any]:
            if self._dynamic_tools_built_this_run >= self.max_dynamic_tools_per_run:
                raise ToolError(
                    f"Dynamic tool budget exhausted for this run "
                    f"(max_dynamic_tools_per_run={self.max_dynamic_tools_per_run}). "
                    "Reuse an existing tool or finish the task with what is available."
                )
            generated = builder.build(
                ToolBuildRequest(
                    capability=capability,
                    tool_name=tool_name,
                    input_schema=input_schema,
                    permissions=permissions or [],
                )
            )
            self.registry.replace(generated.spec, generated)
            self._dynamic_tools_built_this_run += 1
            return {
                "registered": True,
                "tool": {
                    "name": generated.spec.name,
                    "description": generated.spec.description,
                    "input_schema": generated.spec.input_schema,
                    "permissions": generated.spec.permissions,
                },
            }

        self.registry.replace(
            spec=_create_python_tool_spec(),
            handler=create_python_tool,
        )

    def enable_evolution(
        self,
        runtime: Any,
        *,
        capabilities: set[str] | None = None,
    ) -> Any:
        from .evolution import enable_software_evolution

        return enable_software_evolution(self, runtime, capabilities=capabilities)

    def register_recall_tool(
        self,
        *,
        name: str = "recall",
        description: str | None = None,
        default_k: int = 5,
    ) -> None:
        """Register a ``recall`` tool that wraps ``self.memory.recall``.

        Use this when ``memory`` is a semantic store (vector-backed,
        summary-indexed, ...) and you want to give the agent explicit
        access to forgotten details. The tool returns the matching past
        messages as a JSON-serialisable list of ``{role, content}``.
        It is a no-op when no memory is configured — the registration
        is silently skipped.
        """
        if self.memory is None:
            return

        previewed = description or (
            "Search past conversation by semantic query and return matching messages. "
            "Use when you need a detail you no longer have in the current context."
        )

        # Look up self.memory dynamically at call time so reassigning
        # `agent.memory` after registration is honoured. Closing over
        # the value at registration time would silently shadow later
        # reassignments — a subtle and surprising bug for hosts that
        # swap memories between runs.
        agent_self = self

        def _recall(query: str, k: int = default_k) -> dict[str, Any]:
            mem = agent_self.memory
            if mem is None:
                return {"matches": [], "error": "Memory has been detached from the agent."}
            try:
                matches = mem.recall(query, k=k)
            except Exception as exc:
                return {"matches": [], "error": f"{type(exc).__name__}: {exc}"}
            return {
                "matches": [
                    {
                        "role": m.role,
                        "content": truncate_preview(m.content, limit=2000),
                    }
                    for m in matches
                ]
            }

        self.registry.replace(
            ToolSpec(
                name=name,
                description=previewed,
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Free-text question used for semantic match against past turns.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Maximum number of past messages to return.",
                            "default": default_k,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            handler=_recall,
        )

    def as_tool(
        self,
        *,
        name: str,
        description: str,
        request_description: str = "La demande à traiter, formulée en langage naturel.",
    ) -> Callable[..., Any]:
        """Expose THIS agent as a tool for another agent (0.10.0).

        The minimal multi-agent primitive — supervisor/specialist
        hierarchies in two lines::

            expert = Agent(cheap_provider, system_prompt="Expert comptage...")
            supervisor.add_tool(expert.as_tool(
                name="analyser_comptage",
                description="Délègue les questions de comptage à l'expert.",
            ))

        Semantics:
          * Each call starts a FRESH conversation on the sub-agent (its
            system prompt + the request) — stateless delegation. Give the
            sub-agent a ``memory`` if it should remember across calls.
          * The sub-agent keeps its own provider, tools, ``token_budget``
            and ``trace`` — share one ``TraceEmitter`` to see the whole
            swarm in a single trace tree.
          * The parent's ``context`` dict is forwarded to the sub-agent's
            run (host handles stay reachable).
          * Sub-agent failures (``MaxStepsExceeded``, ``ProviderError``…)
            surface as a TOOL ERROR to the parent LLM, which can react —
            they never crash the parent run.
          * The returned dict carries ``output``, ``steps`` and ``tokens``
            (when the provider reports usage) so the parent — and your
            transcript — see the delegation cost.

        Thread-safety: an ``Agent`` instance serves ONE caller at a time.
        If the parent uses ``parallel_tool_calls=True``, give each
        delegation tool its OWN sub-agent instance.
        """
        agent_self = self

        def handler(request: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
            result = agent_self.run(request, context=context)
            payload: dict[str, Any] = {"output": result.output, "steps": result.steps}
            if result.usage is not None:
                payload["tokens"] = result.usage.total_tokens
            return payload

        handler.__name__ = name
        handler.__autoagent_tool_spec__ = ToolSpec(  # type: ignore[attr-defined]
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": request_description},
                },
                "required": ["request"],
                "additionalProperties": False,
            },
        )
        return handler

    def render_system_prompt(self) -> str:
        """Resolve ``self.system_prompt`` to a string for the next run.

        Static strings are returned as-is. Callables are invoked with no
        arguments and their return value is coerced to ``str``. A buggy
        callable that raises is caught and logged — the run proceeds
        with the default prompt rather than crashing the agent. This
        matches the resilience contract of the other host-supplied
        callables (``post_turn_hook``, ``trace.emit``).

        Hosts that persist conversations across HTTP requests (FastAPI
        chat sessions, queue workers, ...) should call this on each turn
        and replace the system message in their stored history so the
        LLM always sees fresh state from the prompt callable.
        """
        prompt = self.system_prompt
        if callable(prompt):
            try:
                resolved = prompt()
            except Exception:
                _log.exception("system_prompt callable raised; falling back to DEFAULT_SYSTEM_PROMPT")
                return DEFAULT_SYSTEM_PROMPT
            return str(resolved) if resolved is not None else DEFAULT_SYSTEM_PROMPT
        return prompt

    def run(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
    ) -> AgentResult:
        messages = [
            Message(role="system", content=self.render_system_prompt()),
            Message(role="user", content=prompt),
        ]
        return self.run_messages(messages, context=context, cancel_token=cancel_token)

    def run_messages(
        self,
        messages: list[Message],
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
    ) -> AgentResult:
        # Thin wrapper over the single loop implementation (_run_loop).
        # Intermediate events (tool_start/tool_end/correction) are ignored;
        # exceptions raised inside the generator (AgentCancelled,
        # MaxStepsExceeded, provider/tool errors) propagate unchanged, so
        # the non-streaming contract is identical to pre-0.10 behaviour.
        for event in self._run_loop(
            messages, context=context, cancel_token=cancel_token, streaming=False
        ):
            if event.type == "done":
                return AgentResult(
                    output=event.output,
                    messages=event.messages,
                    steps=event.steps,
                    usage=event.usage,
                )
        raise AutoAgentError("agent loop ended without a result")  # pragma: no cover

    def run_stream(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        """Streaming counterpart of ``run``.

        Yields ``StreamEvent`` objects: ``text`` deltas as the model
        emits them, ``tool_start`` / ``tool_end`` around each tool
        execution, ``correction`` when the post_turn_hook injects one,
        and a final ``done`` (or ``error``) event. The ``done`` event
        carries the full ``output`` text, the complete ``messages``
        list (persist this), and the ``steps`` count — so a host that
        only wants the result can just consume until ``done``.

        Providers without native streaming degrade gracefully: their
        ``stream()`` fallback (in ``LLMProvider``) emits the whole
        answer as one ``text`` event then the final response.
        """
        messages = [
            Message(role="system", content=self.render_system_prompt()),
            Message(role="user", content=prompt),
        ]
        yield from self.run_messages_stream(
            messages, context=context, cancel_token=cancel_token
        )

    def run_messages_stream(
        self,
        messages: list[Message],
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        """Streaming counterpart of ``run_messages`` — see ``run_stream``."""
        # Same single loop as run_messages, but failures become terminal
        # ``error`` events: streaming consumers read events, they don't catch.
        try:
            yield from self._run_loop(
                messages, context=context, cancel_token=cancel_token, streaming=True
            )
        except AgentCancelled as exc:
            yield StreamEvent(type="error", error="cancelled", steps=getattr(exc, "step", 0))
        except MaxStepsExceeded as exc:
            yield StreamEvent(
                type="error",
                error=f"max_steps={self.max_steps} exceeded",
                messages=getattr(exc, "messages", []),
                steps=self.max_steps,
            )
        except TokenBudgetExceeded as exc:
            yield StreamEvent(
                type="error",
                error=f"token_budget={self.token_budget} exceeded (spent={getattr(exc, 'spent', '?')})",
                messages=getattr(exc, "messages", []),
            )
        except Exception as exc:
            yield StreamEvent(type="error", error=f"{type(exc).__name__}: {exc}")

    def _run_loop(
        self,
        messages: list[Message],
        *,
        context: dict[str, Any] | None,
        cancel_token: threading.Event | None,
        streaming: bool,
    ) -> Iterator[StreamEvent]:
        """THE agent loop — single implementation behind both public entry
        points (0.10.0; previously ``run_messages`` and
        ``run_messages_stream`` were ~150-line near-twins that had to be
        edited in lockstep).

        Yields ``StreamEvent``s: ``text`` deltas only when ``streaming``
        (they come from the provider's native stream), ``tool_start`` /
        ``tool_end`` / ``correction`` always, then exactly one ``done``.
        Failures RAISE — ``run_messages`` lets them propagate to the
        caller, ``run_messages_stream`` converts them into terminal
        ``error`` events. Trace emission is identical on both paths,
        modulo the ``streaming`` flag in run_start/llm_request payloads.
        """
        working_messages = list(messages)
        if self.memory is not None:
            # Compact ONCE before the loop. Doing it per-iteration would
            # invalidate turn_start mid-run and complicate the
            # post_turn_hook accounting. Hosts that need finer control
            # can call memory.compact() themselves before passing the
            # messages in.
            try:
                working_messages = list(self.memory.compact(working_messages))
            except Exception:
                _log.exception("memory.compact raised; using messages unchanged")
        self._dynamic_tools_built_this_run = 0
        corrections = 0
        turn_start = len(working_messages)
        spent_in = 0
        spent_out = 0
        have_usage = False
        model = getattr(getattr(self.provider, "config", None), "model", None)
        run_start_payload: dict[str, Any] = {
            "max_steps": self.max_steps,
            "model": model,
            "message_count": len(working_messages),
            "tool_count": len(self.registry.specs()),
        }
        if streaming:
            run_start_payload["streaming"] = True
        run_span = self._emit("run_start", run_start_payload)
        try:
            for step in range(1, self.max_steps + 1):
                # Cooperative cancellation: the host may set `cancel_token` to
                # abort the run between iterations. We check BEFORE the next
                # provider call so we don't waste a request when the user has
                # already pressed "Cancel".
                if cancel_token is not None and cancel_token.is_set():
                    self._emit("cancelled", {"step": step}, parent_id=run_span)
                    cancelled = AgentCancelled(f"Agent cancelled by caller at step {step}")
                    cancelled.step = step  # consumed by run_messages_stream
                    raise cancelled

                # Token budget: checked BEFORE the next provider call — the
                # call that crossed the line completed normally, we just
                # refuse to issue another one.
                spent = spent_in + spent_out
                if self.token_budget is not None and spent >= self.token_budget:
                    self._emit(
                        "token_budget_exceeded",
                        {"token_budget": self.token_budget, "spent": spent, "step": step},
                        parent_id=run_span,
                    )
                    self._emit(
                        "run_end",
                        {"status": "token_budget", "steps": step - 1},
                        parent_id=run_span,
                    )
                    exhausted = TokenBudgetExceeded(
                        f"Run token budget exhausted: spent {spent} >= budget {self.token_budget}"
                    )
                    exhausted.messages = working_messages  # consumed by run_messages_stream
                    exhausted.spent = spent
                    raise exhausted

                request_payload: dict[str, Any] = {
                    "step": step,
                    "message_count": len(working_messages),
                    "tool_count": len(self.registry.specs()),
                }
                if streaming:
                    request_payload["streaming"] = True
                req_span = self._emit("llm_request", request_payload, parent_id=run_span)

                request = LLMRequest(
                    messages=working_messages,
                    tools=self.registry.specs(),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if streaming:
                    # Drive the provider's streaming generator, re-emitting
                    # text deltas and capturing the assembled final response.
                    final_response: LLMResponse | None = None
                    for chunk in self.provider.stream(request):
                        if chunk.type == "text" and chunk.text:
                            yield StreamEvent(type="text", text=chunk.text)
                        elif chunk.type == "final":
                            final_response = chunk.response
                    if final_response is None:
                        # Provider yielded no final chunk — treat as empty answer.
                        final_response = LLMResponse(content="", model=model)
                    response = final_response
                else:
                    response = self.provider.complete(request)

                self._emit(
                    "llm_response",
                    {
                        "step": step,
                        "content_preview": truncate_preview(response.content),
                        "tool_call_count": len(response.tool_calls),
                        "has_reasoning": response.reasoning_content is not None,
                        "input_tokens": response.usage.input_tokens if response.usage else None,
                        "output_tokens": response.usage.output_tokens if response.usage else None,
                    },
                    parent_id=req_span,
                )
                if response.usage is not None:
                    have_usage = True
                    spent_in += response.usage.input_tokens or 0
                    spent_out += response.usage.output_tokens or 0
                working_messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                        reasoning_content=response.reasoning_content,
                    )
                )

                if not response.tool_calls:
                    # Would normally end the run. If a post_turn_hook is
                    # configured and we still have correction budget, give it
                    # a chance to request another iteration.
                    correction = self._maybe_invoke_post_turn_hook(
                        working_messages, turn_start, corrections, parent_id=req_span
                    )
                    if correction is not None:
                        working_messages.append(correction)
                        corrections += 1
                        turn_start = len(working_messages)
                        yield StreamEvent(type="correction", text=correction.content)
                        continue
                    self._emit(
                        "run_end",
                        {
                            "status": "ok",
                            "steps": step,
                            "output_preview": truncate_preview(response.content),
                        },
                        parent_id=run_span,
                    )
                    yield StreamEvent(
                        type="done",
                        output=response.content,
                        messages=working_messages,
                        steps=step,
                        usage=(
                            TokenUsage(input_tokens=spent_in, output_tokens=spent_out)
                            if have_usage
                            else None
                        ),
                    )
                    return

                if self.parallel_tool_calls and len(response.tool_calls) > 1:
                    # Concurrent execution (opt-in). Starts are announced in
                    # call order, every call runs in a thread pool, then ends
                    # and transcript messages follow the SAME call order —
                    # the conversation stays deterministic whatever the
                    # completion order.
                    spans: list[str | None] = []
                    for call in response.tool_calls:
                        spans.append(self._emit_tool_start(call, req_span))
                        yield StreamEvent(type="tool_start", tool_name=call.name)

                    def _timed(call: ToolCall) -> tuple[Any, int]:
                        started_at = time.monotonic()
                        result = self.registry.execute(call, context=context)
                        return result, int((time.monotonic() - started_at) * 1000)

                    with ThreadPoolExecutor(
                        max_workers=min(len(response.tool_calls), 8),
                        thread_name_prefix="autoagent-tool",
                    ) as pool:
                        outcomes = list(pool.map(_timed, response.tool_calls))
                    for call, tool_span, (tool_result, duration_ms) in zip(
                        response.tool_calls, spans, outcomes
                    ):
                        self._emit_tool_end(call, tool_span, tool_result, duration_ms)
                        yield StreamEvent(
                            type="tool_end",
                            tool_name=call.name,
                            tool_status="ok" if tool_result.ok else "error",
                        )
                        working_messages.append(_tool_message(call, tool_result))
                else:
                    for call in response.tool_calls:
                        tool_span = self._emit_tool_start(call, req_span)
                        yield StreamEvent(type="tool_start", tool_name=call.name)
                        started_at = time.monotonic()
                        tool_result = self.registry.execute(call, context=context)
                        duration_ms = int((time.monotonic() - started_at) * 1000)
                        self._emit_tool_end(call, tool_span, tool_result, duration_ms)
                        yield StreamEvent(
                            type="tool_end",
                            tool_name=call.name,
                            tool_status="ok" if tool_result.ok else "error",
                        )
                        working_messages.append(_tool_message(call, tool_result))

            self._emit("max_steps_exceeded", {"max_steps": self.max_steps}, parent_id=run_span)
            self._emit(
                "run_end",
                {"status": "max_steps", "steps": self.max_steps},
                parent_id=run_span,
            )
            exceeded = MaxStepsExceeded(f"Agent exceeded max_steps={self.max_steps}")
            exceeded.messages = working_messages  # consumed by run_messages_stream
            raise exceeded
        except AgentCancelled:
            self._emit("run_end", {"status": "cancelled"}, parent_id=run_span)
            raise
        except (MaxStepsExceeded, TokenBudgetExceeded):
            raise  # run_end already emitted at the raise site
        except Exception:
            self._emit("run_end", {"status": "error"}, parent_id=run_span)
            raise

    def _emit_tool_start(self, call: ToolCall, req_span: str | None) -> str | None:
        return self._emit(
            "tool_call_start",
            {
                "name": call.name,
                "call_id": call.id,
                "arguments_preview": truncate_preview(call.arguments),
            },
            parent_id=req_span,
        )

    def _emit_tool_end(
        self, call: ToolCall, tool_span: str | None, tool_result: Any, duration_ms: int
    ) -> None:
        self._emit(
            "tool_call_end",
            {
                "name": call.name,
                "call_id": call.id,
                "status": "ok" if tool_result.ok else "error",
                "duration_ms": duration_ms,
                "content_preview": truncate_preview(
                    tool_result.result if tool_result.ok else tool_result.error
                ),
            },
            parent_id=tool_span,
        )

    def _emit(
        self,
        type_: str,
        payload: dict[str, Any] | None = None,
        *,
        parent_id: str | None = None,
    ) -> str | None:
        """Forward a trace event to the configured emitter, if any.

        Returns the emitted ``span_id`` (or ``None`` if no emitter is
        configured). Trace failures never propagate to the caller — the
        emitter already swallows its own errors, but we add a second
        guard here in case the emitter itself raises.
        """
        if self.trace is None:
            return None
        try:
            return self.trace.emit(type_, payload, parent_id=parent_id)
        except Exception:
            _log.exception("trace emit failed; continuing")
            return None

    def _maybe_invoke_post_turn_hook(
        self,
        working_messages: list[Message],
        turn_start: int,
        corrections: int,
        *,
        parent_id: str | None = None,
    ) -> Message | None:
        """Invoke the user-supplied post_turn_hook if eligible.

        Eligible means: a hook is configured AND the correction budget
        is not yet exhausted. Hook exceptions are caught and logged so
        that a buggy verifier cannot break the agent for the caller.
        """
        if self.post_turn_hook is None:
            return None
        if corrections >= self.max_corrections_per_run:
            return None
        new_messages = working_messages[turn_start:]
        tool_calls: list[ToolCall] = []
        for msg in new_messages:
            if msg.role == "assistant":
                tool_calls.extend(msg.tool_calls)
        ctx = AgentTurnContext(
            messages=list(working_messages),
            new_messages=list(new_messages),
            tool_calls=tool_calls,
            correction_count=corrections,
        )
        hook_span = self._emit(
            "post_turn_hook_invoked",
            {"correction_count": corrections},
            parent_id=parent_id,
        )
        try:
            correction = self.post_turn_hook(ctx)
        except Exception:
            _log.exception("post_turn_hook raised; ignoring correction")
            return None
        if correction is not None:
            self._emit(
                "post_turn_hook_correction",
                {"content_preview": truncate_preview(correction.content)},
                parent_id=hook_span,
            )
        return correction


def _create_python_tool_spec():
    from .schema import ToolSpec

    return ToolSpec(
        name="create_python_tool",
        description=(
            "Create and register a new small Python tool when a required capability is missing. "
            "After this tool succeeds, call the newly registered tool by name."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "The missing capability the new tool must provide.",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Optional snake_case name for the new tool.",
                },
                "input_schema": {
                    "type": "object",
                    "description": "Optional JSON schema for the new tool arguments.",
                },
                "permissions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required permissions, e.g. filesystem.read or network.",
                },
            },
            "required": ["capability"],
            "additionalProperties": False,
        },
        permissions=[],
    )
