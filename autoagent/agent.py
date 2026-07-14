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
    ApprovalRequired,
    AutoAgentError,
    MaxStepsExceeded,
    TokenBudgetExceeded,
    ToolError,
)
from .logging import get_logger
from .memory import Memory
from .providers import create_provider
from .providers.base import LLMProvider
from .registry import ToolRegistry, ToolResult
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

__all__ = [
    "Agent",
    "AgentResult",
    "AgentTurnContext",
    "CheckpointHook",
    "PostTurnHook",
    "RunState",
    "ToolPolicy",
    "ToolPolicyContext",
]

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
class RunState:
    """A resumable snapshot of an agent run, taken at a step boundary (0.11.0).

    Produced by the ``checkpoint`` callback of ``run``/``run_messages``
    (and their streaming twins) after every completed step, and attached
    as ``.state`` to ``MaxStepsExceeded`` / ``TokenBudgetExceeded`` /
    ``AgentCancelled``. Feed it to ``Agent.resume`` to continue the run
    where it stopped — after a crash, a process restart, or with a
    raised ``max_steps`` / ``token_budget``.

    JSON round-trip via ``to_dict`` / ``from_dict`` (messages use the
    lossless ``Message.to_dict`` from 0.7.0)::

        path.write_text(json.dumps(state.to_dict()))
        ...
        agent.resume(RunState.from_dict(json.loads(path.read_text())))

    Attributes:
        messages: Full conversation at the snapshot point (consistent:
            every tool result of the last step is included).
        step: Completed steps so far — resume continues at ``step + 1``
            and still honours the agent's ``max_steps``.
        corrections: post_turn_hook corrections already injected.
        turn_start: Index where the current user turn begins (used by
            the post_turn_hook accounting; do not edit).
        input_tokens / output_tokens: Token spend so far, so a resumed
            run keeps honouring ``token_budget``.
        have_usage: Whether any provider call reported usage.
    """

    messages: list[Message]
    step: int = 0
    corrections: int = 0
    turn_start: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    have_usage: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "step": self.step,
            "corrections": self.corrections,
            "turn_start": self.turn_start,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "have_usage": self.have_usage,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        return cls(
            messages=[Message.from_dict(m) for m in data.get("messages") or []],
            step=data.get("step", 0),
            corrections=data.get("corrections", 0),
            turn_start=data.get("turn_start", 0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            have_usage=data.get("have_usage", False),
        )


CheckpointHook = Callable[["RunState"], None]


@dataclass
class ToolPolicyContext:
    """What a ``tool_policy`` hook sees for ONE pending tool call (0.11.0).

    Attributes:
        call: The pending ``ToolCall`` (name, arguments, id). The ``id``
            is stable across pause/resume — key your approval store on it.
        spec: The registered ``ToolSpec`` (``spec.permissions`` is where
            declarative permissions live), or ``None`` for unknown tools.
        step: Current loop step (1-based).
        messages: The conversation so far (treat as READ-ONLY).
        context: The host ``context`` dict passed to ``run`` — the natural
            place for a user id, quotas, or an approval store handle.
    """

    call: ToolCall
    spec: ToolSpec | None
    step: int
    messages: list[Message]
    context: dict[str, Any]


# None = allow. A str = deny with that reason (the model sees it as a tool
# error and re-plans). Raise ApprovalRequired to PAUSE the run resumably.
ToolPolicy = Callable[[ToolPolicyContext], "str | None"]


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
        tool_policy: Optional hook consulted for EVERY pending tool call
            BEFORE anything of that turn executes (0.11.0). Receives a
            ``ToolPolicyContext``; return ``None`` to allow, a ``str``
            to deny with that reason (surfaced to the model as a tool
            error), or raise ``ApprovalRequired`` to PAUSE the run with
            a resumable ``RunState`` attached (``exc.state``) — the
            approval-gate case. A policy that itself crashes DENIES the
            call (fail-closed: this is a security boundary, unlike
            trace/checkpoint callbacks which fail-open).
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
        tool_policy: ToolPolicy | None = None,
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
        self.tool_policy = tool_policy
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

    def register_remember_tool(
        self,
        *,
        name: str = "remember",
        description: str | None = None,
    ) -> None:
        """Register a ``remember`` tool wrapping ``self.memory.remember`` (0.12.0).

        The write-side twin of ``register_recall_tool``: the agent can
        DELIBERATELY store a durable fact (« notez que je pars en août »)
        instead of hoping it survives compaction. The call shows up in
        the trace like any tool call. No-op unless the configured memory
        exposes a ``remember(fact, subject=)`` method (``FactMemory``
        does; bring-your-own memories can too).
        """
        if self.memory is None or not hasattr(self.memory, "remember"):
            return

        previewed = description or (
            "Store one short, self-contained fact in durable memory (a decision, "
            "a preference, a value, a commitment). Use it when the user states "
            "something worth remembering across conversations."
        )

        # Même contrat que register_recall_tool : self.memory est relu à
        # CHAQUE appel, pour honorer un memory remplacé après coup.
        agent_self = self

        def _remember(fact: str, subject: str = "") -> dict[str, Any]:
            mem = agent_self.memory
            if mem is None or not hasattr(mem, "remember"):
                return {"stored": False, "error": "No fact-capable memory is attached."}
            try:
                stored = mem.remember(fact, subject=subject or None)
            except Exception as exc:
                return {"stored": False, "error": f"{type(exc).__name__}: {exc}"}
            return {"stored": True, "fact": stored}

        self.registry.replace(
            ToolSpec(
                name=name,
                description=previewed,
                input_schema={
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "One short, self-contained fact to remember.",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Optional topic tag (e.g. 'rdv', 'contact').",
                        },
                    },
                    "required": ["fact"],
                    "additionalProperties": False,
                },
            ),
            handler=_remember,
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
        checkpoint: CheckpointHook | None = None,
    ) -> AgentResult:
        messages = [
            Message(role="system", content=self.render_system_prompt()),
            Message(role="user", content=prompt),
        ]
        return self.run_messages(
            messages, context=context, cancel_token=cancel_token, checkpoint=checkpoint
        )

    def run_messages(
        self,
        messages: list[Message],
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
        checkpoint: CheckpointHook | None = None,
    ) -> AgentResult:
        # Thin wrapper over the single loop implementation (_run_loop).
        # Intermediate events (tool_start/tool_end/correction) are ignored;
        # exceptions raised inside the generator (AgentCancelled,
        # MaxStepsExceeded, provider/tool errors) propagate unchanged, so
        # the non-streaming contract is identical to pre-0.10 behaviour.
        for event in self._run_loop(
            messages,
            context=context,
            cancel_token=cancel_token,
            streaming=False,
            checkpoint=checkpoint,
        ):
            if event.type == "done":
                return AgentResult(
                    output=event.output,
                    messages=event.messages,
                    steps=event.steps,
                    usage=event.usage,
                )
        raise AutoAgentError("agent loop ended without a result")  # pragma: no cover

    def resume(
        self,
        state: RunState,
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
        checkpoint: CheckpointHook | None = None,
    ) -> AgentResult:
        """Continue an interrupted run from a ``RunState`` snapshot (0.11.0).

        The loop restarts at ``state.step + 1`` with the snapshot's
        conversation and counters (corrections, token spend), so
        ``max_steps`` and ``token_budget`` keep their run-wide meaning.
        To resume past the limit that stopped the run, raise the
        agent's ``max_steps`` / ``token_budget`` first (or resume on an
        agent built with bigger ones — provider and tools may differ).

        Memory compaction is SKIPPED on resume: the snapshot is mid-run
        and its ``turn_start`` index must stay valid. The usual pattern::

            try:
                result = agent.run(prompt, checkpoint=save_state)
            except TokenBudgetExceeded as exc:
                agent.token_budget *= 2
                result = agent.resume(exc.state)
        """
        for event in self._run_loop(
            state.messages,
            context=context,
            cancel_token=cancel_token,
            streaming=False,
            checkpoint=checkpoint,
            resume_from=state,
        ):
            if event.type == "done":
                return AgentResult(
                    output=event.output,
                    messages=event.messages,
                    steps=event.steps,
                    usage=event.usage,
                )
        raise AutoAgentError("agent loop ended without a result")  # pragma: no cover

    def resume_stream(
        self,
        state: RunState,
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
        checkpoint: CheckpointHook | None = None,
    ) -> Iterator[StreamEvent]:
        """Streaming counterpart of ``resume`` — see ``run_messages_stream``
        for the error-event contract."""
        try:
            yield from self._run_loop(
                state.messages,
                context=context,
                cancel_token=cancel_token,
                streaming=True,
                checkpoint=checkpoint,
                resume_from=state,
            )
        except AgentCancelled as exc:
            yield StreamEvent(type="error", error="cancelled", steps=getattr(exc, "step", 0))
        except ApprovalRequired as exc:
            state = getattr(exc, "state", None)
            yield StreamEvent(
                type="error",
                error=f"approval_required: {exc}",
                messages=getattr(state, "messages", []),
                steps=getattr(state, "step", 0),
                state=state,
            )
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

    def run_stream(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
        checkpoint: CheckpointHook | None = None,
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
            messages, context=context, cancel_token=cancel_token, checkpoint=checkpoint
        )

    def run_messages_stream(
        self,
        messages: list[Message],
        *,
        context: dict[str, Any] | None = None,
        cancel_token: threading.Event | None = None,
        checkpoint: CheckpointHook | None = None,
    ) -> Iterator[StreamEvent]:
        """Streaming counterpart of ``run_messages`` — see ``run_stream``."""
        # Same single loop as run_messages, but failures become terminal
        # ``error`` events: streaming consumers read events, they don't catch.
        try:
            yield from self._run_loop(
                messages,
                context=context,
                cancel_token=cancel_token,
                streaming=True,
                checkpoint=checkpoint,
            )
        except AgentCancelled as exc:
            yield StreamEvent(type="error", error="cancelled", steps=getattr(exc, "step", 0))
        except ApprovalRequired as exc:
            state = getattr(exc, "state", None)
            yield StreamEvent(
                type="error",
                error=f"approval_required: {exc}",
                messages=getattr(state, "messages", []),
                steps=getattr(state, "step", 0),
                state=state,
            )
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
        checkpoint: CheckpointHook | None = None,
        resume_from: RunState | None = None,
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
        if self.memory is not None and resume_from is None:
            # Compact ONCE before the loop. Doing it per-iteration would
            # invalidate turn_start mid-run and complicate the
            # post_turn_hook accounting. Hosts that need finer control
            # can call memory.compact() themselves before passing the
            # messages in. SKIPPED on resume: a snapshot is mid-run and
            # compaction would shift the persisted turn_start index.
            try:
                working_messages = list(self.memory.compact(working_messages))
            except Exception:
                _log.exception("memory.compact raised; using messages unchanged")
        self._dynamic_tools_built_this_run = 0
        corrections = resume_from.corrections if resume_from else 0
        turn_start = resume_from.turn_start if resume_from else len(working_messages)
        spent_in = resume_from.input_tokens if resume_from else 0
        spent_out = resume_from.output_tokens if resume_from else 0
        have_usage = resume_from.have_usage if resume_from else False
        start_step = (resume_from.step if resume_from else 0) + 1

        def _snapshot(completed_step: int) -> RunState:
            return RunState(
                messages=list(working_messages),
                step=completed_step,
                corrections=corrections,
                turn_start=turn_start,
                input_tokens=spent_in,
                output_tokens=spent_out,
                have_usage=have_usage,
            )

        def _checkpoint(completed_step: int) -> None:
            if checkpoint is None:
                return
            try:
                checkpoint(_snapshot(completed_step))
            except Exception:
                # Same resilience contract as trace callbacks: persistence
                # trouble must not kill the run it is trying to protect.
                _log.exception("checkpoint callback raised; run continues")

        def _policy_overrides(
            calls: list[ToolCall], step: int, req_span: str | None
        ) -> dict[str, ToolResult]:
            """Consult tool_policy for the WHOLE turn before any side effect.

            Returns {call_id: denial ToolResult} for denied calls. Raises
            ApprovalRequired (with a resumable snapshot attached) BEFORE
            anything of the turn has executed — a pause must never land
            after a side effect.
            """
            overrides: dict[str, ToolResult] = {}
            if self.tool_policy is None:
                return overrides
            for call in calls:
                spec = next((s for s in self.registry.specs() if s.name == call.name), None)
                policy_ctx = ToolPolicyContext(
                    call=call, spec=spec, step=step,
                    messages=working_messages, context=context or {},
                )
                try:
                    verdict = self.tool_policy(policy_ctx)
                except ApprovalRequired as pause:
                    pause.state = _snapshot(step)  # LLM call done, zero tools executed
                    pause.calls = list(calls)
                    self._emit(
                        "approval_required",
                        {
                            "step": step,
                            "call_id": call.id,
                            "names": [c.name for c in calls],
                            "reason": truncate_preview(str(pause)),
                        },
                        parent_id=req_span,
                    )
                    raise
                except Exception as exc:
                    # Fail-CLOSED: a buggy policy denies. This hook is a
                    # security boundary — the opposite contract of trace/
                    # checkpoint callbacks, which fail-open.
                    _log.exception("tool_policy raised; denying %r (fail-closed)", call.name)
                    verdict = f"policy error: {type(exc).__name__}: {exc}"
                if verdict is None:
                    continue
                if not isinstance(verdict, str):
                    verdict = "policy returned an unsupported verdict type"
                overrides[call.id] = ToolResult(ok=False, error=f"ToolPolicyDenied: {verdict}")
                self._emit(
                    "tool_policy_deny",
                    {"name": call.name, "call_id": call.id, "step": step,
                     "reason": truncate_preview(verdict)},
                    parent_id=req_span,
                )
            return overrides

        def _run_turn_tools(
            calls: list[ToolCall], step: int, req_span: str | None
        ) -> Iterator[StreamEvent]:
            """Execute one turn's tool calls (policy-checked), append results."""
            overrides = _policy_overrides(calls, step, req_span)

            def _timed(call: ToolCall) -> tuple[Any, int]:
                denied = overrides.get(call.id)
                if denied is not None:
                    return denied, 0
                started_at = time.monotonic()
                result = self.registry.execute(call, context=context)
                return result, int((time.monotonic() - started_at) * 1000)

            if self.parallel_tool_calls and len(calls) > 1:
                # Concurrent execution (opt-in). Starts are announced in
                # call order, every call runs in a thread pool, then ends
                # and transcript messages follow the SAME call order —
                # the conversation stays deterministic whatever the
                # completion order.
                spans: list[str | None] = []
                for call in calls:
                    spans.append(self._emit_tool_start(call, req_span))
                    yield StreamEvent(type="tool_start", tool_name=call.name)
                with ThreadPoolExecutor(
                    max_workers=min(len(calls), 8),
                    thread_name_prefix="autoagent-tool",
                ) as pool:
                    outcomes = list(pool.map(_timed, calls))
                for call, tool_span, (tool_result, duration_ms) in zip(calls, spans, outcomes):
                    self._emit_tool_end(call, tool_span, tool_result, duration_ms)
                    yield StreamEvent(
                        type="tool_end",
                        tool_name=call.name,
                        tool_status="ok" if tool_result.ok else "error",
                    )
                    working_messages.append(_tool_message(call, tool_result))
            else:
                for call in calls:
                    tool_span = self._emit_tool_start(call, req_span)
                    yield StreamEvent(type="tool_start", tool_name=call.name)
                    tool_result, duration_ms = _timed(call)
                    self._emit_tool_end(call, tool_span, tool_result, duration_ms)
                    yield StreamEvent(
                        type="tool_end",
                        tool_name=call.name,
                        tool_status="ok" if tool_result.ok else "error",
                    )
                    working_messages.append(_tool_message(call, tool_result))

        model = getattr(getattr(self.provider, "config", None), "model", None)
        run_start_payload: dict[str, Any] = {
            "max_steps": self.max_steps,
            "model": model,
            "message_count": len(working_messages),
            "tool_count": len(self.registry.specs()),
        }
        if streaming:
            run_start_payload["streaming"] = True
        if resume_from is not None:
            run_start_payload["resumed_from_step"] = resume_from.step
        run_span = self._emit("run_start", run_start_payload)
        try:
            if (
                resume_from is not None
                and working_messages
                and working_messages[-1].role == "assistant"
                and working_messages[-1].tool_calls
            ):
                # The snapshot was taken by an approval gate: the last step's
                # LLM call is in the transcript but NONE of its tools ran.
                # Finish that step first — each pending call goes through the
                # policy AGAIN (still unapproved → pauses again, idempotent).
                yield from _run_turn_tools(
                    list(working_messages[-1].tool_calls), resume_from.step, run_span
                )
                _checkpoint(resume_from.step)

            for step in range(start_step, self.max_steps + 1):
                # Cooperative cancellation: the host may set `cancel_token` to
                # abort the run between iterations. We check BEFORE the next
                # provider call so we don't waste a request when the user has
                # already pressed "Cancel".
                if cancel_token is not None and cancel_token.is_set():
                    self._emit("cancelled", {"step": step}, parent_id=run_span)
                    cancelled = AgentCancelled(f"Agent cancelled by caller at step {step}")
                    cancelled.step = step  # consumed by run_messages_stream
                    cancelled.state = _snapshot(step - 1)  # resumable via Agent.resume
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
                    exhausted.state = _snapshot(step - 1)  # resumable via Agent.resume
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
                        _checkpoint(step)
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

                yield from _run_turn_tools(list(response.tool_calls), step, req_span)

                # Step boundary: every tool result of this step is in the
                # transcript — the run is resumable from exactly here.
                _checkpoint(step)

            self._emit("max_steps_exceeded", {"max_steps": self.max_steps}, parent_id=run_span)
            self._emit(
                "run_end",
                {"status": "max_steps", "steps": self.max_steps},
                parent_id=run_span,
            )
            exceeded = MaxStepsExceeded(f"Agent exceeded max_steps={self.max_steps}")
            exceeded.messages = working_messages  # consumed by run_messages_stream
            exceeded.state = _snapshot(self.max_steps)  # resumable after raising max_steps
            raise exceeded
        except AgentCancelled:
            self._emit("run_end", {"status": "cancelled"}, parent_id=run_span)
            raise
        except ApprovalRequired:
            self._emit("run_end", {"status": "approval_required"}, parent_id=run_span)
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
