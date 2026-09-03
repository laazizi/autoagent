from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .bounds import Bounds
from .dynamic import DynamicToolBuilder, ToolBuildRequest
from .errors import (
    AgentCancelled,
    ApprovalRequired,
    AutoAgentError,
    MaxStepsExceeded,
    TokenBudgetExceeded,
    ToolError,
)
from .guards import (  # noqa: F401 — re-exportés pour les tests
    TurnGuards,
    _call_signature,
    _count_call_signatures,
)
from .logging import get_logger
from .memory import Memory
from .providers import create_provider
from .providers.base import LLMProvider
from .registry import ToolRegistry
from .schema import (
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolSpec,
    is_tainted,
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
        tainted: Whether untrusted external content has entered the run
            (0.17.0). Persisted so taint survives resume AND memory
            compaction — a monotonic flag, not just a transcript scan.
    """

    messages: list[Message]
    step: int = 0
    corrections: int = 0
    turn_start: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0          # part de l'entree servie par le cache (0.19.0)
    have_usage: bool = False
    tainted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "step": self.step,
            "corrections": self.corrections,
            "turn_start": self.turn_start,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "have_usage": self.have_usage,
            "tainted": self.tainted,
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
            # defaut 0 : un instantane anterieur a 0.19.0 se reprend sans erreur
            cached_tokens=data.get("cached_tokens", 0),
            have_usage=data.get("have_usage", False),
            tainted=data.get("tainted", False),
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
        tainted: ``True`` when UNTRUSTED external content (output of a tool
            declared ``untrusted=True``) already entered the conversation
            (0.15.0). The classic policy: tainted + sensitive permission →
            deny or ``ApprovalRequired``. Derived from the transcript, so
            it survives checkpoint/resume. NB: evaluated BEFORE the turn's
            tools run — a fetch and a sensitive call requested in the SAME
            turn both see the pre-turn taint state.
    """

    call: ToolCall
    spec: ToolSpec | None
    step: int
    messages: list[Message]
    context: dict[str, Any]
    tainted: bool = False
    egress: bool = False


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


# Cadrage des sorties d'outils UNTRUSTED : marqueurs + helper `is_tainted`
# vivent dans schema.py (partagés avec memory.py sans import circulaire).
# La teinte du run est désormais un FLAG MONOTONE persisté dans RunState
# (0.17) — plus seulement un scan de transcript, donc robuste à la
# compaction mémoire qui effacerait le marqueur (trou de la 0.15).


_TRUNCATION_NOTE = (
    "\n[TRUNCATED — {omitted} of {total} characters omitted in the middle. "
    "Narrow your query, filter, or request one portion at a time.]\n"
)


def _truncate_tool_result(content: str, max_chars: int) -> tuple[str, bool]:
    """Bound ONE tool result to ``max_chars``, keeping the head and the tail.

    Middle-out on purpose: the head carries the shape of the payload (keys,
    headers, first rows) and the tail carries what a truncated read would
    otherwise hide (totals, error trailers, "next page" cursors).

    The marker counts against the budget — a bound that can be exceeded is
    not a bound. Returns ``(content, truncated?)``.
    """
    total = len(content)
    if max_chars <= 0 or total <= max_chars:
        return content, total > max_chars
    note = _TRUNCATION_NOTE.format(omitted=total - max_chars, total=total)
    budget = max_chars - len(note)
    if budget <= 0:  # budget smaller than the marker itself: hard cut, no marker
        return content[:max_chars], True
    head = budget - budget // 3
    tail = budget - head
    # Recompute the omitted count now that head/tail are known, then re-fit:
    # the note's own length shifts by a few digits, so keep the total bounded
    # by trimming the head rather than letting the marker push us over.
    note = _TRUNCATION_NOTE.format(omitted=total - head - tail, total=total)
    head = max(0, max_chars - len(note) - tail)
    out = content[:head] + note + (content[-tail:] if tail else "")
    return out[:max_chars], True


_FIND_TOOLS_NAME = "find_tools"


def _find_tools_spec() -> ToolSpec:
    """Meta-tool of progressive disclosure (see ``Agent.enable_tool_search``)."""
    return ToolSpec(
        name=_FIND_TOOLS_NAME,
        description=(
            "Search the tools available to you by keyword when the tool you need is "
            "not already listed. Returns matching tool names and descriptions; their "
            "full schemas then become available so you can call them directly."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you are trying to do, or a tool name "
                                   "(e.g. 'read a file', 'send email').",
                }
            },
            "required": ["query"],
        },
    )


def _words(text: str) -> set[str]:
    """Words of 3+ characters, lowercased — the unit of the lexical match."""
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) >= 3}


def _score_tools(query: str, specs: Sequence[ToolSpec]) -> list[ToolSpec]:
    """Rank specs by word overlap with the query: name hits count double.

    Lexical on purpose — zero dependency, deterministic, and debuggable. Hosts
    that want semantic search can pre-filter the registry themselves.
    """
    terms = _words(query)
    if not terms:
        return []
    scored: list[tuple[int, int, ToolSpec]] = []
    for index, spec in enumerate(specs):
        name_words = _words(spec.name)
        desc_words = _words(spec.description)
        score = 2 * len(terms & name_words) + len(terms & desc_words)
        # A query naming the tool outright ("use read_file") must always win.
        if spec.name.lower() in (query or "").lower():
            score += 5
        if score:
            scored.append((-score, index, spec))  # index keeps ties stable
    scored.sort()
    return [spec for _, _, spec in scored]


_PRUNE_MARK = "[PRUNED"
_PRUNE_NOTE = (
    "[PRUNED — the {chars}-character result of `{name}` was dropped from this "
    "history to keep the context bounded. It was VALID when produced; nothing "
    "about it failed. Call the tool again if you still need that data.]"
)


def _prune_tool_results(
    messages: list[Message], keep: int, batch: int = 1
) -> tuple[list[Message], int, int]:
    """Replace the content of OLD tool results by a short marker.

    A tool result is the biggest and least reusable object in a transcript.
    The model needed those 40 000 characters at step 3; by step 12 it only
    needs the conclusion it drew from them — yet the whole payload is re-sent
    at EVERY step until it falls off the memory's tail, and it is never in the
    provider's cached prefix because the history changes each turn.
    ``max_tool_result_chars`` bounds a result's WIDTH; this bounds its LIFETIME.

    The ``keep`` most recent tool messages are left untouched. Older ones keep
    their role and ``tool_call_id`` — the conversation stays well-formed for
    every provider — and lose only their payload. The marker says the result
    was VALID, because a model told merely that something is "removed" tends to
    re-plan around a failure that never happened.

    Three invariants carry the whole difficulty:

    * **Taint survives.** A pruned result that carried the UNTRUSTED framing is
      re-framed. Dropping the marker would silently un-taint the run and
      disarm the trifecta guard — the 0.15 hole, re-opened by the back door.
    * **Pruning never GROWS the transcript.** A result shorter than its own
      marker is left alone: a bound that costs context is not a bound.
    * **The record is not touched.** The input list is not mutated; only the
      VIEW handed to the provider is pruned, so the trace, the returned
      messages and any checkpoint keep the full result.

    ``batch`` (0.21.0) makes pruning CACHE-FRIENDLY. Pruning at every step
    mutates the view at every step, so the provider's prompt cache — which
    matches on a byte-identical prefix — restarts from zero each turn
    (TokenPilot, arXiv 2606.17016, measures cache-miss tokens dropping
    5.9 M → 1.6 M once compaction is done in batches at stable boundaries).
    With ``batch=K`` the number of pruned results is always a multiple of K:
    the view only changes once every K tool results, and is byte-stable in
    between. Same token saving over the run, K times fewer prefix breaks.
    Stateless — derived from the transcript alone — so it survives resume and
    replay. ``batch=1`` is the 0.19.0 behaviour.

    Returns ``(view, pruned_count, chars_saved)``.
    """
    if keep < 0:
        return messages, 0, 0
    positions = [i for i, m in enumerate(messages) if m.role == "tool"]
    if len(positions) <= keep:
        return messages, 0, 0
    a_elaguer = len(positions) - keep
    if batch > 1:
        a_elaguer = (a_elaguer // batch) * batch     # multiples de K : vue stable entre lots
        if a_elaguer == 0:
            return messages, 0, 0
    view = list(messages)
    pruned = saved = 0
    for i in positions[:a_elaguer]:
        message = view[i]
        content = message.content or ""
        if _PRUNE_MARK in content:  # idempotent: never prune a marker again
            continue
        note = _PRUNE_NOTE.format(chars=len(content), name=message.name or "tool")
        if UNTRUSTED_OPEN in content:
            note = f"{UNTRUSTED_OPEN}\n{note}\n{UNTRUSTED_CLOSE}"
        if len(note) >= len(content):
            continue
        view[i] = replace(message, content=note)
        pruned += 1
        saved += len(content) - len(note)
    return view, pruned, saved


def _tool_message(call: ToolCall, tool_result: Any, *, untrusted: bool = False,
                  max_chars: int | None = None) -> Message:
    """The transcript message carrying one tool result back to the LLM.

    ``max_chars`` bounds the RESULT (before the untrusted framing, which is a
    fixed-size safety marker and must never be cut).
    """
    content = tool_result.to_message_content()
    if max_chars is not None:
        content, _ = _truncate_tool_result(content, max_chars)
    if untrusted:
        content = f"{UNTRUSTED_OPEN}\n{content}\n{UNTRUSTED_CLOSE}"
    return Message(
        role="tool",
        name=call.name,
        tool_call_id=call.id,
        content=content,
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
        max_tool_result_chars: Optional CODE-LEVEL bound on how much of
            ONE tool result enters the transcript (0.18.0). ``None``
            (default) keeps the historical behaviour: whatever the tool
            returns is injected verbatim. Set it and an oversized result
            is truncated MIDDLE-OUT (head + tail kept) with an explicit
            marker telling the model what happened, so it can narrow its
            query instead of silently working on a cut payload. Bounding
            is code: a single unbounded tool (an HTTP fetch, a wide SQL
            SELECT) otherwise blows the context window, burns the
            ``token_budget``, and degrades attention. The bound covers
            the result only — the untrusted framing markers are never
            cut. See also ``ProjectWorkspace(max_read_chars=...)``, which
            bounds workspace reads specifically.
        prune_tool_results_after: Optional CODE-LEVEL bound on how LONG a
            tool result stays in the transcript (0.19.0). ``None``
            (default) keeps the historical behaviour: every result is
            re-sent verbatim at every step until it falls off the
            memory's tail. Set to N and only the N most recent tool
            results keep their payload; older ones are replaced — in the
            view sent to the provider only — by a marker naming the tool
            and the size dropped, so the model can call it again. This is
            the complement of ``max_tool_result_chars``: that one bounds
            a result's WIDTH, this one bounds its LIFETIME. It also ages
            out stale tool ERRORS, which otherwise keep instructing the
            model long after the condition that produced them is gone.
            The untrusted framing is preserved on a pruned result, so
            pruning can never un-taint a run.
        prune_batch: Prune in batches of K results instead of one at a
            time (0.21.0). Pruning at every step rewrites the view at
            every step, which breaks the provider's prompt cache each
            turn. With K > 1 the view only changes once every K tool
            results and is byte-identical in between — same saving, K
            times fewer cache misses. Default 1 = 0.19.0 behaviour.
        bounds: The eight bounds as ONE object (0.21.0) — ``Bounds(max_steps=,
            token_budget=, max_tool_result_chars=, prune_tool_results_after=,
            prune_batch=, max_repeated_tool_calls=, trifecta_guard=,
            shadow_guards=)``. Readable at a glance, shareable between agents,
            serialisable into a trace via ``agent.bounds.to_dict()``. An
            explicit keyword that differs from its default overrides the
            corresponding field. ``agent.bounds`` is a snapshot of the
            bounds IN FORCE (the attributes stay plain and writable).
        shadow_guards: OBSERVE the built-in guards instead of enforcing
            them (0.20.0). ``False`` (default) keeps the historical
            behaviour: a guard that fires refuses the call. Set it and
            the verdict is still computed and TRACED — as
            ``loop_guard_would_block`` / ``trifecta_would_block`` — but
            the call goes through, and ``run_end`` carries
            ``would_block`` (how many were observed). The radar
            photographs without fining.
            Why it exists: turning a bound on is otherwise a leap of
            faith — you cannot know whether it will refuse something
            legitimate until it does, in production. Run a week in
            shadow, read "this bound would have refused 3 calls out of
            41 000, here they are", then enable it knowing. Combined
            with ``ReplaySession``, the same question is answered on
            LAST month's recorded runs, offline, for free.
            Scope, deliberately: this covers the LIBRARY's guards only.
            ``tool_policy`` is the host's own boundary and is never
            shadowed — a library flag must not be able to switch off
            code the host wrote to say no. A host that wants the same
            can return ``None`` and log.
            ⚠️ Shadow mode does NOT protect the run. It is a
            measurement mode, not a safe default.
        max_repeated_tool_calls: Optional loop guard (0.18.0). ``None``
            (default) keeps the historical behaviour. Set to N and the
            (N+1)-th IDENTICAL call — same tool, same arguments — is not
            executed: the model receives a deterministic ``RepeatedCall``
            tool error on the same channel a policy denial uses, so it
            re-plans instead of burning the budget. Two things this buys
            beyond ``max_steps``: the side effect stops repeating (an
            HTTP POST, an e-mail, a sub-agent run), and the trace names
            the failure (``loop_guard_block``) instead of ending on a
            mute ``max_steps``. Counted from the transcript, so it
            survives checkpoint/resume.
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
        max_tool_result_chars: int | None = None,
        prune_tool_results_after: int | None = None,
        prune_batch: int = 1,
        shadow_guards: bool = False,
        max_repeated_tool_calls: int | None = None,
        trifecta_guard: str = "deny",
        bounds: Bounds | None = None,
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
        self.max_tool_result_chars = max_tool_result_chars
        self.prune_tool_results_after = prune_tool_results_after
        self.prune_batch = max(1, int(prune_batch))
        self.shadow_guards = shadow_guards
        self.max_repeated_tool_calls = max_repeated_tool_calls
        self.trifecta_guard = trifecta_guard
        if bounds is not None:
            # `Bounds` pose les huit bornes d'un coup ; un kwarg donné
            # EXPLICITEMENT (différent de sa valeur par défaut) l'emporte —
            # l'intention la plus locale gagne (0.21.0).
            defauts = Bounds()
            explicites = {
                nom: valeur for nom, valeur in (
                    ("max_steps", max_steps), ("token_budget", token_budget),
                    ("max_tool_result_chars", max_tool_result_chars),
                    ("prune_tool_results_after", prune_tool_results_after),
                    ("prune_batch", prune_batch),
                    ("max_repeated_tool_calls", max_repeated_tool_calls),
                    ("trifecta_guard", trifecta_guard), ("shadow_guards", shadow_guards),
                ) if valeur != getattr(defauts, nom)
            }
            bounds.apply_to(self, explicites)
            self.prune_batch = max(1, int(self.prune_batch))
        self.dynamic_builder: DynamicToolBuilder | None = None
        self._dynamic_tools_built_this_run = 0
        # Divulgation progressive (opt-in via enable_tool_search) : désactivée,
        # donc `_visible_specs` renvoie tout — comportement historique.
        self._tool_search = False
        self._tool_search_threshold = 15
        self._tool_search_always: set[str] = set()
        self._tool_search_max_results = 5
        self._revealed_tools: set[str] = set()

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

    @property
    def bounds(self) -> Bounds:
        """La photo des huit bornes EN VIGUEUR sur cet agent (0.21.0)."""
        return Bounds.from_agent(self)

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        permissions: list[str] | None = None,
        untrusted: bool = False,
        egress: bool = False,
        idempotent: bool = False,
    ):
        return self.registry.register(
            func,
            name=name,
            description=description,
            input_schema=input_schema,
            permissions=permissions,
            untrusted=untrusted,
            egress=egress,
            idempotent=idempotent,
        )

    def add_tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        return self.registry.add_function(func)

    def enable_tool_search(
        self,
        *,
        threshold: int = 15,
        always: Sequence[str] = (),
        max_results: int = 5,
    ) -> None:
        """Progressive disclosure of tool SCHEMAS — opt-in (0.18.0).

        Every request normally carries the full schema of every registered
        tool. Mount two MCP servers and that prefix alone can dominate the
        request: schemas are re-sent at EVERY step of EVERY run, and a model
        given 100 tools also picks worse than one given 6. The industry
        converged on the same fix in 2026 (Anthropic's tool-search tool,
        code-execution-with-MCP, Cloudflare's code mode): stop shipping
        schemas the model has not asked for.

        With this enabled, a run that has more than ``threshold`` tools sends
        only a ``find_tools`` meta-tool plus the schemas of the tools already
        revealed (and those named in ``always``). ``find_tools(query)`` returns
        matching names + descriptions — a cheap catalogue, no schemas — and
        reveals them for the remainder of the run, so the very next step can
        call them.

        Deliberately lexical (stdlib only, no embeddings): scoring is a word
        overlap on names and descriptions. A query that matches nothing returns
        the bare catalogue of names rather than an empty result, so the model
        can never be cornered.

        Under the threshold nothing changes — no meta-tool, no filtering, same
        bytes on the wire as before. Revealed tools are re-derived from the
        transcript at the start of each run, so ``resume`` after an approval
        pause keeps the tools the model had already loaded.

        Args:
            threshold: Send everything while the registry holds at most this
                many tools (the meta-tool itself is not counted).
            always: Tool names that stay visible without being searched for —
                the handful an agent needs on every task.
            max_results: How many matches ``find_tools`` returns at once.
        """
        self._tool_search = True
        self._tool_search_threshold = threshold
        self._tool_search_always = set(always)
        self._tool_search_max_results = max_results

        def find_tools(query: str) -> dict[str, Any]:
            catalogue = [s for s in self.registry.specs() if s.name != _FIND_TOOLS_NAME]
            matches = _score_tools(query, catalogue)[: self._tool_search_max_results]
            if not matches:
                # Never corner the model: hand back the bare list of names.
                return {
                    "matches": [],
                    "available": sorted(s.name for s in catalogue),
                    "hint": "No name/description matched. Pick one from `available` "
                            "and search for it by name.",
                }
            self._revealed_tools.update(s.name for s in matches)
            return {
                "matches": [{"name": s.name, "description": s.description} for s in matches],
                "hint": "Their full schemas are now available — call them directly.",
            }

        self.registry.replace(spec=_find_tools_spec(), handler=find_tools)

    def _visible_specs(self) -> list[ToolSpec]:
        """Specs to advertise on the NEXT request (progressive disclosure).

        Returns every spec unless tool search is enabled AND the registry is
        above the threshold. `tool_policy`, taint lookups and execution always
        see the FULL registry — visibility bounds what the model is offered,
        never what the host can govern.
        """
        specs = self.registry.specs()
        if not self._tool_search:
            return specs
        catalogue = [s for s in specs if s.name != _FIND_TOOLS_NAME]
        if len(catalogue) <= self._tool_search_threshold:
            return [s for s in specs if s.name != _FIND_TOOLS_NAME]
        keep = self._revealed_tools | self._tool_search_always
        visible = [s for s in specs if s.name == _FIND_TOOLS_NAME or s.name in keep]
        return visible

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

    def register_forget_tool(
        self,
        *,
        name: str = "forget",
        description: str | None = None,
        confirm: bool = True,
    ) -> None:
        """Register a ``forget`` tool wrapping ``memory.forget_matching`` (0.18.0).

        The third side of memory: ``recall`` reads, ``remember`` writes, this one
        ERASES — in natural language (« oublie tout ce qui concerne mon ancien
        employeur »). No-op unless the configured memory exposes
        ``forget_matching``.

        ``confirm=True`` (default) makes the tool a DRY RUN: it reports what would
        be erased and erases nothing. Deleting a user's data on a model's decision
        alone is not something a library should default to — the host wires the
        second step (a UI confirmation, or an `apply=True` policy) once it has
        shown the list to a human. Set ``confirm=False`` for an agent explicitly
        trusted to erase.
        """
        if self.memory is None or not hasattr(self.memory, "forget_matching"):
            return

        agent_self = self
        described = description or (
            "Erase durable memories matching an instruction in plain language "
            "(e.g. 'forget everything about my previous employer'). "
            + ("Returns what WOULD be erased; a human confirms before anything is "
               "deleted." if confirm else "Erases immediately — use with care.")
        )

        def _forget(instruction: str) -> dict[str, Any]:
            mem = agent_self.memory
            if mem is None or not hasattr(mem, "forget_matching"):
                return {"erased": False, "error": "No fact-capable memory is attached."}
            try:
                touched = mem.forget_matching(instruction, dry_run=confirm)
            except Exception as exc:
                return {"erased": False, "error": f"{type(exc).__name__}: {exc}"}
            return {
                "erased": not confirm and bool(touched),
                "pending_confirmation": confirm and bool(touched),
                "facts": [{"id": f["id"], "fact": f["fact"]} for f in touched],
                "count": len(touched),
            }

        self.registry.replace(
            ToolSpec(
                name=name,
                description=described,
                input_schema={
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "What to forget, in plain language.",
                        },
                    },
                    "required": ["instruction"],
                    "additionalProperties": False,
                },
            ),
            handler=_forget,
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
            # La dépense du sous-agent est REMISE À ZÉRO avant le run : ce qui
            # reste ici après un run raté ne doit pas être facturé au parent au
            # tour suivant.
            handler.__autoagent_usage__ = None  # type: ignore[attr-defined]
            result = agent_self.run(request, context=context)
            payload: dict[str, Any] = {"output": result.output, "steps": result.steps}
            if result.usage is not None:
                payload["tokens"] = result.usage.total_tokens
            # Canal de retour vers la comptabilité du PARENT (0.19.0). Le chiffre
            # dans `payload` est pour le MODÈLE (il le lit dans le transcript) ;
            # celui-ci est pour le CODE. Sans lui, un superviseur qui délègue
            # dépensait sans que `token_budget` ne voie rien passer — un plafond
            # qu'il suffisait de contourner en déléguant.
            handler.__autoagent_usage__ = result.usage  # type: ignore[attr-defined]
            return payload

        handler.__autoagent_usage__ = None  # type: ignore[attr-defined]
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
        # Divulgation progressive : les outils déjà révélés sont RE-DÉRIVÉS du
        # transcript, pas gardés en mémoire d'instance. Un `resume` après une
        # pause d'approbation retrouve donc ce que le modèle avait chargé, sans
        # nouveau champ dans RunState.
        if self._tool_search:
            self._revealed_tools = {
                call.name
                for m in working_messages
                for call in (m.tool_calls or [])
                if call.name != _FIND_TOOLS_NAME
            }
        corrections = resume_from.corrections if resume_from else 0
        turn_start = resume_from.turn_start if resume_from else len(working_messages)
        spent_in = resume_from.input_tokens if resume_from else 0
        spent_out = resume_from.output_tokens if resume_from else 0
        spent_cached = resume_from.cached_tokens if resume_from else 0
        # « 0 jeton servi par le cache » est une MESURE ; « rien rapporte » est
        # une absence. Sans ce drapeau les deux se confondraient en None, et on
        # ne saurait jamais distinguer un cache qui ne mord pas d'un fournisseur
        # qui se tait — donc jamais si activer le cache a servi a quelque chose.
        saw_cached = bool(resume_from and resume_from.cached_tokens)
        have_usage = resume_from.have_usage if resume_from else False
        start_step = (resume_from.step if resume_from else 0) + 1
        # La compaction mémoire (résumé / extraction de faits) appelle SON
        # propre LLM. On compte ce coût dans le budget et l'usage rapporté —
        # sinon `token_budget` sous-estimait la dépense réelle (0.17).
        mem_usage = getattr(self.memory, "last_usage", None) if resume_from is None else None
        if mem_usage is not None:
            spent_in += mem_usage.input_tokens or 0
            spent_out += mem_usage.output_tokens or 0
            if mem_usage.cached_tokens is not None:
                spent_cached += mem_usage.cached_tokens
                saw_cached = True
            have_usage = True
        # Teinte = flag MONOTONE (cellule mutable pour les closures). Semé du
        # RunState à la reprise, sinon du transcript initial (marqueur/sentinelle
        # d'un historique persisté). Passe à True dès qu'une sortie untrusted
        # entre — et le reste. Robuste à la compaction (≠ scan seul).
        taint = [resume_from.tainted if resume_from else is_tainted(working_messages)]

        # Dépense des SOUS-AGENTS (`Agent.as_tool`). Cellule mutable, comme la
        # teinte : la collecte se fait dans `_run_turn_tools`, une fonction
        # imbriquée qui ne peut pas réaffecter `spent_in`. Le drainage a lieu
        # après chaque tour d'outils, donc AVANT la vérification de budget de
        # l'étape suivante — sinon `token_budget` se contournerait en déléguant
        # (le trou : un superviseur plafonné à 5 000 jetons pouvait en brûler
        # dix fois plus via ses spécialistes sans que rien ne s'en aperçoive).
        # Compteur du MODE TÉMOIN : combien de refus ont été observés sans
        # être appliqués. Remonté dans `run_end` — c'est le rapport de fin de
        # semaine (« cette borne aurait refusé N appels »).
        temoin = [0]

        # Exécution AU FIL DU FLUX (0.21.0) : futures indexées par call.id, et le
        # compte de ceux qui ont vraiment servi (remonté dans run_end).
        en_avance: dict[str, Any] = {}
        en_avance_utilises = [0]
        en_avance_pool: list[Any] = []       # l'exécuteur, créé au premier usage

        def _lancer_en_avance(call: ToolCall, step: int, req_span: str | None) -> None:
            handler = self.registry.handler_for(call.name)
            if handler is None or call.id in en_avance:
                return
            spec = next((s for s in self.registry.specs() if s.name == call.name), None)
            if spec is None or not spec.idempotent:
                return                              # effet de bord possible : on attend
            # Les MÊMES gardes que le chemin normal, sur cet appel seul. Un
            # refus ici = pas de lancement ; le chemin normal le refusera de
            # nouveau (mêmes entrées, même verdict) et le tracera.
            # `pending=True` : l'appel n'est pas encore dans le transcript ; la
            # garde anti-boucle le compte quand même, pour rendre le MÊME verdict
            # que la vérification réelle du tour (0.21.0).
            refus = guards.builtin([call], step, req_span, pending=True)
            refus.update(guards.policy([call], step, req_span))
            if refus:
                return
            if not en_avance_pool:
                en_avance_pool.append(ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="autoagent-early"))
            en_avance[call.id] = en_avance_pool[0].submit(
                self.registry.execute, call, context=context)
            self._emit("tool_call_early_start",
                       {"step": step, "name": call.name, "call_id": call.id},
                       parent_id=req_span)

        def _avec_temoin(payload: dict[str, Any]) -> dict[str, Any]:
            """Ajoute le bilan du mode témoin au `run_end`, et seulement alors.

            Une clé absente en mode normal, plutôt qu'un zéro : « aucune garde
            n'aurait bloqué » et « le mode n'était pas actif » ne veulent pas
            dire la même chose — c'est la même règle que pour `cached_tokens`.
            """
            if self.shadow_guards:
                payload["shadow_guards"] = True
                payload["would_block"] = temoin[0]
            if en_avance_pool:
                # Fin de run : les résultats anticipés non consommés (flux cassé,
                # appel absent de la réponse finale) sont JETÉS — idempotents, ils
                # n'ont rien changé. L'exécuteur est fermé sans attendre.
                en_avance.clear()
                en_avance_pool[0].shutdown(wait=False)
                en_avance_pool.clear()
                payload["early_tool_calls"] = en_avance_utilises[0]
            return payload
        delegations: list[TokenUsage] = []

        def _absorber_delegations() -> None:
            nonlocal spent_in, spent_out, spent_cached, saw_cached, have_usage
            while delegations:
                usage = delegations.pop(0)
                spent_in += usage.input_tokens or 0
                spent_out += usage.output_tokens or 0
                if usage.cached_tokens is not None:
                    spent_cached += usage.cached_tokens
                    saw_cached = True
                have_usage = True

        def _snapshot(completed_step: int) -> RunState:
            return RunState(
                messages=list(working_messages),
                step=completed_step,
                corrections=corrections,
                turn_start=turn_start,
                input_tokens=spent_in,
                output_tokens=spent_out,
                cached_tokens=spent_cached,
                have_usage=have_usage,
                tainted=taint[0],
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

        # Les gardes vivent dans `guards.py` (0.21.0) : mêmes verdicts, mêmes
        # événements de trace, même ordre — sorties de la boucle pour qu'on
        # puisse relire les deux. Construites une fois par run avec l'état
        # qu'elles lisaient en fermeture (transcript, teinte, témoin, snapshot).
        guards = TurnGuards(self, working_messages, context, taint, temoin, _snapshot)
        _policy_overrides = guards.policy

        def _run_turn_tools(
            calls: list[ToolCall], step: int, req_span: str | None
        ) -> Iterator[StreamEvent]:
            """Execute one turn's tool calls (policy-checked), append results."""
            # Gardes intégrées d'abord (anti-boucle, trifecta — avec le mode
            # témoin), puis la politique de l'hôte en DERNIER : elle ne peut
            # qu'AJOUTER des refus (retourner None n'efface rien), donc elle ne
            # peut jamais dé-bloquer une garde intégrée — l'hôte reste souverain
            # sans pouvoir affaiblir la frontière par inadvertance.
            overrides = guards.builtin(calls, step, req_span)
            overrides.update(guards.policy(calls, step, req_span))

            def _timed(call: ToolCall) -> tuple[Any, int, Any]:
                """Exécute UN appel et rend (résultat, durée, dépense déléguée).

                Le troisième membre n'est renseigné que pour un outil qui est en
                fait un AGENT (`Agent.as_tool()`). On le RELÈVE ici mais on ne
                l'additionne pas : en mode parallèle cette fonction tourne dans
                un thread, et cumuler depuis plusieurs threads ferait perdre des
                jetons. L'addition a lieu dans la boucle ordonnée, en aval.
                """
                denied = overrides.get(call.id)
                if denied is not None:
                    return denied, 0, None
                started_at = time.monotonic()
                avance = en_avance.pop(call.id, None)
                if avance is not None:
                    # Lancé au fil du flux : on ATTEND le résultat, on ne relance
                    # pas. La durée rapportée est celle de l'attente restante —
                    # c'est le temps réellement gagné qui manque ici, et c'est
                    # voulu : la trace dit ce que le tour a coûté en latence.
                    result = avance.result()
                    en_avance_utilises[0] += 1
                else:
                    result = self.registry.execute(call, context=context)
                duration_ms = int((time.monotonic() - started_at) * 1000)
                handler = self.registry.handler_for(call.name)
                delegue = getattr(handler, "__autoagent_usage__", None)
                if delegue is not None and handler is not None:
                    # Consommé : un deuxième appel du même outil ne doit pas
                    # refacturer la dépense du premier.
                    handler.__autoagent_usage__ = None
                return result, duration_ms, delegue

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
                for call, tool_span, (tool_result, duration_ms, delegue) in zip(
                        calls, spans, outcomes, strict=True):
                    if delegue is not None:
                        delegations.append(delegue)
                    self._emit_tool_end(call, tool_span, tool_result, duration_ms)
                    yield StreamEvent(
                        type="tool_end",
                        tool_name=call.name,
                        tool_status="ok" if tool_result.ok else "error",
                    )
                    untrusted = self._is_untrusted(call)
                    if untrusted:
                        taint[0] = True                     # teinte monotone
                    working_messages.append(
                        _tool_message(call, tool_result, untrusted=untrusted,
                                      max_chars=self.max_tool_result_chars)
                    )
            else:
                for call in calls:
                    tool_span = self._emit_tool_start(call, req_span)
                    yield StreamEvent(type="tool_start", tool_name=call.name)
                    tool_result, duration_ms, delegue = _timed(call)
                    if delegue is not None:
                        delegations.append(delegue)
                    self._emit_tool_end(call, tool_span, tool_result, duration_ms)
                    yield StreamEvent(
                        type="tool_end",
                        tool_name=call.name,
                        tool_status="ok" if tool_result.ok else "error",
                    )
                    untrusted = self._is_untrusted(call)
                    if untrusted:
                        taint[0] = True                     # teinte monotone
                    working_messages.append(
                        _tool_message(call, tool_result, untrusted=untrusted,
                                      max_chars=self.max_tool_result_chars)
                    )

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
                _absorber_delegations()
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
                        _avec_temoin({"status": "token_budget", "steps": step - 1}),
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

                # L'élagage porte sur la VUE envoyée au fournisseur, jamais sur
                # `working_messages` : la teinte, la garde anti-boucle, les
                # outils révélés, la trace et le snapshot continuent de lire le
                # transcript COMPLET. Élaguer le registre lui-même reviendrait à
                # perdre des preuves pour économiser des jetons.
                view = working_messages
                if self.prune_tool_results_after is not None:
                    view, pruned_count, chars_saved = _prune_tool_results(
                        working_messages, self.prune_tool_results_after,
                        self.prune_batch,
                    )
                    if pruned_count:
                        self._emit(
                            "context_pruned",
                            {"step": step, "pruned": pruned_count,
                             "chars_saved": chars_saved,
                             "kept": self.prune_tool_results_after,
                             "batch": self.prune_batch},
                            parent_id=req_span,
                        )

                request = LLMRequest(
                    messages=view,
                    tools=self._visible_specs(),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if streaming:
                    # Drive the provider's streaming generator, re-emitting
                    # text deltas and capturing the assembled final response.
                    #
                    # EXÉCUTION AU FIL DU FLUX (0.21.0). Un chunk `tool_call`
                    # arrive dès qu'un appel est complet, souvent bien avant la
                    # fin du message. Si l'outil est déclaré `idempotent` ET que
                    # les gardes le laissent passer, on le lance TOUT DE SUITE en
                    # arrière-plan : le modèle parle, l'outil travaille. Trois
                    # bornes qui rendent ça sûr :
                    #   * jamais sans `idempotent=True` — un flux qui casse jette
                    #     le résultat, et un outil idempotent n'a rien changé ;
                    #   * les gardes (anti-boucle, trifecta, politique) passent
                    #     AVANT, exactement comme sur le chemin normal ;
                    #   * `_run_turn_tools` CONSOMME le résultat au lieu de
                    #     ré-exécuter : un seul appel, ordre des événements et du
                    #     transcript inchangé.
                    final_response: LLMResponse | None = None
                    for chunk in self.provider.stream(request):
                        if chunk.type == "text" and chunk.text:
                            yield StreamEvent(type="text", text=chunk.text)
                        elif chunk.type == "tool_call" and chunk.tool_call is not None:
                            _lancer_en_avance(chunk.tool_call, step, req_span)
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
                    if response.usage.cached_tokens is not None:
                        spent_cached += response.usage.cached_tokens
                        saw_cached = True
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
                        _avec_temoin({
                            "status": "ok",
                            "steps": step,
                            "output_preview": truncate_preview(response.content),
                        }),
                        parent_id=run_span,
                    )
                    yield StreamEvent(
                        type="done",
                        output=response.content,
                        messages=working_messages,
                        steps=step,
                        usage=(
                            TokenUsage(input_tokens=spent_in, output_tokens=spent_out,
                                       cached_tokens=spent_cached if saw_cached
                                       else None)
                            if have_usage
                            else None
                        ),
                    )
                    return

                yield from _run_turn_tools(list(response.tool_calls), step, req_span)
                _absorber_delegations()

                # Step boundary: every tool result of this step is in the
                # transcript — the run is resumable from exactly here.
                _checkpoint(step)

            self._emit("max_steps_exceeded", {"max_steps": self.max_steps}, parent_id=run_span)
            self._emit(
                "run_end",
                _avec_temoin({"status": "max_steps", "steps": self.max_steps}),
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

    def _is_untrusted(self, call: ToolCall) -> bool:
        """La sortie de cet outil est-elle déclarée non fiable (spec 0.15) ?"""
        spec = next((s for s in self.registry.specs() if s.name == call.name), None)
        return bool(spec is not None and spec.untrusted)

    def _is_egress(self, call: ToolCall) -> bool:
        """Cet outil peut-il faire SORTIR de l'information du système (spec 0.18) ?"""
        spec = next((s for s in self.registry.specs() if s.name == call.name), None)
        return bool(spec is not None and spec.egress)

    def audit_trifecta(self) -> list[str]:
        """Lint de configuration : la « lethal trifecta » est-elle réunie ? (0.18.0)

        Renvoie une liste de constats lisibles (vide = rien à signaler). À appeler
        AU DÉMARRAGE, pas en boucle : c'est une vérification de câblage, pas un
        contrôle d'exécution. Réunir « contenu non fiable » et « capacité de
        sortie » dans le même agent, c'est le motif exact des exfiltrations par
        injection indirecte — mieux vaut le voir au boot qu'après l'incident.
        """
        untrusted = sorted(s.name for s in self.registry.specs() if s.untrusted)
        egress = sorted(s.name for s in self.registry.specs() if s.egress)
        if not (untrusted and egress):
            return []
        return [
            f"lethal trifecta: untrusted input tools {untrusted} coexist with egress "
            f"tools {egress} in the same agent. trifecta_guard={self.trifecta_guard!r} "
            f"governs what happens once the run is tainted "
            f"({'blocked' if self.trifecta_guard == 'deny' else self.trifecta_guard})."
        ]

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


def delegate_to(
    specialistes: dict[str, Agent],
    *,
    name: str = "delegate",
    description: str | None = None,
    max_parallel: int = 4,
) -> Callable[..., Any]:
    """Un outil qui interroge PLUSIEURS sous-agents EN MÊME TEMPS (0.20.0).

    ``Agent.as_tool()`` expose un spécialiste ; un superviseur qui en consulte
    trois les fait passer l'un après l'autre, et attend la somme des durées.
    Ici le modèle envoie plusieurs demandes en un seul appel, elles partent
    ensemble, et il récupère tout d'un coup ::

        superviseur.add_tool(delegate_to({
            "comptage":  expert_comptage,
            "juridique": expert_juridique,
        }))

    LE CONTRAT QUI REND ÇA SÛR : l'appel **ne rend la main que lorsque TOUS les
    spécialistes ont fini**. C'est la différence entre paralléliser et rendre le
    système asynchrone, et elle n'est pas cosmétique. ``token_budget`` est
    vérifié avant chaque appel LLM, sur la dépense déjà connue : avec des
    sous-agents encore en vol, le plafond ne bornerait plus que ce qui a atterri,
    pas ce qui est engagé — et le chiffre manquant n'existerait pas encore, donc
    aucune comptabilité ne le rattraperait. On gagne le temps, on ne perd pas la
    borne. Idem pour la teinte, qui suppose un ordre.

    Deux détails qui sont des bugs si on les oublie :

    * **Un ``Agent`` ne sert qu'un appelant à la fois.** Deux demandes adressées
      au MÊME spécialiste sont donc exécutées l'une après l'autre ; seuls des
      spécialistes DIFFÉRENTS tournent en parallèle.
    * **L'ordre des réponses suit l'ordre des demandes**, jamais l'ordre
      d'arrivée : le transcript reste déterministe, donc rejouable.

    L'échec d'un spécialiste n'annule pas les autres : sa réponse porte
    ``error``, les autres passent. Et si le run d'un spécialiste a vu du contenu
    externe non fiable, sa sortie est rendue ENCADRÉE — sans quoi déléguer
    laverait la teinte.
    """
    if not specialistes:
        raise ValueError("delegate_to attend au moins un spécialiste")

    noms = sorted(specialistes)
    described = description or (
        "Ask one or several specialists. Requests aimed at DIFFERENT specialists "
        "run IN PARALLEL: group them in a single call rather than calling this tool "
        f"several times. Available specialists: {', '.join(noms)}."
    )

    def _une(cible: str, demande: str,
             context: dict[str, Any] | None) -> tuple[dict[str, Any], Any]:
        agent = specialistes[cible]
        try:
            resultat = agent.run(demande, context=context)
        except Exception as exc:                       # remonte au LLM, pas au parent
            return {"specialist": cible, "error": f"{type(exc).__name__}: {exc}"}, None
        sortie = resultat.output
        if is_tainted(resultat.messages):
            # Le spécialiste a lu du contenu externe : sa sortie peut le citer.
            # Sans ce cadre, déléguer LAVERAIT la teinte — le run parent
            # redeviendrait « propre » et la garde trifecta se désarmerait.
            sortie = "\n".join([UNTRUSTED_OPEN, sortie, UNTRUSTED_CLOSE])
        reponse: dict[str, Any] = {
            "specialist": cible,
            "output": sortie,
            "steps": resultat.steps,
        }
        if resultat.usage is not None:
            reponse["tokens"] = resultat.usage.total_tokens
        return reponse, resultat.usage

    def handler(requests: list[dict[str, Any]],
                context: dict[str, Any] | None = None) -> dict[str, Any]:
        handler.__autoagent_usage__ = None             # type: ignore[attr-defined]
        if not isinstance(requests, list) or not requests:
            return {"responses": [], "error": "`requests` must be a non-empty list."}

        # Regroupement par spécialiste : un Agent ne sert qu'un appelant à la
        # fois, donc deux demandes pour la même cible ne peuvent PAS partir
        # ensemble. On parallélise ENTRE cibles, on sérialise À L'INTÉRIEUR.
        groupes: dict[str, list[int]] = {}
        reponses: list[Any] = [None] * len(requests)
        usages: list[Any] = [None] * len(requests)
        for index, item in enumerate(requests):
            cible = (item or {}).get("specialist") if isinstance(item, dict) else None
            if cible not in specialistes:
                reponses[index] = {
                    "specialist": cible,
                    "error": f"Unknown specialist. Available: {', '.join(noms)}.",
                }
                continue
            groupes.setdefault(cible, []).append(index)

        def _groupe(cible: str) -> None:
            # Chaque index n'est écrit que par UN thread : pas de verrou requis.
            for index in groupes[cible]:
                demande = str((requests[index] or {}).get("request", ""))
                reponses[index], usages[index] = _une(cible, demande, context)

        if len(groupes) > 1:
            with ThreadPoolExecutor(
                max_workers=min(len(groupes), max_parallel),
                thread_name_prefix="autoagent-delegate",
            ) as pool:
                list(pool.map(_groupe, list(groupes)))   # `list` = on ATTEND tout
        else:
            for cible in groupes:
                _groupe(cible)

        # Comptabilité : la dépense de TOUS les spécialistes remonte au parent par
        # le même canal que `as_tool`, donc elle entre dans `token_budget`. Tout
        # est terminé ici — c'est ce qui rend le plafond exact.
        entree = sortie = cache = 0
        vu = vu_cache = False
        for usage in usages:
            if usage is None:
                continue
            vu = True
            entree += usage.input_tokens or 0
            sortie += usage.output_tokens or 0
            if usage.cached_tokens is not None:
                cache += usage.cached_tokens
                vu_cache = True
        if vu:
            handler.__autoagent_usage__ = TokenUsage(   # type: ignore[attr-defined]
                input_tokens=entree, output_tokens=sortie,
                cached_tokens=cache if vu_cache else None)
        return {"responses": reponses, "tokens": entree + sortie if vu else None}

    handler.__autoagent_usage__ = None                 # type: ignore[attr-defined]
    handler.__name__ = name
    handler.__autoagent_tool_spec__ = ToolSpec(        # type: ignore[attr-defined]
        name=name,
        description=described,
        input_schema={
            "type": "object",
            "properties": {
                "requests": {
                    "type": "array",
                    "description": "The requests to handle — in parallel when they "
                                   "target different specialists.",
                    "items": {
                        "type": "object",
                        "properties": {
                            # Volontairement PAS un `enum` : la validation de
                            # schéma rejette la requête ENTIÈRE au premier nom
                            # inconnu, donc une coquille annulerait les autres
                            # délégations, déjà valides. Le nom est vérifié
                            # entrée par entrée : la faute coûte UNE réponse,
                            # pas le lot. Les noms valides sont dans la
                            # description et rappelés dans le message d'erreur.
                            "specialist": {"type": "string",
                                           "description": f"One of: {', '.join(noms)}."},
                            "request": {"type": "string"},
                        },
                        "required": ["specialist", "request"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["requests"],
            "additionalProperties": False,
        },
    )
    return handler
