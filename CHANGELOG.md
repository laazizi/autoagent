# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.0] - 2026-07-14

### Added
- **`FactMemory` + `Agent.register_remember_tool()`** — fact-based memory
  kept UP TO DATE instead of a rolling summary. Old turns go through an LLM
  extraction that maintains a list of short atomic facts via add / update /
  delete operations — a contradiction REPLACES the stale fact instead of
  piling up next to it. Context gets the fact list (dense) rather than raw
  messages; `recall()` does lexical search over facts; `remember()` stores a
  fact directly (no LLM), exposed to the agent by `register_remember_tool()`
  (the write-side twin of `register_recall_tool` — deliberate, traced
  memorization). Optional `path=` gives a human-readable JSON store per
  identity (per caller, per customer) — auditable, hand-correctable, and
  GDPR-friendly (forget someone = delete their file). Extraction failures
  skip compaction (nothing silently truncated); malformed operations are
  ignored, never fatal.

## [0.11.0] - 2026-07-13

### Added
- **`MCPClient` (`autoagent/mcp.py`)** — zero-dependency MCP client over the
  stdio transport (newline-delimited JSON-RPC 2.0 to a server subprocess).
  Server tools become ordinary autoagent tools: `mcp.mount(agent, prefix=,
  include=, exclude=)` registers each one as a handler carrying
  `__autoagent_tool_spec__` (server `inputSchema` validated by the registry
  like any local tool). `tools/list` pagination followed; server-initiated
  pings answered; notifications ignored; `structuredContent` returned as-is,
  otherwise text parts joined as `{"text": ...}`; a result flagged `isError`
  raises `ToolError` (surfaces to the LLM as a tool error). Thread-safe
  (requests correlated by id — works under `parallel_tool_calls=True`).
  Transport/protocol failures raise the new `MCPError`.
- **`OTelTraceExporter` (`autoagent/otel.py`)** — OpenTelemetry exporter for
  `TraceEmitter` (`on_event` callback, callable + context manager). Rebuilds
  the agent's span tree as real OTel spans (agent.run → llm → tool.<name>)
  with durations, `autoagent.*` attributes from event payloads (already
  secret-redacted), ERROR status on `error`/`cancelled`/`max_steps`, and
  point events attached to the nearest open span. `close()` ends spans left
  open by interrupted runs; a broken backend can never break the agent loop.
  `opentelemetry-api` is imported lazily at construction — the core keeps
  its zero-dependency contract; without the package a clear `AutoAgentError`
  explains what to install.
- **`RunState` + `checkpoint=` + `Agent.resume`** — long-running agents. The
  run loop accepts a `checkpoint` callback (all four entry points) invoked
  with a `RunState` snapshot after every completed step and every post-turn
  correction; `to_dict`/`from_dict` give a lossless JSON round-trip (built on
  `Message.to_dict` 0.7.0). `Agent.resume(state)` / `resume_stream(state)`
  continue at `state.step + 1` with restored counters — `max_steps` and
  `token_budget` keep their run-wide meaning across resumes. The `.state`
  attribute on `MaxStepsExceeded` / `TokenBudgetExceeded` / `AgentCancelled`
  is a ready-to-resume snapshot (raise the limit, then `resume(exc.state)`).
  A raising checkpoint callback is logged and ignored (same resilience
  contract as trace callbacks); memory compaction is skipped on resume so
  the persisted `turn_start` stays valid.
- **`Agent(tool_policy=)` + `ApprovalRequired`** — one execution-policy hook
  covering the enterprise quartet: allow / deny / ask-a-human / audit-quota.
  Consulted for EVERY pending tool call BEFORE anything of the turn executes
  (also under `parallel_tool_calls`): return `None` to allow, a `str` to deny
  with that reason (the model sees `ToolPolicyDenied: …` as a tool error and
  re-plans), or raise `ApprovalRequired` to pause the run resumably — the
  exception carries `.state` (a `RunState`) and `.calls` (nothing executed).
  On `resume()` the pending calls go through the policy again: unapproved
  pauses again (idempotent), rejected surfaces to the model, approved runs
  exactly once. A crashing policy DENIES (fail-closed — a security boundary,
  the opposite contract of trace/checkpoint callbacks). New trace events
  `tool_policy_deny` / `approval_required`; streaming emits a terminal
  `error` event carrying the snapshot in `ev.state`.
- **`Agent.as_tool(name=, description=)`** — the minimal multi-agent
  primitive: expose an agent as a TOOL of another (supervisor/specialist
  hierarchies in two lines). Stateless delegation, parent `context`
  forwarded, sub-agent failures surface as tool errors (never crash the
  parent), result carries `{output, steps, tokens}`.
- **`SummarizingMemory(provider, max_messages=, keep_recent=)`** — folds turns
  beyond the threshold into an INCREMENTAL LLM summary (injected as a system
  message) instead of dropping them. Failure-safe: if the summary call fails,
  compaction is skipped (nothing silently truncated). Re-absorbs its own
  in-band summary when hosts persist compacted history. `recall()` does
  lexical retrieval over folded messages (works with `register_recall_tool`).
- **`Agent(token_budget=N)`** — hard cap on a run's cumulative token usage
  (input+output as reported by the provider), checked BEFORE each provider
  call. Raises `TokenBudgetExceeded` (streaming: terminal `error` event);
  emits a `token_budget_exceeded` trace event. `AgentResult.usage` (and the
  `done` stream event) now carry the run's aggregated `TokenUsage`.
- **`Agent(parallel_tool_calls=True)`** — when the model requests several
  tools in one turn they execute concurrently (thread pool, capped at 8).
  Opt-in: handlers must be thread-safe and share the `context` dict. The
  transcript stays deterministic (results appended in the model's call
  order); stream events and trace spans keep the same order.
- **`LLMRequest.response_format`** — structured output. OpenAI-compatible:
  passed through verbatim (`{"type": "json_object"}` or `json_schema`);
  Gemini: `responseMimeType: application/json`; Anthropic: strict
  "JSON only" system instruction (no native mode — best effort).
  `DynamicToolBuilder` now requests JSON mode, killing the ```json-fence
  failure class at the source (the tolerant parser stays as a backstop).
- **`TokenUsage`** on `LLMResponse.usage` — token accounting extracted from all
  three wire formats (OpenAI `usage`, Anthropic `usage`, Gemini `usageMetadata`),
  in `complete()` AND in the streaming `final` chunk. `llm_response` trace events
  now carry `input_tokens`/`output_tokens`.
- **Native SSE streaming for OpenAI-compatible providers** (OpenAI, DeepSeek,
  Groq, vLLM…): text deltas, `reasoning_content` deltas, incremental tool-call
  assembly. Previously these providers fell back to non-streamed `complete()`.
- **`RoutingProvider.stream()`** — routing now preserves the chosen provider's
  native streaming (was silently degrading to the non-streaming fallback).
- `ProviderError.status_code` / `ProviderError.retryable` — programmatic error
  metadata (no more parsing the message text to detect a 429).

### Changed
- **The agent loop is now a single implementation** (`Agent._run_loop`):
  `run_messages` and `run_messages_stream` are thin wrappers over it.
  They were ~150-line near-twins that had to be edited in lockstep — the
  main source of future divergence. Public contracts are unchanged
  (non-streaming raises, streaming yields terminal `error` events).

### Fixed
- `post_json` now retries 429/500/502/503/504 with backoff (honouring
  `Retry-After`), not just transient network errors; `post_sse` retries the
  initial connection with the same policy (mid-stream errors still propagate).
- `tool_choice` is now honoured by Anthropic (`any`/`tool`/drop-tools-on-none)
  and Gemini (`toolConfig.functionCallingConfig`) — it was OpenAI-only.
- Gemini: tool results now recover their `functionResponse.name` from the
  assistant `tool_calls` via `tool_call_id` when `Message.name` is missing
  (the `"tool"` fallback broke matching as soon as two tools existed).
- Streaming `final` chunks now carry `raw` (summary) and `usage` — parity with
  `complete()`.
- `ToolRegistry`: the JSON-Schema validator and handler signature are now built
  once at registration instead of on every `execute()` (hot-path win);
  `schema_from_callable` no longer advertises bogus `args`/`kwargs` properties
  for `*args`/`**kwargs` handlers.
- `SecretRedactingFilter` no longer coerces non-string log args to `str`
  (numeric format specs like `%d` would have raised `TypeError`).
- Anthropic: `max_tokens=0` no longer silently becomes 2048.

## [0.9.0] - 2026-06-09

### Added
- **`Orchestrator`** (public) — the library's second core primitive,
  alongside `Agent`. Where `Agent` hands control to the model (right
  for open-ended work), `Orchestrator` inverts it for CERTIFIED
  processes : the host owns the state machine (`current_steps()` +
  `record()`), and the LLM performs two bounded micro-tasks per turn —
  INTERPRET the user's reply into typed values (strict JSON, fail-safe
  to « unclear ») and PHRASE the current step naturally (streamed).
  The LLM cannot advance, skip, or invent steps : it never sees slots
  the host didn't expose. A garbage LLM output degrades one turn's
  wording, never the flow.
- Built-in turn mechanics : faithful acknowledgments (only what was
  ACTUALLY recorded, resolved through a host `describe` hook so raw
  option IDs never reach the phrasing model), horizon fills for
  compound replies (« Julie 55 and me, 54 » records several slots in
  one turn, but only host-exposed ones), anti-loop escalation
  (consecutive non-answers on the same step raise `stuck_count` and
  switch the phrasing strategy), `on_offtopic` hook, and host-injectable
  prompts/payload builders for full language control.
- New public types : `Step`, `TurnEvent`, `PhraseSignals`,
  `InterpretOutcome`, plus `DEFAULT_INTERPRET_SYSTEM` /
  `DEFAULT_PHRASE_SYSTEM`.

### Notes
- Pattern extracted from `examples/cati_chat` (certified Cerema
  mobility survey, 100+ questions, conditional filters, nested loops,
  typed validation), where an autonomous agent + verifier still
  drifted ; the example now consumes the library primitive.
- `stuck_slot` / `stuck_count` are plain attributes the host persists
  with its session state when rebuilding the Orchestrator per request.

## [0.8.0] - 2026-06-09

### Added
- **Streaming** — `Agent.run_stream(prompt)` and
  `Agent.run_messages_stream(messages)` yield `StreamEvent` objects as
  the model produces output: `text` deltas (token-by-token), `tool_start`
  / `tool_end` around each tool execution, `correction` when the
  post_turn_hook injects one, and a final `done` (carrying `output`,
  the full `messages` list to persist, and `steps`) or `error`. The
  full tool-use loop, memory compaction, post_turn_hook and tracing all
  work identically to the non-streaming `run_messages`.
- **`LLMProvider.stream(request)`** — yields `StreamChunk` objects
  (`text` deltas then one `final` chunk with the assembled
  `LLMResponse`). The base class provides a NON-STREAMING FALLBACK
  (calls `complete()`, emits the whole content as one chunk) so every
  provider supports the streaming API; OpenAI/DeepSeek degrade
  gracefully until they gain native support.
- **Native SSE streaming for Anthropic and Gemini** —
  `AnthropicProvider.stream` parses `content_block_delta` events
  (text + `input_json_delta` for tool args); `GeminiProvider.stream`
  uses `streamGenerateContent?alt=sse`. Both assemble the same
  `LLMResponse` the non-streaming path would return, so tool calling is
  unaffected.
- **`post_sse`** in `autoagent.http` — a generator that POSTs a JSON
  body and yields parsed `data:` events from a Server-Sent Events
  stream. Skips `event:` / comment / blank lines and the `[DONE]`
  sentinel; tolerates malformed `data:` payloads.
- New public exports: `StreamChunk`, `StreamEvent`.

### Notes
- Backward compatible: `run` / `run_messages` / `complete` are
  unchanged. Streaming is purely additive.
- Cancellation and max-steps surface as a terminal `error` StreamEvent
  (not a raised exception) — streaming consumers read events.
- The `examples/cati_chat/` example gains a `POST /api/chat/stream`
  SSE endpoint and a progressive-rendering frontend; the survey
  question now appears token-by-token instead of after a full pause.

## [0.7.0] - 2026-06-09

### Added
- **Dynamic system prompts** — `Agent.system_prompt` now accepts either a
  `str` (static, as before) OR a zero-arg `Callable[[], str]`
  re-evaluated at the start of every `run()`. Lets hosts inject live
  state into the prompt on every turn: form progress in a survey app,
  current step in a workflow, list of remaining questions, etc. The
  alternative — putting state into messages — gets trimmed at memory
  compaction, but a dynamic system prompt is always fresh.
- **`Agent.render_system_prompt()`** (public) — resolves the prompt to
  its current string value. Hosts that persist conversations across
  HTTP requests (FastAPI chat sessions, queue workers, ...) call this
  on each turn and replace the stale system message in their stored
  history so the LLM always sees the freshly-rendered state.
- **`Message.to_dict()` / `Message.from_dict()`** — symmetric
  serialisation to plain JSON-safe dicts, with empty optional fields
  omitted on output (compact snapshots) and tolerant of missing
  optional fields on input (forward-compatible). Same on `ToolCall`
  and `ImageAttachment`. Lossless round-trip through `json.dumps` /
  `json.loads`, so a full chat history can be persisted to SQLite /
  Redis / a JSON file and rehydrated turn after turn.

### Notes
- Backward compatible: existing code passing `system_prompt="..."`
  works unchanged. The type annotation widened from `str` to
  `str | Callable[[], str]`.
- Resilience: a callable that raises is caught and logged; the run
  falls back to `DEFAULT_SYSTEM_PROMPT` rather than crashing — same
  contract as `post_turn_hook` and `trace.emit`. A callable that
  returns `None` is also treated as the default; a non-string return
  is coerced via `str()` (covers numeric counters, template objects).
- The `examples/cati_chat/` example (added alongside 0.7.0) uses both
  features as the canonical pattern: the form state is rendered into
  the system prompt every turn, and the conversation history is
  persisted as JSON between FastAPI requests.

## [0.6.1] - 2026-06-07

### Added
- **`RoutingProvider`** (public) — wraps multiple `LLMProvider` instances
  and dispatches each request based on content. Default policy: route
  messages with image attachments to a vision-capable provider, route
  text-only requests to a cheaper text provider. The web_app_evolution
  example gets new `--vision-provider` / `--vision-model` CLI flags
  that activate the routing.
- Stripping of historical `attachments` from messages going to text-only
  providers, so a previously-vision conversation doesn't break the next
  text turn.
- **`ToolCall.thought_signature`** (public) — optional field carrying
  the encrypted reasoning signature emitted by Gemini 3+ thinking
  models on each function call. Captured on response, echoed back on
  the next assistant message. Non-Gemini providers ignore it.

### Fixed
- **Gemini tool-use rejected by API** — `ToolSpec.as_gemini_declaration()`
  now sanitises the JSON Schema before sending: `additionalProperties`,
  `$schema`, `$id`, `$ref`, `$defs`, `definitions`, `patternProperties`,
  `unevaluatedProperties` are stripped recursively (Gemini uses a
  subset of OpenAPI 3.0 Schema, not full JSON Schema). Previously any
  tool with `additionalProperties: false` (most of ours) was rejected
  with a 400 `"Unknown name"` error.
- **Gemini 3+ thoughtSignature support** — Gemini 3 thinking models
  emit a `thoughtSignature` on each function call AND require it to be
  echoed back on the next request. Previously the lib dropped it on
  parse and Gemini 3 rejected the next turn with 400 `"Function call
  is missing a thought_signature"`. Captured + echoed automatically
  now. Note the asymmetric format: Gemini RESPONSES nest the signature
  inside `functionCall`, but REQUESTS expect it at the PART level
  (sibling of `functionCall`) — putting it inside the request's
  `functionCall` yields a different 400 (`"Unknown name
  thoughtSignature ... Cannot find field"`).
- `ImageAttachment.as_base64()` rejects non-base64 data URLs explicitly
  (`data:image/png,<urlencoded>`) instead of silently returning
  corrupted bytes to the provider. Previously the MIME parser also
  truncated by one char on this corner case.
- `examples/web_app_evolution.py` lowered the per-image size cap from
  8 MB to 4 MB (so 4-image messages stay under Gemini's ~20 MB inline
  limit) and removed `image/gif` from the allowlist (not officially
  supported by Gemini's inline_data; animated GIFs are systematically
  refused).

## [0.6.0] - 2026-06-06

### Added
- **`Memory` protocol** (public) — two methods, `compact(messages)` and
  `recall(query, k)`, that let a host shape what the agent sees on
  each call. Implementations are free to truncate, summarise,
  project state from artifacts, embed-and-index for later recall, or
  anything else. The library stays opinion-free on backend choice
  (vector store, embedding provider, summarisation model).
- **`BufferMemory(max_messages=20)`** — trivial implementation that
  keeps every `system` message plus the last N non-system messages.
  Safe truncation: the tail is anchored on the first `user` message
  so we never leave an orphan `tool` head that strict providers
  reject.
- **`Agent(memory=...)`** — new optional keyword. When configured,
  `run_messages` calls `memory.compact(messages)` ONCE before the
  loop. Compaction errors are isolated; the run proceeds with the
  original messages and logs the failure via the `autoagent.agent`
  logger.
- **`Agent.register_recall_tool(name='recall')`** — helper that
  registers a `recall` tool wrapping `memory.recall(query, k)`. Use
  it with vector-backed or summary-backed memories to give the
  agent explicit access to forgotten details on demand. Silent
  no-op when no memory is configured.

### Notes
- `Agent(memory=None)` (the default) is exactly equivalent to 0.5.0
  — no behaviour change, no overhead. Hosts on 0.5.0 require no
  migration.
- The lib ships only the baseline `BufferMemory`. Richer
  implementations — vector-backed semantic memory, recursive
  summarisation, code-state projection — live in `examples/` so
  they can pull their own opinionated stack (chromadb, OpenAI
  embeddings, sentence-transformers, ...) without weighing down
  the library.
- For agents that evolve code (the `EvolutionRuntime` use case),
  consider a `Memory` that projects current state from artifacts
  rather than summarising chat history — see the new
  `examples/qt_webview_evolution.py` integration for a worked
  example.

## [0.5.0] - 2026-06-05

### Added
- **Structured event tracing** — new `TraceEmitter` and `TraceEvent`
  (both public) plus a new `Agent(trace=...)` keyword argument. When
  a trace emitter is configured the agent emits typed events at every
  lifecycle point of a run:
    - `run_start` / `run_end` (status: `ok` | `cancelled` |
      `max_steps` | `error`)
    - `llm_request` / `llm_response`
    - `tool_call_start` / `tool_call_end` (with `duration_ms` and
      `status`)
    - `post_turn_hook_invoked` / `post_turn_hook_correction`
    - `cancelled` / `max_steps_exceeded`
  Every event carries `type`, `span_id`, `parent_id`, `ts`, and a
  per-type `payload`, so a consumer can rebuild the call tree and
  group tool calls under their owning LLM step.
- Two persistence modes, composable:
    - `TraceEmitter(file="trace.jsonl")` — appends one JSON object per
      line. The emitter owns and closes the handle on `close()` /
      context-manager exit.
    - `TraceEmitter(on_event=callback)` — synchronous callback called
      for each event. Exceptions raised by the callback are caught
      and logged; they cannot break the agent loop.
  Both can be set at the same time. Open file handles passed in are
  treated as host-owned and never closed by the emitter.
- `truncate_preview(value)` helper for hosts that want to add their
  own bounded fields onto events without duplicating the rule used
  by the agent. The helper applies the same secret-redaction patterns
  as `autoagent.logging` (Bearer tokens, `x-api-key` /
  `x-goog-api-key`, `api_key` JSON fields, legacy `?key=` URL form)
  so credentials in tool arguments, tool errors, LLM responses, and
  post-turn-hook corrections cannot leak into the trace file or an
  external observability backend.

### Security
- Trace event payloads are filtered through the same
  `SecretRedactingFilter` patterns as the logger. The redaction is
  applied at the `truncate_preview` layer, so every `*_preview` field
  is protected uniformly (`arguments_preview`, `content_preview`,
  `output_preview`). Hosts forwarding traces to Langfuse / Phoenix /
  Jaeger can rely on the same baseline as their existing log
  pipeline. Hosts with stricter requirements (PII, customer ids,
  ...) should still layer their own filter on top of `on_event`.

### Notes
- `Agent(trace=None)` (the default) emits nothing and behaves
  exactly as in 0.4.0 — zero overhead, zero behaviour change. Hosts
  on 0.4.0 require no migration.
- The emitter never propagates an error to the agent: file-write
  failures and callback exceptions are swallowed and logged via the
  `autoagent.trace` namespace. The agent additionally guards against
  a misbehaving emitter object itself.
- Use the JSONL output to plug Langfuse, Phoenix, Jaeger, or an
  OpenTelemetry exporter in a few lines on the host side — autoagent
  imposes no transport, only the event shape.

## [0.4.0] - 2026-05-24

### Added
- **Multimodal messages** — nouveau `ImageAttachment` (public) et nouveau
  champ `Message.attachments: list[ImageAttachment]`. Permet d'attacher
  une ou plusieurs images à un message utilisateur. Chaque provider
  sérialise dans son format natif :
    - OpenAI : `content` devient une liste `[{type: text}, {type: image_url}, …]`
    - Anthropic : bloc `{type: image, source: {type: base64, media_type, data}}`
    - Gemini : part `{inline_data: {mime_type, data}}`
  Les hôtes n'ont pas à connaître les différences.
- `ImageAttachment.as_data_url()` et `as_base64()` helpers pour les hôtes
  qui doivent décoder eux-mêmes.
- **Web example** : bouton 📎, glisser-déposer, et coller-image (`Ctrl+V`)
  dans la chat panel. Vignettes de preview avant envoi, suppression
  individuelle. Validation côté serveur : 4 images max, 8 MB chacune,
  MIME types `image/{jpeg,png,webp,gif}`. Les images apparaissent dans
  la bulle utilisateur du chat après envoi.

### Notes
- Modèle requis pour la vision : Claude Sonnet 4.5+, GPT-4o (pas mini),
  Gemini 2+. DeepSeek-chat et les modèles texte-seul ignorent
  silencieusement les images dans le payload.

## [0.3.2] - 2026-05-24

### Added
- **`Message.reasoning_content`** et **`LLMResponse.reasoning_content`** :
  champs optionnels (`str | None`) qui transportent la trace de raisonnement
  émise par les modèles de "thinking mode" (DeepSeek v4 pro, OpenAI o-series
  avec reveal). Champ par défaut `None`, non-breaking pour le code existant.

### Fixed
- **OpenAI provider** : capture `reasoning_content` depuis la réponse et le
  ré-injecte dans le payload assistant au tour suivant. Sans ça, DeepSeek
  v4 pro rejette les requêtes multi-tours avec
  `The reasoning_content in the thinking mode must be passed back to the API`.
  Le champ est ignoré silencieusement par les modèles non-reasoning.
- **Agent loop** : propage `response.reasoning_content` dans le `Message`
  ajouté à l'historique, pour que les tours suivants puissent l'echo-back.

## [0.3.1] - 2026-05-24

### Fixed
- **OpenAI provider** : envoie `max_completion_tokens` au lieu de `max_tokens`
  pour les modèles `gpt-5*`, `o1*`, `o3*`, `o4*`. OpenAI a fait évoluer son
  API en 2025 ; les nouveaux modèles rejettent `max_tokens` avec
  `unsupported_parameter`. Les modèles legacy (`gpt-3.5*`, `gpt-4*`,
  `gpt-4o*`) continuent d'utiliser `max_tokens` comme avant.
- Test paramétré couvrant les deux familles de modèles pour éviter une
  régression silencieuse à la prochaine évolution d'API.

## [0.3.0] - 2026-05-24

### Added
- **Cooperative cancellation** via `cancel_token: threading.Event` on
  `Agent.run` and `Agent.run_messages`. When the host sets the event,
  the agent loop raises the new `AgentCancelled` exception at the start
  of the next iteration, before the next provider call. In-flight HTTP
  requests are NOT interrupted — cancellation happens at the next safe
  loop boundary. New public symbol: `AgentCancelled` (subclass of
  `AutoAgentError`).

  Use case: a UI "Cancel" button while the agent is working. The host
  passes a `threading.Event` to `run_messages` and sets it on click;
  the worker catches `AgentCancelled` and reports a clean stop.

## [0.2.0] - 2026-05-24

### Added
- **`post_turn_hook`** on `Agent` for post-execution verification. The hook
  is called every time the LLM emits a final text response (would normally
  end the run). It receives an `AgentTurnContext` snapshot and can return
  either `None` to confirm the turn, or a `Message` to inject as a
  correction and trigger another agent turn. Bounded by the new
  `max_corrections_per_run` parameter (default 1) to prevent loops. New
  public symbols: `AgentTurnContext`, `PostTurnHook`.

  Solves a common class of bugs where the tool succeeded (file written)
  but the downstream effect failed (host crashed loading the file).
  Hosts now express their own verification logic instead of each
  reimplementing a feedback loop.

  Hook exceptions are caught and logged; the agent's turn ends normally
  rather than propagating an internal error to the user.

## [0.1.0] - 2026-05-23

First publishable release. The API surface exported from
`autoagent/__init__.py` is the SemVer-stable contract from this version
forward: anything imported via that module is covered by SemVer, anything
underscored or imported from a submodule path is internal and may change.

### Security
- Gemini provider sends the API key via the `x-goog-api-key` header
  instead of `?key=...` query string. Prevents leakage through HTTP
  access logs, proxies, browser history, and crash dumps.
- `validate_generated_tool_code` blocks references to `__builtins__`,
  `__import__`, `getattr`, `setattr`, `delattr`, `globals`, `locals`,
  `vars`, `importlib`, `__loader__`, and `__spec__` anywhere in the
  AST (Name or Attribute). Closes known bypasses where a malicious
  tool hides `eval` or dangerous imports behind one indirection level.
  `importlib` is now also always-banned as an import root.
- New `autoagent.logging` module: every internal logger has a
  `SecretRedactingFilter` that scrubs Bearer tokens, `x-api-key` /
  `x-goog-api-key` headers, and `?key=...` URL fragments before any
  handler receives the record.

### Added
- Public API frozen and documented in `autoagent/__init__.py`. The
  top-level exports now include `Agent`, `AgentResult`, `ToolRegistry`,
  `ToolSpec`, `ToolCall`, `Message`, `LLMRequest`, `LLMResponse`,
  `ModelConfig`, `LLMProvider`, `OpenAIProvider`, `AnthropicProvider`,
  `DeepSeekProvider`, `GeminiProvider`, `create_provider`,
  `ProjectWorkspace`, `PipelineManager`, `DynamicToolBuilder`,
  `ToolBuildRequest`, `EvolutionRuntime`, `enable_software_evolution`,
  `EVOLUTION_CAPABILITIES`, `tool`, `get_logger`, and the error
  hierarchy (`AutoAgentError`, `ProviderError`, `ToolError`,
  `ToolValidationError`, `MaxStepsExceeded`).
- `__all__` on every module to fence internal helpers off the public API.
- Thread-safety: `threading.RLock` guards `ToolRegistry._tools`,
  `ProjectWorkspace._changes`, and workspace file mutations. The
  registry is safe to read concurrently with `add`/`replace`; the
  workspace serializes (read-before, write, append-record) so the
  change history stays coherent under concurrent writes to the same
  path.
- `jsonschema` runtime dependency for tool argument validation.
- Comprehensive test suite (229 tests, 95% coverage) including
  `hypothesis`-based property tests for the LLM-output JSON parser
  and stress tests for thread-safety.

### Fixed
- `RegisteredTool.execute` no longer calls `asyncio.run()` when the
  current thread already has a running event loop. The coroutine now
  runs on a dedicated worker thread, making the registry safe to use
  from FastAPI, Jupyter, aiohttp, Discord bots, and any other modern
  async host.
- Tool arguments are validated against `ToolSpec.input_schema` BEFORE
  the handler is invoked. Invalid arguments produce a structured
  `ValidationError: ...` message that lets the LLM self-correct,
  instead of crashing the host with a Python `TypeError`.
- `_extract_first_json_object` no longer treats quotes in surrounding
  LLM prose as JSON string delimiters. A stray `"` in the model's
  preamble previously corrupted the parser state and caused it to
  miss the JSON block that followed. Discovered by a `hypothesis`
  property-based test now in the suite.
- `enable_software_evolution` is idempotent: calling it twice on the
  same agent no longer crashes with `ToolError: already registered`.
  Tools already present in the registry are skipped.

### Infrastructure
- Fixed broken `build-backend` in `pyproject.toml`
  (`setuptools.backends._legacy:_Backend` → `setuptools.build_meta`).
  `pip install -e .` works again.
- Removed the `sys.path.insert(0, ".")` hack from `tests/conftest.py`;
  the test suite now relies on `pip install -e .`.
- Added `pytest-asyncio`, `pytest-cov`, `pytest-timeout`, `hypothesis`
  to the `[dev]` optional dependencies.
- CI invariants: `ruff check`, `ruff format --check`, `mypy autoagent/`,
  and `pytest` are all green.

[Unreleased]: https://github.com/anomalyco/alyce_autoagent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anomalyco/alyce_autoagent/releases/tag/v0.1.0
