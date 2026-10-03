# AGENTS.md — context for coding agents working on this repo

This file is the single source of agent-facing guidance. `CLAUDE.md`,
`GEMINI.md` and `.github/copilot-instructions.md` just point here.

## What this is

**autoagent** — a zero-dependency LLM agent runtime (published on PyPI as
[`autoagent-core`](https://pypi.org/project/autoagent-core/), imports as
`import autoagent`). The loop LLM ↔ tools plus code-level guardrails:
bounded workspace, Docker/AST sandbox, tool policies with human approval
gates, durable runs, fact memory, MCP client, OpenTelemetry export.
**It is a library, not a framework** — hosts own the control flow.

## Hard rules (violating these fails review)

1. **The CODE is the source of truth.** Docs have lagged before. When in
   doubt, read `autoagent/*.py` — every module has `__all__` and rich
   docstrings. Never invent an API from memory.
2. **Zero runtime dependencies.** The core imports stdlib ONLY — since
   0.21.0 even JSON-Schema validation is internal (`autoagent/validation.py`,
   differential-tested against `jsonschema`, which is now a `dev` extra).
   Never add a dependency. Optional integrations (OpenTelemetry)
   use lazy imports inside the class + an extra in `pyproject.toml`.
   Provider adapters speak raw wire formats via `urllib` — no SDKs.
3. **Sync by design.** No `async def` in the public API. Streaming is a
   sync iterator. Hosts wrap with threads if they need asyncio.
4. **Failure contracts are deliberate — keep them:**
   | Layer | Contract |
   |---|---|
   | Observability (trace, checkpoint, OTel, memory compaction) | **fail-open**: log and continue, never break the run |
   | Security (`tool_policy`, workspace, sandbox, approval manifest) | **fail-closed**: a crashing policy DENIES |
   | Tool handlers | exceptions become tool errors the model sees — never a crash |
5. **Run the tests**: `pytest tests -q` (1330 tests, about a minute — about 60 s on Windows, where each sandbox subprocess costs ~0.1 s; the 12 `test_constructeur` tests need Node and are skipped without it). Docker
   sandbox tests skip themselves when no daemon. `tests/conftest.py`
   provides `FakeLLMProvider` (records requests in `.calls`).
6. **Sync the docs with any change** — the recurring failure mode of this
   repo is stale summaries. Checklist: `autoagent-dev-doc.md` (section +
   Annexe A/B + version table), `CHANGELOG.md` `[Unreleased]`, README
   feature/comparison tables, the visual builder
   (`constructeur_autoagent.html` presets/blocks), demos README.
   Annexe B of the dev-doc is executable — its import block must run.
7. **Commits**: plain messages, no AI co-author trailers. Never commit
   `.env`, tokens, or `.context/` (personal, gitignored).

## Map

| Path | Contents |
|---|---|
| `autoagent/agent.py` | the loop; `Agent`, `RunState`, `tool_policy`, `as_tool`, recall/remember tools |
| `autoagent/memory.py` | `Memory` protocol, `BufferMemory`, `SummarizingMemory`, `FactMemory` (facts kept up to date; sleep-time `background=True`; semantic `embed_fn`) |
| `autoagent/mcp.py` | zero-dep MCP client (stdio), `mount(agent)` |
| `autoagent/otel.py` | OpenTelemetry exporter (optional extra `[otel]`) |
| `autoagent/registry.py` / `schema.py` | tool registry + JSON-schema generation; wire types |
| `autoagent/workspace.py` / `sandbox.py` / `approval.py` | bounded writes; Docker/AST sandbox (`SubprocessSandbox(warm=True)`: one persistent worker per tool, ~6 ms/call vs ~107 — opt-in, globals survive between calls); hash-manifest promotion |
| `autoagent/dynamic.py` | `DynamicToolBuilder` — tools written by the model: `max_repairs` (refusal fed back to the builder), `persist` (catalogue + reload across runs, sha256-checked), `host_functions` (bridge), tolerant self-tests (`self_test_rel_tol`); `PythonRunner` / `agent.enable_run_python()` — ephemeral, validated, sandboxed snippet (0.22.0). Builder tokens reach `token_budget` through `last_build_usage`. The AST validator is a denylist, NOT a boundary — Docker is |
| `autoagent/orchestrator.py` | host-driven deterministic flows |
| `autoagent/validation.py` | internal JSON-Schema validator (0.21.0) — replaces `jsonschema`; the subset the library generates + MCP/model schemas; differential-tested against `jsonschema` when the `dev` extra is present |
| `autoagent/synthesis.py` | `synthesize_tool` — program synthesis by example (0.21.0): host truth split shown/held-out, the sandbox judges, held-out cases never reach the model |
| `autoagent/guards.py` | `TurnGuards` — loop guard, trifecta, host policy, moved out of the loop (0.21.0); incremental signature counter |
| `autoagent/bounds.py` | `Bounds` — the eight bounds as one frozen dataclass; `agent.bounds` snapshot (0.21.0) |
| `autoagent/_fichiers.py` | `atomic_write_text` / `quarantine` — state files (facts, vectors, manifest) never left truncated nor overwritten when unreadable (0.22.0) |
| `autoagent/cascade.py` | `cascade` — cheap tier first, escalate only when the host's `check` refuses (0.21.0); failed tiers paid; `ApprovalRequired` propagates |
| `autoagent/trace_metrics.py` | `summarize_trace` — efficiency counters from a JSONL trace, runs rebuilt from span parentage (0.21.0) |
| `autoagent/journal.py` | `Journal` — durable, append-only, hash-chained JSONL, ONE writer (OS lock on `<file>.lock`, freed at process death). INTENT fsync'd BEFORE the effect, result AFTER (`Agent._run_loop._executer`); `Agent(journal=)` + `resume_from_journal()`: completed calls re-injected (never re-run, not re-policed), unknown outcomes raise `OutcomeUnknown` BEFORE any tool of the step runs (`idempotent=True` tools re-run), `journal.resolve(...)` is the host's decision. **Fail-CLOSED** (no intent written, no effect) — the opposite of the trace, so they stay two mechanisms. `on_boundary` hook = kill points for tests. `tests/test_journal.py` (crash at each of 10 boundaries; 3 mutants must go red) + `tests/test_journal_kill.py` (real `os._exit`). Power loss NOT tested |
| `autoagent/policy.py` | `ToolPolicySpec` — tool policy as versionable JSON; `compile()` → the existing `tool_policy` signature; monotonic containment (`narrow`/`expand`); `when: {"source": …}` |
| `autoagent/gate.py` | `ActionGate` — THE decision for every path that acts: the host-function bridge (`run_python`, generated tools), native `call_host`, `call_host_function`, and sub-agents with `inherit_policy=True`. Policy (`evaluate_policy` — the ONLY place `tool_policy` is called) + trifecta (`trifecta_blocks`) + taint; fail-closed; an approval request DENIES (no pause mid-program). The loop sets the gate around every tool execution (`_executer`); sinks read `active_gate()`. `tests/test_gate.py` AST-checks that every host-function sink decides BEFORE it runs — a new execution path must fail it, not bypass the policy. Proven on 0.22.0: the bridge let a refused mail leave |
| `autoagent/eval.py` | `run_k` — `pass^k` reliability measurement with a host-supplied DETERMINISTIC judge (never an LLM judge); `_tentative` is the one measured attempt both `run_k` and `compare_configs` use |
| `autoagent/compare.py` | `compare_configs` — two configurations on the same tasks, deterministic judges, arms ALTERNATING each repetition. The verdict needs TWO calculations to agree: an exact stratified permutation test (false-alert probability ≤ α by construction) and a Wilson–Newcombe interval; default answer "indistinguishable". Measured by exact enumeration: the interval alone declared a winner on IDENTICAL configurations up to 14.6 % of the time — never simplify the rule to it (`tests/test_compare_calibration.py` pins this). A/A control, suite/arm fingerprints, cost with bootstrap, `detectable_difference` (what the plan can see). Pure functions `wilson_interval`, `paired_interval`, `paired_p_value` |
| `autoagent/providers/` | OpenAI, Anthropic, DeepSeek, Gemini (raw wire) + `RoutingProvider`; `base.py` holds `synthetic_call_id` and `parse_tool_arguments` (invalid JSON escapes repaired, truncated JSON still refused) |
| `examples_autoagent/` | 37 runnable demos (French), one facet each — `_common.py` picks the provider from `.env` |
| `examples/` | the 55-line vs 164-line before/after argument |
| `constructeur_autoagent.html` | offline visual builder → generates Python; presets must compile (see harness note in git history) |
| `autoagent-dev-doc.md` | the full reference (§1–40) |

## Release process (maintainer-triggered only)

Bump `autoagent/__init__.py.__version__` + `pyproject.toml` → move
`[Unreleased]` to a dated section in `CHANGELOG.md` → full test pass →
commit + tag `vX.Y.Z` → `python -m build` → `twine check` → `twine
upload` → GitHub release. Never republish an existing version.

## Style

Match the file you touch: French docstrings/comments in `memory.py` and
demos, English in `mcp.py`/`otel.py`/README. Guard-rail comments explain
*constraints*, not what the next line does.
