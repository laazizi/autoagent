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
2. **Zero runtime dependencies.** The core imports stdlib + `jsonschema`
   only. Never add a dependency. Optional integrations (OpenTelemetry)
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
5. **Run the tests**: `pytest tests -q` (536+ tests, <30 s). Docker
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
| `autoagent/workspace.py` / `sandbox.py` / `approval.py` | bounded writes; Docker/AST sandbox; hash-manifest promotion |
| `autoagent/orchestrator.py` | host-driven deterministic flows |
| `autoagent/providers/` | OpenAI, Anthropic, DeepSeek, Gemini (raw wire) + `RoutingProvider` |
| `examples_autoagent/` | 18 runnable demos (French), one facet each — `_common.py` picks the provider from `.env` |
| `examples/` | the 55-line vs 164-line before/after argument |
| `constructeur_autoagent.html` | offline visual builder → generates Python; presets must compile (see harness note in git history) |
| `autoagent-dev-doc.md` | the full reference (§1–21) |

## Release process (maintainer-triggered only)

Bump `autoagent/__init__.py.__version__` + `pyproject.toml` → move
`[Unreleased]` to a dated section in `CHANGELOG.md` → full test pass →
commit + tag `vX.Y.Z` → `python -m build` → `twine check` → `twine
upload` → GitHub release. Never republish an existing version.

## Style

Match the file you touch: French docstrings/comments in `memory.py` and
demos, English in `mcp.py`/`otel.py`/README. Guard-rail comments explain
*constraints*, not what the next line does.
