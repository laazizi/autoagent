---
name: autoagent-dev
description: >
  Working on the autoagent library codebase (this repo). Loads the verified
  API surface, the hard rules (zero dependencies, fail-open/fail-closed
  contracts, docs-sync checklist) and the release process. Use it whenever
  reading or modifying autoagent/*.py, the demos, the visual builder or the
  developer docs.
---

# autoagent-dev — contributing to the library

**First: read `AGENTS.md` at the repo root.** It is the master context file
(hard rules, module map, failure contracts, release process) and it wins
over anything else, including this skill.

Working method that has proven itself on this codebase:

1. **Read the module before touching it** — every `autoagent/*.py` has
   `__all__` and rich docstrings; the code is the source of truth, docs
   have lagged before.
2. **Prove changes with tests, then in real conditions.** `pytest tests -q`
   must stay green; new behavior gets a test that would have caught the
   bug. For LLM-facing behavior, a real-provider smoke run has repeatedly
   found bugs unit tests missed (new-conversation detection, embedding
   model 404s…).
3. **Never add a dependency.** stdlib + `jsonschema`. Optional
   integrations = lazy import + `pyproject` extra (see `otel.py`).
4. **Keep the failure contracts** — observability fails open, security
   fails closed, tool exceptions become tool errors.
5. **Sweep the summaries after every change** : dev-doc section + Annexe
   A/B + version table, `CHANGELOG.md [Unreleased]`, README tables, the
   builder's presets, the demos README. Stale summaries are this repo's
   recurring failure mode.
6. Demos live in `examples_autoagent/` (French, `_common.py` resolves the
   provider from `.env`); the before/after argument lives in `examples/`.
