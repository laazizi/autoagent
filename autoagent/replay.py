"""Record / replay déterministe des runs (0.16.0) — reproductibilité.

Le non-déterminisme des agents (2 à 4 trajectoires différentes sur 10 runs,
même à température 0) rend les bugs de prod irreproductibles. Ce module gèle
un vrai run dans un « fixture » JSONL, puis le REJOUE à l'identique.

Deux modes, chacun pour un métier (rien à configurer d'autre que la présence
ou non du registre rejoué) :

* **Replay TOTAL** (LLM + outils gelés) — hors-ligne, zéro réseau, zéro effet
  de bord : pour la CI et la non-régression (pas de clé API, pas d'envoi réel).
* **Replay LLM SEUL** (outils ré-exécutés) — pour déboguer / développer un
  outil pendant que le LLM redit la même chose.

Le replay ne teste NI le LLM NI les outils (gelés) : il exerce TON code —
boucle, `tool_policy`, `post_turn_hook`, mémoire, bornement, parsing, taint.
C'est là que vivent tes bugs et qu'arrivent tes changements. Une divergence
(prompt/code modifié) lève ``ReplayMismatch`` en pointant l'étape exacte.

    from autoagent import Agent, RecordSession, ReplaySession

    with RecordSession("run.jsonl") as rec:                 # enregistre un vrai run
        agent = Agent(rec.provider(vrai_provider), registry=rec.registry())
        for t in outils: agent.add_tool(t)
        agent.run("…")

    with ReplaySession("run.jsonl") as rep:                 # rejoue, hors-ligne
        agent = Agent(rep.provider(), registry=rep.registry())
        result = agent.run("…")                             # trajectoire identique

Zéro dépendance : JSONL + urllib-free. Aucune modification du cœur de la
boucle — uniquement les points d'extension publics `provider=` / `registry=`.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .errors import ReplayMismatch
from .logging import get_logger, redact
from .registry import ToolRegistry, ToolResult
from .schema import LLMRequest, LLMResponse, ModelConfig, StreamChunk, ToolCall

__all__ = [
    "RecordSession",
    "ReplaySession",
    "RecordingProvider",
    "RecordingRegistry",
    "ReplayProvider",
    "ReplayRegistry",
]

_log = get_logger("replay")

_PREVIEW = 200


def _signature(request: LLMRequest, do_redact: bool) -> dict[str, Any]:
    """Signature LÉGÈRE d'une requête LLM — assez pour détecter une
    divergence, pas le prompt entier (fixture léger, moins de PII)."""
    last_user = next(
        (m.content for m in reversed(request.messages) if m.role == "user"), ""
    )
    last_user = (last_user or "")[:_PREVIEW]
    if do_redact:
        last_user = redact(last_user)
    return {
        "messages": len(request.messages),
        "last_user": last_user,
        "tools": sorted(t.name for t in request.tools),
    }


def _maybe_redact(obj: Any, do_redact: bool) -> Any:
    """Redaction récursive des secrets évidents dans les chaînes d'un payload."""
    if not do_redact:
        return obj
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: _maybe_redact(v, True) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_maybe_redact(v, True) for v in obj]
    return obj


# ── Enregistrement ───────────────────────────────────────────────────────────

class _Recorder:
    """Écrit les événements du run dans un fixture JSONL (thread-safe :
    `parallel_tool_calls` enregistre depuis plusieurs threads)."""

    def __init__(self, path: str | Path, *, redact_secrets: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self._redact = redact_secrets
        self._seq = 0
        self._lock = threading.Lock()

    def _write(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            event["seq"] = self._seq
            self._f.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._f.flush()

    def llm(self, request: LLMRequest, response: LLMResponse) -> None:
        self._write({
            "kind": "llm",
            "request": _signature(request, self._redact),
            "response": _maybe_redact(response.to_dict(), self._redact),
        })

    def tool(self, call: ToolCall, result: ToolResult) -> None:
        self._write({
            "kind": "tool",
            "call_id": call.id,
            "name": call.name,
            "result": _maybe_redact(result.to_dict(), self._redact),
        })

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.close()


class RecordingProvider:
    """Enveloppe un provider réel : passe tout au vrai provider et enregistre
    chaque échange. Duck-typé comme un ``LLMProvider`` (``config``/``complete``
    /``stream``)."""

    def __init__(self, wraps: Any, recorder: _Recorder) -> None:
        self._wraps = wraps
        self._rec = recorder
        self.config = getattr(wraps, "config", None)

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._wraps.complete(request)
        self._rec.llm(request, response)
        return response

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        final: LLMResponse | None = None
        for chunk in self._wraps.stream(request):
            if chunk.type == "final" and chunk.response is not None:
                final = chunk.response
            yield chunk
        if final is None:  # provider sans chunk final — filet
            final = LLMResponse(content="")
        self._rec.llm(request, final)


class RecordingRegistry(ToolRegistry):
    """Registre normal qui enregistre CHAQUE résultat d'outil (par call_id)."""

    def __init__(self, recorder: _Recorder) -> None:
        super().__init__()
        self._rec = recorder

    def execute(self, call: ToolCall, context: dict[str, Any] | None = None) -> ToolResult:
        result = super().execute(call, context=context)
        self._rec.tool(call, result)
        return result


class RecordSession:
    """Façade d'enregistrement. Context manager : ferme le fixture en sortie.

    ``provider(wraps)`` enveloppe ton vrai provider ; ``registry()`` te donne
    un registre enregistreur à peupler comme d'habitude (``agent.add_tool`` /
    ``@agent.tool``). ``redact`` (défaut True) scrube les secrets évidents du
    fixture — utile si tu le commits pour la CI.
    """

    def __init__(self, path: str | Path, *, redact: bool = True) -> None:
        self._recorder = _Recorder(path, redact_secrets=redact)

    def provider(self, wraps: Any) -> RecordingProvider:
        return RecordingProvider(wraps, self._recorder)

    def registry(self) -> RecordingRegistry:
        return RecordingRegistry(self._recorder)

    def close(self) -> None:
        self._recorder.close()

    def __enter__(self) -> "RecordSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ── Relecture ────────────────────────────────────────────────────────────────

class _Player:
    """Lit un fixture et sert les événements : LLM par POSITION (les appels
    provider sont séquentiels dans la boucle), outils par CALL_ID (robuste aux
    tool calls parallèles). Une divergence lève ``ReplayMismatch``."""

    def __init__(self, path: str | Path, *, strict: bool = True) -> None:
        self.strict = strict
        self._llm: list[dict[str, Any]] = []
        self._tools: dict[str, dict[str, Any]] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("kind") == "llm":
                self._llm.append(event)
            elif event.get("kind") == "tool":
                self._tools[event["call_id"]] = event
        self._llm_cursor = 0
        self._lock = threading.Lock()
        self.model = next((e["response"].get("model") for e in self._llm), None)

    def next_llm(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            if self._llm_cursor >= len(self._llm):
                raise ReplayMismatch(
                    f"le run demande un appel LLM #{self._llm_cursor + 1} mais le "
                    f"fixture n'en contient que {len(self._llm)} — le run a divergé "
                    "après la fin de l'enregistrement."
                )
            event = self._llm[self._llm_cursor]
            self._llm_cursor += 1
        cursor = self._llm_cursor
        if self.strict:
            recorded = event["request"]
            actual = _signature(request, do_redact=True)
            if recorded.get("tools") != actual.get("tools") or recorded.get("messages") != actual.get("messages"):
                raise ReplayMismatch(
                    f"divergence à l'appel LLM #{cursor} : "
                    f"enregistré {recorded}, obtenu {actual}. "
                    "Le comportement de l'agent a changé depuis l'enregistrement."
                )
        return LLMResponse.from_dict(event["response"])

    def tool(self, call: ToolCall) -> ToolResult:
        event = self._tools.get(call.id)
        if event is None:
            raise ReplayMismatch(
                f"l'outil « {call.name} » (id {call.id}) n'a pas d'enregistrement "
                "dans le fixture — le run a divergé de la trajectoire enregistrée."
            )
        if self.strict and event.get("name") != call.name:
            raise ReplayMismatch(
                f"divergence d'outil pour l'id {call.id} : enregistré "
                f"« {event.get('name')} », obtenu « {call.name} »."
            )
        return ToolResult.from_dict(event["result"])


class ReplayProvider:
    """Rejoue les réponses LLM enregistrées, sans réseau. Duck-typé LLMProvider."""

    def __init__(self, player: _Player) -> None:
        self._player = player
        self.config = ModelConfig(provider="replay", model=player.model or "replay")

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self._player.next_llm(request)

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        response = self._player.next_llm(request)
        if response.content:
            yield StreamChunk(type="text", text=response.content)
        yield StreamChunk(type="final", response=response)


class ReplayRegistry(ToolRegistry):
    """Rejoue les résultats d'outils enregistrés — les handlers ne sont JAMAIS
    exécutés (zéro effet de bord). Mode replay TOTAL."""

    def __init__(self, player: _Player) -> None:
        super().__init__()
        self._player = player

    def execute(self, call: ToolCall, context: dict[str, Any] | None = None) -> ToolResult:
        return self._player.tool(call)


class ReplaySession:
    """Façade de relecture. ``provider()`` rejoue le LLM ; ``registry()``
    rejoue les outils (mode TOTAL hors-ligne). N'appeler QUE ``provider()`` et
    passer un vrai registre à l'agent = mode LLM-seul (outils ré-exécutés).

    ``strict`` (défaut True) : lève ``ReplayMismatch`` dès qu'une requête dévie
    de la trajectoire enregistrée. ``strict=False`` = positionnel best-effort.
    """

    def __init__(self, path: str | Path, *, strict: bool = True) -> None:
        self._player = _Player(path, strict=strict)

    def provider(self) -> ReplayProvider:
        return ReplayProvider(self._player)

    def registry(self) -> ReplayRegistry:
        return ReplayRegistry(self._player)

    def __enter__(self) -> "ReplaySession":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None
