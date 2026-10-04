from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from autoagent.errors import ProviderError
from autoagent.http import is_retryable_status
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, StreamChunk

# Codes TEXTUELS qui disent « transitoire » (OpenAI : `rate_limit_exceeded`, `server_error` ; Anthropic :
# `overloaded_error` ; Gemini : `UNAVAILABLE`, `RESOURCE_EXHAUSTED`…). Une meilleure estimation, pas une table
# officielle : un code textuel inconnu n'est PAS réessayable.
_CODES_TRANSITOIRES = frozenset({
    "rate_limit_exceeded", "rate_limit_error", "rate_limit", "server_error", "overloaded", "overloaded_error",
    "service_unavailable", "unavailable", "timeout", "timeout_error", "api_error", "internal_error", "internal",
    "resource_exhausted", "deadline_exceeded",
})


def _statut_http(brut: Any) -> int | None:
    """Un statut HTTP PLAUSIBLE (100-599) lu dans un champ `code` : un entier, un flottant entier ou une chaîne
    de chiffres. NaN, l'infini, un nombre hors plage ou une chaîne comme « ² » ne sont pas des statuts — et ne
    font jamais lever l'analyse d'une erreur (c'est elle qu'on est en train de signaler)."""
    if isinstance(brut, bool):
        return None
    try:
        if isinstance(brut, float):
            n = int(brut) if brut.is_integer() else None
        elif isinstance(brut, int):
            n = brut
        elif isinstance(brut, str) and brut.strip().isdecimal() and len(brut.strip()) <= 3:
            n = int(brut.strip())
        else:
            n = None
    except (ValueError, OverflowError):
        return None
    return n if n is not None and 100 <= n <= 599 else None


def fermer_flux(evenements: Any) -> None:
    """Ferme un flux d'événements SSE s'il sait se fermer — un générateur ; un double de test peut rendre une simple liste.

    Les `stream()` des fournisseurs l'appellent dans un `finally` : sur CPython 3.12.3 (le Python système d'Ubuntu 24.04),
    `close()` d'un générateur externe ne referme pas le générateur interne (`post_sse`) — la connexion d'un flux abandonné
    (barge-in, `cancel_token`) restait dans le pool et le serveur continuait d'émettre (mesuré). Sur un flux épuisé,
    `close()` ne fait rien : la connexion normale est gardée pour la réutilisation."""
    fermer = getattr(evenements, "close", None)
    if callable(fermer):
        fermer()


def stream_error(provider: str, error: Any, where: str = "stream") -> ProviderError:
    """Une erreur reçue EN COURS de flux → `ProviderError` typée (0.23.1).

    Le statut HTTP 200 est déjà passé : le fournisseur annonce son échec DANS le flux
    (OpenRouter et les passerelles compatibles : `{"error": {"message", "code"}}` ;
    Gemini : `{"error": {"code", "message", "status"}}`). La 0.23.0 ne regardait que
    `choices` / `candidates` et ignorait l'événement : le run finissait en « succès »
    sur une réponse vide ou coupée. `retryable` suit la même règle que pour une
    erreur HTTP (`is_retryable_status`) — l'hôte branche dessus sans lire le message.
    """
    code: int | None = None
    transitoire = False
    if isinstance(error, dict):
        code = _statut_http(error.get("code"))
        # Un code TEXTUEL (`type`, `code` ou `status` : « server_error », « UNAVAILABLE »…) peut dire « transitoire »
        # quand il n'y a pas de statut numérique.
        textes = {str(error[k]).strip().lower() for k in ("code", "type", "status")
                  if isinstance(error.get(k), str)}
        transitoire = bool(textes & _CODES_TRANSITOIRES)
        message = str(error.get("message") or error.get("status") or error)
    else:
        message = str(error)
    etiquette = f" ({code})" if code is not None else ""
    return ProviderError(
        f"{provider} {where} error{etiquette}: {message[:500]}",
        status_code=code,
        retryable=is_retryable_status(code) if code is not None else transitoire,
    )


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``; mutates and returns ``base``.

    Nested dicts are merged key by key; anything else (scalars, lists) is
    replaced by the ``overlay`` value.
    """
    for key, value in overlay.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            deep_merge(current, value)
        else:
            base[key] = value
    return base


def synthetic_call_id(prefix: str, index: int) -> str:
    """Identifiant d'appel d'outil quand le fournisseur n'en donne pas (0.22.0).

    Gemini n'en renvoie jamais, OpenAI-compatibles et Anthropic presque
    toujours. L'ancien repli `f"{prefix}_{index}"` repartait de zéro à CHAQUE
    réponse : le premier appel de chaque tour s'appelait `gemini_tool_call_0`.
    Or tout ce qui se souvient d'un appel le fait par son id — la décision
    humaine d'une approbation (la lib recommande d'y mémoriser le verdict, l'id
    étant stable à travers la pause), le rejeu, l'exécution anticipée. Deux
    appels différents portant le même id, c'est une approbation qui vaut pour
    un autre envoi. Le suffixe aléatoire rend l'id unique dans le run ; le
    préfixe et l'index restent lisibles dans une trace.
    """
    return f"{prefix}_{index}_{uuid.uuid4().hex[:8]}"


_ECHAPPEMENTS_JSON = frozenset('"\\/bfnrt')
_QUATRE_HEX = frozenset("0123456789abcdefABCDEF")


def repair_json_escapes(text: str) -> str:
    """Double le antislash de chaque échappement INVALIDE à l'intérieur d'une chaîne JSON.

    Un modèle qui écrit du code (`re.compile("^\\d+$")`) dans un argument JSON
    met souvent `\\d` au lieu de `\\\\d` : `json.loads` refuse (« Invalid \\escape »)
    et l'appel d'outil partait en `{"_raw": ...}`. L'intention est sans
    ambiguïté — un antislash littéral suivi de `d` — et c'est exactement le
    texte source que le modèle voulait écrire. Seuls les échappements invalides
    sont touchés ; `\\n`, `\\"`, `\\\\`, `\\uXXXX` valides passent tels quels. Ne
    répare PAS un JSON coupé : l'appel garde alors son `_raw` et la validation
    de schéma le refuse (c'est voulu, voir `_assemble`).
    """
    sortie: list[str] = []
    dans_chaine = False
    i, n = 0, len(text)
    while i < n:
        car = text[i]
        if not dans_chaine:
            sortie.append(car)
            dans_chaine = car == '"'
            i += 1
        elif car == '"':
            sortie.append(car)
            dans_chaine = False
            i += 1
        elif car != "\\" or i + 1 >= n:        # caractère ordinaire, ou antislash final (texte coupé)
            sortie.append(car)
            i += 1
        elif text[i + 1] in _ECHAPPEMENTS_JSON:
            sortie.append(text[i:i + 2])
            i += 2
        elif text[i + 1] == "u" and len(text) >= i + 6 and all(c in _QUATRE_HEX for c in text[i + 2:i + 6]):
            sortie.append(text[i:i + 6])
            i += 6
        else:                                  # invalide : l'antislash devient littéral
            sortie.append("\\\\")
            i += 1
    return "".join(sortie)


def loads_tolerant(text: str) -> Any:
    """`json.loads`, puis — SEULEMENT si le texte est refusé — une seconde chance
    avec les échappements invalides réparés et les caractères de contrôle bruts
    (un vrai retour à la ligne dans une chaîne) acceptés. Un JSON valide n'est
    jamais modifié ; un JSON irréparable relève l'erreur ORIGINALE."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as erreur:
        try:
            return json.loads(repair_json_escapes(text), strict=False)
        except json.JSONDecodeError:
            raise erreur from None


def parse_tool_arguments(raw: Any) -> Any:
    """Arguments d'un appel d'outil tels que le fournisseur les a écrits.

    Déjà un objet → rendu tel quel ; vide → `{}` ; texte JSON → l'objet, après
    réparation éventuelle des échappements ; illisible → `{"_raw": texte}` (la
    validation de schéma le refuse et le modèle voit pourquoi — jamais `{}`, qui
    ferait partir l'outil avec ses valeurs par défaut sur des arguments perdus).
    """
    if not isinstance(raw, str):
        return {} if raw is None else raw
    if not raw.strip():
        return {}
    try:
        return loads_tolerant(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


class LLMProvider(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    def _with_extra_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge ``config.extra_body`` into a freshly built payload.

        Escape hatch for model-specific knobs ``LLMRequest`` does not map —
        Gemini's ``thinkingConfig``, OpenAI's ``reasoning_effort``… Every
        provider calls this on the way out of ``_build_payload``, so the
        mechanism is uniform and costs nothing when ``extra_body`` is empty.
        """
        if not self.config.extra_body:
            return payload
        return deep_merge(payload, self.config.extra_body)

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next model response for the agent loop."""

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        """Yield incremental chunks for the next model response.

        Default implementation is a NON-STREAMING FALLBACK: it calls
        ``complete()`` and emits the whole content as a single ``text``
        chunk, then the ``final`` chunk. This keeps the streaming API
        uniform across every provider — those without native SSE
        support degrade gracefully (the user gets the answer in one
        shot instead of token-by-token, but everything still works).

        Providers that support native streaming (Anthropic, Gemini)
        override this to emit real token deltas.
        """
        response = self.complete(request)
        if response.content:
            yield StreamChunk(type="text", text=response.content)
        yield StreamChunk(type="final", response=response)
