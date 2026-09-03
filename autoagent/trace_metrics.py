"""Métriques d'efficacité tirées d'une TRACE — après coup, sans rien ajouter au run (0.21.0).

Chaque run émet déjà une trace JSONL (`TraceEmitter`, §4.5). Elle contient de
quoi répondre à des questions que le taux de réussite seul ne pose pas :

    Combien d'appels d'outils étaient REDONDANTS (même outil, mêmes arguments,
    dans le même run) ? Combien ont été REFUSÉS, et par quelle garde ? Combien de
    jetons par SUCCÈS, échecs compris ? Combien d'outils lancés en avance, combien
    de caractères élagués ?

Le constat de Probe&Prefill (arXiv 2605.09252) — près de la moitié des appels
d'outils sont inutiles sur leur banc — n'est mesurable chez nous qu'ainsi : on
ne lit pas dans le modèle, on lit dans ce qu'il a fait. Et AgentAtlas
(arXiv 2605.20530) rappelle qu'une trajectoire se juge sur ses décisions
(agir, demander, refuser, s'arrêter…), pas sur le seul résultat.

Ce que ce module NE fait pas : juger la QUALITÉ d'une réponse — ça, c'est le
`check` de l'hôte (§25.4). Il compte ce qui est comptable, et rend `None` là où
la trace ne dit rien, jamais un zéro inventé.

Les runs sont reconstitués par la parenté des spans (`parent_id`), pas par
l'ordre des lignes : deux runs entrelacés dans un même fichier sont séparés
correctement.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["TraceMetrics", "summarize_trace"]

_GARDES = ("loop_guard_block", "trifecta_block", "tool_policy_deny")
_TEMOINS = ("loop_guard_would_block", "trifecta_would_block")


@dataclass
class TraceMetrics:
    runs: int = 0
    runs_ok: int = 0
    runs_by_status: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    redundant_tool_calls: int = 0        # même run, même outil, mêmes arguments — déjà vu
    blocked_by_guard: dict[str, int] = field(default_factory=dict)
    would_block: dict[str, int] = field(default_factory=dict)
    early_tool_calls: int = 0
    pruned_chars: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    redundant_by_tool: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    @property
    def tokens_per_success(self) -> float | None:
        """Jetons de TOUS les runs (ratés compris) par run réussi. None si rien
        n'est rapporté ou si rien n'a réussi — pas d'infini, pas de zéro inventé."""
        if self.total_tokens is None or not self.runs_ok or not self.total_tokens:
            return None
        return self.total_tokens / self.runs_ok

    @property
    def redundant_ratio(self) -> float | None:
        return self.redundant_tool_calls / self.tool_calls if self.tool_calls else None

    @property
    def tool_error_ratio(self) -> float | None:
        return self.tool_errors / self.tool_calls if self.tool_calls else None

    def summary(self) -> str:
        lignes = [
            f"runs {self.runs} (ok {self.runs_ok}) · appels LLM {self.llm_calls} · "
            f"appels d'outils {self.tool_calls}"
        ]
        if self.tool_calls:
            lignes.append(
                f"redondants {self.redundant_tool_calls} ({self.redundant_ratio:.0%}) · "
                f"en erreur {self.tool_errors} ({self.tool_error_ratio:.0%})"
            )
        if self.blocked_by_guard:
            lignes.append("refusés : " + ", ".join(f"{k} {v}" for k, v in self.blocked_by_guard.items()))
        if self.would_block:
            lignes.append("auraient été refusés (témoin) : "
                          + ", ".join(f"{k} {v}" for k, v in self.would_block.items()))
        if self.early_tool_calls:
            lignes.append(f"lancés en avance {self.early_tool_calls}")
        if self.pruned_chars:
            lignes.append(f"élagué {self.pruned_chars} caractères")
        if self.total_tokens is not None:
            jps = (f" · {self.tokens_per_success:.0f} jetons/succès"
                   if self.tokens_per_success is not None else "")
            lignes.append(f"jetons {self.total_tokens}{jps}")
        return "\n".join(lignes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs, "runs_ok": self.runs_ok, "runs_by_status": dict(self.runs_by_status),
            "llm_calls": self.llm_calls, "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors, "tool_error_ratio": self.tool_error_ratio,
            "redundant_tool_calls": self.redundant_tool_calls, "redundant_ratio": self.redundant_ratio,
            "redundant_by_tool": dict(self.redundant_by_tool),
            "blocked_by_guard": dict(self.blocked_by_guard), "would_block": dict(self.would_block),
            "early_tool_calls": self.early_tool_calls, "pruned_chars": self.pruned_chars,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens, "tokens_per_success": self.tokens_per_success,
        }


def _events(source: Any) -> list[dict[str, Any]]:
    """Accepte un chemin JSONL, un itérable de dicts, ou d'objets `TraceEvent`."""
    if isinstance(source, (str, Path)):
        out: list[dict[str, Any]] = []
        with open(source, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    out.append(json.loads(ligne))
        return out
    resultat: list[dict[str, Any]] = []
    for e in source:  # type: ignore[union-attr]
        if isinstance(e, dict):
            resultat.append(e)
        elif hasattr(e, "to_dict"):
            resultat.append(e.to_dict())
        else:
            resultat.append({"type": getattr(e, "type", None), "span_id": getattr(e, "span_id", None),
                             "parent_id": getattr(e, "parent_id", None),
                             "payload": dict(getattr(e, "payload", {}) or {})})
    return resultat


def summarize_trace(source: str | Path | Iterable[Any]) -> TraceMetrics:
    """Compte ce qu'une trace permet de compter. Rend `None` où elle est muette."""
    events = _events(source)
    parent: dict[str, str | None] = {}
    types: dict[str, str] = {}
    for e in events:
        sid = e.get("span_id")
        if sid:
            parent[sid] = e.get("parent_id")
            types[sid] = e.get("type", "")

    def racine(e: dict[str, Any]) -> str | None:
        """Le span du `run_start` englobant, par remontée de la parenté."""
        sid = e.get("span_id") if e.get("type") == "run_start" else e.get("parent_id")
        vus = 0
        while sid is not None and vus < 10_000:
            if types.get(sid) == "run_start":
                return str(sid)
            sid = parent.get(sid)
            vus += 1
        return None

    m = TraceMetrics()
    signatures: dict[str | None, set[str]] = {}
    tokens_vus = False
    entree = sortie = 0
    for e in events:
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "run_start":
            m.runs += 1
        elif t == "run_end":
            statut = str(p.get("status", "?"))
            m.runs_by_status[statut] = m.runs_by_status.get(statut, 0) + 1
            if statut == "ok":
                m.runs_ok += 1
            m.early_tool_calls += int(p.get("early_tool_calls") or 0)
        elif t == "llm_request":
            m.llm_calls += 1
        elif t == "llm_response":
            if p.get("input_tokens") is not None or p.get("output_tokens") is not None:
                tokens_vus = True
                entree += int(p.get("input_tokens") or 0)
                sortie += int(p.get("output_tokens") or 0)
        elif t == "tool_call_start":
            m.tool_calls += 1
            run = racine(e)
            sig = f"{p.get('name')}({p.get('arguments_preview', '')})"
            vus = signatures.setdefault(run, set())
            if sig in vus:
                m.redundant_tool_calls += 1
                nom = str(p.get("name"))
                m.redundant_by_tool[nom] = m.redundant_by_tool.get(nom, 0) + 1
            else:
                vus.add(sig)
        elif t == "tool_call_end":
            if p.get("status") == "error":
                m.tool_errors += 1
        elif t in _GARDES:
            m.blocked_by_guard[t] = m.blocked_by_guard.get(t, 0) + 1
        elif t in _TEMOINS:
            m.would_block[t] = m.would_block.get(t, 0) + 1
        elif t == "context_pruned":
            m.pruned_chars += int(p.get("chars_saved") or 0)
    if tokens_vus:
        m.input_tokens, m.output_tokens = entree, sortie
    # tri stable des dicts pour un affichage reproductible
    m.blocked_by_guard = dict(sorted(m.blocked_by_guard.items()))
    m.would_block = dict(sorted(m.would_block.items()))
    m.redundant_by_tool = dict(Counter(m.redundant_by_tool).most_common())
    return m
