"""Aides partagées des tests sur les outils créés par le modèle (carnet, fiche de qualité, apprentissage) :
des fournisseurs scriptés, des outils tout écrits, et la construction d'une bibliothèque persistante."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoagent import DynamicToolBuilder, ToolBuildRequest
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, ToolCall

CODE_DOUBLER = (
    "def run(args, context):\n"
    "    if args['n'] < 0:\n"
    "        raise ValueError('negatif : %d' % args['n'])\n"
    "    return {'double': args['n'] * 2}\n"
)
CODE_HOTE = (
    "def run(args, context):\n"
    "    return {'valeur': context['call_host']('lire', {'cle': args['cle']})}\n"
)


def payload(code: str, nom: str, proprietes: dict[str, Any], tests: list[dict[str, Any]] | None = None) -> str:
    """La réponse d'un modèle constructeur : métadonnées, code, auto-tests (aucun par défaut)."""
    return json.dumps({
        "tool": {"name": nom, "description": f"outil {nom}",
                 "input_schema": {"type": "object", "properties": proprietes, "required": sorted(proprietes)},
                 "permissions": []},
        "code": code,
        "self_tests": tests or [],
    })


DOUBLER = payload(CODE_DOUBLER, "doubler", {"n": {"type": "integer"}})
LECTEUR = payload(CODE_HOTE, "lecteur", {"cle": {"type": "string"}})


class Sequence(LLMProvider):
    """Le modèle CONSTRUCTEUR : rend ses contenus dans l'ordre."""

    def __init__(self, contenus: list[str]) -> None:
        super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
        self.contenus = list(contenus)
        self.requetes: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requetes.append(request)
        return LLMResponse(content=self.contenus.pop(0), model="f")


class Principal(LLMProvider):
    """Le modèle de l'AGENT : rend ses réponses dans l'ordre, puis « fini »."""

    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self.reponses.pop(0) if self.reponses else LLMResponse(content="fini")


def appel(id_: str, nom: str, **args: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=id_, name=nom, arguments=args)])


def construire(dossier: Path, contenu: str = DOUBLER, **kw: Any) -> tuple[DynamicToolBuilder, Any]:
    """Une bibliothèque persistante avec carnet (20 appels par défaut) et UN outil construit."""
    kw.setdefault("record_calls", 20)
    b = DynamicToolBuilder(Sequence([contenu]), tools_dir=dossier, persist=True, **kw)
    return b, b.build(ToolBuildRequest(capability="x"))


class BacEnPanne:
    """Un bac à sable qui échoue lui-même (un démon Docker arrêté)."""

    timeout = 10.0

    def isolation(self) -> dict[str, Any]:
        return {"kind": "faux", "os_boundary": False}

    def run_python_tool(self, *a: Any, **k: Any) -> dict[str, Any]:
        from autoagent.errors import ToolError
        raise ToolError("Docker sandbox runner failed: stdout='' stderr='Cannot connect to the Docker daemon'")
