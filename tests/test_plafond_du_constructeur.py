"""Le plafond de sortie du modèle constructeur — `DynamicToolBuilder(max_build_tokens=)`.

Trouvé par le banc « outils qui cassent » sur un vrai modèle (Gemini, 13 nov. 2026) : à une demande d'amélioration,
le modèle a dépensé 7 861 jetons à RÉFLÉCHIR sur un plafond de sortie de 8 192 ; sa réponse a été coupée (fin :
`length`) avant d'avoir écrit l'outil. La bibliothèque disait « did not contain JSON » — et, avec des reprises,
aurait repayé la même réponse coupée.

* une réponse coupée par le plafond le DIT, et n'est jamais reprise (au même plafond, ce serait la même) ;
* l'hôte peut relever le plafond ; `improve` et le banc gardent celui de l'hôte.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from autoagent import DynamicToolBuilder, ToolBuildRequest
from autoagent.errors import ToolError, ToolValidationError
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, TokenUsage
from autoagent.toolbench import TOOL_TASKS, ScriptedToolProvider, run_tool_bench
from tests._outils_generes import DOUBLER, construire

COUPE = DOUBLER[: len(DOUBLER) // 2]            # la réponse du constructeur, coupée en plein JSON
DOUBLER_PARTOUT = DOUBLER.replace("raise ValueError('negatif : %d' % args['n'])", "return {'double': args['n'] * 2}")


def _reponse(contenu: str, fin: str = "stop", jetons: int = 100) -> LLMResponse:
    return LLMResponse(content=contenu, model="f", finish_reason=fin,
                       usage=TokenUsage(input_tokens=jetons - 10, output_tokens=10, total_tokens=jetons))


class _Reponses(LLMProvider):
    """Un constructeur qui rend ses réponses dans l'ordre, avec leur raison de fin."""

    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
        self.reponses = list(reponses)
        self.requetes: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requetes.append(request)
        return self.reponses.pop(0)


class TestUneReponseCoupeeParLePlafond:
    def test_elle_dit_pourquoi(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(_Reponses([_reponse(COUPE, "length", 9241)]), tools_dir=tmp_path)
        with pytest.raises(ToolValidationError) as info:
            b.build(ToolBuildRequest(capability="x"))
        message = str(info.value)
        assert "ran out of output tokens" in message and "max_tokens=8192" in message and "max_build_tokens" in message
        assert "did not contain JSON" not in message and isinstance(info.value.__cause__, ToolError)
        assert b.last_build_usage is not None and b.last_build_usage.total_tokens == 9241   # payé, donc compté
        assert list(tmp_path.glob("*.py")) == []

    def test_elle_n_est_jamais_reprise(self, tmp_path: Path) -> None:
        # Au même plafond, la reprise rendrait la même réponse coupée : des milliers de jetons pour rien.
        fournisseur = _Reponses([_reponse(COUPE, "length"), _reponse(DOUBLER)])
        b = DynamicToolBuilder(fournisseur, tools_dir=tmp_path, max_repairs=2)
        with pytest.raises(ToolValidationError, match="ran out of output tokens"):
            b.build(ToolBuildRequest(capability="x"))
        assert len(fournisseur.requetes) == 1

    def test_une_reponse_entiere_au_ras_du_plafond_est_acceptee(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(_Reponses([_reponse(DOUBLER, "length")]), tools_dir=tmp_path)
        assert b.build(ToolBuildRequest(capability="x"))(n=4) == {"double": 8}

    def test_un_refus_ordinaire_est_toujours_repris(self, tmp_path: Path) -> None:
        fournisseur = _Reponses([_reponse(COUPE, "stop"), _reponse(DOUBLER)])
        b = DynamicToolBuilder(fournisseur, tools_dir=tmp_path, max_repairs=1)
        assert b.build(ToolBuildRequest(capability="x"))(n=4) == {"double": 8} and len(fournisseur.requetes) == 2

    def test_une_reprise_coupee_s_arrete_la(self, tmp_path: Path) -> None:
        fournisseur = _Reponses([_reponse(COUPE, "stop"), _reponse(COUPE, "length"), _reponse(DOUBLER)])
        b = DynamicToolBuilder(fournisseur, tools_dir=tmp_path, max_repairs=3)
        with pytest.raises(ToolValidationError, match="ran out of output tokens"):
            b.build(ToolBuildRequest(capability="x"))
        assert len(fournisseur.requetes) == 2 and b.last_build_usage is not None and b.last_build_usage.total_tokens == 200


class TestLePlafondSeRegle:
    def test_le_defaut_reste_8192(self, tmp_path: Path) -> None:
        fournisseur = _Reponses([_reponse(DOUBLER)])
        b = DynamicToolBuilder(fournisseur, tools_dir=tmp_path)
        b.build(ToolBuildRequest(capability="x"))
        assert b.max_build_tokens == 8192 and fournisseur.requetes[0].max_tokens == 8192

    def test_l_hote_le_releve(self, tmp_path: Path) -> None:
        fournisseur = _Reponses([_reponse(DOUBLER), _reponse(COUPE, "length")])
        b = DynamicToolBuilder(fournisseur, tools_dir=tmp_path, max_build_tokens=24000)
        b.build(ToolBuildRequest(capability="x"))
        assert fournisseur.requetes[0].max_tokens == 24000
        with pytest.raises(ToolValidationError, match="max_tokens=24000"):
            b.build(ToolBuildRequest(capability="y"))

    @pytest.mark.parametrize("valeur", [0, -1, True, 1.5, "8192", None])
    def test_une_valeur_qui_n_est_pas_un_entier_positif_est_refusee(self, tmp_path: Path, valeur: Any) -> None:
        with pytest.raises(ValueError, match="max_build_tokens"):
            DynamicToolBuilder(_Reponses([]), tools_dir=tmp_path, max_build_tokens=valeur)

    def test_improve_garde_le_plafond_de_l_hote(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, max_build_tokens=24000)
        with pytest.raises(ToolError):
            outil(n=-1)                                              # un plantage : le cas raté
        b.provider.contenus.append(DOUBLER_PARTOUT)                  # type: ignore[attr-defined]
        assert b.improve("doubler").status == "improved"
        assert [r.max_tokens for r in b.provider.requetes] == [24000, 24000]   # type: ignore[attr-defined]

    def test_le_banc_le_transmet_a_chaque_construction(self) -> None:
        t = TOOL_TASKS[0]
        petite = dataclasses.replace(t, easy=t.easy[:2], hard=t.hard[:1], later=t.later[:1])
        fournisseur = ScriptedToolProvider([petite])
        run_tool_bench(fournisseur, tasks=[petite], max_build_tokens=24000)
        assert [r.max_tokens for r in fournisseur.requests] == [24000] * 4
        with pytest.raises(ValueError, match="max_build_tokens"):
            run_tool_bench(fournisseur, tasks=[petite], max_build_tokens=0)
