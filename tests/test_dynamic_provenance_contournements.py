"""Provenance des outils générés : les trois contournements de la nuit du 11 nov. 2026.

Trouvés par l'agent attaquant, reproduits par le juge (`.context/nuit-2026-11-11/attaque.md`) :

1. un parent teinté qui DÉLÈGUE la création à un sous-agent propre (`as_tool`, `delegate_to`) :
   l'outil était enregistré `tainted: False` et rechargé — la délégation blanchissait la teinte ;
2. un outil en quarantaine restait dans le registre et appelable au run suivant du même `Agent` ;
3. `approval.load_tools` lisait `*.py` sans consulter le catalogue et enregistrait l'outil.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoagent import Agent, DynamicToolBuilder, delegate_to
from autoagent.approval import ToolManifest, load_tools
from autoagent.dynamic import ToolBuildRequest
from autoagent.errors import ToolError
from autoagent.schema import LLMResponse

from .conftest import FakeLLMProvider
from .test_dynamic_provenance import Scripte, _appel, _payload, _recharge, _run, lire_page


def _resultats(run: Any, nom: str) -> list[str]:
    return [m.content or "" for m in run.messages if m.role == "tool" and m.name == nom]


def _construit(nom: str) -> list[LLMResponse]:
    return [_appel("create_python_tool", capability="x", tool_name=nom), LLMResponse(content="fait")]


class TestLaDelegationNeBlanchitPas:
    """La teinte du run qui délègue descend dans le run du sous-agent, pour la provenance,
    que le sous-agent hérite de la politique du parent ou non."""

    def _delegation(self, dossier: Path, *, parent_lit: bool, inherit: bool) -> DynamicToolBuilder:
        b = DynamicToolBuilder(FakeLLMProvider([_payload("enfantool", 2)]), tools_dir=dossier, persist=True)
        enfant = Agent(Scripte(_construit("enfantool")), max_steps=5)
        enfant.enable_dynamic_tools(b)
        tours = ([_appel("lire_page", url="u")] if parent_lit else []) + [
            _appel("fabriquer", request="cree un outil"), LLMResponse(content="ok")]
        parent = Agent(Scripte(tours), max_steps=6)
        parent.tool(lire_page, untrusted=True)
        parent.add_tool(enfant.as_tool(name="fabriquer", description="d", inherit_policy=inherit))
        parent.run("go")
        assert enfant._teinte_heritee is False, "l'héritage ne survit pas à la délégation"
        return b

    def test_as_tool_sans_heritage_de_politique(self, tmp_path: Path) -> None:
        b = self._delegation(tmp_path, parent_lit=True, inherit=False)
        prov = b.catalogue()["enfantool"]["provenance"]
        assert prov["tainted"] is True and prov["context"] == "delegated_from_tainted_run"
        assert b.quarantined() == ["enfantool"] and _recharge(tmp_path) == []

    def test_as_tool_avec_heritage_de_politique(self, tmp_path: Path) -> None:
        b = self._delegation(tmp_path, parent_lit=True, inherit=True)
        assert b.quarantined() == ["enfantool"] and _recharge(tmp_path) == []

    def test_parent_propre_enfant_propre(self, tmp_path: Path) -> None:
        b = self._delegation(tmp_path, parent_lit=False, inherit=False)
        assert b.catalogue()["enfantool"]["provenance"]["tainted"] is False
        assert _recharge(tmp_path) == ["enfantool"]

    def test_delegate_to(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(FakeLLMProvider([_payload("specialtool", 2)]), tools_dir=tmp_path, persist=True)
        specialiste = Agent(Scripte(_construit("specialtool")), max_steps=5)
        specialiste.enable_dynamic_tools(b)
        parent = Agent(Scripte([_appel("lire_page", url="u"),
                                _appel("delegate", requests=[{"specialist": "fab", "request": "cree"}]),
                                LLMResponse(content="ok")]), max_steps=6)
        parent.tool(lire_page, untrusted=True)
        parent.add_tool(delegate_to({"fab": specialiste}))
        parent.run("go")
        assert b.quarantined() == ["specialtool"] and _recharge(tmp_path) == []
        assert specialiste._teinte_heritee is False

    def test_delegate_to_en_parallele(self, tmp_path: Path) -> None:
        """Deux spécialistes : ils tournent dans les threads d'un pool, qui ne voient pas la
        porte du parent ; la teinte doit être lue dans le thread de l'appel."""
        b = DynamicToolBuilder(FakeLLMProvider([_payload("una", 2), _payload("unb", 3)]),
                               tools_dir=tmp_path, persist=True)
        sa = Agent(Scripte(_construit("una")), max_steps=5)
        sb = Agent(Scripte(_construit("unb")), max_steps=5)
        sa.enable_dynamic_tools(b)
        sb.enable_dynamic_tools(b)
        parent = Agent(Scripte([_appel("lire_page", url="u"),
                                _appel("delegate", requests=[{"specialist": "a", "request": "x"},
                                                             {"specialist": "b", "request": "y"}]),
                                LLMResponse(content="ok")]), max_steps=6)
        parent.tool(lire_page, untrusted=True)
        parent.add_tool(delegate_to({"a": sa, "b": sb}))
        parent.run("go")
        assert b.quarantined() == ["una", "unb"]

    def test_chaine_de_deux_delegations(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(FakeLLMProvider([_payload("petitool", 2)]), tools_dir=tmp_path, persist=True)
        petit = Agent(Scripte(_construit("petitool")), max_steps=5)
        petit.enable_dynamic_tools(b)
        milieu = Agent(Scripte([_appel("petit", request="cree"), LLMResponse(content="fait")]), max_steps=5)
        milieu.add_tool(petit.as_tool(name="petit", description="d"))
        parent = Agent(Scripte([_appel("lire_page", url="u"), _appel("milieu", request="fais faire"),
                                LLMResponse(content="ok")]), max_steps=6)
        parent.tool(lire_page, untrusted=True)
        parent.add_tool(milieu.as_tool(name="milieu", description="d"))
        parent.run("go")
        assert b.quarantined() == ["petitool"]

class TestLaQuarantaineSAppliqueALAppel:
    """Un outil en quarantaine ne sert plus qu'au run qui l'a construit."""

    def _agent(self, dossier: Path, tours: list[LLMResponse]) -> tuple[Agent, DynamicToolBuilder]:
        b = DynamicToolBuilder(FakeLLMProvider([_payload("persist", 2)]), tools_dir=dossier, persist=True)
        ag = Agent(Scripte(tours), max_steps=6)
        ag.tool(lire_page, untrusted=True)
        ag.enable_dynamic_tools(b)
        return ag, b

    def test_utilisable_dans_le_run_qui_l_a_construit(self, tmp_path: Path) -> None:
        ag, b = self._agent(tmp_path, [_appel("lire_page", url="u"),
                                       _appel("create_python_tool", capability="x", tool_name="persist"),
                                       _appel("persist", n=21), LLMResponse(content="ok")])
        r = ag.run("go")
        assert b.quarantined() == ["persist"]
        assert any("42" in c for c in _resultats(r, "persist"))

    def test_refuse_au_run_suivant_du_meme_agent(self, tmp_path: Path) -> None:
        ag, b = self._agent(tmp_path, [_appel("lire_page", url="u"),
                                       _appel("create_python_tool", capability="x", tool_name="persist"),
                                       LLMResponse(content="ok"), _appel("persist", n=21), LLMResponse(content="fini")])
        ag.run("run1")
        res = _resultats(ag.run("run2"), "persist")
        assert res and not any("42" in c for c in res) and "quarantine" in res[0]

    def test_liberation_rend_l_outil_appelable(self, tmp_path: Path) -> None:
        ag, b = self._agent(tmp_path, [_appel("lire_page", url="u"),
                                       _appel("create_python_tool", capability="x", tool_name="persist"),
                                       LLMResponse(content="ok"), _appel("persist", n=21), LLMResponse(content="fini")])
        ag.run("run1")
        assert b.release("persist", b.catalogue()["persist"]["sha256"])
        assert any("42" in c for c in _resultats(ag.run("run2"), "persist"))

    def test_quarantaine_posee_par_l_hote_sur_un_outil_deja_charge(self, tmp_path: Path) -> None:
        _run(tmp_path, [_payload("doubleur", 2)], lire_avant=False, nom="doubleur")
        b = DynamicToolBuilder(FakeLLMProvider([]), tools_dir=tmp_path, persist=True)
        ag = Agent(Scripte([_appel("doubleur", n=21), LLMResponse(content="a"),
                            _appel("doubleur", n=21), LLMResponse(content="b")]), max_steps=4)
        ag.enable_dynamic_tools(b)                       # recharge la bibliothèque
        assert any("42" in c for c in _resultats(ag.run("1"), "doubleur"))
        assert b.quarantine("doubleur")
        assert not any("42" in c for c in _resultats(ag.run("2"), "doubleur"))

    def test_hors_run(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(FakeLLMProvider([_payload("autre", 3)]), tools_dir=tmp_path, persist=True)
        outil = b.build(ToolBuildRequest(capability="x", tool_name="autre", input_schema=None, permissions=[]))
        assert outil(n=2) == {"r": 6}                    # propre, appelé par l'hôte : inchangé
        b.quarantine("autre")
        with pytest.raises(ToolError, match="quarantine"):
            outil(n=2)

    def test_quarantaine_survit_a_un_catalogue_perdu(self, tmp_path: Path) -> None:
        ag, b = self._agent(tmp_path, [_appel("lire_page", url="u"),
                                       _appel("create_python_tool", capability="x", tool_name="persist"),
                                       LLMResponse(content="ok"), _appel("persist", n=21), LLMResponse(content="fini")])
        ag.run("run1")
        (tmp_path / "catalogue.json").write_text("{corrompu", encoding="utf-8")
        res = _resultats(ag.run("run2"), "persist")
        assert res and not any("42" in c for c in res)

    def test_admission_qui_plante_refuse(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        b = DynamicToolBuilder(FakeLLMProvider([_payload("autre", 3)]), tools_dir=tmp_path, persist=True)
        outil = b.build(ToolBuildRequest(capability="x", tool_name="autre", input_schema=None, permissions=[]))

        def casse(nom: str) -> str | None:
            raise RuntimeError("boum")

        monkeypatch.setattr(outil, "admission", casse)
        with pytest.raises(ToolError, match="admission"):
            outil(n=2)


class TestApprovalConsulteLeCatalogue:
    def _manifeste(self, dossier: Path) -> ToolManifest:
        return ToolManifest(path=dossier / "manifest.json")

    def test_outil_en_quarantaine_non_enregistre(self, tmp_path: Path) -> None:
        _run(tmp_path, [_payload("copieur", 2)], lire_avant=True, nom="copieur")
        ag = Agent(Scripte([]), max_steps=2)
        assert load_tools(ag, tmp_path, self._manifeste(tmp_path)) == [("copieur", "quarantined")]
        assert "copieur" not in [s.name for s in ag.registry.specs()]

    def test_outil_propre_enregistre(self, tmp_path: Path) -> None:
        _run(tmp_path, [_payload("doubleur", 2)], lire_avant=False, nom="doubleur")
        assert load_tools(Agent(Scripte([]), max_steps=2), tmp_path, self._manifeste(tmp_path)) == [("doubleur", "sandbox")]

    def test_approbation_humaine_par_empreinte_l_emporte(self, tmp_path: Path) -> None:
        _run(tmp_path, [_payload("copieur", 2)], lire_avant=True, nom="copieur")
        m = self._manifeste(tmp_path)
        m.approve((tmp_path / "copieur.py").read_text(encoding="utf-8"), name="copieur")
        assert load_tools(Agent(Scripte([]), max_steps=2), tmp_path, m) == [("copieur", "native")]

    def test_fichier_modifie_depuis_la_construction(self, tmp_path: Path) -> None:
        _run(tmp_path, [_payload("doubleur", 2)], lire_avant=False, nom="doubleur")
        chemin = tmp_path / "doubleur.py"
        chemin.write_text(chemin.read_text(encoding="utf-8").replace("* 2", "* 5"), encoding="utf-8")
        assert load_tools(Agent(Scripte([]), max_steps=2), tmp_path, self._manifeste(tmp_path)) == [("doubleur", "quarantined")]

    def test_fichier_sans_entree_dans_un_dossier_gere(self, tmp_path: Path) -> None:
        _run(tmp_path, [_payload("doubleur", 2)], lire_avant=False, nom="doubleur")
        (tmp_path / "intrus.py").write_text(
            (tmp_path / "doubleur.py").read_text(encoding="utf-8").replace("doubleur", "intrus"), encoding="utf-8")
        reg = dict(load_tools(Agent(Scripte([]), max_steps=2), tmp_path, self._manifeste(tmp_path)))
        assert reg == {"doubleur": "sandbox", "intrus": "quarantined"}

    def test_catalogue_illisible_ferme_sans_y_toucher(self, tmp_path: Path) -> None:
        _run(tmp_path, [_payload("doubleur", 2)], lire_avant=False, nom="doubleur")
        (tmp_path / "catalogue.json").write_text("{corrompu", encoding="utf-8")
        assert load_tools(Agent(Scripte([]), max_steps=2), tmp_path, self._manifeste(tmp_path)) == [("doubleur", "quarantined")]
        assert (tmp_path / "catalogue.json").read_text(encoding="utf-8") == "{corrompu"

    def test_dossier_sans_catalogue_inchange(self, tmp_path: Path) -> None:
        _run(tmp_path, [_payload("doubleur", 2)], lire_avant=False, nom="doubleur")
        (tmp_path / "catalogue.json").unlink()
        assert load_tools(Agent(Scripte([]), max_steps=2), tmp_path, self._manifeste(tmp_path)) == [("doubleur", "sandbox")]
