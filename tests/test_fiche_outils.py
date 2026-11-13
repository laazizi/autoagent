"""La fiche de qualité des outils créés par le modèle : MESURER, après avoir surveillé.

Le carnet (`record_calls`) dit ce qui s'est passé. Il ne dit pas ce que vaut l'outil :

* il ne voit pas un résultat FAUX qui ne plante pas, ni une mauvaise entrée que l'outil a eu raison de refuser
  (constaté sur un vrai modèle le 13 nov. 2026 : trois dates mal écrites ont fait retirer un outil juste) ;
* il ne voit pas un appel refusé par le schéma de l'outil avant même qu'il tourne ;
* rien ne dit comment l'outil a été vérifié avant d'être gardé — il peut ne l'avoir jamais été.

Ce qui est fixé ici :

* `verified_by` : l'origine des tests de chaque outil est écrite au catalogue (`none`, `self_tests` écrits par le
  modèle, `examples` de l'hôte dont une part cachée) ;
* les appels refusés par le schéma d'entrée sont notés (`rejected`), à part ;
* `report(nom, id, correct=…)` : le VERDICT DE L'HÔTE sur un appel — c'est lui qui juge, pas le modèle ;
* `health(nom)` : la fiche — issues, taux de défaut JUGÉ avec son intervalle, durée, erreurs regroupées, versions ;
* `reinstate(nom)` : la bibliothèque retire toute seule, seul l'hôte remet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autoagent import Agent, DynamicToolBuilder, ToolBuildRequest
from autoagent.compare import wilson_interval
from autoagent.dynamic import ToolHealth
from autoagent.errors import ToolError
from autoagent.synthesis import Example, synthesize_tool
from tests._outils_generes import (
    CODE_DOUBLER,
    DOUBLER,
    LECTEUR,
    BacEnPanne,
    Principal,
    Sequence,
    appel,
    construire,
    payload,
)

ENTIER = {"n": {"type": "integer"}}


def _echoue(outil: Any, **args: Any) -> None:
    with pytest.raises(ToolError):
        outil(**args)


def _dernier(b: DynamicToolBuilder, nom: str = "doubler") -> dict[str, Any]:
    return b.calls(nom)[-1]


# ---------------------------------------------------------------------------------------------------------

class TestLOrigineDesTestsEstEcrite:
    def test_un_outil_sans_aucun_test(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path)
        assert b.catalogue()["doubler"]["verified_by"] == {"kind": "none"}
        assert outil.verified_by == {"kind": "none"}

    def test_des_auto_tests_ecrits_par_le_modele(self, tmp_path: Path) -> None:
        tests = [{"args": {"n": i}, "expect_equals": {"double": 2 * i}} for i in (1, 2)]
        b, _ = construire(tmp_path, payload(CODE_DOUBLER, "doubler", ENTIER, tests))
        assert b.catalogue()["doubler"]["verified_by"] == {"kind": "self_tests", "cases": 2}

    def test_seuls_les_auto_tests_executes_sont_comptes(self, tmp_path: Path) -> None:
        tests = [{"args": {"n": i}, "expect_equals": {"double": 2 * i}} for i in range(8)]
        b, _ = construire(tmp_path, payload(CODE_DOUBLER, "doubler", ENTIER, tests))
        assert b.catalogue()["doubler"]["verified_by"] == {"kind": "self_tests", "cases": 5}

    def test_des_exemples_de_l_hote_dont_une_part_cachee(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(Sequence([DOUBLER]), tools_dir=tmp_path, persist=True, record_calls=20)
        exemples = [Example({"n": i}, {"double": 2 * i}) for i in range(1, 11)]
        resultat = synthesize_tool(b, "doubler un entier", exemples, holdout=0.4)
        assert resultat.tool is not None
        attendu = {"kind": "examples", "shown": resultat.shown, "hidden": resultat.holdout}
        assert resultat.holdout > 0 and b.catalogue()["doubler"]["verified_by"] == attendu
        assert resultat.tool.verified_by == attendu

    def test_les_passes_de_verification_ne_sont_pas_des_appels(self, tmp_path: Path) -> None:
        """Comme les auto-tests : faire passer des exemples n'est pas se servir de l'outil."""
        b = DynamicToolBuilder(Sequence([DOUBLER]), tools_dir=tmp_path, persist=True, record_calls=20)
        resultat = synthesize_tool(b, "doubler", [Example({"n": i}, {"double": 2 * i}) for i in range(1, 6)])
        assert b.catalogue()["doubler"]["calls"] == 0 and b.calls("doubler") == []
        assert resultat.tool is not None and resultat.tool(n=4) == {"double": 8}
        assert b.catalogue()["doubler"]["calls"] == 1                 # l'usage, lui, est compté

    def test_l_origine_survit_au_rechargement(self, tmp_path: Path) -> None:
        tests = [{"args": {"n": 1}, "expect_equals": {"double": 2}}]
        construire(tmp_path, payload(CODE_DOUBLER, "doubler", ENTIER, tests))
        (recharge,) = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True).load_library()
        assert recharge.verified_by == {"kind": "self_tests", "cases": 1}

    def test_un_catalogue_d_avant_ne_sait_pas(self, tmp_path: Path) -> None:
        b, _ = construire(tmp_path)
        chemin = tmp_path / "catalogue.json"
        donnees = json.loads(chemin.read_text(encoding="utf-8"))
        del donnees["tools"]["doubler"]["verified_by"]
        chemin.write_text(json.dumps(donnees), encoding="utf-8")
        assert b.health("doubler").verified_by == {"kind": "unknown"}


class TestLesAppelsRefusesParLeSchema:
    def _agent(self, dossier: Path, reponses: list[Any], **kw: Any) -> tuple[DynamicToolBuilder, Any]:
        construire(dossier)
        b = DynamicToolBuilder(Sequence([]), tools_dir=dossier, persist=True, record_calls=20, **kw)
        agent = Agent(Principal(reponses), max_steps=6)
        agent.enable_dynamic_tools(b)
        return b, agent.run("go")

    def test_un_argument_du_mauvais_type_est_note_a_part(self, tmp_path: Path) -> None:
        b, resultat = self._agent(tmp_path, [appel("c1", "doubler", n="douze")])
        (note,) = b.calls("doubler")
        assert note["ok"] is False and note["kind"] == "rejected" and note["args"] == {"n": "douze"}
        assert "integer" in note["error"]
        e = b.catalogue()["doubler"]
        assert (e["calls"], e["errors"], e["consecutive_errors"], e["rejected_calls"]) == (0, 0, 0, 1)
        assert any("integer" in (m.content or "") for m in resultat.messages if m.role == "tool")   # le modèle le voit aussi

    def test_des_refus_du_schema_ne_retirent_pas_l_outil(self, tmp_path: Path) -> None:
        reponses = [appel(f"c{i}", "doubler", n="x") for i in range(4)]
        b, _ = self._agent(tmp_path, reponses, retire_after_errors=2)
        e = b.catalogue()["doubler"]
        assert e["retired"] is False and e["rejected_calls"] == 4 and e["unattributed_failures"] == 0

    def test_un_refus_du_schema_ne_remet_pas_la_serie_a_zero(self, tmp_path: Path) -> None:
        b, _ = self._agent(tmp_path, [appel("c1", "doubler", n=-1), appel("c2", "doubler", n="x"),
                                      appel("c3", "doubler", n=-2)], retire_after_errors=2)
        assert b.catalogue()["doubler"]["retired"] is True

    def test_un_outil_de_l_hote_n_est_pas_concerne(self, tmp_path: Path) -> None:
        agent = Agent(Principal([appel("c1", "ajouter", a="x")]), max_steps=3)

        @agent.tool
        def ajouter(a: int) -> int:
            """Ajoute un."""
            return a + 1

        resultat = agent.run("go")
        assert any("integer" in (m.content or "") for m in resultat.messages if m.role == "tool")

    def test_le_crochet_du_registre_est_ouvert_a_tout_outil(self, tmp_path: Path) -> None:
        """Le registre prévient tout outil qui porte `__autoagent_rejected__` — et un crochet qui plante ne
        change rien au refus que le modèle reçoit."""
        vus: list[tuple[Any, str]] = []

        def ajouter(a: int) -> int:
            """Ajoute un."""
            return a + 1

        def soustraire(a: int) -> int:
            """Retire un."""
            return a - 1

        ajouter.__autoagent_rejected__ = lambda args, message: vus.append((args, message))  # type: ignore[attr-defined]
        soustraire.__autoagent_rejected__ = lambda args, message: 1 / 0  # type: ignore[attr-defined]
        agent = Agent(Principal([appel("c1", "ajouter", a="x"), appel("c2", "soustraire", a="y"),
                                 appel("c3", "ajouter", a=1)]), max_steps=5)
        agent.add_tool(ajouter)
        agent.add_tool(soustraire)
        resultat = agent.run("go")
        outils = [m.content or "" for m in resultat.messages if m.role == "tool"]
        assert len(vus) == 1 and vus[0][0] == {"a": "x"} and "integer" in vus[0][1]      # un appel valide ne prévient pas
        assert "integer" in outils[0] and "integer" in outils[1] and "2" in outils[2]

    def test_un_crochet_qui_plante_ne_change_pas_le_refus(self, tmp_path: Path) -> None:
        construire(tmp_path)
        b = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True, record_calls=20)
        agent = Agent(Principal([appel("c1", "doubler", n="x")]), max_steps=3)
        agent.enable_dynamic_tools(b)
        agent.registry.handler_for("doubler").recorder = lambda a: 1 / 0
        resultat = agent.run("go")
        assert any("integer" in (m.content or "") for m in resultat.messages if m.role == "tool")


class TestLeVerdictDeLHote:
    def test_un_resultat_juge_faux(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path)
        outil(n=2)
        id_ = _dernier(b)["id"]
        assert b.report("doubler", id_, correct=False, reason="attendu le triple", expected={"triple": 6}) is True
        verdict = _dernier(b)["verdict"]
        assert verdict["correct"] is False and verdict["reason"] == "attendu le triple"
        assert verdict["expected"] == {"triple": 6} and verdict["ts"].endswith("Z")

    def test_un_refus_juge_legitime(self, tmp_path: Path) -> None:
        """Le cas vu sur un vrai modèle : l'outil a eu RAISON de refuser une entrée invalide."""
        b, outil = construire(tmp_path)
        _echoue(outil, n=-1)
        assert b.report("doubler", _dernier(b)["id"], correct=True, reason="entrée invalide") is True
        assert _dernier(b)["verdict"]["correct"] is True and "expected" not in _dernier(b)["verdict"]

    def test_un_appel_ou_un_outil_inconnu(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path)
        outil(n=1)
        assert b.report("doubler", "pas-un-id", correct=False) is False
        assert b.report("absent", _dernier(b)["id"], correct=False) is False
        assert "verdict" not in _dernier(b)

    def test_le_dernier_verdict_remplace_le_precedent(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path)
        outil(n=1)
        id_ = _dernier(b)["id"]
        b.report("doubler", id_, correct=False, reason="faux")
        b.report("doubler", id_, correct=True)
        assert _dernier(b)["verdict"]["correct"] is True and "reason" not in _dernier(b)["verdict"]
        assert len(b.calls("doubler")) == 1                           # une entrée par appel, pas par verdict

    @pytest.mark.parametrize("valeur", [1, "oui", None])
    def test_correct_est_un_booleen(self, tmp_path: Path, valeur: Any) -> None:
        b, outil = construire(tmp_path)
        outil(n=1)
        with pytest.raises(TypeError, match="correct"):
            b.report("doubler", _dernier(b)["id"], correct=valeur)

    def test_le_verdict_ne_change_ni_les_compteurs_ni_le_statut(self, tmp_path: Path) -> None:
        """La bibliothèque retire toute seule ; un verdict ne remet rien : c'est `reinstate`."""
        b, outil = construire(tmp_path, retire_after_errors=2)
        for n in (-1, -2):
            _echoue(outil, n=n)
        avant = b.catalogue()["doubler"]
        for note in b.calls("doubler"):
            b.report("doubler", note["id"], correct=True)
        apres = b.catalogue()["doubler"]
        assert apres["retired"] is True and (apres["errors"], apres["consecutive_errors"]) == (avant["errors"], 2)

    def test_la_raison_et_l_attendu_passent_par_redact_et_sont_bornes(self, tmp_path: Path) -> None:
        masquer = lambda v: json.loads(json.dumps(v, ensure_ascii=False).replace("Dupont", "<nom>"))  # noqa: E731
        b, outil = construire(tmp_path, redact=masquer, max_record_chars=100)
        outil(n=1)
        id_ = _dernier(b)["id"]
        b.report("doubler", id_, correct=False, reason="Dupont attendait " + "z" * 500, expected={"pour": "Dupont"})
        verdict = _dernier(b)["verdict"]
        assert verdict["reason"].startswith("<nom> attendait") and len(verdict["reason"]) <= 300
        assert verdict["expected"] == {"pour": "<nom>"}
        b.report("doubler", id_, correct=False, expected={"gros": "y" * 500})
        verdict = _dernier(b)["verdict"]
        assert "expected" not in verdict and len(verdict["expected_sha256"]) == 64
        assert "Dupont" not in (tmp_path / "doubler.calls.jsonl").read_text(encoding="utf-8")

    def test_un_redact_qui_plante_n_ecrit_que_le_verdict(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path)
        outil(n=1)
        b.redact = lambda v: 1 / 0
        assert b.report("doubler", _dernier(b)["id"], correct=False, reason="Dupont", expected={"pour": "Dupont"}) is True
        verdict = _dernier(b)["verdict"]
        assert verdict["correct"] is False and not {"reason", "expected", "expected_sha256"} & set(verdict)
        assert "Dupont" not in (tmp_path / "doubler.calls.jsonl").read_text(encoding="utf-8")

    def test_le_carnet_reste_borne_apres_des_verdicts(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, record_calls=3)
        for n in range(8):
            outil(n=n)
            b.report("doubler", _dernier(b)["id"], correct=True)
            assert len((tmp_path / "doubler.calls.jsonl").read_text(encoding="utf-8").splitlines()) <= 6
        assert [a["args"]["n"] for a in b.calls("doubler")] == [5, 6, 7]
        assert all(a["verdict"]["correct"] for a in b.calls("doubler"))

    def test_sans_carnet_pas_de_verdict(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, record_calls=0)
        outil(n=1)
        assert b.report("doubler", "x", correct=True) is False

    def test_un_nom_qui_sort_du_dossier_n_ecrit_rien(self, tmp_path: Path) -> None:
        """Le nom vient de l'appelant : seul un outil du catalogue a un carnet qu'on peut annoter."""
        b, outil = construire(tmp_path / "outils")
        outil(n=1)
        dehors = tmp_path / "secret.calls.jsonl"
        dehors.write_text('{"id": "x", "ok": true}\n', encoding="utf-8")
        assert b.report("../secret", "x", correct=False) is False
        assert "verdict" not in dehors.read_text(encoding="utf-8")


class TestLaFiche:
    def test_un_outil_inconnu(self, tmp_path: Path) -> None:
        b, _ = construire(tmp_path)
        with pytest.raises(KeyError):
            b.health("absent")

    def test_trois_reussites_un_plantage(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path)
        for n in (1, 2, 3):
            outil(n=n)
        _echoue(outil, n=-1)
        fiche = b.health("doubler")
        assert isinstance(fiche, ToolHealth)
        assert (fiche.name, fiche.status, fiche.basis) == ("doubler", "active", "call log")
        assert fiche.version == b.catalogue()["doubler"]["sha256"][:12]
        assert fiche.outcomes == {"ok": 3, "exception": 1} and fiche.window == 4
        assert (fiche.judged, fiche.defects, fiche.defect_rate) == (4, 1, 0.25)
        assert fiche.defect_interval == wilson_interval(1, 4)
        assert fiche.median_seconds is not None and fiche.median_seconds >= 0
        assert (fiche.calls, fiche.errors, fiche.unattributed_failures, fiche.rejected_calls) == (4, 1, 0, 0)

    def test_ce_qui_ne_vient_pas_de_l_outil_n_entre_pas_dans_le_taux(self, tmp_path: Path) -> None:
        def lire(cle: str) -> str:
            raise KeyError(cle)

        b, outil = construire(tmp_path, LECTEUR, host_functions={"lire": lire})
        for _ in range(2):
            _echoue(outil, cle="a")
        bon = outil.sandbox
        outil.sandbox = BacEnPanne()
        _echoue(outil, cle="a")
        outil.sandbox = bon
        fiche = b.health("lecteur")
        assert fiche.outcomes == {"host": 2, "sandbox": 1}
        assert (fiche.judged, fiche.defects, fiche.defect_rate, fiche.defect_interval) == (0, 0, None, None)

    def test_le_verdict_de_l_hote_prime(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path)
        outil(n=1)
        outil(n=2)
        _echoue(outil, n=-1)
        ok1, ok2, plante = b.calls("doubler")
        assert b.health("doubler").defects == 1
        b.report("doubler", plante["id"], correct=True)                # le refus était légitime
        assert (b.health("doubler").defects, b.health("doubler").defect_rate) == (0, 0.0)
        b.report("doubler", ok1["id"], correct=False)                  # un résultat faux qui n'avait pas planté
        b.report("doubler", ok2["id"], correct=True)                   # un résultat confirmé juste
        fiche = b.health("doubler")
        assert (fiche.judged, fiche.defects) == (3, 1) and fiche.verdicts == {"correct": 2, "wrong": 1}

    def test_un_refus_du_schema_juge_a_tort_devient_un_defaut(self, tmp_path: Path) -> None:
        """Un schéma trop strict qui refuse une entrée valide : seul l'hôte peut le dire."""
        construire(tmp_path)
        b = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True, record_calls=20)
        agent = Agent(Principal([appel("c1", "doubler", n="12")]), max_steps=3)
        agent.enable_dynamic_tools(b)
        agent.run("go")
        assert (b.health("doubler").judged, b.health("doubler").defects) == (0, 0)
        b.report("doubler", _dernier(b)["id"], correct=False, reason="« 12 » devait être accepté")
        fiche = b.health("doubler")
        assert (fiche.judged, fiche.defects, fiche.defect_rate) == (1, 1, 1.0)

    def test_les_erreurs_sont_regroupees(self, tmp_path: Path) -> None:
        code = ("def run(args, context):\n"
                "    if args['n'] < 0:\n"
                "        raise ValueError('negatif : %d' % args['n'])\n"
                "    if args['n'] > 100:\n"
                "        raise OverflowError('trop grand')\n"
                "    return args['n']\n")
        b, outil = construire(tmp_path, payload(code, "borne", ENTIER), retire_after_errors=0)
        for n in (500, -1, -22, -333):                               # la plus fréquente n'est pas la première vue
            _echoue(outil, n=n)
        premiere, seconde = b.health("borne").errors_seen
        assert premiere["count"] == 3 and "negatif" in premiere["error"] and premiere["example_args"] == {"n": -333}
        assert seconde["count"] == 1 and "trop grand" in seconde["error"] and seconde["example_args"] == {"n": 500}

    def test_un_echec_excuse_n_est_plus_dans_les_erreurs(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path)
        _echoue(outil, n=-1)
        b.report("doubler", _dernier(b)["id"], correct=True)
        assert b.health("doubler").errors_seen == []

    def test_par_version(self, tmp_path: Path) -> None:
        autre = DOUBLER.replace("* 2", "* 2 + 0")
        b = DynamicToolBuilder(Sequence([DOUBLER, autre]), tools_dir=tmp_path, persist=True, record_calls=20)
        ancien = b.build(ToolBuildRequest(capability="x"))
        ancien(n=1)
        _echoue(ancien, n=-1)
        nouveau = b.build(ToolBuildRequest(capability="x"))
        nouveau(n=2)
        fiche = b.health("doubler")
        v_ancienne, v_nouvelle = (a["version"] for a in (b.calls("doubler")[0], b.calls("doubler")[-1]))
        assert fiche.version == v_nouvelle != v_ancienne
        assert fiche.versions == {v_ancienne: {"judged": 2, "defects": 1}, v_nouvelle: {"judged": 1, "defects": 0}}

    def test_le_statut(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, retire_after_errors=1)
        assert b.health("doubler").status == "active"
        _echoue(outil, n=-1)
        assert b.health("doubler").status == "retired"
        b.quarantine("doubler")
        assert b.health("doubler").status == "quarantined"

    def test_sans_carnet_la_fiche_vient_du_catalogue(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, record_calls=0)
        outil(n=1)
        _echoue(outil, n=-1)
        fiche = b.health("doubler")
        assert (fiche.basis, fiche.window, fiche.outcomes) == ("catalogue", 0, {})
        assert (fiche.judged, fiche.defects, fiche.defect_rate) == (2, 1, 0.5)
        assert fiche.defect_interval == wilson_interval(1, 2) and fiche.median_seconds is None

    def test_le_taux_ne_porte_que_sur_les_appels_juges(self, tmp_path: Path) -> None:
        """Avec ou sans carnet : un échec qui ne vient pas de l'outil ne dilue ni n'aggrave son taux."""
        for carnet in (20, 0):
            dossier = tmp_path / str(carnet)
            b, outil = construire(dossier, record_calls=carnet, retire_after_errors=0)
            outil(n=1)
            _echoue(outil, n=-1)
            bon, outil.sandbox = outil.sandbox, BacEnPanne()
            for _ in range(2):
                _echoue(outil, n=1)
            outil.sandbox = bon
            fiche = b.health("doubler")
            assert (fiche.judged, fiche.defects, fiche.defect_rate) == (2, 1, 0.5), carnet
            assert fiche.defect_interval == wilson_interval(1, 2)

    def test_un_outil_jamais_appele(self, tmp_path: Path) -> None:
        b, _ = construire(tmp_path)
        fiche = b.health("doubler")
        assert (fiche.judged, fiche.defect_rate, fiche.defect_interval, fiche.median_seconds) == (0, None, None, None)

    def test_toutes_les_fiches(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(Sequence([DOUBLER, LECTEUR]), tools_dir=tmp_path, persist=True, record_calls=20)
        b.build(ToolBuildRequest(capability="x"))
        b.build(ToolBuildRequest(capability="y"))
        fiches = b.health_all()
        assert sorted(fiches) == ["doubler", "lecteur"] and all(isinstance(f, ToolHealth) for f in fiches.values())

    def test_le_resume_se_lit_et_la_fiche_se_serialise(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path)
        outil(n=1)
        _echoue(outil, n=-1)
        fiche = b.health("doubler")
        resume = fiche.summary()
        assert "doubler" in resume and "jamais vérifié" in resume and "1 défaut sur 2" in resume and "negatif" in resume
        donnees = json.loads(json.dumps(fiche.to_dict()))
        assert donnees["name"] == "doubler" and donnees["defect_rate"] == 0.5 and donnees["verified_by"] == {"kind": "none"}

    def test_le_resume_dit_d_ou_viennent_les_tests(self, tmp_path: Path) -> None:
        tests = [{"args": {"n": 1}, "expect_equals": {"double": 2}}]
        b, _ = construire(tmp_path, payload(CODE_DOUBLER, "doubler", ENTIER, tests))
        assert "1 auto-test écrit par le modèle" in b.health("doubler").summary()


class TestSeulLHoteRemet:
    def test_un_outil_retire_a_tort_est_remis(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, retire_after_errors=2)
        for n in (-1, -2):
            _echoue(outil, n=n)
        assert b.load_library() == []
        assert b.reinstate("doubler") is True
        e = b.catalogue()["doubler"]
        assert e["retired"] is False and e["consecutive_errors"] == 0 and "retired_reason" not in e
        (recharge,) = b.load_library()
        assert recharge(n=3) == {"double": 6}

    def test_un_outil_inconnu(self, tmp_path: Path) -> None:
        b, _ = construire(tmp_path)
        assert b.reinstate("absent") is False

    def test_remettre_ne_leve_pas_une_quarantaine(self, tmp_path: Path) -> None:
        """La quarantaine est une décision de SÉCURITÉ : elle se lève par `release`, empreinte en main."""
        b, _ = construire(tmp_path)
        b.retire("doubler")
        b.quarantine("doubler")
        assert b.reinstate("doubler") is True
        assert b.catalogue()["doubler"]["quarantined"] is True and b.load_library() == []
