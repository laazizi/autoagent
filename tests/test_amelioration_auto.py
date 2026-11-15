"""La correction AUTOMATIQUE d'un outil créé : `agent.enable_dynamic_tools(builder, auto_improve=...)`.

Jusqu'ici `improve` attendait que l'hôte l'appelle : un outil de la bibliothèque qui échouait en plein run restait
tel quel, et l'agent restait bloqué. Avec `auto_improve`, la boucle lance `improve` dès l'échec, PENDANT le run :

* ``"defects"`` : après un défaut de l'outil (plantage, délai dépassé) ;
* ``"refusals"`` : aussi après un REFUS que personne n'a jugé — `improve(learn_from_refusals=True)`, dont la
  nouvelle lecture reste « non vérifiée ».

Si la nouvelle version est prouvée et installée, elle remplace l'ancienne dans le registre de l'agent, le résultat
de l'appel raté le dit au modèle, et il peut rappeler l'outil. Les bornes : un essai payant par run
(`max_auto_improvements_per_run`), deux par version d'outil, aucun dans un run teinté ; la dépense entre dans le
`token_budget` du run.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from autoagent import Agent, DynamicToolBuilder, ToolBuildRequest
from autoagent.errors import TokenBudgetExceeded, ToolError
from autoagent.schema import LLMRequest, LLMResponse, TokenUsage
from autoagent.trace import TraceEmitter
from tests._outils_generes import BacEnPanne, Principal, Sequence, appel, payload

TEXTE = {"texte": {"type": "string"}}
_DEBUT = r"""
import re

def run(args, context):
    t = args['texte']
    m = re.fullmatch(r'(\d+)h(\d*)', t)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2) or 0)
"""
_REFUS = "    raise ValueError('invalid input: duree illisible')\n"
_MINUTES = r"""    m = re.fullmatch(r'(\d+) min', t)
    if m:
        return int(m.group(1))
"""
V1 = payload(_DEBUT + _REFUS, "duree", TEXTE)                       # REFUSE « 90 min »
V1_PLANTE = payload(_DEBUT + "    return int(t)\n", "duree", TEXTE)  # PLANTE sur « 90 min »
V2 = payload(_DEBUT + _MINUTES + _REFUS, "duree", TEXTE)            # lit « 90 min »
# Refuse « ??? » mais PLANTE sur « 90 min » : un défaut et un refus dans le même outil.
V1_MIXTE = payload(_DEBUT + "    if t == '???':\n" + "    " + _REFUS + "    return int(t)\n", "duree", TEXTE)
V2_CASSE = payload((_DEBUT + _MINUTES + _REFUS).replace("* 60 +", "* 61 +"), "duree", TEXTE)   # change un cas réussi
V2_PLANTE = payload(_DEBUT + _MINUTES + "    return 1 / 0\n", "duree", TEXTE)   # un refus propre devenu plantage


class Constructeur(Sequence):
    """Le modèle constructeur : ses réponses dans l'ordre, 150 jetons déclarés chacune."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        reponse = super().complete(request)
        return LLMResponse(content=reponse.content, model="f",
                           usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150))


def _bibliotheque(dossier: Path, premiere: str = V1, *suivantes: str, **kw: Any) -> DynamicToolBuilder:
    """Une bibliothèque avec l'outil `duree`, utilisé une fois avec succès (« 1h30 », un cas réussi à garder)."""
    kw.setdefault("record_calls", 20)
    b = DynamicToolBuilder(Constructeur([premiere, *suivantes]), tools_dir=dossier, persist=True, **kw)
    outil = b.build(ToolBuildRequest(capability="durée en minutes"))
    assert outil(texte="1h30") == 90
    return b


def _agent(b: DynamicToolBuilder, reponses: list[LLMResponse], *, evts: list | None = None, **kw: Any) -> Agent:
    auto = kw.pop("auto_improve", "refusals")
    plafond = kw.pop("max_auto_improvements_per_run", 1)
    agent = Agent(Principal(reponses), max_steps=8, trace=TraceEmitter(on_event=evts.append) if evts is not None else None,
                  **kw)
    agent.enable_dynamic_tools(b, auto_improve=auto, max_auto_improvements_per_run=plafond)
    return agent


def _resultat_de(resultat: Any, call_id: str) -> dict[str, Any]:
    message = next(m for m in resultat.messages if m.role == "tool" and m.tool_call_id == call_id)
    return json.loads(message.content)


def _appels_au_constructeur(b: DynamicToolBuilder) -> int:
    return len(b.provider.requetes)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------------------------------------

class TestUnOutilQuiRefuseSeCorrigeEnPleinRun:
    def test_l_agent_n_est_plus_bloque(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2)
        agent = _agent(b, [appel("c1", "duree", texte="90 min"), appel("c2", "duree", texte="90 min")])
        resultat = agent.run("Combien de minutes dans 90 min ?")
        premier, second = _resultat_de(resultat, "c1"), _resultat_de(resultat, "c2")
        assert premier["ok"] is False and "invalid input" in premier["error"]
        assert "[autoagent] This tool has just been improved" in premier["error"]       # dit au modèle
        assert second == {"ok": True, "result": 90, "error": None}                     # il rappelle : ça marche
        assert _appels_au_constructeur(b) == 2                                          # la construction + improve
        assert b.catalogue()["duree"]["verified_by"]["extended"] == 1                   # « non vérifiée », écrit

    def test_la_depense_entre_dans_le_run(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2)
        resultat = _agent(b, [appel("c1", "duree", texte="90 min")]).run("x")
        assert resultat.usage is not None and resultat.usage.total_tokens == 150

    def test_et_dans_le_plafond_de_jetons(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2)
        with pytest.raises(TokenBudgetExceeded):
            _agent(b, [appel("c1", "duree", texte="90 min"), appel("c2", "duree", texte="1h")], token_budget=100).run("x")

    def test_la_trace_le_dit(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2)
        evts: list = []
        _agent(b, [appel("c1", "duree", texte="90 min")], evts=evts).run("x")
        (evt,) = [e for e in evts if e.type == "dynamic_tool_auto_improve"]
        assert evt.payload["name"] == "duree" and evt.payload["trigger"] == "refused"
        assert evt.payload["status"] == "improved" and evt.payload["extended"] == 1 and evt.payload["tokens"] == 150
        assert evt.payload["old_version"] != evt.payload["new_version"]

    def test_le_registre_porte_la_nouvelle_version(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2.replace("outil duree", "outil duree, version deux"))
        agent = _agent(b, [appel("c1", "duree", texte="90 min")])
        avant = agent.registry.handler_for("duree")
        agent.run("x")
        apres = agent.registry.handler_for("duree")
        assert apres is not avant and apres.recorder == b._noter_appel                 # surveillée, comme l'ancienne
        assert next(s for s in agent.registry.specs() if s.name == "duree").description == "outil duree, version deux"


class TestCeQuiDeclenche:
    def test_sans_l_option_rien_ne_change(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2)
        agent = Agent(Principal([appel("c1", "duree", texte="90 min"), appel("c2", "duree", texte="90 min")]), max_steps=6)
        agent.enable_dynamic_tools(b)
        resultat = agent.run("x")
        assert _resultat_de(resultat, "c2")["ok"] is False and _appels_au_constructeur(b) == 1
        assert "[autoagent]" not in _resultat_de(resultat, "c1")["error"]

    def test_en_mode_defauts_un_refus_ne_declenche_rien(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2)
        resultat = _agent(b, [appel("c1", "duree", texte="90 min")], auto_improve="defects").run("x")
        assert _appels_au_constructeur(b) == 1 and "[autoagent]" not in _resultat_de(resultat, "c1")["error"]

    def test_en_mode_defauts_un_refus_ne_declenche_rien_meme_avec_un_defaut_au_carnet(self, tmp_path: Path) -> None:
        # Un défaut ancien (appel direct, hors de tout agent) est au carnet : un REFUS en mode « defects » ne doit
        # pas servir d'occasion pour le corriger — ce serait un essai payant que le mode n'a pas demandé.
        b = _bibliotheque(tmp_path, V1_MIXTE, V2)
        (outil,) = b.load_library()
        with pytest.raises(ToolError):
            outil(texte="90 min")                                   # plante : un défaut au carnet
        resultat = _agent(b, [appel("c1", "duree", texte="???")], auto_improve="defects").run("x")
        assert "invalid input" in _resultat_de(resultat, "c1")["error"] and _appels_au_constructeur(b) == 1

    def test_en_mode_defauts_un_plantage_declenche(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1_PLANTE, V2)
        resultat = _agent(b, [appel("c1", "duree", texte="90 min"), appel("c2", "duree", texte="90 min")],
                          auto_improve="defects").run("x")
        assert _resultat_de(resultat, "c2")["result"] == 90 and _appels_au_constructeur(b) == 2

    def test_un_refus_du_schema_compte_comme_un_refus(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2.replace('"type": "string"', '"type": ["string", "integer"]')
                          .replace("t = args['texte']", "t = args['texte'] if isinstance(args['texte'], str) else '%dh' % (args['texte'] // 60)"))
        resultat = _agent(b, [appel("c1", "duree", texte=120), appel("c2", "duree", texte=120)]).run("x")
        assert _resultat_de(resultat, "c2") == {"ok": True, "result": 120, "error": None}

    def test_une_panne_du_bac_ne_declenche_rien(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2)
        b.provider.contenus.append(V2)  # type: ignore[attr-defined]
        agent = _agent(b, [appel("c1", "duree", texte="1h")], auto_improve="defects")
        agent.registry.handler_for("duree").sandbox = BacEnPanne()
        agent.run("x")
        assert _appels_au_constructeur(b) == 1

    def test_un_outil_de_l_hote_n_est_jamais_remplace(self, tmp_path: Path) -> None:
        # L'hôte a SON outil « duree » : celui de la bibliothèque n'est pas chargé. Même si le dernier appel noté sur
        # ce thread est un refus de l'outil de la bibliothèque du même nom, l'échec de l'outil de l'hôte ne
        # déclenche rien — et surtout, l'outil de l'hôte n'est jamais remplacé par un outil écrit par le modèle.
        b = _bibliotheque(tmp_path, V1, V2)
        agent = Agent(Principal([appel("c1", "duree", texte="90 min")]), max_steps=4)

        @agent.tool
        def duree(texte: str) -> int:
            """La durée, par l'hôte."""
            raise ValueError("invalid input: non")

        agent.enable_dynamic_tools(b, auto_improve="refusals")
        (outil,) = b.load_library()
        with pytest.raises(ToolError):
            outil(texte="90 min")                                   # le dernier appel noté : un refus de la bibliothèque
        agent.run("x")
        assert _appels_au_constructeur(b) == 1 and agent.registry.handler_for("duree") is duree

    def test_un_outil_de_l_hote_ne_declenche_rien(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2)
        agent = _agent(b, [appel("c1", "convertir", texte="90 min")])

        @agent.tool
        def convertir(texte: str) -> int:
            """Convertit."""
            raise ValueError("invalid input: non")

        agent.run("x")
        assert _appels_au_constructeur(b) == 1

    def test_un_appel_reussi_ne_declenche_rien(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2)
        _agent(b, [appel("c1", "duree", texte="2h")]).run("x")
        assert _appels_au_constructeur(b) == 1

    def test_un_outil_cree_pendant_le_run_aussi(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(Constructeur([V1, V2]), tools_dir=tmp_path, persist=True, record_calls=20)
        agent = _agent(b, [appel("c0", "create_python_tool", capability="durée en minutes", tool_name="duree"),
                           appel("c1", "duree", texte="1h"), appel("c2", "duree", texte="90 min"),
                           appel("c3", "duree", texte="90 min")])
        resultat = agent.run("x")
        assert _resultat_de(resultat, "c3")["result"] == 90


class TestLesBornes:
    def test_rien_dans_un_run_teinte(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2)
        evts: list = []
        agent = _agent(b, [appel("c0", "lire_page", url="https://exemple.fr"), appel("c1", "duree", texte="90 min")],
                       evts=evts)

        @agent.tool(untrusted=True)
        def lire_page(url: str) -> str:
            """Lit une page web."""
            return "contenu externe"

        resultat = agent.run("x")
        assert _appels_au_constructeur(b) == 1 and "[autoagent]" not in _resultat_de(resultat, "c1")["error"]
        assert not [e for e in evts if e.type == "dynamic_tool_auto_improve"]

    def test_rien_dans_un_run_teinte_meme_avec_un_cas_propre_au_carnet(self, tmp_path: Path) -> None:
        # Un refus PROPRE (appel direct, hors de tout run) est au carnet : il serait montrable. Le run teinté ne
        # doit tout de même rien déclencher — c'est la teinte du RUN qui l'interdit, pas seulement celle du cas.
        b = _bibliotheque(tmp_path, V1, V2)
        (outil,) = b.load_library()
        with pytest.raises(ToolError):
            outil(texte="45 min")
        agent = _agent(b, [appel("c0", "lire_page", url="https://exemple.fr"), appel("c1", "duree", texte="90 min")])

        @agent.tool(untrusted=True)
        def lire_page(url: str) -> str:
            """Lit une page web."""
            return "contenu externe"

        agent.run("x")
        assert _appels_au_constructeur(b) == 1

    def test_un_essai_payant_par_run(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2_CASSE, V2)
        agent = _agent(b, [appel("c1", "duree", texte="90 min"), appel("c2", "duree", texte="90 min"),
                           LLMResponse(content="fini"),
                           appel("c3", "duree", texte="90 min"), appel("c4", "duree", texte="90 min")])
        resultat = agent.run("x")
        assert _appels_au_constructeur(b) == 2                     # le second échec ne relance pas : plafond du run
        assert _resultat_de(resultat, "c2")["ok"] is False
        resultat = agent.run("encore")                             # le MÊME agent, un run neuf : le compte repart
        assert _resultat_de(resultat, "c4")["result"] == 90 and _appels_au_constructeur(b) == 3

    def test_le_plafond_du_run_se_regle(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2_CASSE, V2)
        agent = _agent(b, [appel("c1", "duree", texte="90 min"), appel("c2", "duree", texte="90 min"),
                           appel("c3", "duree", texte="90 min")], max_auto_improvements_per_run=2)
        assert _resultat_de(agent.run("x"), "c3")["result"] == 90 and _appels_au_constructeur(b) == 3

    def test_un_essai_gratuit_ne_compte_pas(self, tmp_path: Path) -> None:
        # Après un essai sans gain, l'entrée examinée n'est plus reproposée : `improve` ne coûte rien, et ne
        # consomme pas le plafond du run — un autre outil peut encore être corrigé.
        b = _bibliotheque(tmp_path, V1, V1.replace("duree illisible", "duree vraiment illisible"))
        _agent(b, [appel("c1", "duree", texte="???")]).run("x")
        assert _appels_au_constructeur(b) == 2
        b.provider.contenus.append(V2)  # type: ignore[attr-defined]
        resultat = _agent(b, [appel("c2", "duree", texte="???"), appel("c3", "duree", texte="90 min"),
                              appel("c4", "duree", texte="90 min")]).run("x")
        assert _resultat_de(resultat, "c4")["result"] == 90 and _appels_au_constructeur(b) == 3

    def test_deux_essais_payants_par_version(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, V2_CASSE, V2_CASSE, V2)
        for i in range(3):
            _agent(b, [appel(f"c{i}", "duree", texte="90 min")]).run("x")
        assert _appels_au_constructeur(b) == 3                     # 1 construction + 2 essais ; le 3e run n'essaie plus
        assert b.catalogue()["duree"]["auto_improve"] == {"version": b.catalogue()["duree"]["sha256"][:12], "attempts": 2}

    def test_le_plafond_d_une_version_ne_bride_pas_une_autre(self, tmp_path: Path) -> None:
        # B (installée par improve) épuise ses deux essais ; un retour arrière remet A, dont l'entrée du catalogue
        # garde la trace de B : elle ne doit pas brider A.
        b = _bibliotheque(tmp_path, V1, V2, V2_PLANTE, V2_PLANTE, V2)
        _agent(b, [appel("c1", "duree", texte="90 min")]).run("x")                  # A → B
        for i in (2, 3):
            _agent(b, [appel(f"c{i}", "duree", texte="2:30")]).run("x")             # B : deux essais refusés
        assert b.catalogue()["duree"]["auto_improve"]["attempts"] == 2 and _appels_au_constructeur(b) == 4
        assert b.rollback("duree") is True                                          # A revient
        _agent(b, [appel("c4", "duree", texte="2:30")]).run("x")
        assert _appels_au_constructeur(b) == 5                                      # A a droit à son essai

    def test_la_decision_ne_lit_que_l_appel_de_cet_outil(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1, payload(_DEBUT + _REFUS, "autre", TEXTE))
        b.build(ToolBuildRequest(capability="un autre outil", tool_name="autre"))     # un AUTRE outil de la bibliothèque
        outil = next(o for o in b.load_library() if o.spec.name == "duree")
        with pytest.raises(ToolError):
            outil(texte="90 min")                                   # le dernier appel noté : « duree » a refusé
        assert b._ameliorer_apres_echec("autre", "refusals") is None                # rien n'est tenté pour « autre »
        assert b._ameliorer_apres_echec("duree", "refusals") is None                # déjà lue : consommée
        assert _appels_au_constructeur(b) == 2                                      # les deux constructions, rien d'autre

    def test_une_nouvelle_version_repart_de_zero(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path, V1_PLANTE, V2)
        _agent(b, [appel("c1", "duree", texte="90 min")], auto_improve="defects").run("x")
        assert "auto_improve" not in b.catalogue()["duree"]

    def test_deux_appels_en_parallele_un_seul_essai(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        b = _bibliotheque(tmp_path, V1, V2, V2)
        vrai, barriere = b.improve, threading.Barrier(2, timeout=5)

        def lent(*a: Any, **k: Any) -> Any:
            try:
                barriere.wait()
            except threading.BrokenBarrierError:
                pass
            return vrai(*a, **k)

        monkeypatch.setattr(b, "improve", lent)
        agent = _agent(b, [LLMResponse(tool_calls=[*appel("c1", "duree", texte="90 min").tool_calls,
                                                     *appel("c2", "duree", texte="45 min").tool_calls])],
                       parallel_tool_calls=True, max_auto_improvements_per_run=2)
        agent.run("x")
        assert _appels_au_constructeur(b) == 2                     # un seul `improve` à la fois sur un même outil


class TestLaConfiguration:
    @pytest.mark.parametrize("valeur", [True, "oui", "defaut", 1, 0, None, ""])
    def test_une_valeur_inconnue_est_refusee(self, tmp_path: Path, valeur: Any) -> None:
        b = _bibliotheque(tmp_path)
        with pytest.raises(ValueError, match="auto_improve"):
            Agent(Principal([])).enable_dynamic_tools(b, auto_improve=valeur)

    def test_il_faut_le_carnet(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(Constructeur([]), tools_dir=tmp_path, persist=True)
        with pytest.raises(ValueError, match="record_calls"):
            Agent(Principal([])).enable_dynamic_tools(b, auto_improve="defects")
        agent = Agent(Principal([]))
        agent.enable_dynamic_tools(b)                               # sans l'option : rien n'est exigé
        assert agent.dynamic_builder is b

    @pytest.mark.parametrize("valeur", [0, -1, True, 1.5, "2"])
    def test_le_plafond_est_un_entier_positif(self, tmp_path: Path, valeur: Any) -> None:
        b = _bibliotheque(tmp_path)
        with pytest.raises(ValueError, match="max_auto_improvements_per_run"):
            Agent(Principal([])).enable_dynamic_tools(b, auto_improve="defects", max_auto_improvements_per_run=valeur)

    def test_une_configuration_refusee_ne_touche_a_rien(self, tmp_path: Path) -> None:
        b = _bibliotheque(tmp_path)
        agent = Agent(Principal([]))
        with pytest.raises(ValueError):
            agent.enable_dynamic_tools(b, auto_improve="oui")
        assert agent.dynamic_builder is None and "create_python_tool" not in agent.registry
