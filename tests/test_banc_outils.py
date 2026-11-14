"""Le banc « outils qui cassent » (`autoagent.toolbench`) : ce que vaut `improve`, mesuré.

Un outil créé par le modèle marche sur ses premières entrées, puis d'autres écritures arrivent et il échoue. Le
banc compare quatre réactions À PARTIR DU MÊME premier outil — ne rien faire, le refaire à l'aveugle, `improve`
quand l'hôte dit l'attendu, `improve` quand l'hôte ne dit rien — et les juge PAR DU CODE sur des entrées que rien
n'a jamais vues.

Ces tests sont le niveau 1 : un constructeur scripté (`ScriptedToolProvider`), hors ligne. Ils vérifient le CODE
du banc — jamais un modèle. Les trois sortes de casse sont couvertes : le premier outil PLANTE, rend une valeur
FAUSSE, ou REFUSE une entrée valide.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from autoagent.dynamic import DynamicToolBuilder
from autoagent.sandbox import SubprocessSandbox
from autoagent.schema import LLMRequest, LLMResponse
from autoagent.toolbench import (
    ARMS,
    TOOL_TASKS,
    ScriptedToolProvider,
    ToolBenchReport,
    ToolBenchRow,
    ToolTask,
    positive_control_tools,
    run_tool_bench,
    same_result,
)

PAR_NOM = {t.name: t for t in TOOL_TASKS}


def _petite(nom: str, durs: set[str], suite: set[str]) -> ToolTask:
    """Une tâche fournie, réduite : deux entrées faciles, les dures et les suivantes nommées (des tests rapides)."""
    t = PAR_NOM[nom]
    return dataclasses.replace(t, easy=t.easy[:2], hard=tuple(c for c in t.hard if c[0] in durs),
                               later=tuple(c for c in t.later if c[0] in suite))


MINUTES = _petite("minutes_de", {"90 min"}, {"3h15", "45 min"})                 # le premier outil PLANTE
MONTANT = _petite("montant", {"1 234,50", "12,5"}, {"99.90", "2 000,00"})       # ... plante, et rend une valeur FAUSSE
POURCENT = _petite("pourcentage", {"0.5 pour cent"}, {"25%", "80 pour cent"})   # ... REFUSE une entrée valide


def _lance(code: str, param: str, texte: str) -> tuple[str, Any]:
    """Le code d'une tâche, exécuté ici, sans bac : ce sont les codes du banc, pas ceux d'un modèle."""
    espace: dict[str, Any] = {}
    exec(code, espace)
    try:
        return "result", espace["run"]({param: texte}, {})
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"


def _lit(code: str, t: ToolTask, cas: tuple[tuple[str, Any], ...]) -> list[bool]:
    return [_lance(code, t.param, texte)[0] == "result" and same_result(attendu, _lance(code, t.param, texte)[1])
            for texte, attendu in cas]


def _ligne(tache: str, arm: str, resolu: bool, *, essai: int = 1, statut: str = "kept", jetons: int = 0,
           durs: int = 0, premier: int = 150) -> ToolBenchRow:
    return ToolBenchRow(tache, essai, arm, statut, 5 if resolu else 2, 5, 0, jetons, durs, 3, premier)


# ---------------------------------------------------------------------------------------------------------

class TestLesTachesFournies:
    def test_huit_taches_aux_noms_distincts(self) -> None:
        assert len(TOOL_TASKS) == 8 and len(PAR_NOM) == 8

    @pytest.mark.parametrize("t", TOOL_TASKS, ids=lambda t: t.name)
    def test_une_entree_n_apparait_qu_une_fois(self, t: ToolTask) -> None:
        entrees = [texte for texte, _ in (*t.easy, *t.hard, *t.later)]
        assert len(set(entrees)) == len(entrees)                       # « suivantes » = jamais vues avant
        assert len(t.easy) >= 3 and len(t.hard) >= 3 and len(t.later) >= 5

    @pytest.mark.parametrize("t", TOOL_TASKS, ids=lambda t: t.name)
    def test_la_demande_ne_decrit_que_le_cas_facile(self, t: ToolTask) -> None:
        assert t.easy[0][0] in t.capability
        assert not any(texte in t.capability for texte, _ in (*t.hard, *t.later))

    @pytest.mark.parametrize("t", TOOL_TASKS, ids=lambda t: t.name)
    def test_la_version_naive_lit_le_facile_et_rien_du_dur(self, t: ToolTask) -> None:
        assert all(_lit(t.naive_code, t, t.easy)) and not any(_lit(t.naive_code, t, t.hard))

    @pytest.mark.parametrize("t", TOOL_TASKS, ids=lambda t: t.name)
    def test_la_version_complete_lit_tout_et_refuse_l_illisible(self, t: ToolTask) -> None:
        assert all(_lit(t.full_code, t, (*t.easy, *t.hard, *t.later)))
        issue, erreur = _lance(t.full_code, t.param, "???")
        assert issue == "error" and erreur.startswith("ValueError: invalid input")

    @pytest.mark.parametrize("t", TOOL_TASKS, ids=lambda t: t.name)
    def test_le_premier_outil_lit_une_partie_de_la_suite(self, t: ToolTask) -> None:
        lues = sum(_lit(t.naive_code, t, t.later))
        assert 0 < lues < len(t.later)             # de quoi voir un gain, ET de quoi voir une entrée perdue

    def test_les_trois_sortes_de_casse_sont_la(self) -> None:
        sortes = set()
        for t in TOOL_TASKS:
            for texte, _ in t.hard:
                issue, valeur = _lance(t.naive_code, t.param, texte)
                sortes.add("valeur fausse" if issue == "result" else
                           "refus" if str(valeur).startswith("ValueError: invalid input") else "plantage")
        assert sortes == {"plantage", "valeur fausse", "refus"}


class TestLeJuge:
    def test_la_valeur_nue_ou_dans_un_objet_a_une_cle(self) -> None:
        assert same_result(90, 90) and same_result(90, {"minutes": 90}) and same_result("JD", {"initiales": "JD"})
        assert not same_result(90, {"minutes": 90, "heures": 1})        # deux clés : ce n'est plus un emballage
        assert not same_result(90, {"minutes": {"valeur": 90}})         # un seul niveau

    def test_les_nombres_a_peu_pres(self) -> None:
        assert same_result(20, 19.999999999999977) and same_result(0, 0.0) and same_result(0.1, 1 / 10)
        assert not same_result(20, 20.001) and not same_result(90, 150)

    def test_un_type_n_en_vaut_pas_un_autre(self) -> None:
        assert not same_result(90, "90") and not same_result("90", 90)
        assert not same_result(1, True) and not same_result(True, 1) and not same_result(0, False)
        assert same_result(True, True) and not same_result(True, False)
        assert not same_result(90, None) and not same_result(90, [90])


# ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def controle() -> tuple[ToolBenchReport, ScriptedToolProvider]:
    """Le contrôle positif sur trois petites tâches — une par sorte de casse. Un seul passage pour tout le module."""
    fournisseur = ScriptedToolProvider([MINUTES, MONTANT, POURCENT])
    return run_tool_bench(fournisseur, tasks=[MINUTES, MONTANT, POURCENT]), fournisseur


class TestLeControlePositif:
    def test_improve_resout_ce_que_les_autres_ne_resolvent_pas(self, controle: Any) -> None:
        rapport, _ = controle
        c = rapport.by_arm()
        assert list(c) == list(ARMS) and all(c[arm]["units"] == 3 for arm in ARMS)
        assert (c["none"]["solved"], c["rebuild"]["solved"], c["improve"]["solved"]) == (0, 0, 3)
        assert (c["none"]["right"], c["rebuild"]["right"], c["improve"]["right"]) == (3, 3, 6)
        assert c["improve"]["total"] == 6 and c["improve"]["statuses"] == {"improved": 3}
        assert c["none"]["statuses"] == {"kept": 3} and c["rebuild"]["statuses"] == {"rebuilt": 3}

    def test_sans_verdict_de_l_hote_seul_le_plantage_s_apprend(self, controle: Any) -> None:
        # La limite connue, MESURÉE : une valeur fausse que personne n'a signalée compte comme un cas réussi (la
        # bonne correction est refusée) ; un refus que personne n'a jugé n'est pas un cas à corriger.
        rapport, _ = controle
        statuts = {r.task: (r.status, r.right) for r in rapport.rows if r.arm == "improve_unjudged"}
        assert statuts == {"minutes_de": ("improved", 2), "montant": ("rejected", 1), "pourcentage": ("nothing_to_fix", 1)}
        assert rapport.by_arm()["improve_unjudged"]["solved"] == 1

    def test_rien_n_est_perdu(self, controle: Any) -> None:
        rapport, _ = controle
        assert all(r.regressions == 0 for r in rapport.rows)

    def test_les_jetons_sont_comptes_par_reaction(self, controle: Any) -> None:
        rapport, fournisseur = controle
        c = rapport.by_arm()
        # 150 jetons déclarés par appel. `nothing_to_fix` n'appelle pas le modèle : 2 appels sur 3 sans verdict.
        assert (c["none"]["tokens"], c["rebuild"]["tokens"], c["improve"]["tokens"]) == (0, 450, 450)
        assert c["improve_unjudged"]["tokens"] == 300
        assert all(r.first_tokens == 150 for r in rapport.rows)
        assert rapport.tokens == 450 + 450 + 450 + 300 == 150 * len(fournisseur.requests)

    def test_le_premier_outil_a_casse_partout(self, controle: Any) -> None:
        rapport, _ = controle
        assert (rapport.units, rapport.broke, rapport.trials, rapport.model) == (3, 3, 1, "scripted-1")
        assert rapport.tasks == ("minutes_de", "montant", "pourcentage")
        assert [(r.task, r.arm) for r in rapport.rows] == [(t, arm) for t in rapport.tasks for arm in ARMS]
        assert all(r.first_hard_right == 0 and r.trial == 1 for r in rapport.rows)
        assert [r.hard_total for r in rapport.rows if r.arm == "none"] == [1, 2, 1]

    def test_le_modele_ne_voit_jamais_les_entrees_suivantes(self, controle: Any) -> None:
        _, fournisseur = controle
        demandes = ["\n".join(m.content or "" for m in d.messages) for d in fournisseur.requests]
        for tache in (MINUTES, MONTANT, POURCENT):
            for texte, _ in tache.later:
                assert not any(json.dumps(texte, ensure_ascii=False) in d for d in demandes), texte
        ameliorations = [d for d in demandes if "IMPROVE AN EXISTING TOOL" in d]
        assert len(ameliorations) == 5                                 # 3 avec verdict + 2 sans (1 n'a rien à corriger)
        assert sum("90 min" in d for d in ameliorations) == 2          # le plantage : montré dans les deux cas
        assert sum("12,5" in d for d in ameliorations) == 1            # la valeur fausse : seulement si l'hôte l'a dite
        assert sum("0.5 pour cent" in d for d in ameliorations) == 1   # le refus : seulement si l'hôte l'a jugé

    def test_le_resume_se_lit(self, controle: Any) -> None:
        resume = controle[0].summary()
        assert "scripted-1 · 3 tâches x 1 essai = 3 unités · le premier outil a cassé sur 3 · 1650 jetons" in resume
        assert "improve (l'hôte dit l'attendu) : 3/3 unités résolues · 6/6 entrées suivantes justes" in resume
        assert "improve (l'hôte ne dit rien)   : 1/3 unités résolues" in resume
        assert "1 improved, 1 nothing_to_fix, 1 rejected" in resume and "Le juge est du code." in resume
        assert "rien → refait à l'aveugle : +0 pt" in resume and "indistinguables" in resume

    def test_le_rapport_s_ecrit_en_json(self, controle: Any) -> None:
        d = json.loads(json.dumps(controle[0].to_dict()))
        assert (d["units"], d["broke"], d["tokens"], d["arms"]) == (3, 3, 1650, list(ARMS))
        assert d["by_arm"]["improve"]["solved"] == 3 and len(d["rows"]) == 12
        assert d["rows"][2] == {"task": "minutes_de", "trial": 1, "arm": "improve", "status": "improved", "right": 2,
                                "total": 2, "regressions": 0, "tokens": 150, "first_hard_right": 0, "hard_total": 1,
                                "first_tokens": 150, "detail": "", "solved": True}


# ---------------------------------------------------------------------------------------------------------

class _Illisible(ScriptedToolProvider):
    """Un constructeur dont la n-ième construction ORDINAIRE (hors `improve`) n'est pas un outil."""

    def __init__(self, tasks: Any, rang: int) -> None:
        super().__init__(tasks)
        self.rang, self.ordinaires = rang, 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        if "IMPROVE AN EXISTING TOOL" not in "".join(m.content or "" for m in request.messages):
            self.ordinaires += 1
            if self.ordinaires == self.rang:
                self.requests.append(request)
                return LLMResponse(content="ceci n'est pas un outil", model="scripted-1")
        return super().complete(request)


class _Gourmand(ScriptedToolProvider):
    """Un constructeur qui s'accorde le réseau."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        reponse = super().complete(request)
        contenu = json.loads(reponse.content or "")
        contenu["tool"]["permissions"] = ["network"]
        return LLMResponse(content=json.dumps(contenu), model="scripted-1")


@dataclasses.dataclass
class _BacEspion(SubprocessSandbox):
    """Le bac choisi par l'hôte : il compte ce qui passe par lui."""

    vus: list[str] = dataclasses.field(default_factory=list)

    def run_python_tool(self, file_path: Any, args: Any, context: Any = None, **kw: Any) -> Any:
        self.vus.append(str(file_path))
        return super().run_python_tool(file_path, args, context, **kw)


class TestLesReactions:
    def test_une_mauvaise_correction_est_refusee_et_rien_n_est_perdu(self) -> None:
        rapport = run_tool_bench(ScriptedToolProvider([MINUTES], fix="regress"), tasks=[MINUTES],
                                 arms=("none", "improve"))
        rien, improve = rapport.rows
        assert (improve.status, improve.right, improve.regressions) == ("rejected", rien.right, 0) and rien.right == 1
        assert improve.tokens == 150 and not improve.solved             # l'essai est payé, l'ancienne version reste
        assert improve.detail == "0 failing case(s) not fixed, 1 passing case(s) changed" and rien.detail == ""

    def test_une_correction_qui_ne_corrige_rien_est_refusee(self) -> None:
        rapport = run_tool_bench(ScriptedToolProvider([MINUTES], fix="same"), tasks=[MINUTES], arms=("improve",))
        (improve,) = rapport.rows
        assert (improve.status, improve.right, improve.total, improve.regressions) == ("rejected", 1, 2, 0)

    def test_sans_premier_outil_l_unite_est_ratee_pour_tous(self) -> None:
        rapport = run_tool_bench(_Illisible([MINUTES], rang=1), tasks=[MINUTES])
        assert [(r.status, r.right, r.total, r.solved, r.tokens) for r in rapport.rows] == [
            ("first_build_failed", 0, 2, False, 0)] * 4
        assert all(r.detail.startswith("ToolValidationError: ") for r in rapport.rows)      # pourquoi, mot pour mot
        assert rapport.broke == 0 and rapport.units == 1 and rapport.by_arm()["improve"]["solved"] == 0
        assert rapport.compare("rebuild", "improve")["verdict"] == "indistinguishable"

    def test_le_modele_ne_peut_s_accorder_aucune_permission(self) -> None:
        # Des outils de calcul pur : un constructeur qui demande le réseau n'obtient pas d'outil.
        rapport = run_tool_bench(_Gourmand([MINUTES]), tasks=[MINUTES], arms=("none", "improve"))
        assert [r.status for r in rapport.rows] == ["first_build_failed", "first_build_failed"]

    def test_un_outil_qu_on_ne_sait_pas_refaire_est_perdu(self) -> None:
        # Le retrait suivi d'une recréation qui échoue : l'agent n'a plus d'outil, ce qu'il lisait est perdu.
        rapport = run_tool_bench(_Illisible([MINUTES], rang=2), tasks=[MINUTES], arms=("none", "rebuild"))
        rien, refait = rapport.rows
        assert (refait.status, refait.right, refait.regressions) == ("build_failed", 0, 1) and rien.right == 1
        c = rapport.by_arm()
        assert (c["none"]["regressions"], c["rebuild"]["regressions"]) == (0, 1)
        assert refait.detail.startswith("ToolValidationError: ") and rien.detail == ""
        assert "0/2 entrées suivantes justes · 1 perdue(s)" in rapport.summary()

    def test_seules_les_reactions_demandees_tournent(self) -> None:
        fournisseur = ScriptedToolProvider([MINUTES])
        rapport = run_tool_bench(fournisseur, tasks=[MINUTES], arms=("improve", "none"))
        assert [r.arm for r in rapport.rows] == ["improve", "none"] and rapport.arms == ("improve", "none")
        assert len(fournisseur.requests) == 2                           # le premier outil, puis `improve`
        assert rapport.rows[0].solved and not rapport.rows[1].solved

    def test_les_entrees_perdues_se_comptent_meme_sans_le_bras_rien(self) -> None:
        rapport = run_tool_bench(_Illisible([MINUTES], rang=2), tasks=[MINUTES], arms=("rebuild",))
        assert [(r.arm, r.regressions) for r in rapport.rows] == [("rebuild", 1)]

    def test_plusieurs_essais(self) -> None:
        rapport = run_tool_bench(ScriptedToolProvider([MINUTES]), tasks=[MINUTES], arms=("none", "improve"), trials=2)
        assert [(r.trial, r.arm) for r in rapport.rows] == [(1, "none"), (1, "improve"), (2, "none"), (2, "improve")]
        assert (rapport.units, rapport.broke) == (2, 2) and rapport.by_arm()["improve"]["solved"] == 2
        assert "1 tâche x 2 essais = 2 unités" in rapport.summary()

    def test_l_attendu_est_dit_dans_la_forme_que_l_outil_rend(self) -> None:
        # Un outil qui rend {"minutes": 90} : l'hôte doit dire {"minutes": 90}, pas 90 — sinon rien n'est prouvé.
        emballe = dataclasses.replace(
            MINUTES,
            naive_code=MINUTES.naive_code.replace("return int(m.group(1)) * 60 + int(m.group(2) or 0)",
                                                  "return {'minutes': int(m.group(1)) * 60 + int(m.group(2) or 0)}"),
            full_code=MINUTES.full_code.replace("return int(m.group(1)) * 60 + int(m.group(2) or 0)",
                                                "return {'minutes': int(m.group(1)) * 60 + int(m.group(2) or 0)}")
                                       .replace("return int(m.group(1))\n", "return {'minutes': int(m.group(1))}\n"))
        assert "{'minutes'" in emballe.naive_code and emballe.full_code.count("{'minutes'") == 2
        fournisseur = ScriptedToolProvider([emballe])
        (improve,) = run_tool_bench(fournisseur, tasks=[emballe], arms=("improve",)).rows
        assert (improve.status, improve.solved) == ("improved", True)
        demande = "\n".join(m.content or "" for m in fournisseur.requests[-1].messages)
        assert '{\\"minutes\\": 90}' in demande                         # l'attendu, tel que l'outil le rend

    def test_un_premier_outil_qui_lit_deja_tout_n_apprend_rien(self) -> None:
        deja = dataclasses.replace(MINUTES, naive_code=MINUTES.full_code)
        fournisseur = ScriptedToolProvider([deja])
        rapport = run_tool_bench(fournisseur, tasks=[deja])
        assert rapport.broke == 0 and all(r.solved and r.first_hard_right == 1 for r in rapport.rows)
        assert {r.arm: r.status for r in rapport.rows} == {
            "none": "kept", "rebuild": "rebuilt", "improve": "nothing_to_fix", "improve_unjudged": "nothing_to_fix"}
        assert {r.arm: r.detail for r in rapport.rows}["improve"] == "no failing call of the current version in the call log"
        assert len(fournisseur.requests) == 2                           # `improve` n'a pas appelé le modèle
        assert rapport.compare("rebuild", "improve")["verdict"] == "indistinguishable"

    def test_tout_le_code_ecrit_par_le_modele_tourne_dans_le_bac_de_l_hote(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bac = _BacEspion()
        monkeypatch.setattr("autoagent.dynamic._bac_par_defaut", lambda *a, **k: pytest.fail("un autre bac a été créé"))
        rapport = run_tool_bench(ScriptedToolProvider([MINUTES]), tasks=[MINUTES], sandbox=bac)
        assert rapport.by_arm()["improve"]["solved"] == 1
        # 3 premiers usages + 2 entrées suivantes par outil final (rien, improve, improve sans verdict, refait)
        # + 3 cas rejoués par chaque `improve` (1 raté, 2 réussis)
        assert len(bac.vus) == 3 + 2 * 4 + 3 * 2

    def test_les_entrees_suivantes_n_entrent_jamais_au_carnet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        notes: list[dict[str, Any]] = []
        vrai = DynamicToolBuilder._noter_appel

        def espion(self: DynamicToolBuilder, appel: dict[str, Any]) -> None:
            notes.append(appel)
            vrai(self, appel)

        monkeypatch.setattr(DynamicToolBuilder, "_noter_appel", espion)
        run_tool_bench(ScriptedToolProvider([MINUTES]), tasks=[MINUTES])
        vues = [a["args"]["texte"] for a in notes]
        assert vues == ["1h30", "2h", "90 min"]                         # les premiers usages, et rien d'autre

    def test_rien_ne_reste_sur_le_disque(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
        run_tool_bench(ScriptedToolProvider([MINUTES]), tasks=[MINUTES], arms=("none", "improve"))
        assert list(tmp_path.iterdir()) == []

    def test_le_controle_positif_en_une_ligne(self) -> None:
        c = positive_control_tools(tasks=[MINUTES]).by_arm()
        assert c["improve"]["solved"] == 1 and c["none"]["solved"] == 0 and c["rebuild"]["solved"] == 0

    @pytest.mark.parametrize("kw", [
        {"arms": ("none", "magie")}, {"arms": ("none", "none")}, {"arms": ()}, {"trials": 0}, {"trials": True},
        {"trials": 1.5}, {"tasks": []}, {"tasks": [MINUTES, MINUTES]},
    ])
    def test_les_demandes_mal_formees_sont_refusees(self, kw: dict[str, Any]) -> None:
        fournisseur = ScriptedToolProvider([MINUTES])
        with pytest.raises(ValueError):
            run_tool_bench(fournisseur, **{"tasks": [MINUTES], **kw})
        assert fournisseur.requests == []                               # refusé AVANT tout appel au modèle

    def test_le_constructeur_scripte_ne_connait_que_trois_corrections(self) -> None:
        with pytest.raises(ValueError):
            ScriptedToolProvider([MINUTES], fix="magie")


# ---------------------------------------------------------------------------------------------------------

class TestLeRapport:
    """Le calcul, sur des lignes écrites à la main : aucun bac, aucun modèle."""

    @staticmethod
    def _rapport(resolus_a: int, resolus_b: int, taches: int = 8) -> ToolBenchReport:
        noms = tuple(f"t{i}" for i in range(taches))
        lignes = []
        for i, nom in enumerate(noms):
            lignes.append(_ligne(nom, "rebuild", i < resolus_a, statut="rebuilt", jetons=100))
            lignes.append(_ligne(nom, "improve", i < resolus_b, statut="improved", jetons=200))
        return ToolBenchReport(rows=lignes, arms=("rebuild", "improve"), trials=1, model="m", tasks=noms)

    def test_huit_taches_sur_huit_c_est_une_difference(self) -> None:
        c = self._rapport(0, 8).compare("rebuild", "improve")
        assert c["verdict"] == "b_better" and c["delta"] == 1.0 and c["p"] == pytest.approx(2 / 256) and c["low"] > 0
        assert (c["a"], c["b"]) == ("rebuild", "improve")
        assert self._rapport(8, 0).compare("rebuild", "improve")["verdict"] == "a_better"
        assert self._rapport(0, 8).compare("improve", "rebuild")["verdict"] == "a_better"

    def test_quatre_taches_ne_suffisent_pas_a_trancher(self) -> None:
        # 4 sur 4 contre 0 sur 4 : p = 2 x (1/2)^4 = 0,125. Le banc ne tranche pas — il faut plus de tâches.
        c = self._rapport(0, 4, taches=4).compare("rebuild", "improve")
        assert c["verdict"] == "indistinguishable" and c["p"] == pytest.approx(0.125) and c["delta"] == 1.0

    def test_deux_bras_egaux_sont_indistinguables(self) -> None:
        c = self._rapport(3, 3).compare("rebuild", "improve")
        assert (c["verdict"], c["delta"], c["p"]) == ("indistinguishable", 0.0, 1.0)

    def test_un_bras_inconnu_est_refuse(self) -> None:
        with pytest.raises(ValueError):
            self._rapport(1, 1).compare("rebuild", "none")

    def test_les_totaux_par_bras(self) -> None:
        rapport = self._rapport(2, 5)
        c = rapport.by_arm()
        assert (c["rebuild"]["solved"], c["improve"]["solved"]) == (2, 5)
        assert c["rebuild"]["right"] == 2 * 5 + 6 * 2 and c["rebuild"]["total"] == 40
        assert (c["rebuild"]["tokens"], c["improve"]["tokens"]) == (800, 1600)
        assert rapport.tokens == 8 * 150 + 800 + 1600                   # la première construction : une fois par unité

    def test_une_unite_n_a_casse_que_si_un_dur_a_ete_mal_lu(self) -> None:
        lignes = [_ligne("a", "none", False, durs=0), _ligne("b", "none", True, durs=3),
                  _ligne("c", "none", False, statut="first_build_failed", durs=0)]
        rapport = ToolBenchReport(rows=lignes, arms=("none",), trials=1, tasks=("a", "b", "c"))
        assert rapport.broke == 1 and rapport.units == 3
        assert "modèle inconnu" in rapport.summary()

    def test_resolu_veut_dire_tout_juste(self) -> None:
        assert ToolBenchRow("t", 1, "none", "kept", 5, 5, 0, 0, 0, 3, 0).solved
        assert not ToolBenchRow("t", 1, "none", "kept", 4, 5, 0, 0, 0, 3, 0).solved
        assert not ToolBenchRow("t", 1, "none", "kept", 0, 0, 0, 0, 0, 3, 0).solved      # rien à lire : pas « résolu »
