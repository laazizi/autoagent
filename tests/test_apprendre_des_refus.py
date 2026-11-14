"""Apprendre des REFUS : `builder.improve(nom, learn_from_refusals=True)`.

Mesuré au banc (13 nov. 2026) : sans verdict de l'hôte, `improve` n'avait RIEN à corriger sur un vrai modèle — ses
outils refusent proprement l'écriture qu'ils ne connaissent pas (« 1:30 ») au lieu de planter, et un refus que
personne n'a jugé n'est pas un défaut. Or l'hôte ne peut pas toujours juger.

Avec l'option, le modèle voit aussi les entrées REFUSÉES que personne n'a jugées, et décide : une demande valide
écrite autrement se traite ; une entrée invalide reste refusée. Ce que le code prouve, et rien de plus :

* chaque cas réussi du carnet donne le même résultat qu'avant ;
* un refus propre ne devient jamais un plantage ;
* un refus que l'hôte a jugé LÉGITIME reste un refus (toujours vérifié, option ou pas).

Ce qu'il ne prouve PAS : que la nouvelle lecture est juste. Elle est comptée à part (`extended`) et dite « non
vérifiée » partout — c'est le même niveau de preuve qu'une première construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoagent import DynamicToolBuilder
from autoagent.dynamic import ImprovementResult, _demande_d_amelioration
from autoagent.errors import ToolError
from tests._outils_generes import Sequence, construire, payload

TEXTE = {"texte": {"type": "string"}}
# La durée en minutes. V1 ne lit que « 1h30 » et REFUSE le reste. V2 lit aussi « 90 min ».
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
V1 = payload(_DEBUT + _REFUS, "duree", TEXTE)
V2 = payload(_DEBUT + _MINUTES + _REFUS, "duree", TEXTE)
V2_DEVINE = payload(_DEBUT + _MINUTES + "    return 0\n", "duree", TEXTE)                  # ne refuse plus rien
V2_PLANTE = payload(_DEBUT + _MINUTES + "    return 1 / 0\n", "duree", TEXTE)              # un refus devenu plantage
V2_CASSE = payload((_DEBUT + _MINUTES + _REFUS).replace("* 60 +", "* 61 +"), "duree", TEXTE)  # change un cas réussi

LU = {"texte": "1h30"}
AUTREMENT = {"texte": "90 min"}
ILLISIBLE = {"texte": "???"}


def _atelier(dossier: Path, *contenus: str, **kw: Any) -> tuple[DynamicToolBuilder, Any]:
    """V1, trois appels au carnet : un réussi, deux refus que personne n'a jugés."""
    b, outil = construire(dossier, V1, **kw)
    assert outil(**LU) == 90
    for args in (AUTREMENT, ILLISIBLE):
        with pytest.raises(ToolError, match="invalid input"):
            outil(**args)
    b.provider.contenus.extend(contenus)  # type: ignore[attr-defined]
    return b, outil


def _appel(b: DynamicToolBuilder, args: dict[str, Any]) -> dict[str, Any]:
    return next(a for a in b.calls("duree") if a["args"] == args)


def _demande(b: DynamicToolBuilder) -> str:
    return "\n".join(m.content or "" for m in b.provider.requetes[-1].messages)  # type: ignore[attr-defined]


def _sha(dossier: Path) -> str:
    import hashlib
    return hashlib.sha256((dossier / "duree.py").read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------

class TestSansLOptionRienNeChange:
    def test_un_refus_non_juge_n_est_pas_un_cas_a_corriger(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2)
        resultat = b.improve("duree")
        assert resultat.status == "nothing_to_fix" and len(b.provider.requetes) == 1      # type: ignore[attr-defined]
        assert (resultat.extended, resultat.still_refused, resultat.extended_cases) == (0, 0, [])

    def test_la_demande_au_modele_est_mot_pour_mot_celle_d_avant(self) -> None:
        # La phrase du constructeur a été mesurée sur un vrai modèle (tâche 3) : sans refus, elle ne bouge pas.
        cas = [{"args": {"n": 1}, "kind": "exception", "error": "ZeroDivisionError: division by zero"}]
        assert _demande_d_amelioration("outil x", "def run(args, context): ...", cas, []) == (
            "IMPROVE AN EXISTING TOOL - keep its name and its purpose: outil x\n\n"
            "Its current code:\ndef run(args, context): ...\n\n"
            'It failed on these REAL calls:\n1. args = {"n": 1} -> error: ZeroDivisionError: division by zero.\n\n'
            "Fix the CAUSE, not only these inputs. Other real calls that succeed today will be replayed against your "
            "new version, and you are not shown them: their results must not change.\n"
            "If an input is truly invalid for this tool, REFUSE it: let input_schema reject it (pattern, enum, minimum, "
            'minItems...) or raise ValueError("invalid input: <why>"). Never return an error value in place of a result.'
        )

    def test_l_option_est_un_booleen(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path)
        with pytest.raises(TypeError, match="learn_from_refusals"):
            b.improve("duree", learn_from_refusals=1)  # type: ignore[arg-type]


class TestUneEntreeRefuseeDevientLisible:
    def test_la_nouvelle_version_est_installee_et_dit_ce_qui_n_est_pas_verifie(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2)
        ancienne = _sha(tmp_path)
        resultat = b.improve("duree", learn_from_refusals=True)
        assert isinstance(resultat, ImprovementResult) and resultat.status == "improved", resultat.reason
        assert (resultat.fixed, resultat.checked, resultat.kept) == (0, 0, 1)
        assert (resultat.extended, resultat.still_refused) == (1, 1)
        assert resultat.extended_cases == [{"args": AUTREMENT, "got": 90}]          # ce que l'hôte peut relire
        assert resultat.still_failing == [] and resultat.regressions == [] and resultat.broken_refusals == []
        assert b.catalogue()["duree"]["verified_by"] == {"kind": "replay", "fixed": 0, "checked": 0, "kept": 1,
                                                         "extended": 1, "from": ancienne[:12]}
        assert resultat.tool is not None and resultat.tool(**AUTREMENT) == 90 and resultat.tool(**LU) == 90
        with pytest.raises(ToolError, match="invalid input"):
            resultat.tool(**ILLISIBLE)                                              # l'illisible reste refusé

    def test_le_compte_rendu_et_la_fiche_le_disent(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2)
        resume = b.improve("duree", learn_from_refusals=True).summary()
        attendu = "1 cas réussi rejoué à l'identique, 1 entrée auparavant refusée maintenant lue (non vérifiée)"
        assert attendu in resume and "cas raté" not in resume and "rollback" in resume
        assert attendu in b.health("duree").summary()

    def test_plusieurs_entrees_s_accordent(self, tmp_path: Path) -> None:
        b, outil = _atelier(tmp_path, V2)
        with pytest.raises(ToolError):
            outil(texte="45 min")
        resultat = b.improve("duree", learn_from_refusals=True)
        assert resultat.extended == 2 and "2 entrées auparavant refusées maintenant lues (non vérifiées)" in resultat.summary()

    def test_l_ancienne_version_se_retrouve(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2)
        ancienne = _sha(tmp_path)
        b.improve("duree", learn_from_refusals=True)
        assert b.rollback("duree") is True and _sha(tmp_path) == ancienne

    def test_un_refus_du_schema_compte_comme_un_refus(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, V1)
        assert outil(**LU) == 90
        outil.__autoagent_rejected__({"texte": 90}, "texte: 90 is not of type 'string'")       # ce que fait le registre
        large = payload((_DEBUT + _REFUS).replace("t = args['texte']", "t = '%dh' % (args['texte'] // 60) if "
                        "isinstance(args['texte'], int) else args['texte']"), "duree", {"texte": {}})
        b.provider.contenus.append(large)  # type: ignore[attr-defined]
        resultat = b.improve("duree", learn_from_refusals=True)
        assert resultat.status == "improved" and resultat.extended_cases == [{"args": {"texte": 90}, "got": 60}]
        assert "refused by the input_schema" in _demande(b)


class TestCeQueLeModeleVoit:
    def test_les_refus_dans_leur_propre_rubrique_avec_la_consigne(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2)
        b.improve("duree", learn_from_refusals=True)
        demande = _demande(b)
        assert "It REFUSED these REAL inputs, and nobody has said whether it was right to:" in demande
        assert '{\\"texte\\": \\"90 min\\"}' in demande and '{\\"texte\\": \\"???\\"}' in demande
        assert "invalid input: duree illisible" in demande
        assert "keep refusing it: never guess a value" in demande
        assert "It failed on these REAL calls" not in demande            # aucun cas raté ici : pas de rubrique vide
        assert '\\"1h30\\"' not in demande                                # le cas réussi reste caché

    def test_un_refus_venu_d_un_run_teinte_n_est_pas_montre_mais_il_est_rejoue(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2)
        self._teinter(b, tmp_path, AUTREMENT)
        resultat = b.improve("duree", learn_from_refusals=True)
        demande = _demande(b)
        assert '\\"90 min\\"' not in demande and '\\"???\\"' in demande
        assert resultat.status == "improved" and resultat.extended == 1   # rejoué : la version le lit, et c'est dit

    def test_si_tous_les_refus_viennent_de_runs_teintes_le_modele_n_est_pas_appele(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2)
        self._teinter(b, tmp_path, AUTREMENT)
        self._teinter(b, tmp_path, ILLISIBLE)
        resultat = b.improve("duree", learn_from_refusals=True)
        assert resultat.status == "no_usable_case" and len(b.provider.requetes) == 1          # type: ignore[attr-defined]

    def test_un_refus_masque_par_redact_n_est_ni_montre_ni_rejoue(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, V1, redact=lambda v: "***" if isinstance(v, str) else v)
        outil(**LU)
        with pytest.raises(ToolError):
            outil(**AUTREMENT)
        b.provider.contenus.append(V2)  # type: ignore[attr-defined]
        resultat = b.improve("duree", learn_from_refusals=True)
        assert (resultat.status, resultat.skipped) == ("no_usable_case", 1)
        assert len(b.provider.requetes) == 1                                                  # type: ignore[attr-defined]

    def test_le_nombre_de_refus_montres_est_borne(self, tmp_path: Path) -> None:
        b, outil = _atelier(tmp_path, V2)
        for i in range(8):
            with pytest.raises(ToolError):
                outil(texte=f"{i} jours")
        b.improve("duree", learn_from_refusals=True, max_cases=3)
        rubrique = _demande(b).split("It REFUSED these REAL inputs")[1].split("Decide for each one")[0]
        assert rubrique.count("args = ") == 3

    @staticmethod
    def _teinter(b: DynamicToolBuilder, dossier: Path, args: dict[str, Any]) -> None:
        import json
        chemin = dossier / "duree.calls.jsonl"
        lignes = [json.loads(ligne) for ligne in chemin.read_text(encoding="utf-8").splitlines()]
        for ligne in lignes:
            if ligne.get("args") == args:
                ligne["tainted"] = True
        chemin.write_text("\n".join(json.dumps(ligne, ensure_ascii=False) for ligne in lignes) + "\n", encoding="utf-8")


class TestCeQueLeCodeRefuse:
    def test_une_version_qui_ne_lit_rien_de_plus_n_est_pas_installee(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V1.replace("duree illisible", "duree vraiment illisible"))
        avant = _sha(tmp_path)
        resultat = b.improve("duree", learn_from_refusals=True)
        assert (resultat.status, resultat.extended, resultat.still_refused) == ("nothing_gained", 0, 2)
        assert _sha(tmp_path) == avant and resultat.tool is None and resultat.new_version is None
        assert b.catalogue()["duree"]["last_improvement"]["status"] == "nothing_gained"
        assert "aucun gain" in resultat.summary() and "l'ancienne reste" in resultat.summary()

    def test_un_refus_propre_ne_devient_pas_un_plantage(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2_PLANTE)
        avant = _sha(tmp_path)
        resultat = b.improve("duree", learn_from_refusals=True)
        assert resultat.status == "rejected" and _sha(tmp_path) == avant
        (casse,) = resultat.broken_refusals
        assert casse["args"] == ILLISIBLE and "ZeroDivisionError" in casse["error"] and "crash" in casse["why"]
        assert resultat.reason == "0 failing case(s) not fixed, 0 passing case(s) changed, 1 refusal(s) broken"
        assert resultat.summary().endswith("l'ancienne reste : 1 refus cassé.")

    def test_un_cas_reussi_ne_change_pas(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2_CASSE)
        resultat = b.improve("duree", learn_from_refusals=True)
        assert resultat.status == "rejected" and len(resultat.regressions) == 1 and resultat.tool is None

    def test_ce_que_le_code_ne_voit_pas_une_version_qui_devine(self, tmp_path: Path) -> None:
        # LA LIMITE, écrite : « ??? » était refusé, la version qui devine rend 0. Rien ne dit au code que c'est faux.
        # Elle est installée — et le compte rendu montre ce qu'elle répond, pour que l'hôte (ou le banc) le voie.
        b, _ = _atelier(tmp_path, V2_DEVINE)
        resultat = b.improve("duree", learn_from_refusals=True)
        assert resultat.status == "improved" and resultat.extended == 2 and resultat.still_refused == 0
        assert {"args": ILLISIBLE, "got": 0} in resultat.extended_cases


class TestLeVerdictDeLHoteCompteToujours:
    def test_un_refus_juge_legitime_n_est_pas_propose_et_doit_rester_un_refus(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2_DEVINE)
        b.report("duree", _appel(b, ILLISIBLE)["id"], correct=True, reason="ce n'est pas une durée")
        avant = _sha(tmp_path)
        resultat = b.improve("duree", learn_from_refusals=True)
        assert '\\"???\\"' not in _demande(b)                             # jugé : ce n'est plus une question
        assert resultat.status == "rejected" and _sha(tmp_path) == avant
        (casse,) = resultat.broken_refusals
        assert casse == {"args": ILLISIBLE, "got": 0, "why": "the host judged this failure legitimate: the input is invalid"}

    def test_meme_sans_l_option(self, tmp_path: Path) -> None:
        # Un cas raté à corriger (un refus que l'hôte a jugé À TORT), et un refus jugé LÉGITIME : la version qui
        # corrige l'un en se mettant à répondre à l'autre est refusée.
        b, _ = _atelier(tmp_path, V2_DEVINE, V2)
        b.report("duree", _appel(b, AUTREMENT)["id"], correct=False, expected=90)
        b.report("duree", _appel(b, ILLISIBLE)["id"], correct=True)
        premier = b.improve("duree")
        assert premier.status == "rejected" and len(premier.broken_refusals) == 1 and premier.fixed == 1
        second = b.improve("duree")
        assert (second.status, second.fixed, second.checked, second.extended) == ("improved", 1, 1, 0)

    def test_un_refus_juge_a_tort_est_un_cas_rate_pas_une_extension(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2)
        b.report("duree", _appel(b, AUTREMENT)["id"], correct=False, expected=90)
        resultat = b.improve("duree", learn_from_refusals=True)
        assert (resultat.status, resultat.fixed, resultat.checked, resultat.extended, resultat.still_refused) == (
            "improved", 1, 1, 0, 1)
        assert "It failed on these REAL calls" in _demande(b) and "It REFUSED these REAL inputs" in _demande(b)


class TestUnRefusDejaExamineNEstPasRepropose:
    """Mesuré sur Gemini (14 nov. 2026) : dans 4 unités sur 8, le seul refus à examiner était « ??? ». Le modèle
    l'a laissé refusé — à raison —, mais l'appel était payé (26 136 jetons sur 62 508), et il l'aurait été de
    nouveau à CHAQUE appel suivant. Une entrée refusée que le modèle a examinée et laissée refusée n'est donc
    plus reproposée : ni pour cette version, ni pour la suivante."""

    V1_BIS = V1.replace("duree illisible", "duree vraiment illisible")      # une « correction » qui ne lit rien de plus

    def test_apres_un_essai_sans_gain_le_modele_n_est_plus_appele(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, self.V1_BIS, V2)
        assert b.improve("duree", learn_from_refusals=True).status == "nothing_gained"
        second = b.improve("duree", learn_from_refusals=True)
        assert second.status == "nothing_to_fix" and "already been examined" in second.reason
        assert len(b.provider.requetes) == 2 and second.usage is None        # type: ignore[attr-defined]
        assert len(b.catalogue()["duree"]["refusals_examined"]) == 2

    def test_un_nouveau_refus_est_propose_seul(self, tmp_path: Path) -> None:
        b, outil = _atelier(tmp_path, self.V1_BIS, V2)
        b.improve("duree", learn_from_refusals=True)             # « 90 min » et « ??? » : examinées, laissées refusées
        with pytest.raises(ToolError):
            outil(texte="45 min")
        resultat = b.improve("duree", learn_from_refusals=True)
        demande = _demande(b)
        assert '\\"45 min\\"' in demande and '\\"90 min\\"' not in demande and '\\"???\\"' not in demande
        # ... mais toutes sont REJOUÉES : la version qui lit « 45 min » lit aussi « 90 min », et c'est compté.
        assert (resultat.status, resultat.extended, resultat.still_refused) == ("improved", 2, 1)

    def test_la_memoire_passe_a_la_version_suivante(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2)
        resultat = b.improve("duree", learn_from_refusals=True)  # « 90 min » lue ; « ??? » examinée, laissée refusée
        assert resultat.tool is not None
        with pytest.raises(ToolError):
            resultat.tool(**ILLISIBLE)                           # la nouvelle version la refuse encore
        suivant = b.improve("duree", learn_from_refusals=True)
        assert suivant.status == "nothing_to_fix" and len(b.provider.requetes) == 2      # type: ignore[attr-defined]
        assert len(b.catalogue()["duree"]["refusals_examined"]) == 1         # « ??? » seulement : « 90 min » est lue

    def test_un_essai_refuse_n_est_pas_un_examen(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, V2_PLANTE, V2)
        assert b.improve("duree", learn_from_refusals=True).status == "rejected"
        assert "refusals_examined" not in b.catalogue()["duree"]
        assert b.improve("duree", learn_from_refusals=True).status == "improved"         # les mêmes refus, reproposés

    def test_l_hote_garde_le_dernier_mot(self, tmp_path: Path) -> None:
        b, _ = _atelier(tmp_path, self.V1_BIS, V2)
        b.improve("duree", learn_from_refusals=True)             # « 90 min » : examinée, laissée refusée
        b.report("duree", _appel(b, AUTREMENT)["id"], correct=False, expected=90)
        resultat = b.improve("duree")                            # un cas raté maintenant, option ou pas
        assert (resultat.status, resultat.fixed, resultat.checked) == ("improved", 1, 1)

    def test_la_memoire_est_bornee(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("autoagent.dynamic._MAX_REFUS_EXAMINES", 1)
        b, _ = _atelier(tmp_path, self.V1_BIS)
        b.improve("duree", learn_from_refusals=True)
        assert len(b.catalogue()["duree"]["refusals_examined"]) == 1


class TestLHotePeutDireQuIlFallaitRefuser:
    """L'hôte savait dire « résultat faux » — ce qui réclame un AUTRE résultat. Il ne savait pas dire « cette entrée
    est invalide, il fallait la refuser » : un outil qui répond à n'importe quoi ne pouvait donc pas être corrigé.
    `report(nom, id, correct=False, expected_refusal=True)`."""

    def _devine(self, dossier: Path, *contenus: str) -> DynamicToolBuilder:
        b, outil = construire(dossier, V2_DEVINE)                # il rend 0 pour ce qu'il ne lit pas
        assert outil(**LU) == 90 and outil(**ILLISIBLE) == 0
        b.provider.contenus.extend(contenus)  # type: ignore[attr-defined]
        b.report("duree", _appel(b, ILLISIBLE)["id"], correct=False, expected_refusal=True)
        return b

    def test_le_verdict_est_ecrit_et_c_est_un_defaut(self, tmp_path: Path) -> None:
        b = self._devine(tmp_path)
        verdict = _appel(b, ILLISIBLE)["verdict"]
        assert verdict["correct"] is False and verdict["expected_refusal"] is True and "expected" not in verdict
        assert b.health("duree").defects == 1

    def test_la_version_qui_refuse_resout_le_cas_et_c_est_verifie(self, tmp_path: Path) -> None:
        b = self._devine(tmp_path, V2)
        resultat = b.improve("duree")
        assert (resultat.status, resultat.fixed, resultat.checked, resultat.kept) == ("improved", 1, 1, 1)
        assert "must REFUSE it" in _demande(b)
        with pytest.raises(ToolError, match="invalid input"):
            resultat.tool(**ILLISIBLE)  # type: ignore[misc]

    def test_un_autre_resultat_ne_resout_rien(self, tmp_path: Path) -> None:
        autre = V2_DEVINE.replace("return 0\\n", "return -1\\n")
        assert autre != V2_DEVINE                                 # la candidate rend -1 au lieu de 0 : un AUTRE résultat
        b = self._devine(tmp_path, autre)
        resultat = b.improve("duree")
        assert resultat.status == "rejected" and len(resultat.still_failing) == 1

    @pytest.mark.parametrize("kw", [{"correct": True, "expected_refusal": True},
                                    {"correct": False, "expected_refusal": True, "expected": 90}])
    def test_les_verdicts_contradictoires_sont_refuses(self, tmp_path: Path, kw: dict[str, Any]) -> None:
        b, outil = construire(tmp_path, V2_DEVINE)
        outil(**ILLISIBLE)
        with pytest.raises(ValueError, match="expected_refusal"):
            b.report("duree", b.calls("duree")[-1]["id"], **kw)
        assert "verdict" not in b.calls("duree")[-1]

    def test_expected_refusal_est_un_booleen(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, V2_DEVINE)
        outil(**ILLISIBLE)
        with pytest.raises(TypeError, match="expected_refusal"):
            b.report("duree", b.calls("duree")[-1]["id"], correct=False, expected_refusal="oui")  # type: ignore[arg-type]


def test_une_bibliotheque_rechargee_apprend_aussi_des_refus(tmp_path: Path) -> None:
    _atelier(tmp_path)
    autre = DynamicToolBuilder(Sequence([V2]), tools_dir=tmp_path, persist=True, record_calls=20)
    resultat = autre.improve("duree", learn_from_refusals=True)
    assert resultat.status == "improved" and resultat.extended == 1
