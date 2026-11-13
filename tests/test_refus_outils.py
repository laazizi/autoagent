"""Un outil créé par le modèle sait dire NON : `raise ValueError("invalid input: …")` est un refus, pas un plantage.

Constaté sur un vrai modèle (13 nov. 2026) : un outil de dates, appelé trois fois avec une date mal écrite, a levé
`ValueError` trois fois — à raison — et s'est fait RETIRER. La bibliothèque ne distinguait pas « l'outil a un
bogue » de « on lui a donné une mauvaise entrée ». Le verdict de l'hôte règle le cas après coup ; ceci le règle à
la source, sans rien demander à l'hôte : l'outil DÉCLARE son refus.

* Une erreur qui commence par `ValueError: invalid input` est notée `refused`, à part : ni un défaut du code, ni
  une réussite ; la série d'erreurs ne bouge pas, l'outil n'est pas retiré.
* Ce n'est pas une porte de sortie cachée : le refus est VISIBLE (le carnet, la fiche, l'erreur rendue au modèle
  qui appelle), il ne produit jamais un résultat faux en silence, et l'hôte peut le juger à tort (`report`).
* Le modèle constructeur est prévenu de la convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoagent import DynamicToolBuilder
from autoagent.errors import ToolError
from tests._outils_generes import Sequence, construire, payload

ENTIER = {"n": {"type": "integer"}}
CODE_POLI = (
    "def run(args, context):\n"
    "    if args['n'] < 0:\n"
    "        raise ValueError('invalid input: n must be >= 0, got %d' % args['n'])\n"
    "    if args['n'] > 1000:\n"
    "        raise ValueError('trop grand')\n"
    "    return {'double': args['n'] * 2}\n"
)
POLI = payload(CODE_POLI, "doubler", ENTIER)


def _echoue(outil: Any, **args: Any) -> str:
    with pytest.raises(ToolError) as erreur:
        outil(**args)
    return str(erreur.value)


class TestUnRefusDeclare:
    def test_il_est_note_a_part(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, POLI)
        assert "invalid input: n must be >= 0" in _echoue(outil, n=-5)          # le modèle qui appelle le voit
        (appel,) = b.calls("doubler")
        assert appel["ok"] is False and appel["kind"] == "refused" and appel["args"] == {"n": -5}
        e = b.catalogue()["doubler"]
        assert (e["calls"], e["errors"], e["consecutive_errors"], e["refused_calls"]) == (1, 0, 0, 1)

    def test_des_refus_ne_retirent_pas_l_outil(self, tmp_path: Path) -> None:
        """Le cas du vrai modèle : trois mauvaises entrées de suite, l'outil juste reste."""
        b, outil = construire(tmp_path, POLI, retire_after_errors=2)
        for n in (-1, -2, -3, -4):
            _echoue(outil, n=n)
        assert b.catalogue()["doubler"]["retired"] is False and b.load_library() != []

    def test_un_refus_ne_touche_pas_a_la_serie_d_erreurs(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, POLI, retire_after_errors=2)
        _echoue(outil, n=5000)                  # un vrai plantage
        _echoue(outil, n=-1)                    # un refus : ni remise à zéro…
        assert b.catalogue()["doubler"]["consecutive_errors"] == 1
        _echoue(outil, n=5000)                  # …ni oubli : deux plantages, l'outil est retiré
        assert b.catalogue()["doubler"]["retired"] is True

    def test_un_plantage_ordinaire_reste_un_plantage(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, POLI)
        _echoue(outil, n=5000)
        assert b.calls("doubler")[0]["kind"] == "exception"

    @pytest.mark.parametrize("texte", [
        "raise RuntimeError('invalid input: x')",          # pas une ValueError
        "raise ValueError('the input is invalid input: x')",   # ne COMMENCE pas par la formule
        "raise ValueError('invalid: x')",
    ])
    def test_seule_la_formule_exacte_compte(self, tmp_path: Path, texte: str) -> None:
        code = f"def run(args, context):\n    {texte}\n"
        b, outil = construire(tmp_path, payload(code, "strict", ENTIER))
        _echoue(outil, n=1)
        assert b.calls("strict")[0]["kind"] == "exception"

    def test_la_casse_de_la_formule_est_libre(self, tmp_path: Path) -> None:
        code = "def run(args, context):\n    raise ValueError('Invalid input: vide')\n"
        b, outil = construire(tmp_path, payload(code, "majuscule", ENTIER))
        _echoue(outil, n=1)
        assert b.calls("majuscule")[0]["kind"] == "refused"

    def test_un_echec_de_l_hote_reste_un_echec_de_l_hote(self, tmp_path: Path) -> None:
        """L'issue `host` est comptée côté hôte : un refus déclaré par l'outil ne la masque pas."""
        code = ("def run(args, context):\n"
                "    try:\n"
                "        context['call_host']('lire', {'cle': 'a'})\n"
                "    except Exception:\n"
                "        raise ValueError('invalid input: je maquille un echec de l hote')\n")

        def lire(cle: str) -> str:
            raise KeyError(cle)

        b, outil = construire(tmp_path, payload(code, "maquilleur", ENTIER), host_functions={"lire": lire})
        _echoue(outil, n=1)
        assert b.calls("maquilleur")[0]["kind"] == "host"


class TestLeRefusResteVisibleEtJugeable:
    def test_la_fiche_le_montre_sans_le_compter_comme_un_defaut(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, POLI)
        outil(n=2)
        _echoue(outil, n=-1)
        fiche = b.health("doubler")
        assert fiche.outcomes == {"ok": 1, "refused": 1} and fiche.refused_calls == 1
        assert (fiche.judged, fiche.defects, fiche.defect_rate) == (1, 0, 0.0)
        assert "1 refused" in fiche.summary()

    def test_l_hote_peut_juger_un_refus_a_tort(self, tmp_path: Path) -> None:
        """Un outil qui refuserait tout ne se cache pas : l'hôte dit que l'entrée était valide."""
        b, outil = construire(tmp_path, POLI)
        _echoue(outil, n=-1)
        b.report("doubler", b.calls("doubler")[0]["id"], correct=False, reason="-1 devait donner -2")
        fiche = b.health("doubler")
        assert (fiche.judged, fiche.defects) == (1, 1) and "-1 devait donner -2" in fiche.errors_seen[0]["error"]

    def test_sans_carnet_un_refus_n_entre_pas_dans_le_taux(self, tmp_path: Path) -> None:
        b, outil = construire(tmp_path, POLI, record_calls=0)
        outil(n=1)
        _echoue(outil, n=-1)
        fiche = b.health("doubler")
        assert (fiche.judged, fiche.defects, fiche.refused_calls) == (1, 0, 1)


def test_le_modele_constructeur_est_prevenu(tmp_path: Path) -> None:
    consigne = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path)._system_prompt()
    assert 'ValueError("invalid input: ' in consigne and "never return an error value" in consigne
