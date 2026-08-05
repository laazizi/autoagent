"""Réparation d'un JSON d'extraction TRONQUÉ (0.18.0).

BUG TROUVÉ EN CONDITIONS RÉELLES (Gemini 3.5, août 2026) : la réponse
d'extraction de faits arrive amputée de son accolade finale —
`{"operations": [ {...} ]` — de façon reproductible et **sans lien avec
max_tokens** (48 jetons de sortie sur un plafond de 800, sortie identique à
2048).

Conséquence avant correctif : `json.loads` échouait, TOUTES les opérations du tour
étaient abandonnées en silence, donc **la contradiction était perdue** et la
mémoire continuait de servir le fait périmé comme courant. Le pire mode d'échec
possible pour une mémoire d'agent.

Asymétrie assumée : on répare l'EXTRACTION (perdre une opération se rattrape au
tour suivant) mais JAMAIS l'oubli (`[123]` tronqué en `[12]` donne un id valide
mais faux → destruction d'une donnée innocente).
"""

from __future__ import annotations

from autoagent.memory import _parse_forget_ids, _parse_operations, _repair_json_tail

# La charge EXACTE observée en réel (accolade finale manquante).
REEL = ('{\n  "operations": [\n    {\n      "op": "update",\n      "id": 2,\n'
        '      "fact": "L\'utilisateur souhaite être rappelé le matin."\n    }\n  ]')


class TestReparateur:
    def test_le_cas_reel(self) -> None:
        repare, coupe = _repair_json_tail(REEL)
        assert coupe is False                      # la coupe est hors chaîne
        import json
        data = json.loads(repare)
        assert data["operations"][0]["fact"].endswith("le matin.")

    def test_coupe_en_pleine_chaine_est_signalee(self) -> None:
        repare, coupe = _repair_json_tail('{"operations": [{"op": "add", "fact": "à moiti')
        assert coupe is True
        import json
        json.loads(repare)                          # réparable, mais signalé douteux

    def test_json_complet_rend_none(self) -> None:
        assert _repair_json_tail('{"operations": []}') is None

    def test_delimiteurs_incoherents_rend_none(self) -> None:
        """Ce n'est pas une troncature : on ne bricole pas."""
        assert _repair_json_tail('{"a": [1, 2}]') is None

    def test_accolade_dans_une_chaine_ne_compte_pas(self) -> None:
        """Un `}` littéral à l'intérieur d'une chaîne ne doit pas fermer un bloc."""
        repare, _ = _repair_json_tail('{"operations": [{"fact": "utilise } et ]"}')
        import json
        assert json.loads(repare)["operations"][0]["fact"] == "utilise } et ]"

    def test_guillemet_echappe(self) -> None:
        repare, coupe = _repair_json_tail('{"operations": [{"fact": "il a dit \\"oui\\""}]')
        assert coupe is False
        import json
        assert json.loads(repare)["operations"][0]["fact"] == 'il a dit "oui"'


class TestExtraction:
    def test_le_cas_reel_est_desormais_applique(self) -> None:
        """LE test de non-régression : la contradiction n'est plus perdue."""
        ops = _parse_operations(REEL)
        assert len(ops) == 1
        assert ops[0]["op"] == "update" and ops[0]["id"] == 2

    def test_json_complet_inchange(self) -> None:
        ops = _parse_operations('{"operations": [{"op": "add", "fact": "x"}]}')
        assert ops == [{"op": "add", "fact": "x"}]

    def test_dernier_element_ecarte_si_coupe_en_valeur(self) -> None:
        """Un fait au texte amputé serait pire que pas de fait."""
        ops = _parse_operations(
            '{"operations": [{"op": "add", "fact": "complet"}, '
            '{"op": "add", "fact": "amput')
        assert len(ops) == 1
        assert ops[0]["fact"] == "complet"

    def test_vraie_absence_de_json_reste_un_no_op(self) -> None:
        assert _parse_operations("je pense qu'il faudrait ajouter un fait") == []

    def test_fences_toujours_gerees(self) -> None:
        ops = _parse_operations('```json\n{"operations": [{"op": "add", "fact": "y"}]}\n```')
        assert ops == [{"op": "add", "fact": "y"}]

    def test_fence_ET_troncature(self) -> None:
        ops = _parse_operations('```json\n{"operations": [{"op": "delete", "id": 3}]')
        assert ops == [{"op": "delete", "id": 3}]


class TestOubliResteStrict:
    def test_aucune_reparation_sur_une_suppression(self) -> None:
        """`[123]` tronqué en `[12]` supprimerait un fait INNOCENT."""
        assert _parse_forget_ids('{"forget": [12') == set()

    def test_json_complet_fonctionne(self) -> None:
        assert _parse_forget_ids('{"forget": [1, 3]}') == {1, 3}
