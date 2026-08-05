"""Bi-temporalité + provenance de `FactMemory` (0.18.0).

Avant : `update` **écrasait** le texte du fait en place. Trois conséquences que la
recherche 2026 documente comme le mode d'échec n°1 des mémoires d'agent :
  * une extraction LLM ratée DÉTRUISAIT silencieusement une donnée juste ;
  * impossible de répondre « depuis quand ? » ;
  * impossible d'arbitrer entre une déclaration de l'utilisateur et une inférence
    de l'agent (aucune provenance).

Désormais une contradiction FERME la fenêtre de validité de l'ancien fait et en
crée un nouveau. Invariant : on ne sert JAMAIS un fait périmé comme courant.

⚠️ Ces tests couvrent aussi la MIGRATION : des fichiers de faits écrits par les
versions 0.12→0.17 existent en production et doivent se charger sans perte.
"""

from __future__ import annotations

import json

from autoagent.memory import FactMemory, _migrate_fact
from autoagent.schema import LLMResponse


class _ProviderScripte:
    """Rend des opérations d'extraction fixées d'avance."""

    def __init__(self, *reponses: str) -> None:
        self.reponses = list(reponses)
        from autoagent.schema import ModelConfig
        self.config = ModelConfig(provider="fake", model="fake")

    def complete(self, request):
        contenu = self.reponses.pop(0) if self.reponses else '{"operations": []}'
        return LLMResponse(content=contenu, model="fake")


def _memoire(provider=None, **kw) -> FactMemory:
    return FactMemory(provider or _ProviderScripte(), **kw)


class TestSupersession:
    def test_l_ancien_fait_survit_marque_perime(self) -> None:
        m = _memoire()
        m.remember("Le client préfère être appelé le soir.")
        m._apply_operations([{"op": "update", "id": 1,
                              "fact": "Le client préfère être appelé le matin."}])
        tous = m.facts(include_invalid=True)
        assert len(tous) == 2
        ancien = next(f for f in tous if f["id"] == 1)
        assert ancien["invalid_at"] is not None       # fenêtre fermée
        assert ancien["superseded_by"] == 2           # pointe vers son successeur
        assert "soir" in ancien["fact"]               # texte d'origine INTACT

    def test_facts_ne_montre_que_le_courant_par_defaut(self) -> None:
        """Rétrocompatibilité : c'est ce que voyaient les consommateurs avant."""
        m = _memoire()
        m.remember("Le client préfère le soir.")
        m._apply_operations([{"op": "update", "id": 1, "fact": "Le client préfère le matin."}])
        courants = m.facts()
        assert len(courants) == 1
        assert "matin" in courants[0]["fact"]

    def test_un_fait_perime_n_est_jamais_remonte_par_recall(self) -> None:
        """L'invariant central : ne jamais servir un fait périmé comme courant."""
        m = _memoire()
        m.remember("Le rendez-vous est fixé au mardi.")
        m._apply_operations([{"op": "update", "id": 1,
                              "fact": "Le rendez-vous est fixé au jeudi."}])
        trouve = m.recall("rendez-vous", k=5)
        assert len(trouve) == 1
        assert "jeudi" in trouve[0].content
        assert "mardi" not in trouve[0].content

    def test_un_fait_perime_n_est_pas_injecte_dans_le_contexte(self) -> None:
        from autoagent.schema import Message
        m = _memoire()
        m.remember("Adresse : 3 rue du Port.")
        m._apply_operations([{"op": "update", "id": 1, "fact": "Adresse : 8 avenue Foch."}])
        assemble = m._assemble([Message(role="system", content="sys")], [])
        bloc = assemble[1].content
        assert "avenue Foch" in bloc and "rue du Port" not in bloc

    def test_le_vecteur_du_perime_est_purge(self) -> None:
        m = _memoire(embed_fn=lambda ts: [[1.0, 0.0] for _ in ts])
        m.remember("Le client aime le thé.")
        m.recall("thé")                                   # peuple les vecteurs
        assert 1 in m._vectors
        m._apply_operations([{"op": "update", "id": 1, "fact": "Le client aime le café."}])
        assert 1 not in m._vectors                        # inutile : jamais remonté

    def test_pas_de_chainage_sur_un_fait_deja_perime(self) -> None:
        """Deux updates sur le MÊME id dans un lot : la 2e est périmée, ignorée."""
        m = _memoire()
        m.remember("Version 1.")
        m._apply_operations([
            {"op": "update", "id": 1, "fact": "Version 2."},
            {"op": "update", "id": 1, "fact": "Version 3."},   # stale
        ])
        assert [f["fact"] for f in m.facts()] == ["Version 2."]

    def test_le_sujet_est_herite(self) -> None:
        m = _memoire()
        m.remember("Rappel à 18h.", subject="rdv")
        m._apply_operations([{"op": "update", "id": 1, "fact": "Rappel à 9h."}])
        assert m.facts()[0]["subject"] == "rdv"


class TestHistorique:
    def test_chaine_complete_dans_l_ordre(self) -> None:
        m = _memoire()
        m.remember("Habite à Lyon.")
        m._apply_operations([{"op": "update", "id": 1, "fact": "Habite à Paris."}])
        m._apply_operations([{"op": "update", "id": 2, "fact": "Habite à Nantes."}])
        chaine = [f["fact"] for f in m.history(1)]
        assert chaine == ["Habite à Lyon.", "Habite à Paris.", "Habite à Nantes."]

    def test_accessible_depuis_n_importe_quel_maillon(self) -> None:
        m = _memoire()
        m.remember("A.")
        m._apply_operations([{"op": "update", "id": 1, "fact": "B."}])
        m._apply_operations([{"op": "update", "id": 2, "fact": "C."}])
        assert [f["fact"] for f in m.history(3)] == ["A.", "B.", "C."]

    def test_fait_inconnu(self) -> None:
        assert _memoire().history(999) == []

    def test_tolere_un_maillon_supprime_par_forget(self) -> None:
        """`forget` supprime DUREMENT (droit à l'effacement) → pointeur pendant."""
        m = _memoire()
        m.remember("A.")
        m._apply_operations([{"op": "update", "id": 1, "fact": "B."}])
        m.forget(2)                                    # le successeur disparaît
        chaine = m.history(1)
        assert [f["fact"] for f in chaine] == ["A."]   # s'arrête proprement


class TestProvenance:
    def test_remember_est_declare_host_par_defaut(self) -> None:
        m = _memoire()
        assert m.remember("Fait posé par le code hôte.")["source"] == "host"

    def test_source_explicite(self) -> None:
        m = _memoire()
        assert m.remember("Déclaré par l'utilisateur.", source="user")["source"] == "user"

    def test_extraction_est_declaree_agent(self) -> None:
        m = _memoire()
        m._apply_operations([{"op": "add", "fact": "Déduit par l'agent."}])
        assert m.facts()[0]["source"] == "agent"

    def test_source_invalide_repliee_sur_agent(self) -> None:
        m = _memoire()
        assert m.remember("x", source="n'importe quoi")["source"] == "agent"


class TestEviction:
    def test_les_perimes_partent_en_premier(self) -> None:
        """Sans cette priorité, la bi-temporalité évincerait des faits COURANTS
        pour garder des périmés — exactement l'inverse du but."""
        m = _memoire(max_facts=3)
        m.remember("Fait A.")
        m._apply_operations([{"op": "update", "id": 1, "fact": "Fait A bis."}])  # 1 périmé
        m.remember("Fait B.")
        m.remember("Fait C.")          # dépasse max_facts=3 → le périmé doit partir
        restants = m.facts(include_invalid=True)
        assert len(restants) == 3
        assert all(f.get("invalid_at") is None for f in restants)

    def test_la_borne_est_tenue(self) -> None:
        m = _memoire(max_facts=5)
        for i in range(20):
            m.remember(f"Fait numéro {i}.")
        assert len(m.facts(include_invalid=True)) == 5


class TestMigrationAncienFormat:
    """Des fichiers 0.12→0.17 existent EN PRODUCTION : ils doivent se charger."""

    ANCIEN = {
        "facts": [
            {"id": 1, "fact": "Le client habite Lyon.", "subject": "adresse",
             "updated": "2026-07-01"},
            {"id": 2, "fact": "Le client aime le thé.", "subject": None,
             "updated": "2026-07-02"},
        ],
        "next_id": 3,
    }

    def test_migration_unitaire(self) -> None:
        migre = _migrate_fact(dict(self.ANCIEN["facts"][0]))
        assert migre["invalid_at"] is None          # aucun fait périmé
        assert migre["superseded_by"] is None
        assert migre["source"] == "agent"           # provenance inconnue
        assert migre["valid_from"] == "2026-07-01"  # dérivé de `updated`
        assert migre["fact"] == "Le client habite Lyon."   # contenu INTACT

    def test_chargement_d_un_fichier_ancien(self, tmp_path) -> None:
        chemin = tmp_path / "faits.json"
        chemin.write_text(json.dumps(self.ANCIEN, ensure_ascii=False), encoding="utf-8")
        m = _memoire(path=chemin)
        assert len(m.facts()) == 2                  # les deux sont COURANTS
        assert m.recall("Lyon", k=1)                # et retrouvables
        assert m._next_id == 3

    def test_le_fichier_est_reecrit_au_format_complet(self, tmp_path) -> None:
        chemin = tmp_path / "faits.json"
        chemin.write_text(json.dumps(self.ANCIEN, ensure_ascii=False), encoding="utf-8")
        m = _memoire(path=chemin)
        m.remember("Nouveau fait.")                 # déclenche une sauvegarde
        recharge = json.loads(chemin.read_text(encoding="utf-8"))
        for fait in recharge["facts"]:
            assert "invalid_at" in fait and "source" in fait and "valid_from" in fait

    def test_supersession_apres_migration(self, tmp_path) -> None:
        """Le scénario réel : un fichier ancien, puis une contradiction."""
        chemin = tmp_path / "faits.json"
        chemin.write_text(json.dumps(self.ANCIEN, ensure_ascii=False), encoding="utf-8")
        m = _memoire(path=chemin)
        m._apply_operations([{"op": "update", "id": 1, "fact": "Le client habite Marseille."}])
        assert [f["fact"] for f in m.facts()] == ["Le client aime le thé.",
                                                  "Le client habite Marseille."]
        assert "Lyon" in m.history(1)[0]["fact"]    # l'historique existe désormais

    def test_champ_source_inconnu_dans_le_fichier(self) -> None:
        migre = _migrate_fact({"id": 9, "fact": "x", "source": "extraterrestre"})
        assert migre["source"] == "agent"

    def test_fichier_deja_au_nouveau_format_inchange(self) -> None:
        deja = {"id": 4, "fact": "y", "subject": None, "updated": "2026-08-01",
                "source": "user", "valid_from": "2026-07-15",
                "invalid_at": "2026-08-01", "superseded_by": 5}
        assert _migrate_fact(dict(deja)) == deja
