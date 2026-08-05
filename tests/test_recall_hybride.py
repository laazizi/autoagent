"""Recall hybride BM25 + fusion RRF dans `FactMemory` (0.18.0).

Avant : `recall()` était un OU exclusif — soit cosinus pur (si `embed_fn`), soit un
repli qui n'était pas un algorithme de retrieval mais une intersection d'ensembles
de mots sur `.split()` : ni IDF, ni normalisation de longueur, ni tokenisation
(« crêpes, » ne matchait pas « crêpes »), et un seuil à 3 caractères jetait « n° »,
« TVA », « ok ».

Les deux signaux échouent sur des requêtes OPPOSÉES : le sémantique perd les
correspondances exactes (n° de contrat, SIREN, plaque), le lexical perd les
synonymes. La fusion RRF rattrape les deux angles morts.
"""

from __future__ import annotations

from autoagent.memory import FactMemory, _rrf_fuse, _tokenize


class _ProviderMuet:
    """FactMemory exige un provider, mais aucun test ici n'appelle le LLM."""

    def complete(self, request):  # pragma: no cover
        raise AssertionError("recall ne doit JAMAIS appeler le LLM")


def _memoire(faits: list[str], **kwargs) -> FactMemory:
    m = FactMemory(_ProviderMuet(), **kwargs)
    for f in faits:
        m.remember(f)
    return m


def _ids(messages) -> list[int]:
    return [int(msg.content.split("#")[1].split("]")[0]) for msg in messages]


class TestTokenisation:
    def test_ponctuation_ne_casse_plus_les_mots(self) -> None:
        assert "crêpes" in _tokenize("Recette de crêpes, sucrées.")

    def test_accents_preserves(self) -> None:
        assert _tokenize("héberger") == ["héberger"]

    def test_mots_de_deux_caracteres_gardes(self) -> None:
        """« n° », « ok », « tva » comptent — l'ancien seuil à 3 les jetait."""
        assert "ok" in _tokenize("c'est ok pour moi")

    def test_vide(self) -> None:
        assert _tokenize("") == [] and _tokenize(None) == []


class TestBM25:
    def test_terme_rare_pese_plus_que_terme_commun(self) -> None:
        """L'IDF : ce que l'intersection de mots ne savait pas faire."""
        m = _memoire([
            "Le client aime le café.",
            "Le client aime le thé.",
            "Le client possède un SIREN 812345678.",
            "Le client habite Lyon.",
        ])
        # « client » est dans TOUS les faits (IDF faible) ; « SIREN » dans un seul.
        res = m.recall("client SIREN", k=1)
        assert "812345678" in res[0].content

    def test_identifiant_exact_retrouve(self) -> None:
        m = _memoire([
            "Le contrat de M. Martin porte le numéro ALY-2026-0042.",
            "M. Martin préfère être appelé le matin.",
        ])
        res = m.recall("ALY-2026-0042", k=1)
        assert "ALY-2026-0042" in res[0].content

    def test_fait_court_pas_noye_par_fait_bavard(self) -> None:
        """Normalisation de longueur : un fait ciblé bat un fait dilué."""
        m = _memoire([
            "Téléphone : 0612345678.",
            ("Compte rendu de la réunion du 3 mars sur la refonte du site, avec de "
             "nombreux détails annexes sans rapport direct, où l'on a évoqué au "
             "passage un téléphone parmi beaucoup d'autres sujets divers."),
        ])
        res = m.recall("téléphone", k=1)
        assert "0612345678" in res[0].content

    def test_aucune_correspondance_rend_vide(self) -> None:
        m = _memoire(["Le client aime le café."])
        assert m.recall("astrophysique quantique") == []

    def test_requete_vide_rend_vide(self) -> None:
        m = _memoire(["Le client aime le café."])
        assert m.recall("") == []

    def test_aucun_appel_llm(self) -> None:
        """Le recall reste gratuit : zéro réseau, zéro token."""
        m = _memoire(["Le client habite Lyon."])
        assert m.recall("Lyon")            # _ProviderMuet lèverait sinon


class TestFusionRRF:
    def test_fusion_de_deux_classements(self) -> None:
        """Propriété de RRF (convexité de 1/(k+r)) : être 1er dans UN classement
        et 3e dans l'autre BAT être 2e dans les deux — 1/61 + 1/63 > 2/62. Un
        signal très confiant est donc récompensé, sans pouvoir balayer l'autre."""
        a = [{"id": 1}, {"id": 2}, {"id": 3}]
        b = [{"id": 3}, {"id": 2}, {"id": 1}]
        fusion = [f["id"] for f in _rrf_fuse(a, b)]
        assert sorted(fusion) == [1, 2, 3]
        assert fusion[-1] == 2                 # 2e partout : dernier ex aequo
        assert set(fusion[:2]) == {1, 3}       # les deux « 1er quelque part »

    def test_un_consensus_bat_un_signal_unique(self) -> None:
        """L'autre moitié du compromis : présent dans les DEUX classements bat
        présent dans un seul, à rang comparable."""
        a = [{"id": 1}, {"id": 2}]
        b = [{"id": 2}]
        fusion = [f["id"] for f in _rrf_fuse(a, b)]
        assert fusion[0] == 2                  # 2 est vu deux fois

    def test_un_seul_classement_conserve_l_ordre(self) -> None:
        a = [{"id": 7}, {"id": 9}]
        assert [f["id"] for f in _rrf_fuse(a)] == [7, 9]

    def test_deduplique(self) -> None:
        a = [{"id": 1}, {"id": 2}]
        b = [{"id": 1}]
        assert len(_rrf_fuse(a, b)) == 2


class TestModes:
    def _memoire_semantique(self, **kwargs) -> FactMemory:
        # embed_fn factice : un vecteur par mot-clé, pour un cosinus prévisible
        def embed(textes: list[str]) -> list[list[float]]:
            out = []
            for t in textes:
                bas = t.lower()
                out.append([
                    1.0 if ("voiture" in bas or "véhicule" in bas) else 0.0,
                    1.0 if "lyon" in bas else 0.0,
                ])
            return out

        return _memoire(["Le client possède deux voitures.",
                         "Le client habite Lyon."], embed_fn=embed, **kwargs)

    def test_hybride_trouve_par_le_sens(self) -> None:
        """« véhicule » n'a AUCUN mot commun avec « voitures » → seul le
        sémantique peut le trouver, et l'hybride doit le laisser passer."""
        m = self._memoire_semantique()
        assert m.recall("véhicule", k=1)[0].content.endswith("deux voitures.")

    def test_hybride_trouve_aussi_par_le_mot_exact(self) -> None:
        m = self._memoire_semantique()
        assert "Lyon" in m.recall("Lyon", k=1)[0].content

    def test_mode_lexical_ignore_les_embeddings(self) -> None:
        m = self._memoire_semantique(recall_mode="lexical")
        assert m.recall("véhicule") == []      # aucun mot commun → rien

    def test_mode_semantique_seul(self) -> None:
        m = self._memoire_semantique(recall_mode="semantic")
        assert m.recall("véhicule", k=1)[0].content.endswith("deux voitures.")

    def test_mode_invalide_refuse(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="recall_mode"):
            FactMemory(_ProviderMuet(), recall_mode="magique")

    def test_embed_fn_en_panne_replie_sur_le_lexical(self) -> None:
        """Contrat inchangé : un embed_fn qui plante ne casse pas le recall."""
        def embed_casse(textes):
            raise RuntimeError("quota dépassé")

        m = _memoire(["Le client habite Lyon."], embed_fn=embed_casse)
        assert "Lyon" in m.recall("Lyon", k=1)[0].content

    def test_sans_embed_fn_bm25_seul_fonctionne(self) -> None:
        """Le gain profite aux projets qui n'ont PAS d'embed_fn."""
        m = _memoire(["Le client habite Lyon.", "Le client aime le café."])
        assert "Lyon" in m.recall("où habite le client ?", k=1)[0].content
