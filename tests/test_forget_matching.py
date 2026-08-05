"""`forget_matching` + `register_forget_tool` — oubli en langue naturelle (0.18.0).

Jusqu'ici la seule décision confiée au LLM était l'ÉCRITURE (extraction dans
`compact`) ; l'oubli côté hôte se limitait à `forget(fact_id)` — un entier. Aucun
chemin pour « oublie tout ce qui concerne mon ancien employeur ».

Les architectures « décision à l'écriture seule » échouent sur la suppression
INTENTIONNELLE : collision de préfixe, faits composés, variantes d'identifiants.
Déplacer la décision au moment de la MUTATION récupère ces cas.

Contrat de sûreté vérifié ici : **fail-CLOSED**. Sur une suppression, tout doute
(LLM en panne, JSON non conforme, id hors du lot soumis) ⇒ on ne supprime RIEN.
"""

from __future__ import annotations

import json

from autoagent import Agent
from autoagent.memory import FactMemory
from autoagent.schema import LLMResponse, ToolCall

from .conftest import FakeLLMProvider


class _ProviderOubli:
    """Rend une décision d'oubli fixée d'avance (aucun vrai LLM)."""

    def __init__(self, contenu: str) -> None:
        self.contenu = contenu
        self.calls: list = []
        from autoagent.schema import ModelConfig
        self.config = ModelConfig(provider="fake", model="fake")

    def complete(self, request):
        self.calls.append(request)
        return LLMResponse(content=self.contenu, model="fake")


class _ProviderQuiPlante:
    def __init__(self) -> None:
        from autoagent.schema import ModelConfig
        self.config = ModelConfig(provider="fake", model="fake")

    def complete(self, request):
        raise RuntimeError("quota dépassé")


def _memoire(provider, faits: list[str], **kwargs) -> FactMemory:
    m = FactMemory(provider, **kwargs)
    for f in faits:
        m.remember(f)
    return m


_FAITS = [
    "Le client travaille chez Acme Corp.",             # id 1
    "Le client aime le thé vert.",                     # id 2
    "L'adresse pro du client est 12 rue Acme, Lyon.",  # id 3
]


class TestSuppression:
    def test_supprime_les_ids_designes(self) -> None:
        m = _memoire(_ProviderOubli('{"forget": [1, 3], "reason": "employeur"}'), _FAITS)
        supprimes = m.forget_matching("oublie tout ce qui concerne mon employeur Acme")
        assert sorted(f["id"] for f in supprimes) == [1, 3]
        assert [f["id"] for f in m.facts()] == [2]

    def test_retourne_les_faits_complets_pour_l_audit(self) -> None:
        """Preuve d'effacement : on renvoie le contenu, pas juste des ids."""
        m = _memoire(_ProviderOubli('{"forget": [2]}'), _FAITS)
        supprimes = m.forget_matching("oublie ses préférences de boisson")
        assert supprimes[0]["fact"] == "Le client aime le thé vert."

    def test_dry_run_ne_touche_a_rien(self) -> None:
        m = _memoire(_ProviderOubli('{"forget": [1, 3]}'), _FAITS)
        prevus = m.forget_matching("oublie Acme", dry_run=True)
        assert len(prevus) == 2
        assert len(m.facts()) == 3          # rien n'a bougé

    def test_aucune_correspondance(self) -> None:
        m = _memoire(_ProviderOubli('{"forget": [], "reason": "rien"}'), _FAITS)
        assert m.forget_matching("oublie ses voyages en Islande") == []
        assert len(m.facts()) == 3

    def test_base_vide_sans_appel_llm(self) -> None:
        prov = _ProviderOubli('{"forget": [1]}')
        m = FactMemory(prov)
        assert m.forget_matching("oublie tout") == []
        assert prov.calls == []              # pas d'appel inutile

    def test_instruction_vide_refusee(self) -> None:
        import pytest
        m = _memoire(_ProviderOubli('{"forget": []}'), _FAITS)
        with pytest.raises(ValueError, match="instruction"):
            m.forget_matching("   ")

    def test_les_vecteurs_sont_purges_aussi(self) -> None:
        """Un fait supprimé ne doit pas laisser son embedding derrière lui."""
        m = _memoire(_ProviderOubli('{"forget": [1]}'), _FAITS,
                     embed_fn=lambda textes: [[1.0, 0.0] for _ in textes])
        m.recall("client")                   # peuple les vecteurs
        assert 1 in m._vectors
        m.forget_matching("oublie Acme")
        assert 1 not in m._vectors


class TestFailClosed:
    def test_llm_en_panne_ne_supprime_rien(self) -> None:
        m = _memoire(_ProviderQuiPlante(), _FAITS)
        assert m.forget_matching("oublie tout ce qui concerne Acme") == []
        assert len(m.facts()) == 3           # intact

    def test_json_non_conforme_ne_supprime_rien(self) -> None:
        m = _memoire(_ProviderOubli("je pense qu'il faut oublier le fait 1"), _FAITS)
        assert m.forget_matching("oublie Acme") == []
        assert len(m.facts()) == 3

    def test_id_inconnu_ignore(self) -> None:
        m = _memoire(_ProviderOubli('{"forget": [1, 999]}'), _FAITS)
        supprimes = m.forget_matching("oublie Acme")
        assert [f["id"] for f in supprimes] == [1]
        assert len(m.facts()) == 2

    def test_booleen_deguise_en_id_ignore(self) -> None:
        """`json.loads` rend True pour `true` ; True == 1 en Python → piège."""
        m = _memoire(_ProviderOubli('{"forget": [true]}'), _FAITS)
        assert m.forget_matching("oublie Acme") == []
        assert len(m.facts()) == 3

    def test_ids_en_chaine_acceptes(self) -> None:
        """Tolérance raisonnable : un modèle qui rend "2" au lieu de 2."""
        m = _memoire(_ProviderOubli('{"forget": ["2"]}'), _FAITS)
        assert [f["id"] for f in m.forget_matching("oublie le thé")] == [2]


class TestOutilExpose:
    def _agent(self, memoire, **kwargs) -> Agent:
        agent = Agent(FakeLLMProvider([
            LLMResponse(tool_calls=[ToolCall(id="c1", name="forget",
                                            arguments={"instruction": "oublie Acme"})]),
            "c'est noté",
        ]), max_steps=4, memory=memoire)
        agent.register_forget_tool(**kwargs)
        return agent

    def test_par_defaut_l_outil_est_un_dry_run(self) -> None:
        """Supprimer des données d'un utilisateur sur la seule décision d'un
        modèle n'est pas un défaut acceptable pour une bibliothèque."""
        m = _memoire(_ProviderOubli('{"forget": [1, 3]}'), _FAITS)
        res = self._agent(m).run("oublie mon employeur")
        charge = json.loads(next(x for x in res.messages if x.role == "tool").content)["result"]
        assert charge["pending_confirmation"] is True
        assert charge["erased"] is False
        assert charge["count"] == 2
        assert len(m.facts()) == 3           # rien n'est parti

    def test_confirm_false_supprime_vraiment(self) -> None:
        m = _memoire(_ProviderOubli('{"forget": [1]}'), _FAITS)
        res = self._agent(m, confirm=False).run("oublie mon employeur")
        charge = json.loads(next(x for x in res.messages if x.role == "tool").content)["result"]
        assert charge["erased"] is True
        assert len(m.facts()) == 2

    def test_no_op_si_la_memoire_ne_sait_pas_oublier(self) -> None:
        from autoagent.memory import BufferMemory
        agent = Agent(FakeLLMProvider(["ok"]), memory=BufferMemory())
        agent.register_forget_tool()
        assert "forget" not in [s.name for s in agent.registry.specs()]

    def test_no_op_sans_memoire(self) -> None:
        agent = Agent(FakeLLMProvider(["ok"]))
        agent.register_forget_tool()
        assert agent.registry.specs() == []
