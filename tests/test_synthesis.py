"""`synthesize_tool` (0.21.0) — le modèle propose, les cas de l'hôte tranchent.

Ce que ces tests verrouillent, dans l'ordre d'importance :

  1. LES CAS CACHÉS NE SORTENT JAMAIS. Ni dans la demande, ni dans un retour
     d'échec. C'est la seule chose qui empêche le modèle d'apprendre le jeu par
     cœur — vérifié en inspectant CHAQUE requête envoyée au fournisseur.
  2. UN OUTIL QUI SUR-APPREND EST REJETÉ. Un outil qui traite les cas montrés un
     par un passe les montrés et rate les cachés : il doit être jeté, et son
     fichier avec.
  3. LE RETOUR D'ÉCHEC SUR LES CACHÉS DIT COMBIEN, JAMAIS LESQUELS.
  4. La coupure est déterministe, laisse au moins un cas montré, et un seul
     exemple = rien de caché (et le résultat le dit).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from autoagent import Agent, Example, synthesize_tool
from autoagent.dynamic import DynamicToolBuilder
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, TokenUsage
from autoagent.synthesis import _split


class Scripte:
    """Rend une SÉQUENCE de payloads (un par essai) et garde chaque requête."""

    def __init__(self, payloads: list[str]) -> None:
        self.config = ModelConfig(provider="fake", model="fake")
        self.payloads = list(payloads)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        contenu = self.payloads.pop(0) if self.payloads else self.payloads_epuises()
        return LLMResponse(content=contenu, model="fake",
                           usage=TokenUsage(input_tokens=100, output_tokens=40))

    @staticmethod
    def payloads_epuises() -> str:
        return "pas du json"


def _payload(code: str, name: str = "doubler") -> str:
    return json.dumps({
        "tool": {"name": name, "description": "double", "permissions": [],
                 "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}},
                                  "required": ["x"]}},
        "code": textwrap.dedent(code).strip(),
        "self_tests": [],
    })


# Dix exemples de la règle « x -> 2x ».
EXEMPLES = [Example({"x": i}, i * 2) for i in range(1, 11)]

GENERAL = """
    def run(args, context):
        return args["x"] * 2
"""


def _surappris(cas_montres: list[Example]) -> str:
    """Un outil qui ne connaît QUE les cas montrés — le piège classique."""
    branches = "\n".join(
        f"    if args['x'] == {ex.args['x']}: return {ex.expected}" for ex in cas_montres)
    return f"def run(args, context):\n{branches}\n    return None\n"


def _builder(tmp_path: Path, provider: Scripte) -> DynamicToolBuilder:
    return DynamicToolBuilder(provider, tools_dir=tmp_path, timeout=15)


def _capability(message) -> str:  # type: ignore[no-untyped-def]
    """Le constructeur encapsule la demande dans un JSON (`_user_prompt`), donc
    les guillemets y sont ÉCHAPPÉS : chercher `{"x": 4} -> 8` en clair dans le
    brut passait À VIDE. On décode pour que le contrôle porte sur le texte que
    le modèle lit vraiment."""
    try:
        return str(json.loads(message.content).get("capability", ""))
    except (ValueError, AttributeError):
        return message.content


def _textes(provider: Scripte, index: int | None = None) -> str:
    """Tout ce que le modèle a LU (ou la seule requête `index`), décodé."""
    requetes = provider.requests if index is None else [provider.requests[index]]
    return "\n".join(_capability(m) for r in requetes for m in r.messages)


class TestCoupure:
    def test_deterministe_et_au_moins_un_montre(self) -> None:
        a = _split(EXEMPLES, 0.4, seed=7)
        b = _split(EXEMPLES, 0.4, seed=7)
        assert a == b
        montres, caches = a
        assert len(caches) == 4 and len(montres) == 6
        assert not set(map(id, montres)) & set(map(id, caches))

    def test_holdout_extreme_laisse_un_montre(self) -> None:
        montres, caches = _split(EXEMPLES, 1.0, seed=0)
        assert len(montres) == 1 and len(caches) == 9

    def test_un_seul_exemple_rien_de_cache(self, tmp_path: Path) -> None:
        provider = Scripte([_payload(GENERAL)])
        res = synthesize_tool(_builder(tmp_path, provider), "double", [Example({"x": 3}, 6)])
        assert res.accepted and res.holdout == 0 and res.shown == 1


class TestLesCachesNeSortentJamais:
    @pytest.mark.timeout(60)
    def test_aucune_requete_ne_contient_un_cas_cache(self, tmp_path: Path) -> None:
        montres, caches = _split(EXEMPLES, 0.4, seed=0)
        provider = Scripte([_payload(_surappris(montres)), _payload(GENERAL)])
        res = synthesize_tool(_builder(tmp_path, provider), "double x", EXEMPLES, seed=0)
        assert res.accepted and len(res.attempts) == 2

        tout = _textes(provider)
        for ex in caches:
            ligne = f'{json.dumps(ex.args)} -> {json.dumps(ex.expected)}'
            assert ligne not in tout, f"le cas caché {ex} a fuité vers le modèle"
        for ex in montres:
            assert f'{json.dumps(ex.args)} -> {json.dumps(ex.expected)}' in tout


class TestSurapprentissage:
    @pytest.mark.timeout(60)
    def test_l_outil_par_coeur_est_rejete_puis_le_general_accepte(self, tmp_path: Path) -> None:
        montres, caches = _split(EXEMPLES, 0.4, seed=0)
        provider = Scripte([_payload(_surappris(montres)), _payload(GENERAL)])
        res = synthesize_tool(_builder(tmp_path, provider), "double x", EXEMPLES, seed=0)

        premier, second = res.attempts
        assert premier.shown_passed == 6 and premier.holdout_passed == 0
        assert not premier.accepted
        assert second.accepted and second.holdout_passed == 4
        assert res.tool is not None and res.tool(x=21) == 42

    @pytest.mark.timeout(60)
    def test_le_retour_dit_combien_jamais_lesquels(self, tmp_path: Path) -> None:
        montres, caches = _split(EXEMPLES, 0.4, seed=0)
        provider = Scripte([_payload(_surappris(montres)), _payload(GENERAL)])
        synthesize_tool(_builder(tmp_path, provider), "double x", EXEMPLES, seed=0)
        seconde_demande = _textes(provider, 1)
        assert "4 of 4 UNSEEN cases fail" in seconde_demande
        for ex in caches:
            assert json.dumps(ex.args) not in seconde_demande.split("rejected")[-1]

    @pytest.mark.timeout(60)
    def test_un_outil_refuse_ne_reste_pas_sur_le_disque(self, tmp_path: Path) -> None:
        montres, _ = _split(EXEMPLES, 0.4, seed=0)
        provider = Scripte([_payload(_surappris(montres), name="rate")])
        res = synthesize_tool(_builder(tmp_path, provider), "double x", EXEMPLES,
                              seed=0, max_attempts=1)
        assert not res.accepted
        assert not (tmp_path / "rate.py").exists(), "un outil non validé est resté chargeable"


class TestBornesEtRobustesse:
    @pytest.mark.timeout(60)
    def test_max_attempts_est_une_borne_dure(self, tmp_path: Path) -> None:
        montres, _ = _split(EXEMPLES, 0.4, seed=0)
        provider = Scripte([_payload(_surappris(montres))] * 10)
        res = synthesize_tool(_builder(tmp_path, provider), "double x", EXEMPLES,
                              seed=0, max_attempts=3)
        assert not res.accepted and len(res.attempts) == 3
        assert len(provider.requests) == 3

    @pytest.mark.timeout(60)
    def test_une_construction_ratee_compte_comme_un_essai(self, tmp_path: Path) -> None:
        provider = Scripte(["ceci n'est pas du JSON", _payload(GENERAL)])
        res = synthesize_tool(_builder(tmp_path, provider), "double x", EXEMPLES, seed=0)
        assert res.accepted and len(res.attempts) == 2
        assert res.attempts[0].error and res.attempts[0].tool_name is None
        assert "could not be built" in _textes(provider, 1)

    @pytest.mark.timeout(60)
    def test_les_cas_montres_rates_sont_renvoyes_avec_leur_contenu(self, tmp_path: Path) -> None:
        faux = "def run(args, context):\n    return args['x'] * 3\n"
        provider = Scripte([_payload(faux), _payload(GENERAL)])
        res = synthesize_tool(_builder(tmp_path, provider), "double x", EXEMPLES,
                              seed=0, feedback_cases=2)
        assert res.accepted
        retour = _textes(provider, 1)
        assert "of the shown examples failed" in retour and "expected=" in retour and "got=" in retour

    @pytest.mark.timeout(60)
    def test_enregistrement_sur_l_agent_et_cout(self, tmp_path: Path) -> None:
        provider = Scripte([_payload(GENERAL)])
        agent = Agent(provider)
        res = synthesize_tool(_builder(tmp_path, provider), "double x", EXEMPLES,
                              seed=0, register_on=agent)
        assert "doubler" in agent.registry
        assert res.usage is not None and res.usage.input_tokens == 100

    def test_exemples_en_tuples_acceptes(self, tmp_path: Path) -> None:
        provider = Scripte([_payload(GENERAL)])
        res = synthesize_tool(_builder(tmp_path, provider), "double x",
                              [({"x": 1}, 2), ({"x": 2}, 4)], seed=0)
        assert res.accepted

    def test_sans_exemple_refus_net(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            synthesize_tool(_builder(tmp_path, Scripte([])), "x", [])

    def test_le_fournisseur_du_builder_est_restaure(self, tmp_path: Path) -> None:
        provider = Scripte([_payload(GENERAL)])
        b = _builder(tmp_path, provider)
        synthesize_tool(b, "double x", EXEMPLES, seed=0)
        assert b.provider is provider
