"""Outils dynamiques — lot B (0.22.0) : reprendre un refus, et ne pas tout reconstruire.

Mesuré sur DeepSeek : une fois `re.compile` libéré, ce qui fait encore échouer
une création, ce sont des self-tests dont le MODÈLE a mal calculé la valeur à
la main (« expected 392.4, got 392.2 ») — le code était bon, l'attendu faux.
Un refus sec perd l'outil ; renvoyé au constructeur, il sait lequel des deux
corriger. Et un outil accepté un jour était reconstruit — et repayé — au run
suivant : la bibliothèque persistante le garde, avec ses compteurs.

Les défauts ne bougent pas : `max_repairs=0`, `persist=False`.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from autoagent import Agent, DynamicToolBuilder, ToolBuildRequest
from autoagent.errors import ToolError, ToolValidationError
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, TokenUsage, ToolCall

USAGE = TokenUsage(input_tokens=1000, output_tokens=500)

CODE_OK = "def run(args, context):\n    return {'double': args['n'] * 2}\n"
CODE_CAPRICIEUX = (
    "def run(args, context):\n"
    "    if args['n'] < 0:\n"
    "        raise ValueError('negatif')\n"
    "    return {'double': args['n'] * 2}\n"
)


def _payload(code: str = CODE_OK, tests: Any = None, nom: str = "doubler", permissions: list[str] | None = None) -> str:
    return json.dumps({
        "tool": {"name": nom, "description": "double un entier",
                 "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
                 "permissions": permissions or []},
        "code": code,
        "self_tests": [{"args": {"n": 2}, "expect_equals": {"double": 4}}] if tests is None else tests,
    })


FAUX = _payload(tests=[{"args": {"n": 2}, "expect_equals": {"double": 5}}])    # attendu faux, code bon
BON = _payload()


class Sequence(LLMProvider):
    """Rend ses contenus dans l'ordre et note chaque requête."""

    def __init__(self, contenus: list[str], usage: TokenUsage | None = USAGE) -> None:
        super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
        self.contenus, self.usage = list(contenus), usage
        self.requetes: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requetes.append(request)
        return LLMResponse(content=self.contenus.pop(0), model="f", usage=self.usage)


def _fichiers(dossier: Path) -> list[str]:
    return sorted(p.name for p in dossier.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


# ── La boucle de réparation ─────────────────────────────────────────────────

class TestReparation:
    def test_par_defaut_un_refus_reste_un_echec_en_un_seul_appel(self, tmp_path: Path) -> None:
        fournisseur = Sequence([FAUX, BON])
        with pytest.raises(ToolValidationError, match="Self-test 0 failed"):
            DynamicToolBuilder(fournisseur, tools_dir=tmp_path).build(ToolBuildRequest(capability="x"))
        assert len(fournisseur.requetes) == 1, "max_repairs=0 : comportement historique"

    def test_un_self_test_faux_est_repris_et_l_outil_est_garde(self, tmp_path: Path) -> None:
        fournisseur = Sequence([FAUX, BON])
        outil = DynamicToolBuilder(fournisseur, tools_dir=tmp_path, max_repairs=1).build(
            ToolBuildRequest(capability="x"))
        assert outil(n=21) == {"double": 42}
        assert len(fournisseur.requetes) == 2
        assert (tmp_path / "doubler.py").is_file()

    def test_le_constructeur_revoit_sa_reponse_et_la_raison_du_refus(self, tmp_path: Path) -> None:
        fournisseur = Sequence([FAUX, BON])
        DynamicToolBuilder(fournisseur, tools_dir=tmp_path, max_repairs=1).build(ToolBuildRequest(capability="x"))
        seconde = fournisseur.requetes[1].messages
        assert [m.role for m in seconde] == ["system", "user", "assistant", "user"]
        assert seconde[2].content == FAUX
        assert "Self-test 0 failed" in seconde[3].content
        assert "expected {'double': 5}, got {'double': 4}" in seconde[3].content
        assert "decide which one is wrong" in seconde[3].content

    def test_les_deux_appels_sont_payes(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(Sequence([FAUX, BON]), tools_dir=tmp_path, max_repairs=1)
        b.build(ToolBuildRequest(capability="x"))
        assert (b.last_build_usage.input_tokens, b.last_build_usage.output_tokens) == (2000, 1000)

    def test_usage_inconnu_n_est_pas_invente(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(Sequence([FAUX, BON], usage=None), tools_dir=tmp_path, max_repairs=1)
        b.build(ToolBuildRequest(capability="x"))
        assert b.last_build_usage is None

    def test_reprises_epuisees_l_erreur_est_la_derniere_et_rien_ne_reste(self, tmp_path: Path) -> None:
        autre_faux = _payload(tests=[{"args": {"n": 2}, "expect_equals": {"double": 6}}])
        fournisseur = Sequence([FAUX, autre_faux, BON])
        b = DynamicToolBuilder(fournisseur, tools_dir=tmp_path, max_repairs=1)
        with pytest.raises(ToolValidationError, match=r"expected \{'double': 6\}"):
            b.build(ToolBuildRequest(capability="x"))
        assert len(fournisseur.requetes) == 2
        assert _fichiers(tmp_path) == []
        assert b.last_build_usage.input_tokens == 2000, "les deux tentatives ratées ont coûté"

    def test_le_refus_du_plafond_de_permissions_n_est_jamais_repris(self, tmp_path: Path) -> None:
        reseau = _payload(permissions=["network"])
        fournisseur = Sequence([reseau, BON])
        b = DynamicToolBuilder(fournisseur, tools_dir=tmp_path, allowed_permissions=set(), max_repairs=3)
        with pytest.raises(ToolValidationError, match="not allowed by the host"):
            b.build(ToolBuildRequest(capability="x"))
        assert len(fournisseur.requetes) == 1, "une décision de l'hôte n'est pas un bug à contourner en boucle"

    def test_un_code_refuse_par_le_validateur_est_repris(self, tmp_path: Path) -> None:
        mauvais = _payload(code="def run(args, context):\n    return eval('1')\n", tests=[])
        outil = DynamicToolBuilder(Sequence([mauvais, BON]), tools_dir=tmp_path, max_repairs=1).build(
            ToolBuildRequest(capability="x"))
        assert outil(n=1) == {"double": 2}

    def test_un_json_illisible_est_repris(self, tmp_path: Path) -> None:
        outil = DynamicToolBuilder(Sequence(["pas du json du tout", BON]), tools_dir=tmp_path, max_repairs=1).build(
            ToolBuildRequest(capability="x"))
        assert outil(n=1) == {"double": 2}

    def test_un_plantage_pendant_le_self_test_est_repris(self, tmp_path: Path) -> None:
        plante = _payload(code="def run(args, context):\n    raise RuntimeError('boum')\n")
        outil = DynamicToolBuilder(Sequence([plante, BON]), tools_dir=tmp_path, max_repairs=1).build(
            ToolBuildRequest(capability="x"))
        assert outil(n=1) == {"double": 2}

    def test_max_repairs_negatif_vaut_zero(self, tmp_path: Path) -> None:
        assert DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, max_repairs=-4).max_repairs == 0


class TestFormesInattendues:
    """Une réponse de forme inattendue est une erreur LISIBLE (donc reprenable), pas un AttributeError."""

    def _echec(self, tmp_path: Path, contenu: str, motif: str) -> None:
        with pytest.raises(ToolValidationError, match=motif):
            DynamicToolBuilder(Sequence([contenu]), tools_dir=tmp_path).build(ToolBuildRequest(capability="x"))

    def test_racine_pas_un_objet(self, tmp_path: Path) -> None:
        self._echec(tmp_path, "[1, 2]", "must be a JSON object")

    def test_tool_pas_un_objet(self, tmp_path: Path) -> None:
        self._echec(tmp_path, json.dumps({"tool": "doubler", "code": CODE_OK}), "'tool' must be a JSON object")

    def test_nom_absent(self, tmp_path: Path) -> None:
        sans_nom = json.dumps({"tool": {"description": "d", "input_schema": {"type": "object"}}, "code": CODE_OK})
        self._echec(tmp_path, sans_nom, r"missing keys: \['name'\]")

    def test_permissions_en_texte(self, tmp_path: Path) -> None:
        meta = {"name": "o", "description": "d", "input_schema": {"type": "object"}, "permissions": "network"}
        self._echec(tmp_path, json.dumps({"tool": meta, "code": CODE_OK}), "must be a list of strings")

    def test_input_schema_pas_un_objet(self, tmp_path: Path) -> None:
        meta = {"name": "o", "description": "d", "input_schema": "n: int"}
        self._echec(tmp_path, json.dumps({"tool": meta, "code": CODE_OK}), "must be a JSON-schema object")


# ── La bibliothèque persistante ─────────────────────────────────────────────

def _construire(dossier: Path, contenus: list[str] | None = None, **kw: Any) -> tuple[DynamicToolBuilder, Any]:
    b = DynamicToolBuilder(Sequence(contenus or [BON]), tools_dir=dossier, persist=True, **kw)
    return b, b.build(ToolBuildRequest(capability="x"))


class TestBibliotheque:
    def test_sans_persist_rien_n_est_ecrit_ni_recharge(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(Sequence([BON]), tools_dir=tmp_path)
        b.build(ToolBuildRequest(capability="x"))
        assert _fichiers(tmp_path) == ["doubler.py"]
        assert b.load_library() == [] and b.catalogue() == {}

    def test_la_construction_inscrit_l_outil_au_catalogue(self, tmp_path: Path) -> None:
        import hashlib
        _b, outil = _construire(tmp_path)
        e = json.loads((tmp_path / "catalogue.json").read_text(encoding="utf-8"))["tools"]["doubler"]
        assert e["sha256"] == hashlib.sha256((tmp_path / "doubler.py").read_bytes()).hexdigest()
        assert (e["calls"], e["errors"], e["retired"]) == (0, 0, False)
        assert e["created"].endswith("Z") and e["description"] == "double un entier"
        assert outil.observer is not None

    def test_les_self_tests_ne_comptent_pas_comme_des_appels(self, tmp_path: Path) -> None:
        b, _ = _construire(tmp_path)
        assert b.catalogue()["doubler"]["calls"] == 0

    def test_les_appels_et_les_erreurs_sont_comptes(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, [_payload(code=CODE_CAPRICIEUX)])
        outil(n=1)
        outil(n=2)
        with pytest.raises(Exception, match="negatif"):
            outil(n=-1)
        e = b.catalogue()["doubler"]
        assert (e["calls"], e["errors"], e["consecutive_errors"]) == (3, 1, 1)
        assert "negatif" in e["last_error"]

    def test_un_succes_remet_la_serie_d_erreurs_a_zero(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, [_payload(code=CODE_CAPRICIEUX)], retire_after_errors=3)
        for _ in range(2):
            with pytest.raises(ToolError, match="negatif"):
                outil(n=-1)
        outil(n=1)
        assert b.catalogue()["doubler"]["consecutive_errors"] == 0
        assert not b.catalogue()["doubler"]["retired"]

    def test_des_erreurs_de_suite_retirent_l_outil(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, [_payload(code=CODE_CAPRICIEUX)], retire_after_errors=2)
        for _ in range(2):
            with pytest.raises(ToolError, match="negatif"):
                outil(n=-1)
        e = b.catalogue()["doubler"]
        assert e["retired"] is True and "2 consecutive errors" in e["retired_reason"]
        assert b.load_library() == []

    def test_retire_after_errors_zero_ne_retire_jamais(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, [_payload(code=CODE_CAPRICIEUX)], retire_after_errors=0)
        for _ in range(5):
            with pytest.raises(ToolError, match="negatif"):
                outil(n=-1)
        assert not b.catalogue()["doubler"]["retired"]

    def test_retrait_manuel(self, tmp_path: Path) -> None:
        b, _ = _construire(tmp_path)
        assert b.retire("doubler", "obsolète") is True
        assert b.retire("inconnu") is False
        assert b.catalogue()["doubler"]["retired_reason"] == "obsolète"
        assert b.load_library() == []

    def test_un_nouveau_constructeur_recharge_et_l_outil_fonctionne(self, tmp_path: Path) -> None:
        _construire(tmp_path)
        neuf = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True)     # aucun appel prévu
        outils = neuf.load_library()
        assert [o.spec.name for o in outils] == ["doubler"]
        assert outils[0](n=21) == {"double": 42}
        assert neuf.catalogue()["doubler"]["calls"] == 1, "les compteurs continuent d'un run à l'autre"

    def test_un_fichier_modifie_depuis_la_validation_n_est_pas_recharge(self, tmp_path: Path, caplog) -> None:
        _construire(tmp_path)
        fichier = tmp_path / "doubler.py"
        fichier.write_text(fichier.read_text(encoding="utf-8") + "\n# retouché\n", encoding="utf-8")
        neuf = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True)
        with caplog.at_level(logging.WARNING, logger="autoagent"):
            assert neuf.load_library() == []
        assert any("changed on disk" in r.message for r in caplog.records)

    def test_un_fichier_disparu_est_ignore(self, tmp_path: Path) -> None:
        _construire(tmp_path)
        (tmp_path / "doubler.py").unlink()
        assert DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True).load_library() == []

    def test_un_plafond_plus_strict_empeche_le_rechargement(self, tmp_path: Path) -> None:
        _construire(tmp_path, [_payload(permissions=["network"])])      # sans plafond : accepté
        strict = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True, allowed_permissions=set())
        assert strict.load_library() == []
        large = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True, allowed_permissions={"network"})
        assert [o.spec.name for o in large.load_library()] == ["doubler"]

    def test_un_fichier_sans_entree_au_catalogue_est_ignore(self, tmp_path: Path) -> None:
        _construire(tmp_path)
        (tmp_path / "inconnu.py").write_text("TOOL = {}\n", encoding="utf-8")
        noms = [o.spec.name for o in DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True).load_library()]
        assert noms == ["doubler"]

    def test_un_catalogue_illisible_est_mis_de_cote_pas_ecrase(self, tmp_path: Path) -> None:
        _construire(tmp_path)
        (tmp_path / "catalogue.json").write_text("{pas du json", encoding="utf-8")
        neuf = DynamicToolBuilder(Sequence([BON]), tools_dir=tmp_path, persist=True)
        assert neuf.load_library() == []
        assert list(tmp_path.glob("catalogue.json.corrompu-*")), "l'ancien catalogue reste récupérable"
        neuf.build(ToolBuildRequest(capability="x"))                      # et la bibliothèque repart
        assert "doubler" in neuf.catalogue()

    def test_reconstruire_le_meme_nom_remet_les_compteurs_a_zero(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, [BON, BON])
        outil(n=1)
        b.build(ToolBuildRequest(capability="x"))
        assert b.catalogue()["doubler"]["calls"] == 0

    def test_un_observateur_qui_plante_ne_casse_pas_l_appel(self, tmp_path: Path) -> None:
        _b, outil = _construire(tmp_path)
        outil.observer = lambda *a: 1 / 0
        assert outil(n=2) == {"double": 4}

    def test_pas_de_perte_de_comptage_en_concurrence(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path)
        fils = [threading.Thread(target=lambda: [outil(n=1) for _ in range(3)]) for _ in range(4)]
        for f in fils:
            f.start()
        for f in fils:
            f.join()
        assert b.catalogue()["doubler"]["calls"] == 12


# ── Dans l'agent ────────────────────────────────────────────────────────────

class Principal(LLMProvider):
    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self.reponses.pop(0)


def _appel(nom: str, **args: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id="c1", name=nom, arguments=args)],
                       usage=TokenUsage(input_tokens=100, output_tokens=10))


def _fin() -> LLMResponse:
    return LLMResponse(content="fini", usage=TokenUsage(input_tokens=120, output_tokens=5))


class TestDansLAgent:
    def test_le_run_suivant_reutilise_l_outil_sans_repayer_le_constructeur(self, tmp_path: Path) -> None:
        # Run 1 : le modèle crée l'outil (le constructeur est payé une fois).
        constructeur = Sequence([BON])
        run1 = Agent(Principal([_appel("create_python_tool", capability="doubler", tool_name="doubler"), _fin()]),
                     max_steps=4)
        run1.enable_dynamic_tools(DynamicToolBuilder(constructeur, tools_dir=tmp_path, persist=True))
        run1.run("go")
        assert len(constructeur.requetes) == 1

        # Run 2 : un AUTRE agent, un AUTRE constructeur — l'outil est déjà là.
        constructeur2 = Sequence([])
        run2 = Agent(Principal([_appel("doubler", n=21), _fin()]), max_steps=4)
        run2.enable_dynamic_tools(DynamicToolBuilder(constructeur2, tools_dir=tmp_path, persist=True))
        assert "doubler" in run2.registry and "doubler" in run2._dynamic_tool_names
        resultat = run2.run("go")
        assert constructeur2.requetes == [], "aucun appel au constructeur"
        outil = next(m for m in resultat.messages if m.role == "tool")
        assert "42" in outil.content

    def test_sans_persist_le_registre_ne_recharge_rien(self, tmp_path: Path) -> None:
        DynamicToolBuilder(Sequence([BON]), tools_dir=tmp_path, persist=True).build(ToolBuildRequest(capability="x"))
        agent = Agent(Principal([_fin()]), max_steps=2)
        agent.enable_dynamic_tools(DynamicToolBuilder(Sequence([]), tools_dir=tmp_path))
        assert "doubler" not in agent.registry

    def test_un_outil_de_la_bibliotheque_ne_remplace_jamais_un_outil_de_l_hote(self, tmp_path: Path, caplog) -> None:
        DynamicToolBuilder(Sequence([BON]), tools_dir=tmp_path, persist=True).build(ToolBuildRequest(capability="x"))
        agent = Agent(Principal([_fin()]), max_steps=2)
        def hote(n: int) -> dict:
            return {"hote": n}

        agent.tool(hote, name="doubler")
        with caplog.at_level(logging.WARNING, logger="autoagent"):
            agent.enable_dynamic_tools(DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True))
        assert agent.registry.execute(ToolCall(id="x", name="doubler", arguments={"n": 3})).result == {"hote": 3}
        assert "doubler" not in agent._dynamic_tool_names
        assert any("belongs to a host tool" in r.message for r in caplog.records)

    def test_les_outils_recharges_ne_comptent_pas_dans_le_plafond_par_run(self, tmp_path: Path) -> None:
        DynamicToolBuilder(Sequence([BON]), tools_dir=tmp_path, persist=True).build(ToolBuildRequest(capability="x"))
        autre = _payload(nom="triple", code="def run(args, context):\n    return {'triple': args['n'] * 3}\n",
                         tests=[{"args": {"n": 1}, "expect_equals": {"triple": 3}}])
        agent = Agent(Principal([_appel("create_python_tool", capability="t", tool_name="triple"), _fin()]),
                      max_steps=4, max_dynamic_tools_per_run=1)
        agent.enable_dynamic_tools(DynamicToolBuilder(Sequence([autre]), tools_dir=tmp_path, persist=True))
        agent.run("go")
        assert "doubler" in agent.registry and "triple" in agent.registry


# ── Self-tests : les nombres sont comparés à tolérance ──────────────────────

class TestToleranceNumerique:
    """Mesuré sur DeepSeek (haversine) : attendu 10007.543398010286, rendu
    10007.543398010288 — un bit de flottant — et un outil JUSTE était rejeté."""

    def _egal(self, a: Any, b: Any, tol: float = 1e-6) -> bool:
        from autoagent.dynamic import _egal
        return _egal(a, b, tol)

    def test_un_bit_de_flottant_n_est_plus_une_erreur(self) -> None:
        assert self._egal({"d": 10007.543398010286}, {"d": 10007.543398010288})

    def test_un_arrondi_a_six_chiffres_passe(self) -> None:
        assert self._egal(343.556063, 343.55606)

    def test_une_vraie_erreur_reste_une_erreur(self) -> None:
        assert not self._egal(100, 101)
        assert not self._egal(1.0, 1.001)
        assert not self._egal(392.4, 392.2)

    def test_zero_et_bruit_de_flottant(self) -> None:
        assert self._egal(0.0, 1e-15)
        assert not self._egal(0.0, 1e-3)

    def test_entier_et_flottant_egaux(self) -> None:
        assert self._egal(3, 3.0)

    def test_recursif_dans_listes_et_dictionnaires(self) -> None:
        assert self._egal({"a": [1.0, {"b": 2.0000000001}]}, {"a": [1, {"b": 2.0}]})
        assert not self._egal({"a": [1.0, 2.0]}, {"a": [1.0]})
        assert not self._egal({"a": 1}, {"a": 1, "b": 2})

    def test_chaines_et_cles_restent_exactes(self) -> None:
        assert not self._egal("a", "A")
        assert not self._egal({"x": "343.556000"}, {"x": "343.556060"})
        assert self._egal(None, None) and not self._egal(None, 0)

    def test_les_booleens_gardent_l_egalite_de_python(self) -> None:
        assert self._egal(True, True) and not self._egal(True, False)

    def test_tolerance_zero_redonne_l_egalite_stricte(self) -> None:
        assert not self._egal(10007.543398010286, 10007.543398010288, tol=0)
        assert self._egal(10007.543398010286, 10007.543398010286, tol=0)

    def test_dans_un_build_un_outil_juste_n_est_plus_rejete(self, tmp_path: Path) -> None:
        code = "def run(args, context):\n    return {'d': 10007.543398010288}\n"
        proche = _payload(code=code, tests=[{"args": {"n": 1}, "expect_equals": {"d": 10007.543398010286}}])
        outil = DynamicToolBuilder(Sequence([proche]), tools_dir=tmp_path).build(ToolBuildRequest(capability="x"))
        assert outil(n=1) == {"d": 10007.543398010288}

    def test_et_en_mode_strict_il_l_est_toujours(self, tmp_path: Path) -> None:
        code = "def run(args, context):\n    return {'d': 10007.543398010288}\n"
        proche = _payload(code=code, tests=[{"args": {"n": 1}, "expect_equals": {"d": 10007.543398010286}}])
        with pytest.raises(ToolValidationError, match="Self-test 0 failed"):
            DynamicToolBuilder(Sequence([proche]), tools_dir=tmp_path, self_test_rel_tol=0).build(
                ToolBuildRequest(capability="x"))

    def test_le_modele_est_prevenu_de_la_tolerance(self, tmp_path: Path) -> None:
        fournisseur = Sequence([BON])
        DynamicToolBuilder(fournisseur, tools_dir=tmp_path).build(ToolBuildRequest(capability="x"))
        assert "relative tolerance of 1e-06" in fournisseur.requetes[0].messages[0].content
        assert "BY HAND" not in fournisseur.requetes[0].messages[0].content


# ── JSON invalide : le modèle écrivait des EXPRESSIONS à la place des nombres ──

class TestJsonInvalideDuConstructeur:
    """Mesuré dans de vrais runs (tâche haversine, DeepSeek) : 9 réponses sur 20
    étaient du JSON invalide parce que le modèle écrivait `6371 * (math.pi / 2)`
    comme valeur attendue — et, à chaque reprise, « Expecting ',' delimiter » ne
    lui disait rien : même faute, jusqu'à 16 000 jetons brûlés."""

    EXPRESSION = (
        '{"tool": {"name": "doubler", "description": "d", "input_schema": {"type": "object"}, "permissions": []}, '
        '"code": "def run(args, context):\n    return {\'double\': 1}\n", '
        '"self_tests": [{"args": {"n": 1}, "expect_equals": {"double": 6371 * (math.pi / 2)}}]}'
    )

    def test_le_prompt_interdit_les_expressions(self, tmp_path: Path) -> None:
        fournisseur = Sequence([BON])
        DynamicToolBuilder(fournisseur, tools_dir=tmp_path).build(ToolBuildRequest(capability="x"))
        systeme = fournisseur.requetes[0].messages[0].content
        assert "ONE valid JSON document" in systeme
        assert "NEVER an expression" in systeme

    def test_la_reponse_avec_expression_est_bien_invalide(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            json.loads(self.EXPRESSION)

    def test_la_reprise_nomme_la_cause_au_lieu_de_la_colonne(self, tmp_path: Path) -> None:
        fournisseur = Sequence([self.EXPRESSION, BON])
        DynamicToolBuilder(fournisseur, tools_dir=tmp_path, max_repairs=1).build(ToolBuildRequest(capability="x"))
        retour = fournisseur.requetes[1].messages[-1].content
        assert "not valid JSON" in retour
        assert "expression (6371 * math.pi / 2) instead of a plain number (10007.54)" in retour
        assert "decide which one is wrong" not in retour, "l'autre consigne ne s'applique pas ici"

    def test_pour_un_self_test_faux_la_consigne_reste_celle_du_code_ou_de_l_attendu(self, tmp_path: Path) -> None:
        fournisseur = Sequence([FAUX, BON])
        DynamicToolBuilder(fournisseur, tools_dir=tmp_path, max_repairs=1).build(ToolBuildRequest(capability="x"))
        retour = fournisseur.requetes[1].messages[-1].content
        assert "decide which one is wrong" in retour and "not valid JSON" not in retour
