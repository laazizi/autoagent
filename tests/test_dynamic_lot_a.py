"""Outils dynamiques — lot A (0.22.0) : ce que la mesure a montré.

Mesuré sur DeepSeek, 4 tâches (luhn, jours, dist, plaques) : 5 créations
d'outil sur 9 échouaient, et les jetons du modèle constructeur étaient
invisibles à `token_budget` (+39 à +44 % de dépense réelle). Causes trouvées :

* l'exemple de self-test du prompt (`expect_equals: {"ok": True}`) était
  recopié tel quel — un self-test faux rejette l'outil entier ;
* `re.compile(...)` était refusé (le nom `compile` était interdit même par
  attribut) alors que c'est l'usage le plus courant d'un outil légitime ;
* `\\d` dans une regex du champ `code` : échappement JSON invalide, l'appel
  partait en `{"_raw": ...}` ;
* un outil refusé restait sur le disque ;
* l'appel au constructeur n'était compté nulle part.

Chaque test ci-dessous échoue sur le code d'avant.
"""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path
from typing import Any

import pytest

from autoagent import Agent, DynamicToolBuilder, TokenBudgetExceeded, ToolBuildRequest
from autoagent.errors import ToolValidationError
from autoagent.providers import anthropic as anthropic_mod
from autoagent.providers import openai as openai_mod
from autoagent.providers.anthropic import AnthropicProvider
from autoagent.providers.base import LLMProvider, loads_tolerant, parse_tool_arguments, repair_json_escapes
from autoagent.providers.openai import OpenAIProvider
from autoagent.sandbox import discard_generated_tool, validate_generated_tool_code
from autoagent.schema import LLMRequest, LLMResponse, Message, ModelConfig, TokenUsage, ToolCall

BS = chr(92)   # un antislash — jamais écrit « \\ » dans un script de test, trop facile à mal échapper

HEAD = "TOOL = {'name': 't', 'description': 'd', 'input_schema': {'type': 'object', 'properties': {}}}\n"


# ── Le validateur : un faux positif et un trou trivial ──────────────────────

class TestValidateur:
    def test_re_compile_est_accepte(self) -> None:
        """`re.compile` est l'usage le plus courant d'un outil légitime : le
        refuser faisait échouer des créations parfaitement saines."""
        code = HEAD + "import re\ndef run(args, context):\n    return bool(re.compile(r'^a+$').match('aaa'))\n"
        validate_generated_tool_code(code)        # ne lève pas

    def test_le_compile_nu_reste_refuse(self) -> None:
        code = HEAD + "def run(args, context):\n    return compile('1', 'x', 'eval')\n"
        with pytest.raises(ToolValidationError, match="compile"):
            validate_generated_tool_code(code)

    @pytest.mark.parametrize("nom", ["eval", "exec", "__import__"])
    def test_les_autres_appels_interdits_le_restent_aussi_par_attribut(self, nom: str) -> None:
        code = HEAD + f"def run(args, context):\n    return args.{nom}('1')\n"
        with pytest.raises(ToolValidationError):
            validate_generated_tool_code(code)

    @pytest.mark.parametrize("ligne", ["import builtins", "from builtins import exec", "import builtins as b"])
    def test_builtins_ne_s_importe_pas(self, ligne: str) -> None:
        """`import builtins` redonnait `exec`/`eval` alors que le nom nu était
        refusé : contournement d'une ligne."""
        code = HEAD + ligne + "\ndef run(args, context):\n    return 1\n"
        with pytest.raises(ToolValidationError, match="builtins"):
            validate_generated_tool_code(code)


# ── Échappements JSON invalides : réparés, mais jamais au prix d'un JSON valide ─

class TestReparationDesEchappements:
    def _regex_mal_echappee(self) -> str:
        # {"code": "re.compile(\"^\d+$\")"}  — `\"` valide, `\d` invalide
        return '{"code": "re.compile(' + BS + '"^' + BS + 'd+$' + BS + '")"}'

    def test_un_echappement_invalide_devient_un_antislash_litteral(self) -> None:
        args = parse_tool_arguments(self._regex_mal_echappee())
        assert args == {"code": 're.compile("^' + BS + 'd+$")'}

    def test_le_texte_source_voulu_par_le_modele_est_preserve(self) -> None:
        """L'intention est sans ambiguïté : le modèle voulait le texte `\\d`."""
        args = parse_tool_arguments('{"p": "' + BS + 'd' + BS + '.' + BS + 's"}')
        assert args == {"p": BS + "d" + BS + "." + BS + "s"}

    def test_un_json_coupe_reste_refuse(self) -> None:
        """Le garde-fou de `_assemble` : réparer ne doit JAMAIS faire passer un
        appel coupé — un `purger(filtre="*")` sur des arguments perdus."""
        coupe = '{"filtre": "dossier_' + BS + 'd45'
        assert parse_tool_arguments(coupe) == {"_raw": coupe}
        assert parse_tool_arguments('{"a": "x' + BS) == {"_raw": '{"a": "x' + BS}

    def test_le_n_importe_quoi_reste_n_importe_quoi(self) -> None:
        assert parse_tool_arguments("NOT-JSON") == {"_raw": "NOT-JSON"}

    def test_vide_et_objet_deja_pret(self) -> None:
        assert parse_tool_arguments("") == {}
        assert parse_tool_arguments("   ") == {}
        assert parse_tool_arguments(None) == {}
        assert parse_tool_arguments({"a": 1}) == {"a": 1}

    def test_un_retour_a_la_ligne_brut_dans_une_chaine_est_accepte(self) -> None:
        assert parse_tool_arguments('{"code": "a\nb"}') == {"code": "a\nb"}

    def test_l_erreur_originale_est_relevee_si_irreparable(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            loads_tolerant('{"a": ')

    def test_un_json_valide_n_est_jamais_modifie(self) -> None:
        """Propriété : la réparation est l'IDENTITÉ sur du JSON valide — y compris
        un antislash échappé suivi de `d`, les séquences `u` + 4 hex et `/` échappé."""
        rng = random.Random(20260310)
        alphabet = ["a", "d", "u", "n", "0", "é", "\n", "\t", '"', BS, "/", "\u2028", "😀", " "]
        for _ in range(400):
            valeur = {"k": "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))}
            for ascii_seul in (True, False):
                texte = json.dumps(valeur, ensure_ascii=ascii_seul)
                assert repair_json_escapes(texte) == texte
                assert parse_tool_arguments(texte) == valeur

    def test_un_antislash_echappe_suivi_d_une_lettre_n_est_pas_double(self) -> None:
        texte = '{"a": "' + BS + BS + 'd"}'          # JSON valide : antislash + d
        assert repair_json_escapes(texte) == texte
        assert json.loads(texte) == {"a": BS + "d"}

    def test_u_sans_quatre_chiffres_hex_est_invalide(self) -> None:
        texte = '{"a": "' + BS + 'uZZZZ"}'
        assert parse_tool_arguments(texte) == {"a": BS + "uZZZZ"}


class TestLesFournisseursReparent:
    def _openai(self) -> OpenAIProvider:
        return OpenAIProvider(ModelConfig(provider="openai", model="m", api_key="k"))

    def test_openai_hors_flux(self) -> None:
        brut = '{"code": "' + BS + 'd+"}'
        appels = self._openai()._parse_tool_calls([{"id": "c1", "function": {"name": "t", "arguments": brut}}])
        assert appels[0].arguments == {"code": BS + "d+"}

    def test_openai_hors_flux_illisible_garde_le_brut(self) -> None:
        appels = self._openai()._parse_tool_calls([{"id": "c1", "function": {"name": "t", "arguments": "NOT-JSON"}}])
        assert appels[0].arguments == {"_raw": "NOT-JSON"}

    def test_openai_en_flux(self, monkeypatch) -> None:
        brut = '{"code": "' + BS + 'd+"}'
        evts = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "t", "arguments": brut[:8]}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": brut[8:]}}]}, "finish_reason": "tool_calls"}]},
        ]
        monkeypatch.setattr(openai_mod, "post_sse", lambda *a, **k: iter(evts))
        final = list(self._openai().stream(LLMRequest(messages=[Message(role="user", content="x")])))[-1].response
        assert final.tool_calls[0].arguments == {"code": BS + "d+"}

    def test_anthropic_en_flux(self, monkeypatch) -> None:
        brut = '{"code": "' + BS + 'd+"}'
        evts = [{"type": "message_start", "message": {"model": "c", "usage": {"input_tokens": 5}}},
                {"type": "content_block_start", "index": 0,
                 "content_block": {"type": "tool_use", "id": "t1", "name": "outil"}},
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "input_json_delta", "partial_json": brut}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}]
        monkeypatch.setattr(anthropic_mod, "post_sse", lambda *a, **k: iter(evts))
        p = AnthropicProvider(ModelConfig(provider="anthropic", model="c", api_key="k"))
        final = list(p.stream(LLMRequest(messages=[Message(role="user", content="x")])))[-1].response
        assert final.tool_calls[0].arguments == {"code": BS + "d+"}


# ── Le constructeur : prompt, JSON, nettoyage, comptabilité ─────────────────

def _payload(*, code: str | None = None, tests: Any = None, nom: str = "doubler") -> dict[str, Any]:
    return {
        "tool": {"name": nom, "description": "double un entier",
                 "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
                 "permissions": []},
        "code": code or "def run(args, context):\n    return {'double': args['n'] * 2}\n",
        "self_tests": [{"args": {"n": 2}, "expect_equals": {"double": 4}}] if tests is None else tests,
    }


class _Constructeur(LLMProvider):
    """Rend un JSON de constructeur, avec un usage déclaré."""

    def __init__(self, contenu: str, usage: TokenUsage | None = None) -> None:
        super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
        self.contenu, self.usage = contenu, usage
        self.requetes: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requetes.append(request)
        return LLMResponse(content=self.contenu, model="f", usage=self.usage)


@pytest.fixture
def avertissements_neufs(monkeypatch):
    from autoagent import logging as journal
    monkeypatch.setattr(journal, "_DEJA_AVERTI", set())


class TestPromptDuConstructeur:
    def test_plus_aucun_exemple_de_valeur_a_recopier(self, tmp_path: Path) -> None:
        fournisseur = _Constructeur(json.dumps(_payload()))
        DynamicToolBuilder(fournisseur, tools_dir=tmp_path).build(ToolBuildRequest(capability="doubler"))
        demande = json.loads(fournisseur.requetes[0].messages[1].content)
        formats = demande["self_test_format"]
        texte = json.dumps(formats)
        assert '"ok": true' not in texte.lower(), "l'ancien exemple {\"ok\": true} était recopié tel quel"
        assert '"example"' not in texte
        for entree in formats:                       # des DESCRIPTIONS, pas des valeurs
            for cle in ("args", "expect_equals", "expect_contains"):
                if cle in entree:
                    assert isinstance(entree[cle], str) and entree[cle].startswith("<")

    def test_le_systeme_autorise_une_liste_vide(self, tmp_path: Path) -> None:
        fournisseur = _Constructeur(json.dumps(_payload()))
        DynamicToolBuilder(fournisseur, tools_dir=tmp_path).build(ToolBuildRequest(capability="doubler"))
        systeme = fournisseur.requetes[0].messages[0].content
        assert "empty list" in systeme


class TestJsonDuConstructeur:
    def test_une_regex_mal_echappee_dans_code_est_lue(self, tmp_path: Path) -> None:
        code = "import re\ndef run(args, context):\n    return {'ok': bool(re.compile(r'^" + BS + "d+$').match(str(args['n'])))}\n"
        # on fabrique le JSON À LA MAIN avec l'échappement invalide, comme le modèle
        corps = json.dumps(_payload(code="@@CODE@@", tests=[]))
        brut = corps.replace("@@CODE@@", code.replace("\n", BS + "n"))
        assert (BS + "d") in brut
        with pytest.raises(json.JSONDecodeError):
            json.loads(brut)                          # la preuve que le JSON est bien invalide
        outil = DynamicToolBuilder(_Constructeur(brut), tools_dir=tmp_path).build(ToolBuildRequest(capability="x"))
        assert outil(n=123) == {"ok": True}
        assert outil(n="abc") == {"ok": False}


class TestOutilRefuseNeResteSurLeDisque:
    def _fichiers(self, dossier: Path) -> list[str]:
        return sorted(p.name for p in dossier.rglob("*") if p.is_file())

    def test_self_test_faux(self, tmp_path: Path) -> None:
        faux = _payload(tests=[{"args": {"n": 2}, "expect_equals": {"double": 5}}])
        with pytest.raises(ToolValidationError, match="Self-test 0 failed"):
            DynamicToolBuilder(_Constructeur(json.dumps(faux)), tools_dir=tmp_path).build(
                ToolBuildRequest(capability="doubler"))
        assert self._fichiers(tmp_path) == []

    def test_self_test_qui_plante(self, tmp_path: Path) -> None:
        plante = _payload(code="def run(args, context):\n    raise RuntimeError('boum')\n")
        with pytest.raises(Exception, match="boum"):
            DynamicToolBuilder(_Constructeur(json.dumps(plante)), tools_dir=tmp_path).build(
                ToolBuildRequest(capability="doubler"))
        assert self._fichiers(tmp_path) == []

    def test_chargement_qui_echoue(self, tmp_path: Path, monkeypatch) -> None:
        import autoagent.dynamic as dyn

        def echec(*a, **k):
            raise ToolValidationError("illisible")

        monkeypatch.setattr(dyn, "load_generated_tool", echec)
        with pytest.raises(ToolValidationError, match="illisible"):
            DynamicToolBuilder(_Constructeur(json.dumps(_payload())), tools_dir=tmp_path).build(
                ToolBuildRequest(capability="doubler"))
        assert self._fichiers(tmp_path) == []

    def test_un_outil_valide_reste(self, tmp_path: Path) -> None:
        DynamicToolBuilder(_Constructeur(json.dumps(_payload())), tools_dir=tmp_path).build(
            ToolBuildRequest(capability="doubler"))
        assert (tmp_path / "doubler.py").is_file()

    def test_discard_accepte_un_chemin_et_nettoie_le_bytecode(self, tmp_path: Path) -> None:
        f = tmp_path / "x.py"
        f.write_text("1\n", encoding="utf-8")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "x.cpython-311.pyc").write_bytes(b"pyc")
        discard_generated_tool(f)
        assert not f.exists() and not list(cache.glob("x.*.pyc"))
        discard_generated_tool(f)                     # déjà absent : sans effet, sans erreur


class TestSelfTestsMalFormes:
    """Le modèle peut recopier le DESCRIPTEUR du prompt au lieu d'une valeur :
    une erreur lisible (qu'il peut corriger) plutôt qu'un TypeError opaque."""

    def _build(self, tmp_path: Path, tests: Any) -> None:
        DynamicToolBuilder(_Constructeur(json.dumps(_payload(tests=tests))), tools_dir=tmp_path).build(
            ToolBuildRequest(capability="doubler"))

    def test_args_texte(self, tmp_path: Path) -> None:
        with pytest.raises(ToolValidationError, match="'args' must be an object"):
            self._build(tmp_path, [{"args": "<an object valid for input_schema>"}])
        assert not list(tmp_path.glob("*.py"))

    def test_self_tests_pas_une_liste(self, tmp_path: Path) -> None:
        with pytest.raises(ToolValidationError, match="must be a list"):
            self._build(tmp_path, {"args": {"n": 1}})

    def test_un_test_pas_un_objet(self, tmp_path: Path) -> None:
        with pytest.raises(ToolValidationError, match="must be an object"):
            self._build(tmp_path, ["n=2"])


class TestAvertissementSandbox:
    def test_subprocess_avertit_une_fois(self, tmp_path, caplog, avertissements_neufs) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="autoagent"):
            DynamicToolBuilder(_Constructeur("{}"), tools_dir=tmp_path)
            DynamicToolBuilder(_Constructeur("{}"), tools_dir=tmp_path)
        messages = [r.message for r in caplog.records if "not an isolation boundary" in r.message]
        assert len(messages) == 1

    def test_docker_n_avertit_pas(self, tmp_path, caplog, avertissements_neufs) -> None:
        import logging

        from autoagent.sandbox import DockerSandbox
        with caplog.at_level(logging.WARNING, logger="autoagent"):
            DynamicToolBuilder(_Constructeur("{}"), tools_dir=tmp_path, sandbox=DockerSandbox())
        assert not any("not an isolation boundary" in r.message for r in caplog.records)


class TestUsageDuConstructeur:
    def test_last_build_usage_apres_un_succes(self, tmp_path: Path) -> None:
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        b = DynamicToolBuilder(_Constructeur(json.dumps(_payload()), usage), tools_dir=tmp_path)
        assert b.last_build_usage is None
        b.build(ToolBuildRequest(capability="doubler"))
        assert b.last_build_usage == usage

    def test_l_appel_est_paye_meme_si_l_outil_est_refuse(self, tmp_path: Path) -> None:
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        faux = _payload(tests=[{"args": {"n": 2}, "expect_equals": {"double": 5}}])
        b = DynamicToolBuilder(_Constructeur(json.dumps(faux), usage), tools_dir=tmp_path)
        with pytest.raises(ToolValidationError):
            b.build(ToolBuildRequest(capability="doubler"))
        assert b.last_build_usage == usage

    def test_la_valeur_est_par_thread(self, tmp_path: Path) -> None:
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        b = DynamicToolBuilder(_Constructeur(json.dumps(_payload()), usage), tools_dir=tmp_path)
        b.build(ToolBuildRequest(capability="doubler"))
        vu: list[Any] = []
        t = threading.Thread(target=lambda: vu.append(b.last_build_usage))
        t.start()
        t.join()
        assert vu == [None] and b.last_build_usage == usage


class _Principal(LLMProvider):
    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self.reponses.pop(0)


def _creation(usage: tuple[int, int]) -> LLMResponse:
    appel = ToolCall(id="c1", name="create_python_tool",
                     arguments={"capability": "doubler", "tool_name": "doubler"})
    return LLMResponse(tool_calls=[appel], usage=TokenUsage(input_tokens=usage[0], output_tokens=usage[1]))


def _fin() -> LLMResponse:
    return LLMResponse(content="fini", usage=TokenUsage(input_tokens=120, output_tokens=5))


class TestLeConstructeurEstCompteDansLeRun:
    """Le cœur du lot : `token_budget` ne voyait pas le constructeur."""

    def _agent(self, tmp_path: Path, contenu: dict[str, Any], *, budget: int | None = None) -> Agent:
        constructeur = _Constructeur(json.dumps(contenu), TokenUsage(input_tokens=1000, output_tokens=500))
        agent = Agent(_Principal([_creation((100, 10)), _fin()]), max_steps=4, token_budget=budget)
        agent.enable_dynamic_tools(DynamicToolBuilder(constructeur, tools_dir=tmp_path))
        return agent

    def test_les_jetons_du_constructeur_s_ajoutent_au_run(self, tmp_path: Path) -> None:
        usage = self._agent(tmp_path, _payload()).run("go").usage
        # principal 100+120 / 10+5, constructeur 1000 / 500
        assert usage.input_tokens == 1220
        assert usage.output_tokens == 515

    def test_le_plafond_voit_le_constructeur(self, tmp_path: Path) -> None:
        with pytest.raises(TokenBudgetExceeded) as exc:
            self._agent(tmp_path, _payload(), budget=500).run("go")
        assert exc.value.spent == 1610, "le compte doit inclure les 1 500 du constructeur"

    def test_un_outil_refuse_a_quand_meme_coute(self, tmp_path: Path) -> None:
        faux = _payload(tests=[{"args": {"n": 2}, "expect_equals": {"double": 5}}])
        resultat = self._agent(tmp_path, faux).run("go")
        assert resultat.usage.input_tokens == 1220          # la dépense d'un ÉCHEC compte aussi
        outil = next(m for m in resultat.messages if m.role == "tool")
        assert "Self-test 0 failed" in outil.content
        assert not list(tmp_path.glob("*.py")), "et l'outil refusé n'est plus sur le disque"

    def test_un_run_sans_outil_dynamique_compte_comme_avant(self) -> None:
        agent = Agent(_Principal([_fin()]), max_steps=2)
        usage = agent.run("go").usage
        assert (usage.input_tokens, usage.output_tokens) == (120, 5)

    def test_un_faux_constructeur_sans_usage_ne_casse_rien(self, tmp_path: Path) -> None:
        """Les constructeurs de test (sans `last_build_usage`) restent valides."""
        class Faux:
            def build(self, request):
                raise ToolValidationError("non")

        agent = Agent(_Principal([_creation((100, 10)), _fin()]), max_steps=4)
        agent.enable_dynamic_tools(Faux())       # type: ignore[arg-type]
        usage = agent.run("go").usage
        assert (usage.input_tokens, usage.output_tokens) == (220, 15)


class TestDechetsApresLeJson:
    """Capturé sur DeepSeek dans de vrais runs : l'objet JSON est COMPLET, puis le
    modèle laisse fuiter son balisage interne d'appel d'outil. `json.loads` brut
    échoue (« Extra data ») ; l'extraction du premier objet équilibré doit réussir."""

    OBJET = json.dumps({"tool": {"name": "o"}, "code": "def run(a, c):\n    return 1\n", "self_tests": []})

    def test_balisage_dsml_apres_l_objet(self) -> None:
        from autoagent.dynamic import _parse_json_object
        barres = chr(0xFF5C) * 2          # deux barres pleine largeur, comme dans la réponse capturée
        queue = "".join(f"</{barres}DSML{barres} {mot}>\n" for mot in ("parameter", "invoke"))
        queue += f"</{barres}DSML{barres} calls>"
        with pytest.raises(json.JSONDecodeError):
            json.loads(self.OBJET + queue)
        assert _parse_json_object(self.OBJET + queue)["tool"] == {"name": "o"}

    def test_accolade_et_guillemet_en_trop(self) -> None:
        from autoagent.dynamic import _parse_json_object
        assert _parse_json_object(self.OBJET + '"}')["self_tests"] == []
