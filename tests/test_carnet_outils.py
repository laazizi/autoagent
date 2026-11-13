"""Le carnet d'appels des outils créés par le modèle : SURVEILLER avant de mesurer et d'apprendre.

La bibliothèque persistante (`DynamicToolBuilder(persist=True)`, 0.22.0) ne gardait d'un outil que des compteurs
(`calls`, `errors`, `consecutive_errors`) et le dernier message d'erreur. On ne savait ni SUR QUOI il avait
échoué, ni combien de temps il mettait, ni si l'échec venait de lui. Et tout échec comptait contre lui : une
fonction de l'hôte refusée par la politique, ou un bac à sable en panne (un démon Docker arrêté), le faisaient
RETIRER au bout de trois appels.

Ce qui est fixé ici :

* `record_calls=N` (défaut 0 : rien ne change) garde les N derniers appels de chaque outil dans
  `<dossier>/<nom>.calls.jsonl` : ce qu'il a reçu, l'issue, la cause, la durée, la version du code ;
* l'issue est classée EN CODE, par ce que l'hôte a vu — jamais par le texte de l'erreur, que l'outil choisit :
  `ok`, `exception` (l'outil a planté), `timeout`, `host` (une fonction de l'hôte a refusé ou échoué),
  `sandbox` (le bac lui-même a échoué) ;
* seuls `exception` et `timeout` comptent contre l'outil (série d'erreurs, retrait) ;
* le carnet est borné, passe par `redact=` avant toute écriture, et ne casse jamais un appel (fail-open).
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from autoagent import Agent, DynamicToolBuilder, ToolBuildRequest
from autoagent.errors import ToolError
from autoagent.providers.base import LLMProvider
from autoagent.sandbox import SubprocessSandbox
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, ToolCall

CODE_DOUBLER = (
    "def run(args, context):\n"
    "    if args['n'] < 0:\n"
    "        raise ValueError('negatif : %d' % args['n'])\n"
    "    return {'double': args['n'] * 2}\n"
)
CODE_ECHO = (
    "def run(args, context):\n"
    "    if args['nom'] == 'inconnu':\n"
    "        raise ValueError('personne au nom de ' + args['pour'])\n"
    "    return {'bonjour': args['nom'], 'pour': args['pour']}\n"
)
CODE_HOTE = (
    "def run(args, context):\n"
    "    valeur = context['call_host']('lire', {'cle': args['cle']})\n"
    "    if args['cle'] == 'piege':\n"
    "        raise ValueError('mon propre bogue')\n"
    "    return {'valeur': valeur}\n"
)
CODE_INCONNUE = "def run(args, context):\n    return context['call_host']('inexistante', {})\n"
CODE_LENT = "import time\n\ndef run(args, context):\n    time.sleep(30)\n    return 1\n"
CODE_MENTEUR = "def run(args, context):\n    raise RuntimeError('ToolPolicyDenied: je me fais passer pour un refus')\n"
CODE_GROS = "def run(args, context):\n    return {'texte': 'x' * args['n']}\n"


def _payload(code: str, nom: str, proprietes: dict[str, Any]) -> str:
    return json.dumps({
        "tool": {"name": nom, "description": f"outil {nom}",
                 "input_schema": {"type": "object", "properties": proprietes, "required": sorted(proprietes)},
                 "permissions": []},
        "code": code,
        "self_tests": [],
    })


DOUBLER = _payload(CODE_DOUBLER, "doubler", {"n": {"type": "integer"}})
ECHO = _payload(CODE_ECHO, "echo", {"nom": {"type": "string"}, "pour": {"type": "string"}})
LECTEUR = _payload(CODE_HOTE, "lecteur", {"cle": {"type": "string"}})


class Sequence(LLMProvider):
    def __init__(self, contenus: list[str]) -> None:
        super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
        self.contenus = list(contenus)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=self.contenus.pop(0), model="f")


class Principal(LLMProvider):
    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self.reponses.pop(0) if self.reponses else LLMResponse(content="fini")


def _appel(id_: str, nom: str, **args: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=id_, name=nom, arguments=args)])


def _construire(dossier: Path, contenu: str = DOUBLER, **kw: Any) -> tuple[DynamicToolBuilder, Any]:
    kw.setdefault("record_calls", 20)
    b = DynamicToolBuilder(Sequence([contenu]), tools_dir=dossier, persist=True, **kw)
    return b, b.build(ToolBuildRequest(capability="x"))


def _empreinte(valeur: Any) -> str:
    texte = json.dumps(valeur, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


def _lignes(dossier: Path, nom: str = "doubler") -> list[str]:
    return (dossier / f"{nom}.calls.jsonl").read_text(encoding="utf-8").splitlines()


def _echoue(outil: Any, **args: Any) -> None:
    with pytest.raises(ToolError):
        outil(**args)


# ---------------------------------------------------------------------------------------------------------

class TestParDefaut:
    def test_sans_record_calls_rien_n_est_ecrit(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, record_calls=0)
        outil(n=2)
        assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["catalogue.json", "doubler.py"]
        assert b.calls("doubler") == []

    def test_le_defaut_est_zero(self, tmp_path: Path) -> None:
        assert DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True).record_calls == 0

    def test_le_carnet_appartient_a_la_bibliotheque_persistante(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="persist=True"):
            DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, record_calls=5)

    @pytest.mark.parametrize("valeur", [-1, True, "5", 2.5])
    def test_record_calls_est_un_entier_positif(self, tmp_path: Path, valeur: Any) -> None:
        with pytest.raises(ValueError, match="record_calls"):
            DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True, record_calls=valeur)

    @pytest.mark.parametrize("valeur", [0, -3, True])
    def test_max_record_chars_est_strictement_positif(self, tmp_path: Path, valeur: Any) -> None:
        with pytest.raises(ValueError, match="max_record_chars"):
            DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True, record_calls=5, max_record_chars=valeur)


class TestCeQuiEstNote:
    def test_un_appel_reussi(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path)
        assert outil(n=2) == {"double": 4}
        (appel,) = b.calls("doubler")
        assert appel["ok"] is True and appel["kind"] == "ok"
        assert appel["args"] == {"n": 2} and appel["result"] == {"double": 4}
        assert appel["args_sha256"] == _empreinte({"n": 2})
        assert appel["result_sha256"] == _empreinte({"double": 4})
        assert appel["version"] == b.catalogue()["doubler"]["sha256"][:12]
        assert isinstance(appel["seconds"], float) and 0 <= appel["seconds"] < 60
        assert appel["ts"].endswith("Z") and len(appel["id"]) >= 8
        assert appel["tainted"] is None                      # hors de tout run : pas de teinte connue
        assert "error" not in appel

    def test_un_appel_qui_plante(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path)
        _echoue(outil, n=-7)
        (appel,) = b.calls("doubler")
        assert appel["ok"] is False and appel["kind"] == "exception"
        assert "negatif : -7" in appel["error"]
        assert appel["args"] == {"n": -7}                    # SUR QUOI il a échoué : ce que les compteurs ne disaient pas
        assert "result" not in appel and "result_sha256" not in appel

    def test_dans_l_ordre_des_appels_avec_des_identifiants_uniques(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path)
        for n in (1, 2, 3):
            outil(n=n)
        appels = b.calls("doubler")
        assert [a["args"]["n"] for a in appels] == [1, 2, 3]
        assert len({a["id"] for a in appels}) == 3

    def test_filtrer_les_echecs_et_limiter(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path)
        outil(n=1)
        _echoue(outil, n=-1)
        outil(n=2)
        _echoue(outil, n=-2)
        assert [a["args"]["n"] for a in b.calls("doubler", failed=True)] == [-1, -2]
        assert [a["args"]["n"] for a in b.calls("doubler", failed=False)] == [1, 2]
        assert [a["args"]["n"] for a in b.calls("doubler", limit=2)] == [2, -2]

    def test_les_auto_tests_de_la_construction_ne_sont_pas_des_appels(self, tmp_path: Path) -> None:
        contenu = json.loads(DOUBLER)
        contenu["self_tests"] = [{"args": {"n": 2}, "expect_equals": {"double": 4}}]
        b, _ = _construire(tmp_path, json.dumps(contenu))
        assert b.calls("doubler") == []
        assert not (tmp_path / "doubler.calls.jsonl").exists()

    def test_un_outil_inconnu_rend_une_liste_vide(self, tmp_path: Path) -> None:
        b, _ = _construire(tmp_path)
        assert b.calls("absent") == []

    def test_un_nom_qui_sort_du_dossier_ne_lit_rien(self, tmp_path: Path) -> None:
        """Le nom vient de l'appelant : seul un outil du catalogue a un carnet."""
        b, outil = _construire(tmp_path / "outils")
        outil(n=1)
        (tmp_path / "secret.calls.jsonl").write_text('{"id": "x", "ok": true}\n', encoding="utf-8")
        assert b.calls("../secret") == []

    def test_reconstruire_le_meme_nom_garde_l_histoire_avec_sa_version(self, tmp_path: Path) -> None:
        autre = DOUBLER.replace("* 2", "* 2 + 0")
        b = DynamicToolBuilder(Sequence([DOUBLER, autre]), tools_dir=tmp_path, persist=True, record_calls=20)
        b.build(ToolBuildRequest(capability="x"))(n=1)
        b.build(ToolBuildRequest(capability="x"))(n=2)
        versions = [a["version"] for a in b.calls("doubler")]
        assert len(versions) == 2 and versions[0] != versions[1]
        assert versions[1] == b.catalogue()["doubler"]["sha256"][:12]


class TestLeCarnetEstBorne:
    def test_seuls_les_n_derniers_appels_restent(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, record_calls=3)
        for n in range(10):
            outil(n=n)
            assert len(_lignes(tmp_path)) <= 6               # le fichier ne grossit jamais au-delà de 2 N lignes
        assert [a["args"]["n"] for a in b.calls("doubler")] == [7, 8, 9]

    def test_un_argument_trop_gros_n_est_garde_que_par_son_empreinte(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, _payload(CODE_ECHO, "echo", {"nom": {"type": "string"}, "pour": {"type": "string"}}),
                               max_record_chars=200)
        outil(nom="x" * 500, pour="a")
        (appel,) = b.calls("echo")
        assert "args" not in appel and appel["truncated"] == ["args", "result"]
        assert appel["args_sha256"] == _empreinte({"nom": "x" * 500, "pour": "a"})
        assert "x" * 100 not in "\n".join(_lignes(tmp_path, "echo"))

    def test_un_resultat_trop_gros_garde_son_empreinte(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, _payload(CODE_GROS, "gros", {"n": {"type": "integer"}}), max_record_chars=200)
        outil(n=5000)
        (appel,) = b.calls("gros")
        assert appel["args"] == {"n": 5000} and "result" not in appel and appel["truncated"] == ["result"]
        assert appel["result_sha256"] == _empreinte({"texte": "x" * 5000})

    def test_le_texte_d_une_erreur_est_borne(self, tmp_path: Path) -> None:
        code = "def run(args, context):\n    raise ValueError('y' * 5000)\n"
        b, outil = _construire(tmp_path, _payload(code, "bavard", {"n": {"type": "integer"}}))
        _echoue(outil, n=1)
        assert len(b.calls("bavard")[0]["error"]) <= 300


class TestDonneesPersonnelles:
    """Le carnet garde ce que l'outil REÇOIT : `redact=` passe avant toute écriture, sur les arguments, le
    résultat et le texte de l'erreur — y compris le `last_error` du catalogue."""

    @staticmethod
    def _masquer(valeur: Any) -> Any:
        return json.loads(json.dumps(valeur, ensure_ascii=False).replace("Dupont", "<nom>"))

    def _tout_le_disque(self, dossier: Path) -> str:
        return "\n".join(p.read_text(encoding="utf-8") for p in dossier.iterdir() if p.is_file() and p.suffix != ".py")

    def test_redact_passe_sur_les_arguments_et_le_resultat(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, ECHO, redact=self._masquer)
        assert outil(nom="Dupont", pour="Dupont") == {"bonjour": "Dupont", "pour": "Dupont"}   # l'outil, lui, voit tout
        (appel,) = b.calls("echo")
        assert appel["args"] == {"nom": "<nom>", "pour": "<nom>"} and appel["result"] == {"bonjour": "<nom>", "pour": "<nom>"}
        assert appel["redacted"] is True
        assert appel["args_sha256"] == _empreinte({"nom": "<nom>", "pour": "<nom>"})   # l'empreinte aussi : celle du masqué
        assert "Dupont" not in self._tout_le_disque(tmp_path)

    def test_redact_passe_sur_le_texte_de_l_erreur_et_sur_le_catalogue(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, ECHO, redact=self._masquer)
        _echoue(outil, nom="inconnu", pour="Dupont")
        assert "personne au nom de <nom>" in b.calls("echo")[0]["error"]
        assert "<nom>" in b.catalogue()["echo"]["last_error"]
        assert "Dupont" not in self._tout_le_disque(tmp_path)

    def test_sans_redact_l_entree_ne_dit_pas_qu_elle_est_masquee(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, ECHO)
        outil(nom="Dupont", pour="x")
        assert "redacted" not in b.calls("echo")[0]

    def test_un_redact_qui_plante_n_ecrit_aucune_donnee(self, tmp_path: Path) -> None:
        """Fail-closed pour la confidentialité : jamais la donnée brute à la place de la donnée masquée."""
        b, outil = _construire(tmp_path, ECHO, redact=lambda valeur: 1 / 0)
        assert outil(nom="Dupont", pour="Dupont") == {"bonjour": "Dupont", "pour": "Dupont"}
        _echoue(outil, nom="inconnu", pour="Dupont")
        reussi, rate = b.calls("echo")
        for appel in (reussi, rate):
            assert appel["redacted"] == "failed"
            assert not {"args", "result", "error", "args_sha256", "result_sha256"} & set(appel)
        assert (reussi["kind"], rate["kind"]) == ("ok", "exception")          # l'issue, elle, reste connue
        assert "Dupont" not in self._tout_le_disque(tmp_path)


class _BacEnPanne:
    timeout = 10.0

    def isolation(self) -> dict[str, Any]:
        return {"kind": "faux", "os_boundary": False}

    def run_python_tool(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise ToolError("Docker sandbox runner failed: stdout='' stderr='Cannot connect to the Docker daemon'")


class TestLIssueEstClasseeParLeCode:
    def _lecteur(self, dossier: Path, lire: Any, **kw: Any) -> tuple[DynamicToolBuilder, Any]:
        return _construire(dossier, LECTEUR, host_functions={"lire": lire}, **kw)

    def test_une_fonction_de_l_hote_qui_echoue(self, tmp_path: Path) -> None:
        def lire(cle: str) -> str:
            raise KeyError(cle)

        b, outil = self._lecteur(tmp_path, lire)
        _echoue(outil, cle="a")
        assert b.calls("lecteur")[0]["kind"] == "host"

    def test_une_fonction_de_l_hote_qui_repond_puis_l_outil_plante(self, tmp_path: Path) -> None:
        b, outil = self._lecteur(tmp_path, lambda cle: "v")
        assert outil(cle="a") == {"valeur": "v"}
        _echoue(outil, cle="piege")
        assert [a["kind"] for a in b.calls("lecteur")] == ["ok", "exception"]

    def test_appeler_une_fonction_qui_n_existe_pas_est_un_defaut_de_l_outil(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, _payload(CODE_INCONNUE, "egare", {"n": {"type": "integer"}}),
                               host_functions={"lire": lambda cle: "v"}, retire_after_errors=2)
        _echoue(outil, n=1)
        _echoue(outil, n=1)
        assert b.calls("egare")[0]["kind"] == "exception"
        assert b.catalogue()["egare"]["retired"] is True

    def test_un_outil_qui_imite_un_refus_reste_une_exception(self, tmp_path: Path) -> None:
        """Le texte de l'erreur est choisi par l'outil : il ne décide pas de l'issue."""
        b, outil = _construire(tmp_path, _payload(CODE_MENTEUR, "menteur", {"n": {"type": "integer"}}))
        _echoue(outil, n=1)
        assert b.calls("menteur")[0]["kind"] == "exception"

    def test_un_delai_depasse(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, _payload(CODE_LENT, "lent", {"n": {"type": "integer"}}),
                               sandbox=SubprocessSandbox(timeout=1.0))
        _echoue(outil, n=1)
        (appel,) = b.calls("lent")
        assert appel["kind"] == "timeout" and appel["seconds"] >= 0.9

    def test_un_bac_en_panne(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path)
        outil.sandbox = _BacEnPanne()
        _echoue(outil, n=1)
        (appel,) = b.calls("doubler")
        assert appel["kind"] == "sandbox" and "Docker" in appel["error"]

    def test_un_refus_de_la_porte_dans_un_run(self, tmp_path: Path) -> None:
        """La politique de l'hôte refuse la fonction que l'outil appelle : ce n'est pas un défaut de l'outil."""
        _construire(tmp_path, LECTEUR, host_functions={"lire": lambda cle: "v"})
        b = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True, record_calls=20)
        agent = Agent(Principal([_appel("c1", "lecteur", cle="a")]), max_steps=3,
                      tool_policy=lambda ctx: "lecture interdite" if ctx.call.name == "lire" else None)
        agent.enable_dynamic_tools(b, host_functions={"lire": lambda cle: "v"})
        agent.run("go")
        (appel,) = b.calls("lecteur")
        assert appel["kind"] == "host" and "lecture interdite" in appel["error"]


class TestSeulsLesDefautsDeLOutilComptentContreLui:
    def test_des_refus_de_l_hote_ne_retirent_pas_l_outil(self, tmp_path: Path) -> None:
        def lire(cle: str) -> str:
            raise PermissionError("refuse")

        b, outil = _construire(tmp_path, LECTEUR, host_functions={"lire": lire}, retire_after_errors=2)
        for _ in range(4):
            _echoue(outil, cle="a")
        e = b.catalogue()["lecteur"]
        assert e["retired"] is False and e["consecutive_errors"] == 0 and e["errors"] == 0
        assert e["calls"] == 4 and e["unattributed_failures"] == 4

    def test_un_bac_en_panne_ne_retire_pas_la_bibliotheque(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, retire_after_errors=2)
        outil.sandbox = _BacEnPanne()
        for _ in range(4):
            _echoue(outil, n=1)
        e = b.catalogue()["doubler"]
        assert e["retired"] is False and e["errors"] == 0 and e["unattributed_failures"] == 4

    def test_un_echec_non_attribue_ne_remet_pas_la_serie_a_zero(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, retire_after_errors=3)
        bon = outil.sandbox
        _echoue(outil, n=-1)
        outil.sandbox = _BacEnPanne()
        _echoue(outil, n=1)
        outil.sandbox = bon
        assert b.catalogue()["doubler"]["consecutive_errors"] == 1
        _echoue(outil, n=-1)
        _echoue(outil, n=-1)
        assert b.catalogue()["doubler"]["retired"] is True

    def test_les_delais_depasses_retirent_toujours(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, _payload(CODE_LENT, "lent", {"n": {"type": "integer"}}),
                               sandbox=SubprocessSandbox(timeout=0.5), retire_after_errors=2)
        for _ in range(2):
            _echoue(outil, n=1)
        assert b.catalogue()["lent"]["retired"] is True

    def test_sans_carnet_la_regle_est_la_meme(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path, record_calls=0, retire_after_errors=2)
        outil.sandbox = _BacEnPanne()
        for _ in range(3):
            _echoue(outil, n=1)
        assert b.catalogue()["doubler"]["retired"] is False


class TestLaTeinteDuRunEstNotee:
    def _agent(self, dossier: Path, reponses: list[LLMResponse]) -> DynamicToolBuilder:
        _construire(dossier)
        b = DynamicToolBuilder(Sequence([]), tools_dir=dossier, persist=True, record_calls=20)
        agent = Agent(Principal(reponses), max_steps=4)
        agent.enable_dynamic_tools(b)

        @agent.tool(untrusted=True)
        def lire_page(url: str) -> str:
            """Lit une page web."""
            return "contenu externe"

        agent.run("go")
        return b

    def test_un_run_propre(self, tmp_path: Path) -> None:
        b = self._agent(tmp_path, [_appel("c1", "doubler", n=1)])
        assert b.calls("doubler")[0]["tainted"] is False

    def test_un_run_qui_a_lu_du_contenu_non_fiable(self, tmp_path: Path) -> None:
        b = self._agent(tmp_path, [_appel("c1", "lire_page", url="https://exemple.fr"), _appel("c2", "doubler", n=1)])
        assert b.calls("doubler")[0]["tainted"] is True


class TestLeCarnetNeCasseJamaisUnAppel:
    def test_un_carnet_qu_on_ne_peut_pas_ecrire(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path)
        (tmp_path / "doubler.calls.jsonl").mkdir()            # un dossier à la place du fichier
        assert outil(n=2) == {"double": 4}
        assert b.catalogue()["doubler"]["calls"] == 1
        assert b.calls("doubler") == []

    def test_une_ligne_abimee_est_ignoree(self, tmp_path: Path) -> None:
        b, outil = _construire(tmp_path)
        outil(n=1)
        with (tmp_path / "doubler.calls.jsonl").open("a", encoding="utf-8") as fh:
            fh.write('{"id": "coupe", "ok": tr\n')
        outil(n=2)
        assert [a["args"]["n"] for a in b.calls("doubler")] == [1, 2]

    def test_un_crochet_de_l_hote_qui_plante(self, tmp_path: Path) -> None:
        _b, outil = _construire(tmp_path)
        outil.recorder = lambda appel: 1 / 0
        assert outil(n=2) == {"double": 4}

    def test_le_crochet_observer_reste_libre_pour_l_hote(self, tmp_path: Path) -> None:
        """`observer(nom, ok, erreur)` (0.22.0) garde sa signature ; la bibliothèque compte par `recorder`."""
        b, outil = _construire(tmp_path)
        vus: list[tuple[str, bool, str | None]] = []
        outil.observer = lambda nom, ok, erreur: vus.append((nom, ok, erreur))
        outil(n=1)
        _echoue(outil, n=-1)
        assert [(v[0], v[1]) for v in vus] == [("doubler", True), ("doubler", False)] and "negatif" in (vus[1][2] or "")
        assert b.catalogue()["doubler"]["calls"] == 2         # le comptage de la bibliothèque n'est pas remplacé


class TestApresUnRedemarrage:
    def test_un_nouveau_constructeur_continue_le_meme_carnet(self, tmp_path: Path) -> None:
        _a, outil = _construire(tmp_path, record_calls=3)
        outil(n=1)
        outil(n=2)
        b = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True, record_calls=3)
        (recharge,) = b.load_library()
        recharge(n=3)
        recharge(n=4)
        assert [a["args"]["n"] for a in b.calls("doubler")] == [2, 3, 4]
        for n in range(5, 12):
            recharge(n=n)
            assert len(_lignes(tmp_path)) <= 6               # la borne tient dès le redémarrage

    def test_un_carnet_reste_lisible_quand_l_enregistrement_est_coupe(self, tmp_path: Path) -> None:
        _a, outil = _construire(tmp_path)
        outil(n=1)
        b = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True)       # record_calls=0
        (recharge,) = b.load_library()
        recharge(n=2)
        assert [a["args"]["n"] for a in b.calls("doubler")] == [1]


def test_pas_de_perte_en_concurrence(tmp_path: Path) -> None:
    b, outil = _construire(tmp_path, record_calls=50)
    fils = [threading.Thread(target=lambda: [outil(n=1) for _ in range(3)]) for _ in range(4)]
    for f in fils:
        f.start()
    for f in fils:
        f.join()
    appels = b.calls("doubler")
    assert len(appels) == 12 and len({a["id"] for a in appels}) == 12
    assert len(_lignes(tmp_path)) == 12                      # chaque ligne est un JSON entier
