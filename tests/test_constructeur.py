"""Le constructeur visuel génère du Python qui tourne sur la lib COURANTE (0.21.0).

`constructeur_autoagent.html` est du JavaScript qui émet du Python. Rien ne le
faisait tourner en CI : il pouvait dériver de la lib en silence (kwarg renommé,
import disparu, champ de fil en français) et personne ne l'aurait vu avant qu'un
utilisateur ne colle le code. Ce test fait tourner `generate()` hors navigateur
(via `tests/constructeur_headless.js`, Node requis — sauté sinon) sur CHAQUE
preset, puis vérifie que le code produit :

  1. n'est pas le message « ajoute un bloc » (un contrôle vide passerait) ;
  2. compile ;
  3. n'importe que des noms qui EXISTENT dans `autoagent` ;
  4. ne passe à `Agent(...)` que des kwargs qui existent ;
  5. ne contient aucun nom de fil périmé (`deleguer`, `demandes`…).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import py_compile
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import autoagent

RACINE = Path(__file__).resolve().parent.parent
HTML = RACINE / "constructeur_autoagent.html"
HARNAIS = Path(__file__).resolve().parent / "constructeur_headless.js"

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node absent : le constructeur ne peut pas tourner hors navigateur")

# Noms de FIL de l'ancien `delegate_to` (0.20.0, français) — pas des noms de
# variables Python : `reponses` est un identifiant légitime dans la démo 11.
PERIMES = ('name="deleguer"', '"demandes"', '"specialiste":', '["reponses"]')


@pytest.fixture(scope="module")
def generes(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    dossier = tmp_path_factory.mktemp("constructeur")
    proc = subprocess.run([NODE, str(HARNAIS), str(HTML), str(dossier)],
                          capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, f"harnais KO : {proc.stderr[-600:]}"
    rapport = json.loads(proc.stdout)
    assert rapport, "aucun preset généré"
    return rapport


@pytest.fixture(scope="module")
def reussis(generes: list[dict]) -> list[dict]:
    """Les presets dont la génération a abouti — les autres sont rapportés par
    `test_tous_les_presets_se_generent`, inutile de replanter chaque test dessus."""
    return [p for p in generes if p["ok"] and p["fichier"]]


def _kwargs_agent(src: str) -> set[str]:
    """Les kwargs passés à `Agent(...)` / `Agent.from_model(...)`, par AST — une
    regex avalait l'appel suivant sur la même ligne (`delegate_to(..., name=)`)."""
    trouves: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        est_agent = (isinstance(f, ast.Name) and f.id == "Agent") or (
            isinstance(f, ast.Attribute) and f.attr == "from_model"
            and isinstance(f.value, ast.Name) and f.value.id == "Agent")
        if est_agent:
            trouves |= {k.arg for k in node.keywords if k.arg}
    return trouves


class TestChaquePreset:
    def test_tous_les_presets_se_generent(self, generes: list[dict]) -> None:
        rates = [p for p in generes if not p["ok"]]
        assert not rates, "presets en échec : " + "; ".join(f"{p['label']} → {p['erreur']}" for p in rates)
        assert len(generes) >= 26

    def test_aucun_placeholder(self, reussis: list[dict]) -> None:
        vides = [p["label"] for p in reussis if Path(p["fichier"]).read_text(encoding="utf-8").lstrip().startswith("# ←")]
        assert not vides, f"generate() a rendu le message « ajoute un bloc » pour : {vides}"

    def test_le_code_compile(self, reussis: list[dict]) -> None:
        for p in reussis:
            py_compile.compile(p["fichier"], doraise=True)

    def test_les_imports_autoagent_existent(self, reussis: list[dict]) -> None:
        manquants: list[str] = []
        for p in reussis:
            src = Path(p["fichier"]).read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("autoagent"):
                    mod = importlib.import_module(node.module)
                    manquants += [f"{p['label']} : {node.module}.{a.name}" for a in node.names if not hasattr(mod, a.name)]
        assert not manquants, manquants

    def test_les_kwargs_d_agent_existent(self, reussis: list[dict]) -> None:
        connus = set(inspect.signature(autoagent.Agent.__init__).parameters) | {"provider", "model"}
        # Agent.from_model(provider, model, **kwargs) accepte les mêmes kwargs qu'Agent.
        inconnus: list[str] = []
        for p in reussis:
            src = Path(p["fichier"]).read_text(encoding="utf-8")
            inconnus += [f"{p['label']} : {k}" for k in sorted(_kwargs_agent(src) - connus)]
        assert not inconnus, inconnus

    def test_aucun_nom_de_fil_perime(self, reussis: list[dict]) -> None:
        fautifs: list[str] = []
        for p in reussis:
            src = Path(p["fichier"]).read_text(encoding="utf-8")
            fautifs += [f"{p['label']} : {mot}" for mot in PERIMES if mot in src]
        assert not fautifs, fautifs


# Ce que chaque preset « nouveau bloc » DOIT émettre : la primitive de la lib,
# pas de la colle. Un preset qui compile mais n'appelle pas sa primitive serait
# un test vide.
ATTENDUS = {
    "politique": ("ToolPolicySpec.from_dict(", "tool_policy=politique", "except ApprovalRequired", ".audit_trifecta()"),
    "rejeu": ("RecordSession(", "registry=session.registry()", "session.close()"),
    "fiab": ("from autoagent.eval import run_k", "run_k(agent,", "rapport.summary()"),
    "bornes": ("Bounds(", "bounds=BORNES", "idempotent=True"),
    "deleg": ("delegate_to(",),
    "casc": ("cascade(", "check=juge"),
    "synth": ("synthesize_tool(", "Example("),
    "evol": ("EvolutionRuntime(", "enable_software_evolution(", "validation_command="),
    "04": ("except TokenBudgetExceeded", "except MaxStepsExceeded", ".resume(borne.state"),
    "11": ("describe=describe", "on_refused=", "on_offtopic="),
    # Bug vu à l'écran : un outil PARTAGÉ perdait ses drapeaux. Ils doivent suivre
    # l'outil sur l'agent ET sur chaque sous-agent.
    "08": ("superviseur.tool(etat_parc, idempotent=True)", "chercheur.tool(etat_parc, idempotent=True)", "redacteur.tool(etat_parc, idempotent=True)"),
}


class TestCeQueLesPresetsEmettent:
    def test_chaque_nouveau_bloc_emet_sa_primitive(self, reussis: list[dict]) -> None:
        par_id = {p["id"]: p for p in reussis}
        manques: list[str] = []
        for pid, morceaux in ATTENDUS.items():
            assert pid in par_id, f"preset {pid!r} absent"
            src = Path(par_id[pid]["fichier"]).read_text(encoding="utf-8")
            manques += [f"{pid} : {m!r}" for m in morceaux if m not in src]
        assert not manques, manques

    def test_aucun_preset_ne_porte_d_avertissement(self, reussis: list[dict]) -> None:
        """Le diagnostic signale les blocs ignorés / incohérents. Un preset livré
        doit en être exempt — sinon il montre à l'utilisateur un assemblage
        que le constructeur lui-même juge bancal."""
        alertes = [f"{p['id']} : {n['txt']}" for p in reussis for n in p["notes"] if n["lvl"] == "warn"]
        assert not alertes, alertes

    def test_les_retours_a_la_ligne_survivent(self, reussis: list[dict]) -> None:
        """Un system_prompt multi-ligne était concaténé SANS ses retours à la
        ligne (« ligne1ligne2 ») : la concaténation implicite Python ne les
        ajoute pas. Les presets 08b et cati en ont."""
        par_id = {p["id"]: p for p in reussis}
        src = Path(par_id["10"]["fichier"]).read_text(encoding="utf-8")
        # Le corps du hook du preset 10 est multi-ligne : il doit rester indenté ligne à ligne.
        assert "if not a_ecrit and ctx.correction_count == 0:\n        return Message(" in src


def test_la_version_visee_est_celle_de_la_lib() -> None:
    """Le constructeur déclare la version de la lib qu'il couvre. Si la lib
    avance sans lui, ce test casse — c'est le but : plus de retard silencieux."""
    m = re.search(r'const LIB_VERSION = "([^"]+)"', HTML.read_text(encoding="utf-8"))
    assert m, "LIB_VERSION absente du constructeur"
    assert m.group(1) == autoagent.__version__, (
        f"le constructeur vise {m.group(1)}, la lib est en {autoagent.__version__} : "
        "mettre la palette à jour puis LIB_VERSION")


def test_le_harnais_est_a_jour_avec_le_html() -> None:
    """Le harnais dépend de quatre noms globaux du HTML : s'ils sont renommés,
    on veut un message clair plutôt qu'un `eval` cryptique."""
    html = HTML.read_text(encoding="utf-8")
    for nom in ("function generate(", "const PRESETS", "let stack"):
        assert nom in html, f"{nom!r} absent du constructeur — adapter tests/constructeur_headless.js"
    assert sys.version_info >= (3, 10)
