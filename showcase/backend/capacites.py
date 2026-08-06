"""Ce que l'agent a ACQUIS, et la preuve qu'il monte en puissance.

Deux choses vivent ici :

* **L'inventaire** — quels outils l'agent s'est écrits, et à quel niveau chacun
  tourne : bac à sable (jetable, sans réseau) ou natif (dans le process, validé
  par un humain). Plus la promotion d'un niveau à l'autre.

* **La mesure** — le seul chiffre honnête pour « il devient de plus en plus
  puissant » : le pourcentage de demandes traitées **sans avoir à créer un
  nouvel outil**. S'il monte, la bibliothèque acquise couvre de plus en plus de
  terrain. S'il stagne, l'accumulation ne sert à rien — et il faut le savoir.

Rien ici n'appelle un LLM : c'est de la comptabilité.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from autoagent.approval import ToolManifest, sha256_of
from autoagent.sandbox import SubprocessSandbox, load_generated_tool

from .paths import MANIFESTE, OUTILS

COMPTEURS = OUTILS.parent / "capacites.json"
_LOCK = threading.RLock()


# ── Inventaire ───────────────────────────────────────────────────────────────

def inventaire() -> list[dict[str, Any]]:
    """Les outils acquis, avec leur niveau d'exécution.

    ``mode`` vaut ``"natif"`` (empreinte dans le manifeste : l'humain a validé) ou
    ``"bac-a-sable"``. L'empreinte est celle du CODE : si l'agent réécrit l'outil,
    l'empreinte change et l'outil **retombe** en bac à sable. C'est voulu — une
    validation porte sur une version précise, pas sur un nom.
    """
    if not OUTILS.is_dir():
        return []
    manifeste = ToolManifest.load(MANIFESTE)
    out: list[dict[str, Any]] = []
    for fichier in sorted(OUTILS.glob("*.py")):
        code = fichier.read_text(encoding="utf-8")
        empreinte = sha256_of(code)
        entree: dict[str, Any] = {
            "fichier": fichier.name,
            "nom": fichier.stem,
            "octets": len(code),
            "modifie": time.strftime("%Y-%m-%d %H:%M",
                                     time.localtime(fichier.stat().st_mtime)),
            "empreinte": empreinte[:12],
            "mode": "natif" if manifeste.contains(empreinte) else "bac-a-sable",
        }
        try:  # la description vue par le modèle, si l'outil se charge encore
            entree["description"] = load_generated_tool(
                fichier, sandbox=SubprocessSandbox(timeout=5)).spec.description
        except Exception:  # noqa: BLE001 — un outil cassé reste listé, marqué invalide
            entree["mode"] = "invalide"
        out.append(entree)
    return out


def promouvoir(fichier: str, *, par: str = "humain") -> dict[str, Any]:
    """Fait passer UN outil du bac à sable au natif (niveau 1 → 2).

    C'est le seul endroit où une capacité s'élargit, et il exige un geste humain.
    On valide l'empreinte du code **tel qu'il est à cet instant** : si l'agent
    réécrit l'outil ensuite, l'empreinte change et il **retombe** tout seul en bac
    à sable. Une validation porte sur une version, jamais sur un nom.
    """
    cible = OUTILS / Path(fichier).name          # jamais de traversée
    if cible.suffix != ".py" or not cible.is_file():
        return {"promu": False, "erreur": f"outil introuvable : {fichier}"}
    code = cible.read_text(encoding="utf-8")
    with _LOCK:
        manifeste = ToolManifest.load(MANIFESTE)
        empreinte = manifeste.approve(code, name=cible.stem, approved_by=par)
        manifeste.save()
    return {"promu": True, "fichier": cible.name, "mode": "natif",
            "empreinte": empreinte[:12]}


def retrograder(fichier: str) -> dict[str, Any]:
    """Remet un outil en bac à sable (niveau 2 → 1).

    L'échelle doit pouvoir descendre : un outil promu qui se révèle douteux se
    retire sans être supprimé — il continue de fonctionner, mais isolé.
    """
    cible = OUTILS / Path(fichier).name
    if not cible.is_file():
        return {"retrograde": False, "erreur": f"outil introuvable : {fichier}"}
    with _LOCK:
        manifeste = ToolManifest.load(MANIFESTE)
        manifeste.revoke(sha256_of(cible.read_text(encoding="utf-8")))
        manifeste.save()
    return {"retrograde": True, "fichier": cible.name, "mode": "bac-a-sable"}


# ── Mesure de montée en puissance ────────────────────────────────────────────

def _lire() -> dict[str, Any]:
    if not COMPTEURS.is_file():
        return {"demandes": 0, "avec_creation": 0, "historique": []}
    try:
        return json.loads(COMPTEURS.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"demandes": 0, "avec_creation": 0, "historique": []}


def enregistrer_demande(*, outil_cree: bool) -> None:
    """Compte une demande traitée, et si elle a coûté un nouvel outil."""
    with _LOCK:
        d = _lire()
        d["demandes"] = d.get("demandes", 0) + 1
        if outil_cree:
            d["avec_creation"] = d.get("avec_creation", 0) + 1
        # Historique glissant : permet de voir la TENDANCE, pas juste le cumul.
        hist = d.get("historique", [])
        hist.append(1 if outil_cree else 0)
        d["historique"] = hist[-100:]
        COMPTEURS.parent.mkdir(parents=True, exist_ok=True)
        COMPTEURS.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def mesure() -> dict[str, Any]:
    """Le chiffre : part des demandes résolues SANS créer d'outil.

    ``autonomie_globale`` est le cumul depuis toujours ; ``autonomie_recente``
    ne regarde que les 20 dernières demandes. C'est la seconde qui compte : si
    elle dépasse la première, l'agent progresse vraiment.
    """
    d = _lire()
    total = d.get("demandes", 0)
    creations = d.get("avec_creation", 0)
    hist = d.get("historique", [])
    recent = hist[-20:]
    return {
        "demandes": total,
        "outils_crees": creations,
        "autonomie_globale": round(1 - creations / total, 3) if total else None,
        "autonomie_recente": round(1 - sum(recent) / len(recent), 3) if recent else None,
        "outils_acquis": len(list(OUTILS.glob("*.py"))) if OUTILS.is_dir() else 0,
    }
