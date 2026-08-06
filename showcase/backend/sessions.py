"""Persistance des conversations, une par session.

Le backend HTTP est SANS ÉTAT entre requêtes : l'historique d'une session vit sur
disque en JSON (via ``Message.to_dict`` / ``from_dict``, prévus pour ça), et le
``RunState`` en attente d'approbation y est persisté aussi — c'est le
checkpoint/resume de la lib appliqué à un vrai service (un run mis en pause
survivrait à un redémarrage du process).

Ce que ce fichier NE stocke PAS : le HTML des pages générées. Il vit déjà dans le
workspace de la session (écrit par l'agent HTML, sous garde). On ne garde ici que
des MÉTADONNÉES {titre, fichier} et on relit le HTML à la demande — sinon chaque
page était sérialisée deux fois dans le JSON, qui atteignait des centaines de Ko
et ralentissait la moindre lecture.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from autoagent import Message

from .paths import PAGES, SESSIONS, dossier_session, fichier_session, slug

DATA = SESSIONS  # compat : ancien nom du dossier des sessions

_LOCK = threading.RLock()  # sérialise les écritures disque (accès concurrents SSE)
_GARDER = object()  # sentinelle : « ne touche pas à ce champ »


def _lire(session_id: str) -> dict[str, Any]:
    f = fichier_session(session_id)
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}  # session corrompue : on repart proprement plutôt que de crasher


def charger(session_id: str) -> list[Message]:
    """Historique complet d'une session (liste vide si inconnue)."""
    return [Message.from_dict(m) for m in _lire(session_id).get("messages", [])]


def sauver(session_id: str, messages: list[Message], *, titre: str | None = None,
           canvas: dict | None = None, pages_ajout: list[dict] | None = None,
           pending: Any = _GARDER) -> None:
    """UNE seule écriture pour tout ce qu'un run produit.

    ``canvas`` / ``pages_ajout`` ne portent que des métadonnées {titre, fichier}
    (le HTML reste dans le workspace). ``pending`` : ``None`` efface l'état en
    attente, une valeur le remplace, absent = inchangé.
    """
    with _LOCK:
        d = _lire(session_id)
        # Allège au passage les entrées de l'ancien format (HTML embarqué).
        pages = [_meta(p, session_id) for p in d.get("pages", [])]
        if pages_ajout:
            pages = (pages + [_meta(p, session_id) for p in pages_ajout])[-12:]
        ancien_canvas = d.get("canvas")
        f = fichier_session(session_id)
        f.write_text(json.dumps({
            "id": session_id,
            "titre": titre or d.get("titre") or _titre_auto(messages),
            "maj": time.time(),
            "cree": d.get("cree", time.time()),
            "messages": [m.to_dict() for m in messages],
            "canvas": _meta(canvas, session_id) if canvas
            else (_meta(ancien_canvas, session_id) if ancien_canvas else None),
            "pages": pages,
            "pending": d.get("pending") if pending is _GARDER else pending,
        }, ensure_ascii=False, indent=1), encoding="utf-8")


def _meta(page: dict, session_id: str) -> dict:
    """Ne retient que l'identité d'une page — jamais son HTML.

    Tolère l'ANCIEN format (HTML embarqué, sans nom de fichier) : le nom est
    dérivé du titre comme l'a fait ``afficher_ecran``. Le HTML n'est abandonné
    QUE si le fichier existe vraiment dans le workspace — sinon on le conserve,
    pas de perte de données.
    """
    titre = page.get("titre", "")
    fichier = page.get("fichier") or (f"{slug(titre) or 'ecran'}.html")
    if (PAGES / Path(fichier).name).is_file():
        return {"titre": titre, "fichier": fichier}
    return {k: v for k, v in page.items() if k in ("titre", "fichier", "html")}


def charger_pages(session_id: str) -> list[dict]:
    """Métadonnées des pages générées (le HTML se lit via ``lire_page``)."""
    return _lire(session_id).get("pages", [])


def charger_canvas(session_id: str) -> dict | None:
    return _lire(session_id).get("canvas")


def lire_page(session_id: str, fichier: str) -> str | None:
    """HTML d'une page générée, relu depuis le workspace de la session.

    ``fichier`` vient de nos propres métadonnées, mais on le re-borne quand même
    au dossier de la session (défense en profondeur : jamais de traversée)."""
    nom = Path(fichier or "").name
    if not nom.endswith(".html"):
        return None
    # Les pages sont PARTAGÉES entre conversations (elles s'accumulent) : on les
    # lit dans le dossier commun, pas dans celui de la session. `session_id` reste
    # dans la signature pour l'API, mais ne borne plus la lecture.
    cible = PAGES / nom
    if not cible.is_file():
        return None
    try:
        return cible.read_text(encoding="utf-8")
    except OSError:
        return None


def sauver_pending(session_id: str, state_dict: dict | None) -> None:
    """Persiste (ou efface avec None) le RunState en attente d'approbation."""
    with _LOCK:
        d = _lire(session_id)
        d["pending"] = state_dict
        d.setdefault("id", session_id)
        fichier_session(session_id).write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def charger_pending(session_id: str) -> dict | None:
    return _lire(session_id).get("pending")


def lister() -> list[dict]:
    """Métadonnées de toutes les sessions, plus récentes d'abord (pour la sidebar)."""
    out = []
    for f in SESSIONS.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            out.append({"id": d["id"], "titre": d.get("titre", d["id"]),
                        "maj": d.get("maj", 0), "tours": _compter_tours(d.get("messages", []))})
        except Exception:  # noqa: BLE001 — une session corrompue ne casse pas la liste
            continue
    out.sort(key=lambda s: s["maj"], reverse=True)
    return out


def supprimer(session_id: str) -> bool:
    """Efface la CONVERSATION : son JSON (chat + historique + état en attente) et
    son dossier de travail (la trace).

    Ce qu'elle NE supprime PAS, volontairement : les outils que l'agent s'est
    écrits, les pages publiées et sa mémoire factuelle. Ils sont PARTAGÉS entre
    conversations — c'est précisément ce que l'agent a acquis. Supprimer un
    échange ne doit pas lui faire perdre une capacité. Pour effacer un souvenir,
    c'est `forget` ; pour retirer un outil, c'est son fichier dans data/outils.
    """
    supprime = False
    with _LOCK:
        f = fichier_session(session_id)
        if f.is_file():
            f.unlink()
            supprime = True
    ws = dossier_session(session_id)
    if ws.is_dir():
        shutil.rmtree(ws, ignore_errors=True)
        supprime = True
    return supprime


def _titre_auto(messages: list[Message]) -> str:
    """Titre = début du premier message utilisateur."""
    for m in messages:
        if m.role == "user" and m.content.strip():
            t = m.content.strip().replace("\n", " ")
            return t[:48] + ("…" if len(t) > 48 else "")
    return "Nouvelle conversation"


def _compter_tours(messages_dict: list[dict]) -> int:
    return sum(1 for m in messages_dict if m.get("role") == "user")


__all__ = ["charger", "charger_canvas", "charger_pages", "charger_pending", "lire_page",
           "lister", "sauver", "sauver_pending", "supprimer", "slug"]
