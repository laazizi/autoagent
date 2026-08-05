"""Chemins et slug — SOURCE DE VÉRITÉ UNIQUE.

Le slug vivait en double (sessions.py + agent_factory.py) alors que la
suppression d'une session DÉPEND de leur égalité : deux définitions qui divergent
= un workspace orphelin jamais nettoyé. Un seul endroit, donc.
"""

from __future__ import annotations

import re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SESSIONS = DATA / "sessions"
WORKSPACE = DATA / "workspace"
SESSIONS.mkdir(parents=True, exist_ok=True)


def slug(s: str) -> str:
    """Nom de fichier/dossier sûr dérivé d'un texte libre."""
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:48]


def dossier_session(session_id: str) -> Path:
    """Workspace d'une session : pages HTML générées, outils dynamiques, trace."""
    return WORKSPACE / slug(session_id)


def fichier_session(session_id: str) -> Path:
    """JSON de session. session_id est opaque (généré serveur) : pas de traversée."""
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
    if not safe:
        raise ValueError("identifiant de session invalide")
    return SESSIONS / f"{safe}.json"
