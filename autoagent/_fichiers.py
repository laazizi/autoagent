"""Écriture de fichiers d'état sans risque de perte (0.22.0).

Un `path.write_text(...)` interrompu (processus tué, coupure, disque plein)
laisse un fichier TRONQUÉ. Pour un magasin de faits, c'était la perte totale :
au redémarrage le JSON illisible était traité comme vide, et la première
écriture suivante écrasait le fichier — prouvé : 5 faits → crash simulé → 1 fait.

Ici on écrit à côté, on force sur disque, puis on REMPLACE d'un coup
(`os.replace` est atomique sur POSIX et sur Windows/NTFS) : le fichier est
toujours soit l'ancien, entier, soit le nouveau, entier.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from .logging import get_logger

__all__ = ["atomic_write_text", "quarantine"]

_log = get_logger("fichiers")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Écrit ``text`` dans ``path`` de façon atomique.

    Sous Windows, un antivirus ou un indexeur qui tient le fichier ouvert peut
    faire échouer `os.replace` un court instant : on réessaie brièvement, puis
    on retombe sur l'écriture directe (le comportement d'avant) plutôt que de
    perdre l'état — le cas est journalisé.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        for tentative in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.05 * (tentative + 1))
        _log.warning("atomic replace of %s kept failing; writing in place", path)
        path.write_text(text, encoding=encoding)
    finally:
        tmp.unlink(missing_ok=True)


def quarantine(path: Path) -> Path | None:
    """Met de côté un fichier d'état illisible au lieu de le laisser écraser.

    Renvoie le nouveau chemin (``<nom>.corrompu-<horodatage>``), ou ``None``
    si le renommage a échoué. Le contenu reste récupérable à la main.
    """
    path = Path(path)
    cible = path.with_name(f"{path.name}.corrompu-{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        path.replace(cible)
    except OSError:
        _log.exception("could not move aside unreadable file %s", path)
        return None
    _log.error("unreadable state file moved aside to %s — recover it manually if needed", cible)
    return cible
