"""Chaque démo et chaque script du dépôt se lit avec la grammaire de Python 3.10.

Le 11 nov. 2026, la démo 38 ne se lançait plus du tout (des guillemets doubles dans une chaîne à
guillemets doubles) et aucun test ne l'a vu : seul ruff l'a trouvé, juste avant la 0.25.0.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
FICHIERS = sorted(
    f for d in ("examples_autoagent", "examples", "evals") if (RACINE / d).is_dir()
    for f in (RACINE / d).rglob("*.py")
)


def test_il_y_a_des_demos() -> None:
    assert len(FICHIERS) > 30


@pytest.mark.parametrize("fichier", FICHIERS, ids=[str(f.relative_to(RACINE)) for f in FICHIERS])
def test_la_demo_se_lit(fichier: Path) -> None:
    ast.parse(fichier.read_text(encoding="utf-8"), filename=str(fichier), feature_version=(3, 10))
