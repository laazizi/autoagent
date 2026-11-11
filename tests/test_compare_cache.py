"""`compare_configs` garde la part de l'entrée servie par le cache (nuit du 11 nov. 2026).

Le juge l'a relevé : les comparaisons du banc d'efficacité ne conservaient que les jetons BRUTS. Or la
part cachée de DeepSeek varie de 38 % à 79 % selon la configuration (mesures antérieures du même banc) :
des jetons en moins ne valent pas le même pourcentage de COÛT, et le texte statique qu'une option
raccourcit est justement un préfixe cachable. Sans cette part, aucun gain de coût ne peut être écrit.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoagent import EvalTask, compare_configs
from autoagent.schema import TokenUsage


class _Res:
    def __init__(self, entree: int | None, cache: int | None) -> None:
        self.output = "ok"
        self.steps = 1
        self.usage = TokenUsage(input_tokens=entree, output_tokens=10,
                                total_tokens=(entree or 0) + 10, cached_tokens=cache)


def _bras(entree: int | None, cache: int | None):
    class _Agent:
        def run(self, prompt: str, context: dict[str, Any] | None = None) -> _Res:
            return _Res(entree, cache)
    return lambda: _Agent()


def _taches() -> list[EvalTask]:
    return [EvalTask(f"t{i}", f"p{i}", lambda r: True) for i in range(3)]


def test_la_part_du_cache_de_chaque_bras() -> None:
    r = compare_configs(_bras(1000, 400), _bras(600, 450), _taches(), repeats=2)
    assert r.cost is not None
    assert r.cost.cached_share_a == pytest.approx(0.4)
    assert r.cost.cached_share_b == pytest.approx(0.75)
    assert "cache 40 % → 75 %" in r.summary(), r.summary()
    assert r.to_dict()["cost"]["cached_share_b"] == pytest.approx(0.75)


def test_un_fournisseur_qui_ne_dit_rien_du_cache_n_invente_pas_un_zero() -> None:
    r = compare_configs(_bras(1000, None), _bras(600, None), _taches(), repeats=2)
    assert r.cost is not None and r.cost.tokens_per_attempt_b == pytest.approx(610)
    assert r.cost.cached_share_a is None and r.cost.cached_share_b is None
    assert "cache" not in r.summary()


def test_un_cache_a_zero_rapporte_est_un_vrai_zero() -> None:
    r = compare_configs(_bras(1000, 0), _bras(600, 0), _taches(), repeats=2)
    assert r.cost is not None and r.cost.cached_share_a == 0.0


def test_entree_inconnue() -> None:
    r = compare_configs(_bras(None, 5), _bras(None, 5), _taches(), repeats=1)
    assert r.cost is not None and r.cost.cached_share_a is None
