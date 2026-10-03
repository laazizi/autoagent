"""Le journal durable (D3) face à de VRAIES morts de processus.

`tests/test_journal.py` simule la mort par une exception. Ici le processus se tue lui-même par
`os._exit(137)` à la frontière visée : aucun `finally`, aucun `atexit`, aucune fermeture de
fichier — comme `kill -9` ou une coupure de courant. Les effets (un mail, un compteur) sont
écrits dans un fichier, `fsync`, par l'OUTIL : c'est ce fichier — pas le journal — qui dit ce qui
est VRAIMENT parti.

Le cas qui fait la thèse : le mail est parti, le processus meurt avant d'en écrire le résultat.
Une reprise naïve repart de l'instantané précédent et le renvoie. Ici : `OutcomeUnknown`, rien
n'est relancé, et le mail reste parti UNE fois.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from autoagent import Journal

SCENARIO = Path(__file__).resolve().parent / "_journal_scenario.py"


def lancer(*args: object, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCENARIO), *map(str, args)], capture_output=True, text=True,
        timeout=120, env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})}, check=False)


def effets(chemin: Path) -> list[str]:
    """Ce qui est VRAIMENT parti, lu dans le fichier que les outils ont écrit et fsync."""
    return [ligne.split()[0] for ligne in chemin.read_text(encoding="utf-8").splitlines()] if chemin.exists() else []


@pytest.fixture
def lieux(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "run.jsonl", tmp_path / "effets.log"


def test_sans_coupure(lieux: tuple[Path, Path]) -> None:
    journal, log = lieux
    assert lancer("run", journal, log, "-").returncode == 0
    assert effets(log) == ["envoyer_mail", "lire_compteur"]


def test_le_mail_est_parti_le_processus_meurt_avant_d_en_ecrire_le_resultat(lieux: tuple[Path, Path]) -> None:
    """LE cas. Le mail est parti (c'est dans le fichier des effets), le résultat n'est pas écrit."""
    journal, log = lieux
    assert lancer("run", journal, log, "effect:after@envoyer_mail").returncode == 137
    assert effets(log) == ["envoyer_mail"]                           # il est VRAIMENT parti
    # L'intention, elle, est durable : écrite et forcée sur le disque avant l'effet, malgré `_exit`.
    types = [json.loads(ligne)["type"] for ligne in journal.read_text(encoding="utf-8").splitlines()]
    assert "intent" in types and "result" not in types

    reprise = lancer("resume", journal, log, "pause")
    assert reprise.returncode == 3, reprise.stderr[-400:]            # OutcomeUnknown
    assert json.loads(reprise.stdout) == {"unknown": [{"id": "c1", "name": "envoyer_mail"}]}
    assert effets(log) == ["envoyer_mail"]                           # RIEN n'a été relancé

    with Journal(journal) as j:                                      # le verrou est libre : le processus est mort
        j.resolve("c1", ok=True, result={"envoye": True})            # l'hôte a vérifié : le mail est bien parti
    assert lancer("resume", journal, log, "pause").returncode == 0
    assert effets(log) == ["envoyer_mail", "lire_compteur"]          # UN mail, pas deux


def test_mort_apres_le_resultat_du_mail_il_n_est_pas_renvoye(lieux: tuple[Path, Path]) -> None:
    journal, log = lieux
    assert lancer("run", journal, log, "result:after@envoyer_mail").returncode == 137
    assert effets(log) == ["envoyer_mail"]
    assert lancer("resume", journal, log, "pause").returncode == 0
    assert effets(log) == ["envoyer_mail", "lire_compteur"]          # le mail n'est PAS relancé


def test_mort_avant_l_effet_apres_resolution_retry_le_mail_part_une_fois_avec_la_meme_cle(
        lieux: tuple[Path, Path]) -> None:
    journal, log = lieux
    assert lancer("run", journal, log, "intent:after@envoyer_mail").returncode == 137
    assert effets(log) == []                                         # l'effet n'avait pas eu lieu
    assert lancer("resume", journal, log, "pause").returncode == 3
    with Journal(journal) as j:
        run = j.run_ids()[0]
        j.resolve("c1", retry=True)                                  # « il n'est pas parti : relance »
    assert lancer("resume", journal, log, "pause").returncode == 0
    lignes = log.read_text(encoding="utf-8").splitlines()
    assert [ligne.split()[0] for ligne in lignes] == ["envoyer_mail", "lire_compteur"]
    assert lignes[0].split()[1] == Journal.key_for(run, "c1")        # la MÊME clé qu'avant la coupure


def test_deux_processus_qui_reprennent_le_meme_run_ne_refont_pas_l_effet_deux_fois(
        lieux: tuple[Path, Path], tmp_path: Path) -> None:
    """« k processus qui reprennent la même pause déclenchent l'effet k fois » (Khan, 2026) :
    ici un seul tient le verrou, l'autre est refusé."""
    journal, log = lieux
    assert lancer("run", journal, log, "state:after#2").returncode == 137    # mort juste avant les outils
    assert effets(log) == []
    marqueur = tmp_path / "verrou_pris"
    premier = subprocess.Popen(
        [sys.executable, str(SCENARIO), "resume", str(journal), str(log), "pause"],
        env={**os.environ, "LENT": "2", "MARQUEUR": str(marqueur), "PYTHONIOENCODING": "utf-8"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(200):                                          # le premier a pris le verrou
            if marqueur.exists():
                break
            time.sleep(0.05)
        assert marqueur.exists(), "le premier processus n'a jamais pris le verrou"
        second = lancer("resume", journal, log, "pause")
        assert second.returncode == 4                                 # JournalLocked : refusé
        assert premier.wait(timeout=60) == 0
    finally:
        if premier.poll() is None:
            premier.kill()
    assert effets(log) == ["envoyer_mail", "lire_compteur"]           # UN mail, UN compteur


def test_une_mort_ne_laisse_aucun_verrou_perime(lieux: tuple[Path, Path]) -> None:
    journal, log = lieux
    assert lancer("run", journal, log, "effect:after@envoyer_mail").returncode == 137
    Journal(journal).close()                                          # rouvrable aussitôt : le système a libéré le verrou
