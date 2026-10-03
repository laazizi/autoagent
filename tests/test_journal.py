"""Le journal durable (D3) : une coupure brutale ne refait jamais un effet.

La mort du processus est SIMULÉE à chaque frontière d'écriture par une `BaseException` levée
dans le crochet `on_boundary` : ni `except Exception` ni le `run_end(error)` de la boucle ne
s'exécutent, comme sur un processus tué. `tests/test_journal_kill.py` refait les cas critiques
avec de VRAIES morts (`os._exit`) dans des sous-processus.

Ce qu'on prouve, frontière par frontière :
  * un appel dont le résultat est écrit n'est JAMAIS relancé ;
  * un appel dont l'issue est inconnue (intention sans résultat) n'est PAS relancé en silence :
    `OutcomeUnknown`, avant qu'aucun outil de l'étape ne tourne ;
  * un outil `idempotent` à l'issue inconnue est relancé (c'est ce que le drapeau promet) ;
  * si l'intention ne peut pas être écrite, l'effet ne part pas (fail-closed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autoagent import (
    Agent,
    AutoAgentError,
    Journal,
    JournalCorrupted,
    JournalError,
    JournalLocked,
    OutcomeUnknown,
    idempotency_key,
)
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMResponse, ModelConfig, ToolCall

EFFETS: list[tuple[str, str | None]] = []


@pytest.fixture(autouse=True)
def _propre() -> None:
    EFFETS.clear()


class Scripte(LLMProvider):
    """Sans état — comme un vrai modèle vu d'un processus neuf : tant qu'aucun résultat d'outil
    n'est dans la conversation il demande les deux outils, ensuite il conclut."""

    def __init__(self) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.appels = 0

    def complete(self, request: Any) -> LLMResponse:
        self.appels += 1
        if any(m.role == "tool" for m in request.messages):
            return LLMResponse(content="fini")
        return LLMResponse(tool_calls=[
            ToolCall(id="c1", name="envoyer_mail", arguments={"destinataire": "moi@example.com"}),
            ToolCall(id="c2", name="lire_compteur", arguments={}),
        ])


def construire(journal: Journal) -> Agent:
    agent = Agent(Scripte(), max_steps=4, journal=journal)

    @agent.tool
    def envoyer_mail(destinataire: str) -> dict:
        """Envoie un mail (effet non idempotent)."""
        EFFETS.append(("envoyer_mail", idempotency_key()))
        return {"envoye": True}

    @agent.tool(idempotent=True)
    def lire_compteur() -> dict:
        """Lit un compteur (idempotent : le relire ne change rien)."""
        EFFETS.append(("lire_compteur", idempotency_key()))
        return {"n": 7}

    return agent


def nombre(nom: str) -> int:
    return sum(1 for n, _k in EFFETS if n == nom)


class _Mort(BaseException):
    """La mort du processus à une frontière (BaseException : rien ne la rattrape)."""


def tuer_a(frontiere: str, *, outil: str | None = None, n: int = 1):
    vu = {"n": 0}

    def crochet(nom: str, infos: dict[str, Any]) -> None:
        if nom == frontiere and (outil is None or infos.get("name") == outil):
            vu["n"] += 1
            if vu["n"] == n:
                raise _Mort(f"{nom}@{outil}#{n}")

    return crochet


def mourir(chemin: Path, frontiere: str, **kw: Any) -> None:
    """Un run qui meurt à `frontiere` ; le journal est refermé (le système libère le verrou)."""
    journal = Journal(chemin, on_boundary=tuer_a(frontiere, **kw))
    with pytest.raises(_Mort):
        construire(journal).run("go")
    journal.close()


def reprendre(chemin: Path, **kw: Any) -> Any:
    with Journal(chemin) as journal:
        return construire(journal).resume_from_journal(**kw)


# ── La matrice de coupures : une ligne par frontière ─────────────────────────

CAS = [
    # frontière, outil, n,            inconnus,         mail avant, résolution, mail, compteur
    ("state:after", None, 1,          [],               0,  None,    1, 1),   # état initial écrit, rien lancé
    ("state:after", None, 2,          [],               0,  None,    1, 1),   # l'assistant a demandé, aucune intention
    ("intent:after", "envoyer_mail", 1, ["envoyer_mail"], 0, "retry", 1, 1),   # intention écrite, effet pas parti
    ("effect:after", "envoyer_mail", 1, ["envoyer_mail"], 1, "done",  1, 1),   # LE CAS : parti, résultat pas écrit
    ("result:after", "envoyer_mail", 1, [],               1,  None,    1, 1),   # mail terminé : réinjecté, pas relancé
    ("intent:after", "lire_compteur", 1, [],              1,  None,    1, 1),   # idempotent en doute : relancé seul
    ("effect:after", "lire_compteur", 1, [],              1,  None,    1, 2),   # idempotent déjà parti : relancé (déclaré sûr)
    ("result:after", "lire_compteur", 1, [],              1,  None,    1, 1),   # tout terminé, état d'après pas écrit
    ("state:after", None, 3,          [],               1,  None,    1, 1),   # état d'après-étape écrit
    ("end:before", None, 1,          [],               1,  None,    1, 1),   # réponse finale, run_end pas écrit
]


@pytest.mark.parametrize("frontiere,outil,n,inconnus,mail_avant,resolution,mail,compteur", CAS)
def test_une_coupure_a_chaque_frontiere_ne_refait_jamais_un_effet(
    tmp_path: Path, frontiere: str, outil: str | None, n: int, inconnus: list[str],
    mail_avant: int, resolution: str | None, mail: int, compteur: int,
) -> None:
    chemin = tmp_path / "run.jsonl"
    mourir(chemin, frontiere, outil=outil, n=n)
    assert nombre("envoyer_mail") == mail_avant                   # ce que le processus mort a eu le temps de faire

    if inconnus:
        # L'issue est INCONNUE : rien n'est relancé, et on le sait AVANT qu'aucun outil de l'étape ne tourne.
        with pytest.raises(OutcomeUnknown) as info:
            reprendre(chemin)
        assert [c.name for c in info.value.calls] == inconnus
        assert nombre("envoyer_mail") == mail_avant               # PAS relancé en silence
        with Journal(chemin) as journal:                         # l'hôte dit ce qui s'est passé
            for appel in info.value.calls:
                if resolution == "done":
                    journal.resolve(appel.id, ok=True, result={"envoye": True})
                else:
                    journal.resolve(appel.id, retry=True)

    resultat = reprendre(chemin)
    assert resultat.output == "fini"
    assert nombre("envoyer_mail") == mail, f"mail : {nombre('envoyer_mail')} au lieu de {mail}"
    assert nombre("lire_compteur") == compteur
    with Journal(chemin) as journal:
        assert journal.verify().ok
        assert journal.last_resumable_run() is None               # le run est terminé, plus rien à reprendre


def test_la_cle_d_idempotence_survit_a_la_coupure(tmp_path: Path) -> None:
    """Le mail relancé après « il n'est pas parti » porte la MÊME clé que l'intention écrite avant
    la coupure : le service externe peut dédoublonner."""
    chemin = tmp_path / "run.jsonl"
    mourir(chemin, "intent:after", outil="envoyer_mail")
    with Journal(chemin) as journal:
        run = journal.run_ids()[0]
        cle_ecrite = next(r["data"]["key"] for r in journal.records() if r["type"] == "intent")
        assert cle_ecrite == Journal.key_for(run, "c1")
        journal.resolve("c1", retry=True)
    reprendre(chemin)
    assert [k for n, k in EFFETS if n == "envoyer_mail"] == [cle_ecrite]


def test_on_unknown_retry_relance_tout_au_moins_une_fois(tmp_path: Path) -> None:
    """Le choix explicite de l'at-least-once : l'effet qui était déjà parti repart."""
    chemin = tmp_path / "run.jsonl"
    mourir(chemin, "effect:after", outil="envoyer_mail")
    assert nombre("envoyer_mail") == 1
    reprendre(chemin, on_unknown="retry")
    assert nombre("envoyer_mail") == 2


def test_une_seconde_coupure_apres_un_retry_ne_relance_pas_en_silence(tmp_path: Path) -> None:
    """Après « relance », une NOUVELLE intention redevient une issue inconnue : la machine à
    états ne marque pas l'appel comme « à relancer » pour toujours."""
    chemin = tmp_path / "run.jsonl"
    mourir(chemin, "effect:after", outil="envoyer_mail")                 # mail parti, rien d'écrit
    with Journal(chemin) as journal:
        journal.resolve("c1", retry=True)                               # l'hôte (à tort) dit : relance
    # La reprise relance le mail — et MEURT encore une fois entre l'effet et le résultat.
    journal = Journal(chemin, on_boundary=tuer_a("effect:after", outil="envoyer_mail"))
    with pytest.raises(_Mort):
        construire(journal).resume_from_journal()
    journal.close()
    assert nombre("envoyer_mail") == 2
    with pytest.raises(OutcomeUnknown):                                  # troisième reprise : on ne relance PAS
        reprendre(chemin)
    assert nombre("envoyer_mail") == 2


def test_un_outil_deja_execute_ne_repasse_pas_par_la_politique(tmp_path: Path) -> None:
    """La décision a été prise à l'époque, l'effet a eu lieu : le rejouer sous une politique qui
    refuserait aujourd'hui serait mentir sur ce qui s'est passé."""
    chemin = tmp_path / "run.jsonl"
    mourir(chemin, "result:after", outil="envoyer_mail")
    vus: list[str] = []

    def politique(ctx: Any) -> str | None:
        vus.append(ctx.call.name)
        return "plus d'envoi" if ctx.call.name == "envoyer_mail" else None

    with Journal(chemin) as journal:
        agent = construire(journal)
        agent.tool_policy = politique
        agent.resume_from_journal()
    assert "envoyer_mail" not in vus and "lire_compteur" in vus
    assert nombre("envoyer_mail") == 1


# ── Le journal lui-même ──────────────────────────────────────────────────────


class TestChaine:
    def _ecrire(self, chemin: Path) -> None:
        with Journal(chemin) as journal:
            construire(journal).run("go")

    def test_la_chaine_se_verifie(self, tmp_path: Path) -> None:
        chemin = tmp_path / "run.jsonl"
        self._ecrire(chemin)
        with Journal(chemin) as journal:
            rapport = journal.verify()
            assert rapport.ok and rapport.records == len(journal.records()) and rapport.head == journal.head()

    def test_un_enregistrement_modifie_au_milieu_est_refuse(self, tmp_path: Path) -> None:
        chemin = tmp_path / "run.jsonl"
        self._ecrire(chemin)
        lignes = chemin.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lignes[3])
        rec["data"]["call"]["arguments"] = {"destinataire": "evil@example.com"}      # on réécrit l'histoire
        lignes[3] = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")
        with pytest.raises(JournalCorrupted) as info:
            Journal(chemin)
        assert info.value.seq == 3

    def test_un_enregistrement_retire_au_milieu_est_refuse(self, tmp_path: Path) -> None:
        chemin = tmp_path / "run.jsonl"
        self._ecrire(chemin)
        lignes = chemin.read_text(encoding="utf-8").splitlines()
        del lignes[4]
        chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")
        with pytest.raises(JournalCorrupted):
            Journal(chemin)

    def test_une_queue_a_moitie_ecrite_est_reparee(self, tmp_path: Path) -> None:
        """Coupure PENDANT l'écriture : la dernière ligne est incomplète. On la retire — et
        seulement elle."""
        chemin = tmp_path / "run.jsonl"
        self._ecrire(chemin)
        brut = chemin.read_bytes()
        n_avant = brut.count(b"\n")
        chemin.write_bytes(brut + b'{"data":{"call_id":"c9"},"hash":"abc')      # une ligne tronquée en plus
        with Journal(chemin) as journal:
            assert len(journal.records()) == n_avant and journal.verify().ok
        assert chemin.read_bytes() == brut                                    # fichier ramené à l'état valide

    def test_la_derniere_ligne_abimee_est_retiree_mais_pas_celle_du_milieu(self, tmp_path: Path) -> None:
        chemin = tmp_path / "run.jsonl"
        self._ecrire(chemin)
        lignes = chemin.read_bytes().split(b"\n")
        lignes[-2] = lignes[-2][:40] + b"\x00\x00\x00"                        # page de disque à moitié persistée
        chemin.write_bytes(b"\n".join(lignes))
        with Journal(chemin) as journal:
            assert journal.verify().ok
            assert journal.records()[-1]["type"] != "run_end"                 # la dernière a été retirée


class TestEcrivainUnique:
    def test_un_second_ouvreur_est_refuse(self, tmp_path: Path) -> None:
        chemin = tmp_path / "run.jsonl"
        with Journal(chemin), pytest.raises(JournalLocked):
            Journal(chemin)
        Journal(chemin).close()                                              # libéré à la fermeture

    def test_la_lecture_reste_possible_pendant_l_ecriture(self, tmp_path: Path) -> None:
        chemin = tmp_path / "run.jsonl"
        with Journal(chemin) as journal:
            construire(journal).run("go")
            assert len(chemin.read_text(encoding="utf-8").splitlines()) == len(journal.records())


class TestFailClosed:
    def test_si_l_intention_ne_s_ecrit_pas_l_effet_ne_part_pas(self, tmp_path: Path) -> None:
        journal = Journal(tmp_path / "run.jsonl")
        agent = construire(journal)

        class Casse:
            """Un disque qui lâche au moment d'écrire l'intention."""
            def __init__(self, vrai: Any) -> None:
                self.vrai, self.fileno = vrai, vrai.fileno
                self.n = 0

            def write(self, octets: bytes) -> int:
                if b'"type":"intent"' in octets:
                    raise OSError("disque plein")
                return self.vrai.write(octets)

            def __getattr__(self, nom: str) -> Any:
                return getattr(self.vrai, nom)

        journal._fh = Casse(journal._fh)  # type: ignore[assignment]
        with pytest.raises(JournalError, match="disque plein"):
            agent.run("go")
        assert EFFETS == []                                                  # AUCUN effet : l'intention n'est pas écrite
        with pytest.raises(JournalError):                                    # et plus rien ne s'écrit après un échec
            journal.append("x", "r", {})

    def test_resoudre_un_appel_inconnu_est_refuse(self, tmp_path: Path) -> None:
        with Journal(tmp_path / "run.jsonl") as journal, pytest.raises(JournalError, match="aucune intention"):
            journal.resolve("c42", ok=True)


class TestOutilsEnParallele:
    def test_le_journal_reste_coherent_quand_les_outils_tournent_en_parallele(self, tmp_path: Path) -> None:
        """`parallel_tool_calls=True` : les intentions et résultats s'écrivent depuis des threads ;
        l'écrivain unique les sérialise, la chaîne reste valide, rien n'est écrit en double."""
        chemin = tmp_path / "run.jsonl"
        with Journal(chemin) as journal:
            agent = construire(journal)
            agent.parallel_tool_calls = True
            agent.run("go")
            types = [r["type"] for r in journal.records()]
            assert types.count("intent") == 2 and types.count("result") == 2
            assert journal.verify().ok and journal.last_resumable_run() is None
        assert nombre("envoyer_mail") == 1 and nombre("lire_compteur") == 1


class TestSansJournal:
    def test_sans_journal_rien_ne_change(self) -> None:
        agent = Agent(Scripte(), max_steps=4)
        assert agent.journal is None
        with pytest.raises(AutoAgentError, match="needs a Journal"):
            agent.resume_from_journal()

    def test_idempotency_key_est_none_hors_d_un_run_journalise(self) -> None:
        assert idempotency_key() is None
