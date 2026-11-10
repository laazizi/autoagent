"""Le confinement par le noyau du code écrit par le modèle (`SubprocessSandbox(confine=Confinement())`).

Le banc suppose la liste d'interdits AST CONTOURNÉE : les charges sont exécutées directement par
`run_python_tool`, sans passer par `validate_generated_tool_code`. Deux lectures de chaque charge :

* sans confinement elle RÉUSSIT (contrôle positif — sinon le banc est aveugle) ;
* avec le confinement elle échoue, et son effet n'a pas eu lieu.

Les tests du noyau tournent sous Linux x86_64 (WSL2 Ubuntu 24.04, noyau 6.6, Landlock ABI 3). Le filtre
d'appels système, lui, est une fonction pure : il est simulé ici instruction par instruction, sur toute
plateforme.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from autoagent import sandbox as sandbox_mod
from autoagent.errors import ToolError
from autoagent.sandbox import Confinement, SandboxLimits, SubprocessSandbox

CONFINABLE = sys.platform.startswith("linux") and platform.machine() == "x86_64"
sous_noyau = pytest.mark.skipif(
    not CONFINABLE, reason="confinement par le noyau : appliqué et mesuré sous Linux x86_64 seulement")

ARCH_X86_64 = 0xC000003E
PERMIS = 0x7FFF0000
REFUS_EPERM = 0x00050000 | 1
REFUS_ENOSYS = 0x00050000 | 38
TUER = 0x80000000


def _outil(dossier: Path, corps: str, nom: str = "outil.py") -> Path:
    """Un fichier d'outil minimal : `corps` est le corps de `run(args, context)`."""
    lignes = [f"    {ligne}" for ligne in corps.strip().splitlines()]
    fichier = dossier / nom
    fichier.write_text("def run(args, context):\n" + "\n".join(lignes) + "\n", encoding="utf-8")
    return fichier


# ---------------------------------------------------------------------------
# Le monde autour du bac : un fichier privé, un autre processus, deux ports ouverts
# ---------------------------------------------------------------------------

@dataclass
class Monde:
    outils: Path
    prive: Path
    temoin: Path
    secret: str
    canari: str
    processus: subprocess.Popen[bytes]
    port_tcp: int
    port_udp: int
    recu_tcp: list[bytes] = field(default_factory=list)
    udp: socket.socket | None = None

    def recu_udp(self) -> bytes | None:
        assert self.udp is not None
        try:
            return self.udp.recv(64)
        except (TimeoutError, BlockingIOError, OSError):
            return None

    def attendre_tcp(self, delai: float = 2.0) -> list[bytes]:
        fin = time.monotonic() + delai
        while not self.recu_tcp and time.monotonic() < fin:
            time.sleep(0.02)
        return list(self.recu_tcp)


@pytest.fixture
def monde(tmp_path: Path) -> Iterator[Monde]:
    outils = tmp_path / "outils"
    prive = tmp_path / "prive"
    outils.mkdir()
    prive.mkdir()
    secret = f"SECRET-{uuid.uuid4().hex}"
    temoin = prive / "temoin.txt"
    temoin.write_text(secret, encoding="utf-8")
    temoin.chmod(0o600)
    canari = f"cle-temoin-{uuid.uuid4().hex}"
    # Un AUTRE processus du même compte, avec une clé dans son environnement : il tient le rôle de l'hôte.
    processus = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], env={"CLE_API_TEMOIN": canari})

    serveur = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serveur.bind(("127.0.0.1", 0))
    serveur.listen(4)
    serveur.settimeout(0.2)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(0.5)
    le_monde = Monde(outils, prive, temoin, secret, canari, processus,
                     serveur.getsockname()[1], udp.getsockname()[1], udp=udp)
    stop = threading.Event()

    def accepter() -> None:
        while not stop.is_set():
            try:
                client, _ = serveur.accept()
            except OSError:
                continue
            with client:
                client.settimeout(1.0)
                try:
                    le_monde.recu_tcp.append(client.recv(64))
                except OSError:
                    pass

    fil = threading.Thread(target=accepter, daemon=True)
    fil.start()
    try:
        yield le_monde
    finally:
        stop.set()
        fil.join(timeout=2)
        serveur.close()
        udp.close()
        processus.kill()
        processus.wait(timeout=5)


# ---------------------------------------------------------------------------
# Les charges hostiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Charge:
    nom: str
    corps: str
    arguments: Callable[[Monde], dict[str, Any]]
    a_reussi: Callable[[Monde, dict[str, Any]], bool]      # sans confinement : la charge a fait son effet
    sans_effet: Callable[[Monde], bool]                    # avec : l'effet n'a PAS eu lieu


def _mode(chemin: Path) -> int:
    return stat.S_IMODE(chemin.stat().st_mode)


CHARGES = [
    Charge(
        "lire_un_fichier_prive",
        "return open(args['chemin']).read()",
        lambda m: {"chemin": str(m.temoin)},
        lambda m, res: res["result"] == m.secret,
        lambda m: True,
    ),
    Charge(
        "lire_l_environnement_d_un_autre_processus",
        "return open('/proc/%d/environ' % args['pid'], 'rb').read().decode('latin-1')",
        lambda m: {"pid": m.processus.pid},
        lambda m, res: m.canari in res["result"],
        lambda m: True,
    ),
    Charge(
        "lire_l_environnement_de_son_parent",
        "import os\nreturn len(open('/proc/%d/environ' % os.getppid(), 'rb').read())",
        lambda m: {},
        lambda m, res: res["result"] > 0,
        lambda m: True,
    ),
    Charge(
        "lister_un_dossier_prive",
        "import os\nreturn sorted(os.listdir(args['dossier']))",
        lambda m: {"dossier": str(m.prive)},
        lambda m, res: res["result"] == ["temoin.txt"],
        lambda m: True,
    ),
    Charge(
        "ecrire_hors_des_dossiers_accordes",
        "open(args['chemin'], 'w').write('pirate')\nreturn 'ecrit'",
        lambda m: {"chemin": str(m.prive / "nouveau.txt")},
        lambda m, res: (m.prive / "nouveau.txt").read_text() == "pirate",
        lambda m: not (m.prive / "nouveau.txt").exists(),
    ),
    Charge(
        "vider_un_fichier",
        "import os\nos.truncate(args['chemin'], 0)\nreturn 'vide'",
        lambda m: {"chemin": str(m.temoin)},
        lambda m, res: m.temoin.stat().st_size == 0,
        lambda m: m.temoin.read_text() == m.secret,
    ),
    Charge(
        "supprimer_un_fichier",
        "import os\nos.remove(args['chemin'])\nreturn 'supprime'",
        lambda m: {"chemin": str(m.temoin)},
        lambda m, res: not m.temoin.exists(),
        lambda m: m.temoin.exists(),
    ),
    Charge(
        "changer_les_droits_d_un_fichier",
        "import os\nos.chmod(args['chemin'], 0o777)\nreturn 'ouvert'",
        lambda m: {"chemin": str(m.temoin)},
        lambda m, res: _mode(m.temoin) == 0o777,
        lambda m: _mode(m.temoin) == 0o600,
    ),
    Charge(
        "se_connecter_en_tcp",
        "import socket\ns = socket.create_connection(('127.0.0.1', args['port']), 2)\n"
        "s.sendall(b'fuite')\ns.close()\nreturn 'envoye'",
        lambda m: {"port": m.port_tcp},
        lambda m, res: m.attendre_tcp() == [b"fuite"],
        lambda m: m.attendre_tcp(0.4) == [],
    ),
    Charge(
        "envoyer_un_datagramme_udp",
        "import socket\ns = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "s.sendto(b'fuite', ('127.0.0.1', args['port']))\nreturn 'envoye'",
        lambda m: {"port": m.port_udp},
        lambda m, res: m.recu_udp() == b"fuite",
        lambda m: m.recu_udp() is None,
    ),
    Charge(
        "lancer_un_programme",
        "import subprocess\n"
        "return subprocess.run(['/bin/sh', '-c', 'echo hors > ' + args['chemin']]).returncode",
        lambda m: {"chemin": str(m.outils / "lance.txt")},
        lambda m, res: (m.outils / "lance.txt").exists(),
        lambda m: not (m.outils / "lance.txt").exists(),
    ),
    Charge(
        "tuer_un_autre_processus",
        "import os\nos.kill(args['pid'], 9)\nreturn 'tue'",
        lambda m: {"pid": m.processus.pid},
        lambda m, res: m.processus.wait(timeout=5) == -9,
        lambda m: m.processus.poll() is None,
    ),
    Charge(
        "lire_les_limites_d_un_autre_processus",
        "import resource\nreturn list(resource.prlimit(args['pid'], resource.RLIMIT_NOFILE))",
        lambda m: {"pid": m.processus.pid},
        lambda m, res: len(res["result"]) == 2,
        lambda m: True,
    ),
]

IDS = [charge.nom for charge in CHARGES]


@sous_noyau
class TestBancDeCharges:
    @pytest.mark.parametrize("charge", CHARGES, ids=IDS)
    def test_sans_confinement_la_charge_reussit(self, monde: Monde, charge: Charge) -> None:
        """Contrôle positif : le bac d'aujourd'hui laisse passer. Si ce test rougit, le banc ne mesure rien."""
        res = SubprocessSandbox(timeout=30).run_python_tool(
            _outil(monde.outils, charge.corps), charge.arguments(monde))
        assert res["ok"] is True, res
        assert charge.a_reussi(monde, res)

    @pytest.mark.parametrize("charge", CHARGES, ids=IDS)
    def test_confinee_la_charge_echoue(self, monde: Monde, charge: Charge) -> None:
        bac = SubprocessSandbox(timeout=30, confine=Confinement())
        res = bac.run_python_tool(_outil(monde.outils, charge.corps), charge.arguments(monde))
        assert res["ok"] is False, res
        assert "PermissionError" in res["error"], res
        assert monde.secret not in json.dumps(res) and monde.canari not in json.dumps(res)
        assert charge.sans_effet(monde)

    def test_un_enfant_cree_par_fork_herite_du_confinement(self, monde: Monde) -> None:
        corps = (
            "import os\n"
            "lecture, ecriture = os.pipe()\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    try:\n"
            "        open(args['chemin']).read()\n"
            "        os.write(ecriture, b'lu')\n"
            "    except PermissionError:\n"
            "        os.write(ecriture, b'refuse')\n"
            "    os._exit(0)\n"
            "os.waitpid(pid, 0)\n"
            "return os.read(lecture, 16).decode()"
        )
        bac = SubprocessSandbox(timeout=30, confine=Confinement())
        res = bac.run_python_tool(_outil(monde.outils, corps), {"chemin": str(monde.temoin)})
        assert res == {"ok": True, "result": "refuse", "stdout": ""}

    def test_le_pont_et_le_worker_chaud_sont_confines_eux_aussi(self, monde: Monde) -> None:
        corps = "return open(args['chemin']).read()"
        fichier = _outil(monde.outils, corps)
        bac = SubprocessSandbox(timeout=30, confine=Confinement())
        par_le_pont = bac.run_python_tool(
            fichier, {"chemin": str(monde.temoin)}, host_functions={"ping": lambda: "pong"})
        assert par_le_pont["ok"] is False and "PermissionError" in par_le_pont["error"]
        with SubprocessSandbox(timeout=30, warm=True, confine=Confinement()) as chaud:
            a_chaud = chaud.run_python_tool(fichier, {"chemin": str(monde.temoin)})
        assert a_chaud["ok"] is False and "PermissionError" in a_chaud["error"]


# ---------------------------------------------------------------------------
# Le légitime ne change pas
# ---------------------------------------------------------------------------

LEGITIMES = {
    "json_re_math": "import json, re, math\n"
                    "return {'j': json.dumps({'a': 1}), 'r': re.findall(r'\\d+', 'a1b22'), 'm': math.sqrt(16)}",
    "heure_locale": "import datetime\n"
                    "return datetime.datetime(2026, 7, 1, 12).astimezone().utcoffset().total_seconds()",
    "fuseaux": "import datetime, zoneinfo\n"
               "paris = zoneinfo.ZoneInfo('Europe/Paris')\n"
               "return str(datetime.datetime(2026, 7, 1, 12, tzinfo=paris).utcoffset())",
    "decimal_statistics_random": "import decimal, statistics, random\n"
                                 "random.seed(7)\n"
                                 "return [str(decimal.Decimal('1.1') + decimal.Decimal('2.2')),\n"
                                 "        statistics.median([3, 1, 2]), random.randint(0, 1000)]",
    "hashlib_csv_io": "import hashlib, csv, io\n"
                      "lignes = list(csv.reader(io.StringIO('a,b\\n1,2\\n')))\n"
                      "return [hashlib.sha256(b'x').hexdigest()[:8], lignes]",
    "fils_d_execution": "from concurrent.futures import ThreadPoolExecutor\n"
                        "with ThreadPoolExecutor(4) as fils:\n"
                        "    return list(fils.map(lambda x: x * x, range(8)))",
    "asyncio": "import asyncio\nreturn asyncio.run(asyncio.sleep(0, 'ok'))",
    "sqlite_en_memoire": "import sqlite3\nreturn sqlite3.connect(':memory:').execute('select 6 * 7').fetchone()[0]",
    "alea_du_noyau": "import os\nreturn len(os.urandom(8))",
    "un_signal_a_soi_meme": "import os\nos.kill(os.getpid(), 0)\nreturn 'moi'",
    "types_mime": "import mimetypes\nreturn mimetypes.guess_type('rapport.html')[0]",
    "ses_propres_limites": "import resource\nreturn len(resource.getrlimit(resource.RLIMIT_NOFILE))",
}


@sous_noyau
class TestLeLegitimeNeChangePas:
    @pytest.mark.parametrize("nom", sorted(LEGITIMES))
    def test_meme_resultat_avec_et_sans_confinement(self, tmp_path: Path, nom: str) -> None:
        fichier = _outil(tmp_path, LEGITIMES[nom])
        libre = SubprocessSandbox(timeout=30).run_python_tool(fichier, {})
        confine = SubprocessSandbox(timeout=30, confine=Confinement()).run_python_tool(fichier, {})
        assert libre["ok"] is True, libre
        assert confine == libre

    def test_le_pont_repond_sous_confinement(self, tmp_path: Path) -> None:
        fichier = _outil(tmp_path, "return context['call_host']('double', {'x': 21})")
        bac = SubprocessSandbox(timeout=30, confine=Confinement())
        res = bac.run_python_tool(fichier, {}, host_functions={"double": lambda x: 2 * x})
        assert res["ok"] is True and res["result"] == 42

    def test_le_worker_chaud_sert_plusieurs_appels_sous_confinement(self, tmp_path: Path) -> None:
        fichier = _outil(tmp_path, "return args['x'] + 1")
        with SubprocessSandbox(timeout=30, warm=True, confine=Confinement()) as bac:
            assert [bac.run_python_tool(fichier, {"x": i})["result"] for i in range(3)] == [1, 2, 3]
            assert next(iter(bac._workers.values())).calls == 3, "le MÊME worker a servi les trois appels"

    def test_les_plafonds_de_ressources_se_combinent_au_confinement(self, tmp_path: Path) -> None:
        bac = SubprocessSandbox(timeout=30, confine=Confinement(), limits=SandboxLimits(memory_mb=256))
        ok = bac.run_python_tool(_outil(tmp_path, "return sum(range(1000))"), {})
        assert ok["ok"] is True and ok["result"] == 499500
        trop = bac.run_python_tool(_outil(tmp_path, "return len(bytearray(600 * 1024 * 1024))", "gros.py"), {})
        assert trop["ok"] is False and "MemoryError" in trop["error"]


# ---------------------------------------------------------------------------
# Les dossiers accordés par l'hôte, et le réseau rouvert
# ---------------------------------------------------------------------------

@sous_noyau
class TestCeQueLHoteAccorde:
    def test_un_dossier_en_lecture_se_lit_et_ne_s_ecrit_pas(self, monde: Monde) -> None:
        bac = SubprocessSandbox(timeout=30, confine=Confinement(read=[str(monde.prive)]))
        lu = bac.run_python_tool(_outil(monde.outils, "return open(args['c']).read()"), {"c": str(monde.temoin)})
        assert lu["ok"] is True and lu["result"] == monde.secret
        ecrit = bac.run_python_tool(
            _outil(monde.outils, "open(args['c'], 'w').write('x')\nreturn 1", "ecrire.py"),
            {"c": str(monde.prive / "nouveau.txt")})
        assert ecrit["ok"] is False and "PermissionError" in ecrit["error"]
        assert not (monde.prive / "nouveau.txt").exists()

    def test_un_dossier_en_ecriture_s_ecrit_se_relit_et_borne_le_reste(self, monde: Monde, tmp_path: Path) -> None:
        sorties = tmp_path / "sorties"
        sorties.mkdir()
        corps = (
            "import os\n"
            "os.makedirs(args['d'] + '/sous', exist_ok=True)\n"
            "chemin = args['d'] + '/sous/rapport.txt'\n"
            "open(chemin + '.tmp', 'w').write('bilan')\n"
            "os.replace(chemin + '.tmp', chemin)\n"
            "return open(chemin).read()"
        )
        bac = SubprocessSandbox(timeout=30, confine=Confinement(write=[str(sorties)]))
        res = bac.run_python_tool(_outil(monde.outils, corps), {"d": str(sorties)})
        assert res["ok"] is True and res["result"] == "bilan"
        assert (sorties / "sous" / "rapport.txt").read_text() == "bilan"
        ailleurs = bac.run_python_tool(
            _outil(monde.outils, "return open(args['c']).read()", "lire.py"), {"c": str(monde.temoin)})
        assert ailleurs["ok"] is False and "PermissionError" in ailleurs["error"]

    def test_le_reseau_ne_se_rouvre_que_si_l_hote_ET_l_outil_le_disent(self, monde: Monde) -> None:
        corps = CHARGES[IDS.index("se_connecter_en_tcp")].corps
        fichier = _outil(monde.outils, corps)
        args = {"port": monde.port_tcp}
        ferme = SubprocessSandbox(timeout=30, confine=Confinement())
        refuse = ferme.run_python_tool(fichier, args, allow_network=True)
        assert refuse["ok"] is False, "Confinement() coupe le réseau même pour un outil qui a la permission"
        ouvert = SubprocessSandbox(timeout=30, confine=Confinement(network=True))
        sans_permission = ouvert.run_python_tool(fichier, args, allow_network=False)
        assert sans_permission["ok"] is False, "network=True ne suffit pas : l'outil doit porter la permission"
        assert monde.attendre_tcp(0.4) == []
        passe = ouvert.run_python_tool(fichier, args, allow_network=True)
        assert passe["ok"] is True and monde.attendre_tcp() == [b"fuite"]

    def test_reseau_ouvert_les_noms_et_les_certificats_se_lisent_comme_sans_confinement(self, tmp_path: Path) -> None:
        corps = (
            "import socket, ssl\n"
            "adresses = sorted({info[4][0] for info in socket.getaddrinfo('localhost', 80)})\n"
            "return [adresses, ssl.create_default_context().cert_store_stats()['x509_ca'] > 0]"
        )
        fichier = _outil(tmp_path, corps)
        libre = SubprocessSandbox(timeout=30).run_python_tool(fichier, {})
        ouvert = SubprocessSandbox(timeout=30, confine=Confinement(network=True)).run_python_tool(
            fichier, {}, allow_network=True)
        assert libre["ok"] is True, libre
        assert ouvert == libre


# ---------------------------------------------------------------------------
# Refuser plutôt que tourner sans frontière (fail-closed)
# ---------------------------------------------------------------------------

class TestRefuserPlutotQueSeReplier:
    def test_hors_linux_x86_64_la_construction_est_refusee(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_mod, "_confinement_impossible", lambda: "Linux only (this platform: win32)")
        with pytest.raises(ToolError, match="Linux only"):
            SubprocessSandbox(confine=Confinement())

    @pytest.mark.skipif(CONFINABLE, reason="sur une machine confinable, la construction réussit")
    def test_sur_cette_machine_non_confinable_le_refus_est_reel(self) -> None:
        with pytest.raises(ToolError, match="confine"):
            SubprocessSandbox(confine=Confinement())

    def test_confine_doit_etre_un_confinement(self) -> None:
        with pytest.raises(TypeError, match="Confinement"):
            SubprocessSandbox(confine=True)  # type: ignore[arg-type]

    @pytest.mark.parametrize("mauvais", [{"read": "/srv"}, {"write": ["relatif/dossier"]}, {"network": "oui"}])
    def test_un_confinement_mal_ecrit_est_refuse_a_la_construction(self, mauvais: dict[str, Any]) -> None:
        with pytest.raises((TypeError, ValueError)):
            Confinement(**mauvais)

    @sous_noyau
    def test_un_dossier_accorde_doit_exister(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="existing directory"):
            SubprocessSandbox(confine=Confinement(read=[str(tmp_path / "absent")]))

    @sous_noyau
    def test_si_le_confinement_ne_se_pose_pas_l_outil_ne_tourne_pas(self, tmp_path: Path) -> None:
        accorde = tmp_path / "accorde"
        accorde.mkdir()
        bac = SubprocessSandbox(timeout=30, confine=Confinement(read=[str(accorde)]))
        accorde.rmdir()                                    # la règle ne peut plus être posée dans l'enfant
        appels: list[str] = []
        fichier = _outil(tmp_path, "return context['call_host']('marque', {})")
        with pytest.raises(ToolError, match="confinement"):
            bac.run_python_tool(fichier, {}, host_functions={"marque": lambda: appels.append("x")})
        assert appels == [], "l'outil n'a pas tourné"
        with pytest.raises(ToolError, match="confinement"):
            bac.run_python_tool(_outil(tmp_path, "return 1", "simple.py"), {})

    def test_un_enfant_mort_avant_de_lire_rend_une_erreur_typee(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Trouvé par le banc sur le Python système d'Ubuntu : un préambule qui refuse de démarrer tue l'enfant
        avant qu'il lise son entrée, et le pont laissait remonter une `BrokenPipeError` brute — une fois sur
        deux, selon la vitesse des deux processus. Ici l'entrée dépasse le tampon du tube : l'écriture échoue
        à coup sûr."""
        monkeypatch.setattr(sandbox_mod, "_BRIDGE_RUNNER_CODE", "raise SystemExit('mort avant de lire')")
        fichier = _outil(tmp_path, "return 1")
        with pytest.raises(ToolError, match="mort avant de lire"):
            SubprocessSandbox(timeout=30).run_python_tool(
                fichier, {"gros": "a" * 300_000}, host_functions={"ping": lambda: "pong"})

    def test_sans_confinement_rien_ne_change(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        vus: list[list[str]] = []
        reel = subprocess.run

        def espion(cmd: list[str], *a: Any, **k: Any) -> Any:
            vus.append(list(cmd))
            return reel(cmd, *a, **k)

        monkeypatch.setattr(sandbox_mod.subprocess, "run", espion)
        bac = SubprocessSandbox(timeout=30)
        res = bac.run_python_tool(_outil(tmp_path, "return 1"), {})
        assert res["ok"] is True and vus[0][-2] == sandbox_mod.RUNNER_CODE
        assert bac.isolation()["os_boundary"] is False and "confinement" not in bac.isolation()


@sous_noyau
class TestSousLaBoucle:
    """Le confinement branché là où le modèle écrit du code : `PythonRunner` (ce que lancent `run_python`
    et `enable_code_action`)."""

    def test_un_programme_du_modele_tourne_confine_et_appelle_l_hote_par_le_pont(self) -> None:
        from autoagent.dynamic import PythonRunner

        lanceur = PythonRunner(SubprocessSandbox(timeout=30, confine=Confinement()),
                               host_functions={"compter": lambda capteur: {"capteur": capteur, "total": 42}})
        sortie = lanceur("def run(args, context):\n"
                         "    return sum(context['call_host']('compter', {'capteur': c})['total'] for c in args['c'])",
                         {"c": ["A", "B", "C"]})
        assert sortie["result"] == 126

    def test_confine_le_lanceur_n_avertit_plus_qu_il_n_y_a_pas_de_frontiere(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from autoagent import logging as journal
        from autoagent.dynamic import PythonRunner

        monkeypatch.setattr(journal, "_DEJA_AVERTI", set())
        with caplog.at_level("WARNING"):
            PythonRunner(SubprocessSandbox(confine=Confinement()))
        assert not [r for r in caplog.records if "not an isolation boundary" in r.getMessage()]
        with caplog.at_level("WARNING"):
            PythonRunner(SubprocessSandbox())
        assert [r for r in caplog.records if "not an isolation boundary" in r.getMessage()], \
            "sans confinement, l'avertissement part toujours"


# ---------------------------------------------------------------------------
# Le confinement automatique : quand l'hôte ne choisit pas de bac
# ---------------------------------------------------------------------------

ECRIRE_SANS_OPEN = (
    "def run(args, context):\n"
    "    import sqlite3\n"
    "    base = sqlite3.connect(args['chemin'])\n"
    "    base.execute('create table t (x)')\n"
    "    base.commit()\n"
    "    return 'ecrit'\n"
)

GABARIT_D_OUTIL = (
    "TOOL = {{'name': '{nom}', 'description': 'd', 'input_schema': {{'type': 'object', 'properties': {{}}}},"
    " 'permissions': {permissions}}}\n\ndef run(args, context):\n    return 1\n"
)


@pytest.fixture
def avertissements_neufs(monkeypatch: pytest.MonkeyPatch) -> None:
    """`warn_once` se souvient par processus : chaque test repart de zéro."""
    from autoagent import logging as journal

    monkeypatch.setattr(journal, "_DEJA_AVERTI", set())


def _sans_frontiere(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "not an isolation boundary" in r.getMessage()]


@sous_noyau
@pytest.mark.usefixtures("avertissements_neufs")
class TestConfinementAutomatique:
    """Règle : quand l'hôte ne choisit pas de bac, le noyau fait respecter ce que les permissions disent déjà."""

    def test_le_lanceur_par_defaut_est_confine_et_ferme_un_trou_du_validateur(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """`sqlite3.connect(chemin)` écrit un fichier SANS `open()` ni module interdit : le validateur le
        laisse passer alors que le lanceur n'a aucune permission de fichiers. Le noyau, lui, refuse."""
        from autoagent.dynamic import PythonRunner

        avant = PythonRunner(SubprocessSandbox(timeout=30))               # le bac d'hier, choisi explicitement
        assert avant(ECRIRE_SANS_OPEN, {"chemin": str(tmp_path / "hier.db")})["result"] == "ecrit"
        assert (tmp_path / "hier.db").exists(), "contrôle positif : sans frontière, le fichier est écrit"

        caplog.clear()
        with caplog.at_level("WARNING"):
            lanceur = PythonRunner(timeout=30)
        assert lanceur.sandbox.isolation()["kind"] == "subprocess+confine"
        assert _sans_frontiere(caplog) == []
        with pytest.raises(ToolError, match="OperationalError|PermissionError"):
            lanceur(ECRIRE_SANS_OPEN, {"chemin": str(tmp_path / "demain.db")})
        assert not (tmp_path / "demain.db").exists()
        assert lanceur("def run(args, context):\n    return sum(args['v'])", {"v": [1, 2, 3]})["result"] == 6

    def test_des_fichiers_accordes_sans_dire_lesquels_le_lanceur_tourne_comme_avant(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from autoagent.dynamic import PythonRunner

        (tmp_path / "note.txt").write_text("contenu", encoding="utf-8")
        with caplog.at_level("WARNING"):
            lanceur = PythonRunner(timeout=30, permissions={"filesystem.read"})
        assert lanceur.sandbox.isolation()["os_boundary"] is False
        assert _sans_frontiere(caplog), "sans frontière, l'avertissement part toujours"
        lu = lanceur("def run(args, context):\n    return open(args['c']).read()", {"c": str(tmp_path / "note.txt")})
        assert lu["result"] == "contenu"

    def test_le_bac_choisi_par_l_hote_n_est_jamais_remplace(self) -> None:
        from autoagent.dynamic import PythonRunner

        choisi = SubprocessSandbox(timeout=3)
        assert PythonRunner(choisi).sandbox is choisi

    def test_le_reseau_du_bac_automatique_suit_la_permission_de_l_outil(self) -> None:
        from autoagent.dynamic import PythonRunner

        dit = PythonRunner().sandbox.isolation()
        assert dit["confinement"]["network"] is True, "rouvert pour les seuls outils porteurs de la permission"

    def test_un_outil_charge_sans_bac_suit_ses_propres_permissions(self, tmp_path: Path) -> None:
        sobre = tmp_path / "sobre.py"
        sobre.write_text(GABARIT_D_OUTIL.format(nom="sobre", permissions=[]), encoding="utf-8")
        fichiers = tmp_path / "fichiers.py"
        fichiers.write_text(GABARIT_D_OUTIL.format(nom="fichiers", permissions=["filesystem.read"]), encoding="utf-8")
        assert sandbox_mod.load_generated_tool(sobre).sandbox.isolation()["os_boundary"] is True
        assert sandbox_mod.load_generated_tool(fichiers).sandbox.isolation()["os_boundary"] is False
        assert sandbox_mod.load_generated_tool(sobre)() == 1

    def test_le_constructeur_d_outils_choisit_le_bac_outil_par_outil(self, tmp_path: Path) -> None:
        from autoagent.dynamic import DynamicToolBuilder

        from .conftest import FakeLLMProvider

        constructeur = DynamicToolBuilder(FakeLLMProvider(), tools_dir=tmp_path / "outils")
        assert constructeur.sandbox.isolation()["os_boundary"] is True
        assert constructeur._bac_pour([]) is constructeur.sandbox
        assert constructeur._bac_pour(["network"]) is constructeur.sandbox
        assert constructeur._bac_pour(["filesystem.write"]).isolation()["os_boundary"] is False
        choisi = SubprocessSandbox(timeout=3)
        impose = DynamicToolBuilder(FakeLLMProvider(), tools_dir=tmp_path / "autres", sandbox=choisi)
        assert impose._bac_pour(["filesystem.write"]) is choisi and impose._bac_pour([]) is choisi

    def test_un_outil_construit_puis_recharge_tourne_dans_le_bac_de_ses_permissions(self, tmp_path: Path) -> None:
        """Le choix se fait à la CONSTRUCTION de l'outil et de nouveau à son RECHARGEMENT depuis la bibliothèque."""
        from autoagent.dynamic import DynamicToolBuilder, ToolBuildRequest
        from autoagent.providers.base import LLMProvider
        from autoagent.schema import LLMResponse, ModelConfig

        def charge(nom: str, permissions: list[str]) -> str:
            return json.dumps({
                "tool": {"name": nom, "description": "double un entier", "permissions": permissions,
                         "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}},
                "code": "def run(args, context):\n    return {'double': args['n'] * 2}\n",
                "self_tests": [{"args": {"n": 2}, "expect_equals": {"double": 4}}],
            })

        class Constructeur(LLMProvider):
            def __init__(self, contenus: list[str]) -> None:
                super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
                self.contenus = list(contenus)

            def complete(self, request: Any) -> LLMResponse:
                return LLMResponse(content=self.contenus.pop(0), model="f")

        fournisseur = Constructeur([charge("sobre", []), charge("lecteur", ["filesystem.read"])])
        constructeur = DynamicToolBuilder(fournisseur, tools_dir=tmp_path, persist=True)
        sobre = constructeur.build(ToolBuildRequest(capability="doubler"))
        lecteur = constructeur.build(ToolBuildRequest(capability="doubler en lisant"))
        assert sobre.sandbox.isolation()["os_boundary"] is True and sobre(n=4) == {"double": 8}
        assert lecteur.sandbox.isolation()["os_boundary"] is False and lecteur(n=4) == {"double": 8}

        recharges = {o.spec.name: o for o in DynamicToolBuilder(Constructeur([]), tools_dir=tmp_path, persist=True).load_library()}
        assert recharges["sobre"].sandbox.isolation()["os_boundary"] is True
        assert recharges["lecteur"].sandbox.isolation()["os_boundary"] is False


@pytest.mark.usefixtures("avertissements_neufs")
class TestChoixDuBacSansDocker:
    """`make_sandbox()` : Docker, sinon le sous-processus confiné, sinon le sous-processus simple — qui le dit."""

    def test_sans_docker_ni_confinement_le_repli_est_celui_d_avant_et_il_avertit(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(sandbox_mod, "docker_available", lambda: False)
        monkeypatch.setattr(sandbox_mod, "_confinement_impossible", lambda: "pas ici")
        with caplog.at_level("WARNING"):
            bac = sandbox_mod.make_sandbox(timeout=7)
        assert bac.isolation()["kind"] == "subprocess" and bac.timeout == 7
        assert _sans_frontiere(caplog)

    def test_exiger_une_frontiere_refuse_quand_il_n_y_en_a_aucune(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sandbox_mod, "docker_available", lambda: False)
        monkeypatch.setattr(sandbox_mod, "_confinement_impossible", lambda: "pas ici")
        with pytest.raises(ToolError, match="require_boundary"):
            sandbox_mod.make_sandbox(require_boundary=True)

    def test_le_sous_processus_choisi_en_connaissance_de_cause_reste_simple(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(sandbox_mod, "docker_available", lambda: True)
        with caplog.at_level("WARNING"):
            bac = sandbox_mod.make_sandbox(prefer_docker=False)
        assert bac.isolation()["kind"] == "subprocess" and _sans_frontiere(caplog) == []

    def test_sur_une_machine_non_confinable_le_lanceur_par_defaut_est_celui_d_avant(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from autoagent.dynamic import PythonRunner

        monkeypatch.setattr(sandbox_mod, "_confinement_impossible", lambda: "pas ici")
        with caplog.at_level("WARNING"):
            lanceur = PythonRunner()
        assert lanceur.sandbox.isolation()["kind"] == "subprocess" and _sans_frontiere(caplog)

    @sous_noyau
    def test_sans_docker_le_repli_est_confine_et_n_avertit_pas(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(sandbox_mod, "docker_available", lambda: False)
        with caplog.at_level("WARNING"):
            bac = sandbox_mod.make_sandbox(timeout=7)
            exige = sandbox_mod.make_sandbox(require_boundary=True)
        assert bac.isolation()["kind"] == "subprocess+confine" and bac.timeout == 7
        assert exige.isolation()["os_boundary"] is True
        assert _sans_frontiere(caplog) == []
        with pytest.raises(ToolError, match="Docker"):
            sandbox_mod.make_sandbox(require_docker=True)


@sous_noyau
class TestIsolationDitLaVerite:
    def test_le_bac_confine_declare_ce_qu_il_applique(self, tmp_path: Path) -> None:
        bac = SubprocessSandbox(timeout=7, confine=Confinement(write=[str(tmp_path)]),
                                limits=SandboxLimits(memory_mb=256))
        dit = bac.isolation()
        assert dit["kind"] == "subprocess+confine"
        assert dit["os_boundary"] is True and dit["filesystem_isolated"] is True and dit["network_isolated"] is True
        assert dit["confinement"]["landlock_abi"] >= 3
        assert dit["confinement"]["seccomp"] == "denylist" and dit["confinement"]["arch"] == "x86_64"
        assert dit["confinement"]["write"] == [os.path.realpath(tmp_path)]
        assert dit["confinement"]["network"] is False
        assert dit["limits"] == {"timeout_s": 7.0, "memory_mb": 256}

    def test_reseau_ouvert_n_est_pas_declare_isole(self) -> None:
        dit = SubprocessSandbox(confine=Confinement(network=True)).isolation()
        assert dit["network_isolated"] is False and dit["confinement"]["network"] is True


# ---------------------------------------------------------------------------
# Le filtre d'appels système : une fonction pure, simulée sur toute plateforme
# ---------------------------------------------------------------------------

def _simuler(filtre: bytes, nr: int, *, arch: int = ARCH_X86_64, args: tuple[int, ...] = ()) -> int:
    """Exécute le programme BPF comme le noyau : rend l'action décidée pour cet appel."""
    complets = tuple(args) + (0,) * (6 - len(args))
    donnees = struct.pack("<IIQ6Q", nr & 0xFFFFFFFF, arch, 0, *complets)
    instructions = [struct.unpack_from("<HBBI", filtre, i) for i in range(0, len(filtre), 8)]
    accumulateur, pc = 0, 0
    while True:
        code, si_vrai, si_faux, k = instructions[pc]
        pc += 1
        if code == 0x20:
            accumulateur = struct.unpack_from("<I", donnees, k)[0]
        elif code == 0x15:
            pc += si_vrai if accumulateur == k else si_faux
        elif code == 0x25:
            pc += si_vrai if accumulateur > k else si_faux
        elif code == 0x45:
            pc += si_vrai if accumulateur & k else si_faux
        elif code == 0x06:
            return k
        else:
            raise AssertionError(f"instruction BPF inattendue : {code:#x}")


MOI = 4242


def _filtre(reseau_ouvert: bool = False) -> bytes:
    programme, positions = sandbox_mod._filtre_seccomp(reseau_ouvert)
    return sandbox_mod._poser_pid(programme, positions, MOI)


class TestFiltreDAppelsSysteme:
    def test_chaque_appel_refuse_rend_eperm(self) -> None:
        filtre = _filtre()
        for nom, numero in sandbox_mod._APPELS_REFUSES.items():
            assert _simuler(filtre, numero) == REFUS_EPERM, nom
        for nom, numero in sandbox_mod._APPELS_RESEAU.items():
            assert _simuler(filtre, numero) == REFUS_EPERM, nom

    def test_les_appels_critiques_sont_refuses_avec_des_numeros_ecrits_ici(self) -> None:
        """Indépendant de la table du module : retirer `execve` de la liste ne se voyait nulle part ailleurs,
        Landlock refusant déjà de lancer un programme (trouvé par mutation). Deux barrières, deux tests."""
        filtre = _filtre()
        critiques = {"execve": 59, "execveat": 322, "ptrace": 101, "process_vm_readv": 310, "mount": 165,
                     "unshare": 272, "setns": 308, "bpf": 321, "io_uring_setup": 425, "chmod": 90, "fchownat": 260,
                     "utimensat": 280, "socket": 41, "connect": 42, "bind": 49, "pidfd_open": 434, "tkill": 200,
                     "init_module": 175, "kexec_load": 246, "chroot": 161, "shmat": 30}
        for nom, numero in critiques.items():
            assert _simuler(filtre, numero) == REFUS_EPERM, nom

    def test_les_appels_ordinaires_passent(self) -> None:
        filtre = _filtre()
        ordinaires = {"read": 0, "write": 1, "close": 3, "mmap": 9, "openat": 257, "futex": 202, "fork": 57,
                      "socketpair": 53, "getrandom": 318, "exit_group": 231, "landlock_restrict_self": 446}
        for nom, numero in ordinaires.items():
            assert _simuler(filtre, numero) == PERMIS, nom

    def test_reseau_ouvert_seuls_les_appels_reseau_se_rouvrent(self) -> None:
        ouvert = _filtre(reseau_ouvert=True)
        for nom, numero in sandbox_mod._APPELS_RESEAU.items():
            assert _simuler(ouvert, numero) == PERMIS, nom
        for nom, numero in sandbox_mod._APPELS_REFUSES.items():
            assert _simuler(ouvert, numero) == REFUS_EPERM, nom

    def test_un_appel_inconnu_du_filtre_est_refuse_par_defaut(self) -> None:
        filtre = _filtre()
        assert _simuler(filtre, sandbox_mod._DERNIER_APPEL_CONNU) == PERMIS
        assert _simuler(filtre, sandbox_mod._DERNIER_APPEL_CONNU + 1) == REFUS_ENOSYS
        assert _simuler(filtre, 0x40000000 | 59) == REFUS_ENOSYS, "la table x32 est refusée entière"

    def test_une_autre_architecture_tue_le_processus(self) -> None:
        assert _simuler(_filtre(), 11, arch=0x40000003) == TUER, "execve en 32 bits (int 0x80) porte le numéro 11"

    def test_un_signal_ne_part_que_vers_soi(self) -> None:
        filtre = _filtre()
        kill, tgkill = 62, 234
        assert _simuler(filtre, kill, args=(MOI, 15)) == PERMIS
        for cible in (MOI + 1, 0, 1, 0xFFFFFFFF, (1 << 32) - MOI):
            assert _simuler(filtre, kill, args=(cible, 9)) == REFUS_EPERM, cible
        assert _simuler(filtre, tgkill, args=(MOI, 77, 6)) == PERMIS
        assert _simuler(filtre, tgkill, args=(MOI + 1, 77, 6)) == REFUS_EPERM

    def test_les_limites_ne_se_lisent_que_pour_soi(self) -> None:
        filtre = _filtre()
        prlimit64 = 302
        assert _simuler(filtre, prlimit64, args=(0, 7)) == PERMIS
        assert _simuler(filtre, prlimit64, args=(MOI, 7)) == PERMIS
        assert _simuler(filtre, prlimit64, args=(MOI + 1, 7)) == REFUS_EPERM

    def test_un_fil_passe_un_espace_de_noms_non(self) -> None:
        filtre = _filtre()
        clone, clone3 = 56, 435
        fil = 0x00010000 | 0x00000100 | 0x00000200 | 0x00000400 | 0x00000800       # CLONE_THREAD | VM | FS | FILES | SIGHAND
        assert _simuler(filtre, clone, args=(fil,)) == PERMIS
        for espace in (0x00020000, 0x10000000, 0x20000000, 0x40000000, 0x04000000, 0x08000000, 0x02000000, 0x80):
            assert _simuler(filtre, clone, args=(fil | espace,)) == REFUS_EPERM, hex(espace)
        assert _simuler(filtre, clone3) == REFUS_ENOSYS, "clone3 cache ses drapeaux : la libc retombe sur clone"

    def test_le_pid_est_pose_a_chaque_position_annoncee(self) -> None:
        programme, positions = sandbox_mod._filtre_seccomp(False)
        assert len(positions) == 3 and len(programme) % 8 == 0
        for position in positions:
            assert struct.unpack_from("<I", programme, position)[0] == sandbox_mod._PID_A_POSER
        assert sandbox_mod._poser_pid(programme, positions, 99) != sandbox_mod._poser_pid(programme, positions, 100)

    @pytest.mark.skipif(not Path("/usr/include/x86_64-linux-gnu/asm/unistd_64.h").exists(),
                        reason="en-têtes du noyau absents de cette machine")
    def test_les_numeros_concordent_avec_les_en_tetes_du_noyau(self) -> None:
        texte = Path("/usr/include/x86_64-linux-gnu/asm/unistd_64.h").read_text()
        officiels = {nom: int(numero) for nom, numero in re.findall(r"#define __NR_(\w+) (\d+)", texte)}
        connus = {**sandbox_mod._APPELS_REFUSES, **sandbox_mod._APPELS_RESEAU,
                  "kill": 62, "tgkill": 234, "prlimit64": 302, "clone": 56, "clone3": 435}
        ecarts = {nom: (numero, officiels[nom]) for nom, numero in connus.items()
                  if nom in officiels and officiels[nom] != numero}
        assert ecarts == {}
        assert len([nom for nom in connus if nom in officiels]) >= len(connus) - 6, "trop de noms absents des en-têtes"
