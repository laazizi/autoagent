"""Le client WebSocket synchrone (`autoagent.ws`, RFC 6455, bibliothèque standard seulement).

Chaque test parle à `tests/_fake_ws.py`, un serveur qui produit des trames BRUTES : le client doit tenir face à
une fragmentation, un ping au milieu d'un message, une trame coupée en deux par un délai, et refuser les trames
invalides plutôt que de les interpréter (le même contrat que les fournisseurs HTTP : un échec est TYPÉ, jamais
un message tronqué rendu comme un succès).
"""

from __future__ import annotations

import base64
import socket
import struct
import threading
import time

import pytest

from autoagent.ws import WebSocketClient, WebSocketClosed, WebSocketError
from tests._fake_ws import FakeWSServer


@pytest.fixture
def serveur():
    crees: list[FakeWSServer] = []

    def _creer(script) -> FakeWSServer:
        s = FakeWSServer(script)
        crees.append(s)
        return s

    yield _creer
    for s in crees:
        s.close()
    for s in crees:
        assert not s.errors, f"le scénario du serveur a échoué : {s.errors[0]!r}"


def _client(serveur: FakeWSServer, chemin: str = "/", **options) -> WebSocketClient:
    options.setdefault("timeout", 3.0)
    return WebSocketClient(serveur.url(chemin), **options)


def _accepte_puis_attend(conn) -> None:
    conn.accept()
    conn.attendre_la_fin()


# ── la poignée de main ───────────────────────────────────────────────────────────────
def test_la_poignee_de_main_envoie_les_en_tetes_du_protocole_et_les_en_tetes_de_l_hote(serveur):
    vu: dict = {}

    def script(conn):
        vu.update(conn.headers, chemin=conn.path)
        conn.accept()
        conn.attendre_la_fin()
    s = serveur(script)
    c = _client(s, "/chemin?x=1", headers={"x-test": "valeur"})
    c.connect()
    c.close()
    assert vu["chemin"] == "/chemin?x=1"
    assert vu["upgrade"].lower() == "websocket"
    assert "upgrade" in vu["connection"].lower()
    assert vu["sec-websocket-version"] == "13"
    assert len(base64.b64decode(vu["sec-websocket-key"])) == 16
    assert vu["host"] == f"127.0.0.1:{s.port}"
    assert vu["x-test"] == "valeur"


def test_deux_connexions_n_utilisent_jamais_la_meme_cle(serveur):
    cles: list[str] = []

    def script(conn):
        cles.append(conn.headers["sec-websocket-key"])
        _accepte_puis_attend(conn)
    s = serveur(script)
    for _ in range(2):
        c = _client(s)
        c.connect()
        c.close()
    assert len(cles) == 2 and cles[0] != cles[1]


def test_une_reponse_avec_une_mauvaise_cle_d_acceptation_est_refusee(serveur):
    def script(conn):
        conn.accept(accept="Y2xlLWZhdXNzZQ==")
        conn.attendre_la_fin()
    c = _client(serveur(script))
    with pytest.raises(WebSocketError, match="Sec-WebSocket-Accept"):
        c.connect()
    assert c.closed


def test_un_refus_http_donne_le_statut_sans_jamais_reveler_la_requete(serveur):
    """La clé d'API voyage dans la requête (`?key=`) : aucun message d'erreur ne doit la contenir."""
    def script(conn):
        conn.refuse("403 Forbidden", b"clef refusee")
    c = _client(serveur(script), "/ws?key=SECRET123")
    with pytest.raises(WebSocketError) as e:
        c.connect()
    assert "403" in str(e.value)
    assert e.value.status == 403                           # structuré : l'hôte décide « réessayable ? » sans lire le texte
    assert "SECRET123" not in str(e.value) and "key=" not in str(e.value)


def test_une_erreur_sans_reponse_http_n_a_pas_de_statut():
    libre = socket.socket()
    libre.bind(("127.0.0.1", 0))
    port = libre.getsockname()[1]
    libre.close()
    with pytest.raises(WebSocketError) as e:
        WebSocketClient(f"ws://127.0.0.1:{port}/", timeout=2.0).connect()
    assert e.value.status is None


def test_une_connexion_refusee_ne_revele_pas_non_plus_la_requete():
    libre = socket.socket()
    libre.bind(("127.0.0.1", 0))
    port = libre.getsockname()[1]
    libre.close()                                          # plus personne n'écoute sur ce port
    c = WebSocketClient(f"ws://127.0.0.1:{port}/ws?key=SECRET123", timeout=2.0)
    with pytest.raises(WebSocketError) as e:
        c.connect()
    assert "SECRET123" not in str(e.value)


@pytest.mark.parametrize("url", ["http://exemple.org/x", "ftp://exemple.org", "ws://", "exemple.org/x", ""])
def test_une_url_qui_n_est_pas_websocket_est_refusee_tout_de_suite(url):
    with pytest.raises(ValueError):
        WebSocketClient(url)


def test_un_serveur_qui_ne_parle_pas_tls_ne_fait_pas_attendre_indefiniment_en_wss(serveur):
    def script(conn):
        conn.attendre_la_fin()                             # il lit le ClientHello et ne répond jamais
    s = serveur(script)
    c = WebSocketClient(f"wss://127.0.0.1:{s.port}/", timeout=1.0)
    debut = time.monotonic()
    with pytest.raises(WebSocketError):
        c.connect()
    assert time.monotonic() - debut < 5.0


# ── envoi ────────────────────────────────────────────────────────────────────────────
def test_un_message_envoye_arrive_en_trame_masquee_texte(serveur):
    recu: dict = {}

    def script(conn):
        conn.accept()
        fin, opcode, charge, masquee = conn.recevoir_trame()
        recu.update(fin=fin, opcode=opcode, charge=charge, masquee=masquee)
    c = _client(serveur(script))
    c.connect()
    c.send_text("bonjour é")
    time.sleep(0.3)
    c.close()
    assert recu == {"fin": True, "opcode": 1, "charge": "bonjour é".encode(), "masquee": True}


@pytest.mark.parametrize("taille", [0, 1, 125, 126, 127, 65535, 65536, 70000])
def test_les_trois_longueurs_d_en_tete_aller_et_retour(serveur, taille):
    """7 bits, 16 bits, 64 bits : les frontières 125/126 et 65535/65536 sont celles où l'on se trompe."""
    donnees = bytes(i % 251 for i in range(taille))

    def script(conn):
        conn.accept()
        _, opcode, charge, _ = conn.recevoir_trame()
        conn.envoyer(opcode, charge)                        # écho, non masqué
        conn.attendre_la_fin()
    c = _client(serveur(script))
    c.connect()
    c.send_binary(donnees)
    assert c.recv(timeout=3.0) == donnees
    c.close()


class _SocketLente:
    """Envoie chaque paquet en DEUX moitiés séparées par une pause : sans verrou d'envoi, les trames de deux fils
    s'entrelacent à coup sûr. (Sur une vraie socket, un `sendall` complet ne s'entrelace presque jamais : un test
    sans cette lenteur passait même quand on supprimait le verrou — vérifié par mutation.)"""

    def __init__(self, sock):
        self._s = sock

    def sendall(self, donnees):
        milieu = len(donnees) // 2
        self._s.sendall(donnees[:milieu])
        time.sleep(0.002)
        self._s.sendall(donnees[milieu:])

    def __getattr__(self, nom):
        return getattr(self._s, nom)


def test_des_envois_concurrents_de_plusieurs_fils_ne_melangent_jamais_les_trames(serveur):
    recus: list[str] = []

    def script(conn):
        conn.accept()
        conn.sock.settimeout(8)                    # des trames mêlées ne doivent pas faire attendre le test sans fin
        for _ in range(100):
            recus.append(conn.recevoir_texte())
        conn.attendre_la_fin()
    c = _client(serveur(script))
    c.connect()
    c._sock = _SocketLente(c._sock)
    fils = [threading.Thread(target=lambda i=i: [c.send_text(f"t{i}-{k}-" + "x" * 3000) for k in range(25)])
            for i in range(4)]
    for f in fils:
        f.start()
    for f in fils:
        f.join()
    time.sleep(0.5)
    c.close()
    assert sorted(recus) == sorted(f"t{i}-{k}-" + "x" * 3000 for i in range(4) for k in range(25))


# ── réception ────────────────────────────────────────────────────────────────────────
def test_le_texte_est_rendu_en_str_et_le_binaire_en_bytes(serveur):
    def script(conn):
        conn.accept()
        conn.texte("é — texte")
        conn.binaire(b"\x00\x01\xff")
        conn.attendre_la_fin()
    c = _client(serveur(script))
    c.connect()
    assert c.recv(timeout=3.0) == "é — texte"
    assert c.recv(timeout=3.0) == b"\x00\x01\xff"
    c.close()


def test_un_message_fragmente_est_reassemble_meme_avec_un_ping_au_milieu(serveur):
    vu: dict = {}

    def script(conn):
        conn.accept()
        conn.texte("bon", fin=False)
        conn.ping(b"coucou")
        conn.envoyer(0, b"jo", fin=False)
        conn.envoyer(0, b"ur", fin=True)
        _, opcode, charge, _ = conn.recevoir_trame()
        vu.update(opcode=opcode, charge=charge)
        conn.attendre_la_fin()
    c = _client(serveur(script))
    c.connect()
    assert c.recv(timeout=3.0) == "bonjour"
    time.sleep(0.3)
    c.close()
    assert vu == {"opcode": 10, "charge": b"coucou"}        # le pong rend la charge du ping


def test_un_ping_recoit_un_pong_sans_que_l_appelant_ait_rien_a_faire(serveur):
    vu: dict = {}

    def script(conn):
        conn.accept()
        conn.ping(b"abc")
        conn.texte("apres")
        _, opcode, charge, _ = conn.recevoir_trame()
        vu.update(opcode=opcode, charge=charge)
        conn.attendre_la_fin()
    c = _client(serveur(script))
    c.connect()
    assert c.recv(timeout=3.0) == "apres"
    time.sleep(0.3)
    c.close()
    assert vu == {"opcode": 10, "charge": b"abc"}


def test_recv_rend_none_au_delai_et_la_connexion_reste_utilisable(serveur):
    liberer = threading.Event()

    def script(conn):
        conn.accept()
        liberer.wait(5)
        conn.texte("tard")
        conn.attendre_la_fin()
    c = _client(serveur(script))
    c.connect()
    assert c.recv(timeout=0.15) is None
    assert not c.closed
    liberer.set()
    assert c.recv(timeout=3.0) == "tard"
    c.close()


def test_une_trame_coupee_en_deux_par_un_delai_n_est_pas_perdue(serveur):
    """Le délai de `recv` tombe au MILIEU d'une trame : les octets déjà lus doivent rester en mémoire."""
    moitie, suite = threading.Event(), threading.Event()

    def script(conn):
        conn.accept()
        brut = conn.trame(1, "message complet".encode())
        conn.envoyer_brut(brut[:5])
        moitie.set()
        suite.wait(5)
        conn.envoyer_brut(brut[5:])
        conn.attendre_la_fin()
    c = _client(serveur(script))
    c.connect()
    assert moitie.wait(3)
    assert c.recv(timeout=0.2) is None
    suite.set()
    assert c.recv(timeout=3.0) == "message complet"
    c.close()


def test_plusieurs_messages_dans_un_seul_paquet_tcp_sont_tous_livres(serveur):
    def script(conn):
        conn.accept()
        conn.envoyer_brut(b"".join(conn.trame(1, f"m{i}".encode()) for i in range(5)))
        conn.attendre_la_fin()
    c = _client(serveur(script))
    c.connect()
    assert [c.recv(timeout=3.0) for _ in range(5)] == [f"m{i}" for i in range(5)]
    c.close()


# ── fermeture ────────────────────────────────────────────────────────────────────────
def test_la_fermeture_du_serveur_donne_le_code_et_la_raison_et_le_client_repond(serveur):
    vu: dict = {}

    def script(conn):
        conn.accept()
        conn.fermer_proprement(1008, "politique")
        _, opcode, charge, _ = conn.recevoir_trame()
        vu.update(opcode=opcode, code=struct.unpack("!H", charge[:2])[0])
    c = _client(serveur(script))
    c.connect()
    with pytest.raises(WebSocketClosed) as e:
        c.recv(timeout=3.0)
    assert (e.value.code, e.value.reason) == (1008, "politique")
    assert c.closed
    time.sleep(0.3)
    assert vu == {"opcode": 8, "code": 1008}                # le client a rendu une trame de fermeture
    with pytest.raises(WebSocketClosed):
        c.send_text("trop tard")


def test_une_coupure_sans_trame_de_fermeture_est_signalee_comme_telle(serveur):
    def script(conn):
        conn.accept()                                       # puis le serveur ferme le socket brutalement
    c = _client(serveur(script))
    c.connect()
    with pytest.raises(WebSocketClosed) as e:
        c.recv(timeout=3.0)
    assert e.value.code == 1006


def test_close_depuis_un_autre_fil_reveille_un_recv_bloque(serveur):
    c = _client(serveur(_accepte_puis_attend))
    c.connect()
    resultat: dict = {}

    def lecteur():
        try:
            c.recv(timeout=None)
        except WebSocketError as e:
            resultat["erreur"] = e
    fil = threading.Thread(target=lecteur)
    fil.start()
    time.sleep(0.3)
    c.close()
    fil.join(3)
    assert not fil.is_alive(), "le recv bloqué n'a pas été réveillé par close()"
    assert isinstance(resultat["erreur"], WebSocketClosed)


def test_close_est_idempotent(serveur):
    c = _client(serveur(_accepte_puis_attend))
    c.connect()
    c.close()
    c.close()
    assert c.closed


# ── trames invalides : refusées, jamais interprétées ───────────────────────────────────
def _trames_invalides():
    from tests._fake_ws import ServerConn as C
    return {
        "bit RSV posé (aucune extension négociée)": (C.trame(1, b"A", rsv=0x40), 1002),
        "trame du serveur masquée": (C.trame(1, b"A", masquee=True), 1002),
        "opcode inconnu": (C.trame(3, b"A"), 1002),
        "ping fragmenté": (C.trame(9, b"A", fin=False), 1002),
        "ping trop long": (C.trame(9, b"x" * 126), 1002),
        "continuation sans message en cours": (C.trame(0, b"A"), 1002),
        "texte qui n'est pas de l'UTF-8": (C.trame(1, b"\xff\xfe\xfd"), 1007),
    }


@pytest.mark.parametrize("nom", list(_trames_invalides()))
def test_une_trame_invalide_est_refusee_avec_le_bon_code_de_fermeture(serveur, nom):
    brut, code_attendu = _trames_invalides()[nom]
    vu: dict = {}

    def script(conn):
        conn.accept()
        conn.envoyer_brut(brut)
        _, opcode, charge, _ = conn.recevoir_trame()
        vu.update(opcode=opcode, code=struct.unpack("!H", charge[:2])[0])
    c = _client(serveur(script))
    c.connect()
    with pytest.raises(WebSocketError):
        c.recv(timeout=3.0)
    assert c.closed
    time.sleep(0.3)
    assert vu == {"opcode": 8, "code": code_attendu}, nom


def test_un_nouveau_message_pendant_une_fragmentation_est_refuse(serveur):
    def script(conn):
        conn.accept()
        conn.texte("debut", fin=False)
        conn.texte("intrus")                                # un message neuf avant la fin du premier
        conn.attendre_la_fin()
    c = _client(serveur(script))
    c.connect()
    with pytest.raises(WebSocketError):
        c.recv(timeout=3.0)


def test_une_longueur_annoncee_trop_grande_est_refusee_sans_attendre_la_charge(serveur):
    """Le client ne doit pas allouer ni attendre 10 Mo parce qu'un en-tête le dit."""
    vu: dict = {}

    def script(conn):
        conn.accept()
        conn.envoyer_brut(conn.trame(2, b"", longueur_annoncee=10_000_000))   # l'en-tête seul
        _, opcode, charge, _ = conn.recevoir_trame()
        vu.update(opcode=opcode, code=struct.unpack("!H", charge[:2])[0])
    c = _client(serveur(script), max_message_bytes=1024)
    c.connect()
    debut = time.monotonic()
    with pytest.raises(WebSocketError, match="trop grand"):
        c.recv(timeout=3.0)
    assert time.monotonic() - debut < 2.0
    time.sleep(0.3)
    assert vu == {"opcode": 8, "code": 1009}


def test_un_message_fragmente_qui_depasse_la_limite_au_total_est_refuse(serveur):
    def script(conn):
        conn.accept()
        for _ in range(5):
            conn.envoyer(2 if _ == 0 else 0, b"x" * 400, fin=False)
        conn.attendre_la_fin()
    c = _client(serveur(script), max_message_bytes=1024)
    c.connect()
    with pytest.raises(WebSocketError, match="trop grand"):
        c.recv(timeout=3.0)


def test_recv_avant_connect_est_une_erreur_claire():
    c = WebSocketClient("ws://127.0.0.1:1/")
    with pytest.raises(WebSocketClosed):
        c.recv(timeout=0.1)
    with pytest.raises(WebSocketClosed):
        c.send_text("x")
