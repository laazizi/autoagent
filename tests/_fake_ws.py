"""Serveur WebSocket de test, bibliothèque standard seulement (le client sous test est `autoagent.ws`).

Il fait la poignée de main puis laisse le scénario parler en TRAMES BRUTES : fragmentation, ping, trames
invalides (RSV, masquée, UTF-8 cassé, trop longue), coupure. Un scénario est une fonction `script(conn)` qui
tourne dans son propre fil ; une assertion qui y échoue est gardée dans `serveur.errors` et re-levée par la
fixture à la fin du test — jamais perdue dans un fil.

Le serveur n'implémente PAS le protocole pour l'utilisateur : c'est voulu, un serveur « correct » ne pourrait
pas produire les fautes qu'on veut voir refusées.
"""

from __future__ import annotations

import base64
import hashlib
import socket
import struct
import threading

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class ServerConn:
    def __init__(self, sock: socket.socket, fin: threading.Event) -> None:
        self.sock = sock
        self._fin = fin
        self.buf = bytearray()
        self.headers: dict[str, str] = {}
        self.request_line = ""
        self._lire_requete()

    # ── poignée de main ────────────────────────────────────────────────────────
    def _lire_requete(self) -> None:
        while b"\r\n\r\n" not in self.buf:
            morceau = self.sock.recv(4096)
            if not morceau:
                raise ConnectionError("le client a fermé avant la fin de la requête")
            self.buf += morceau
        tete, _, reste = bytes(self.buf).partition(b"\r\n\r\n")
        self.buf = bytearray(reste)
        lignes = tete.decode("latin-1").split("\r\n")
        self.request_line = lignes[0]
        for ligne in lignes[1:]:
            cle, _, valeur = ligne.partition(":")
            self.headers[cle.strip().lower()] = valeur.strip()

    @property
    def path(self) -> str:
        return self.request_line.split(" ")[1]

    def accept(self, *, accept: str | None = None, status: str = "101 Switching Protocols",
               extra: tuple[str, ...] = ()) -> None:
        cle = self.headers["sec-websocket-key"]
        bonne = base64.b64encode(hashlib.sha1((cle + GUID).encode()).digest()).decode()
        lignes = [f"HTTP/1.1 {status}", "Upgrade: websocket", "Connection: Upgrade",
                  f"Sec-WebSocket-Accept: {accept or bonne}", *extra, "", ""]
        self.sock.sendall("\r\n".join(lignes).encode())

    def refuse(self, status: str = "403 Forbidden", corps: bytes = b"refus") -> None:
        tete = f"HTTP/1.1 {status}\r\nContent-Length: {len(corps)}\r\nConnection: close\r\n\r\n".encode()
        self.sock.sendall(tete + corps)

    # ── trames ────────────────────────────────────────────────────────────────
    @staticmethod
    def trame(opcode: int, charge: bytes = b"", *, fin: bool = True, rsv: int = 0, masquee: bool = False,
              longueur_annoncee: int | None = None) -> bytes:
        n = len(charge) if longueur_annoncee is None else longueur_annoncee
        octet0 = (0x80 if fin else 0) | (rsv & 0x70) | (opcode & 0x0F)
        masque_bit = 0x80 if masquee else 0
        if n < 126:
            tete = bytes([octet0, masque_bit | n])
        elif n < 65536:
            tete = bytes([octet0, masque_bit | 126]) + struct.pack("!H", n)
        else:
            tete = bytes([octet0, masque_bit | 127]) + struct.pack("!Q", n)
        if masquee:
            masque = b"\x01\x02\x03\x04"
            return tete + masque + bytes(b ^ masque[i % 4] for i, b in enumerate(charge))
        return tete + charge

    def envoyer_brut(self, octets: bytes) -> None:
        self.sock.sendall(octets)

    def envoyer(self, opcode: int, charge: bytes = b"", **options) -> None:
        self.sock.sendall(self.trame(opcode, charge, **options))

    def texte(self, texte: str, **options) -> None:
        self.envoyer(1, texte.encode("utf-8"), **options)

    def binaire(self, donnees: bytes, **options) -> None:
        self.envoyer(2, donnees, **options)

    def ping(self, charge: bytes = b"") -> None:
        self.envoyer(9, charge)

    def fermer_proprement(self, code: int = 1000, raison: str = "") -> None:
        self.envoyer(8, struct.pack("!H", code) + raison.encode("utf-8"))

    # ── réception (les trames d'un client sont toujours masquées) ───────────────
    def _lire(self, n: int) -> bytes:
        while len(self.buf) < n:
            morceau = self.sock.recv(65536)
            if not morceau:
                raise ConnectionError("le client a fermé la connexion")
            self.buf += morceau
        sortie = bytes(self.buf[:n])
        del self.buf[:n]
        return sortie

    def recevoir_trame(self) -> tuple[bool, int, bytes, bool]:
        """(fin, opcode, charge démasquée, masquée ?)"""
        octet0, octet1 = self._lire(2)
        n = octet1 & 0x7F
        if n == 126:
            n = struct.unpack("!H", self._lire(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", self._lire(8))[0]
        masquee = bool(octet1 & 0x80)
        masque = self._lire(4) if masquee else b""
        charge = self._lire(n)
        if masquee:
            charge = bytes(b ^ masque[i % 4] for i, b in enumerate(charge))
        return bool(octet0 & 0x80), octet0 & 0x0F, charge, masquee

    def recevoir_texte(self) -> str:
        fin, opcode, charge, _ = self.recevoir_trame()
        assert fin and opcode == 1, f"trame texte attendue, reçu opcode {opcode}"
        return charge.decode("utf-8")

    def attendre_la_fin(self, delai: float = 10.0) -> None:
        """Garde la connexion ouverte jusqu'à la fin du test (sinon le serveur la coupe avant que le client lise)."""
        self._fin.wait(delai)


class FakeWSServer:
    def __init__(self, script) -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(8)
        self.port: int = self._srv.getsockname()[1]
        self.script = script
        self.errors: list[BaseException] = []
        self.conns: list[ServerConn] = []
        self._fin = threading.Event()
        threading.Thread(target=self._accepter, daemon=True).start()

    def url(self, chemin: str = "/") -> str:
        return f"ws://127.0.0.1:{self.port}{chemin}"

    def _accepter(self) -> None:
        while True:
            try:
                sock, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._servir, args=(sock,), daemon=True).start()

    def _servir(self, sock: socket.socket) -> None:
        try:
            conn = ServerConn(sock, self._fin)
            self.conns.append(conn)
            self.script(conn)
        except ConnectionError:
            pass                                            # le client est parti : normal en fin de test
        except BaseException as exc:
            self.errors.append(exc)
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def close(self) -> None:
        self._fin.set()
        try:
            self._srv.close()
        except OSError:
            pass
