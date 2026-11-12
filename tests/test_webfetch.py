"""`autoagent.webfetch.http_fetch_tool` : lire le web PAR L'HÔTE, à une liste de domaines, sans ouvrir le bac.

Le réseau du bac est en tout ou rien : un outil qui a besoin d'un seul site obligeait à ouvrir tout le réseau,
et chaque hôte écrivait sa propre lecture (`urllib.request.urlopen`), sans garde. Cet outil lit pour le modèle,
côté hôte ; le code du modèle l'appelle par `call_host`, donc par la porte unique.

Ce qui est refusé, EN CODE, avant toute connexion : un schéma autre que https (http sur option), un port autre
que celui du schéma, un hôte hors de la liste, une URL ambiguë (barre inverse, identifiants, blancs), une adresse
RÉSOLUE privée, de bouclage, link-local, de métadonnées, ou une IPv4 privée logée dans une IPv6. La connexion se
fait sur l'adresse vérifiée (pas de seconde résolution). Une redirection repasse les mêmes règles. La lecture est
bornée en taille et en temps. Seul GET part, sans cookie ni identifiant.

Drapeaux : `untrusted=True` (ce qui revient est de la donnée), PAS `egress` — sinon la garde trifecta refuserait la
deuxième lecture d'un run (la première le teinte). La fuite par l'URL est tenue par la liste de domaines et, pour
l'URL exacte, par `Agent(url_guard="deny")` (testé ici).
"""

from __future__ import annotations

import http.server
import threading
import time
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from autoagent import Agent, webfetch
from autoagent.errors import ToolError
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, ToolCall, is_tainted
from autoagent.webfetch import http_fetch_tool

# ---------------------------------------------------------------------------------------------------------
# Doublures : un résolveur et une connexion scriptés. Aucun test ici ne touche le réseau.
# ---------------------------------------------------------------------------------------------------------

class _Reponse:
    def __init__(self, status: int, entetes: dict[str, str], corps: bytes, *, lent: float = 0.0) -> None:
        self.status = status
        self._entetes = {k.lower(): v for k, v in entetes.items()}
        self._corps = corps
        self._lent = lent
        self.octets_lus = 0

    def getheader(self, nom: str, defaut: str | None = None) -> str | None:
        return self._entetes.get(nom.lower(), defaut)

    def read(self, n: int = -1) -> bytes:
        if self._lent:
            time.sleep(self._lent)
        if self._corps == b"*infini*":
            self.octets_lus += n
            return b"x" * n
        morceau = self._corps[:n] if n >= 0 else self._corps
        self._corps = self._corps[len(morceau):]
        self.octets_lus += len(morceau)
        return morceau


class _Reseau:
    """Le résolveur et les connexions : ce qui a été résolu, ouvert, demandé."""

    def __init__(self, dns: dict[str, list[str]], pages: dict[tuple[str, str], _Reponse]) -> None:
        self.dns = dns
        self.pages = pages
        self.resolutions: list[str] = []
        self.ouvertures: list[tuple[str, str, str, int]] = []
        self.requetes: list[tuple[str, str, str, dict[str, str]]] = []

    def resoudre(self, hote: str, port: int) -> list[str]:
        self.resolutions.append(hote)
        if hote not in self.dns:
            raise OSError(f"nom inconnu : {hote}")
        return self.dns[hote]

    def ouvrir(self, schema: str, hote: str, ip: str, port: int, delai: float) -> Any:
        self.ouvertures.append((schema, hote, ip, port))
        reseau = self

        class _Connexion:
            def request(self, methode: str, chemin: str, headers: dict[str, str]) -> None:
                reseau.requetes.append((methode, hote, chemin, dict(headers)))
                self._cle = (hote, chemin)

            def getresponse(self) -> _Reponse:
                if self._cle not in reseau.pages:
                    return _Reponse(404, {"Content-Type": "text/plain"}, b"absent")
                return reseau.pages[self._cle]

            def close(self) -> None:
                pass

        return _Connexion()


PUBLIQUE = "93.184.216.34"


def _page(texte: str = "bonjour", *, type_: str = "text/html; charset=utf-8", status: int = 200) -> _Reponse:
    return _Reponse(status, {"Content-Type": type_}, texte.encode("utf-8"))


def _redirection(vers: str, status: int = 302) -> _Reponse:
    return _Reponse(status, {"Location": vers}, b"")


@pytest.fixture
def reseau(monkeypatch: pytest.MonkeyPatch) -> _Reseau:
    faux = _Reseau(dns={"exemple.fr": [PUBLIQUE], "api.exemple.fr": [PUBLIQUE]},
                   pages={("exemple.fr", "/"): _page()})
    monkeypatch.setattr(webfetch, "_resoudre", faux.resoudre)
    monkeypatch.setattr(webfetch, "_ouvrir", faux.ouvrir)
    return faux


# ---------------------------------------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------------------------------------

class TestConstruction:
    @pytest.mark.parametrize("hotes", [[], (), [""], ["*"], ["*.exemple.fr"], ["https://exemple.fr"],
                                       ["exemple.fr/chemin"], ["exemple.fr:8443"], ["moi@exemple.fr"],
                                       ["exe mple.fr"], ["exemple.fr\\x"]])
    def test_une_liste_de_noms_d_hote_nus(self, hotes: list[str]) -> None:
        with pytest.raises(ValueError, match="allowed_hosts"):
            http_fetch_tool(hotes)

    def test_une_chaine_seule_est_refusee(self) -> None:
        """`"exemple.fr"` serait lu lettre par lettre comme une liste de 10 hôtes."""
        with pytest.raises(TypeError, match="allowed_hosts"):
            http_fetch_tool("exemple.fr")  # type: ignore[arg-type]

    def test_casse_et_point_initial_normalises(self) -> None:
        outil = http_fetch_tool([" Exemple.FR ", ".autre.org"])
        assert "exemple.fr" in outil.__autoagent_tool_spec__.description
        assert "autre.org" in outil.__autoagent_tool_spec__.description

    @pytest.mark.parametrize("reglage", [{"max_bytes": 0}, {"timeout": 0}, {"max_redirects": -1}])
    def test_bornes_positives(self, reglage: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            http_fetch_tool(["exemple.fr"], **reglage)

    def test_drapeaux_de_l_outil(self) -> None:
        spec = http_fetch_tool(["exemple.fr"], name="lire_web").__autoagent_tool_spec__
        assert spec.name == "lire_web"
        assert spec.untrusted is True                # ce qui revient est de la donnée
        assert spec.egress is False                  # sinon une seule lecture par run teinté
        assert spec.idempotent is True
        assert spec.input_schema["required"] == ["url"]


# ---------------------------------------------------------------------------------------------------------
# L'URL : refusée AVANT toute résolution
# ---------------------------------------------------------------------------------------------------------

class TestLURL:
    @pytest.mark.parametrize("url", [
        "http://exemple.fr/",                       # http sans l'option
        "ftp://exemple.fr/", "file:///etc/passwd", "javascript:alert(1)", "gopher://exemple.fr/",
        "exemple.fr/sans-schema", "//exemple.fr/",
        "https://evil.com/",                        # hors liste
        "https://evilexemple.fr/",                  # suffixe sans point
        "https://exemple.fr.evil.com/",             # préfixe
        "https://exemple.fr@evil.com/",             # identifiants : l'hôte réel est evil.com
        "https://moi@exemple.fr/",                  # identifiants, même sur le bon hôte
        "https://moi:secret@exemple.fr/",
        "https://exemple.fr\\@evil.com/",           # barre inverse : deux analyseurs, deux lectures
        "https://exemple.fr:8443/",                 # port
        "https://exemple.fr:80/",
        "https://exemple.fr:/",
        "https://exemple.fr:abc/",
        "https:///chemin",                          # pas d'hôte
        "https://exemple.fr/a\\b",                  # barre inverse dans le chemin
        "https://exemple.fr/a b",                   # blanc
        "https://exemple.fr/a\tb", "https://exemple.fr/a\nb", "https://exemple.fr/\x00",
        " https://exemple.fr/",
        "https://93.184.216.34/",                   # une adresse n'est pas un nom de la liste
        "https://[::1]/",
        "",
    ])
    def test_refusee_sans_rien_resoudre(self, reseau: _Reseau, url: str) -> None:
        with pytest.raises(ToolError):
            http_fetch_tool(["exemple.fr"])(url)
        assert reseau.resolutions == [] and reseau.ouvertures == []

    def test_pas_une_chaine(self, reseau: _Reseau) -> None:
        with pytest.raises(ToolError):
            http_fetch_tool(["exemple.fr"])(["https://exemple.fr/"])  # type: ignore[arg-type]

    @pytest.mark.parametrize("url", ["https://exemple.fr/", "https://EXEMPLE.fr/", "https://exemple.fr:443/",
                                     "https://exemple.fr./", "https://exemple.fr"])
    def test_acceptee(self, reseau: _Reseau, url: str) -> None:
        assert http_fetch_tool(["exemple.fr"])(url)["text"] == "bonjour"

    def test_un_sous_domaine(self, reseau: _Reseau) -> None:
        reseau.pages[("api.exemple.fr", "/v1?q=1")] = _page("{}", type_="application/json")
        assert http_fetch_tool(["exemple.fr"])("https://api.exemple.fr/v1?q=1#ancre")["status"] == 200

    def test_http_sur_option(self, reseau: _Reseau) -> None:
        rendu = http_fetch_tool(["exemple.fr"], allow_http=True)("http://exemple.fr/")
        assert rendu["text"] == "bonjour"
        assert reseau.ouvertures == [("http", "exemple.fr", PUBLIQUE, 80)]
        with pytest.raises(ToolError):
            http_fetch_tool(["exemple.fr"], allow_http=True)("http://exemple.fr:443/")


# ---------------------------------------------------------------------------------------------------------
# L'adresse résolue
# ---------------------------------------------------------------------------------------------------------

class TestLAdresse:
    @pytest.mark.parametrize("ip", [
        "127.0.0.1", "127.8.8.8", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.1.1",
        "169.254.169.254",                           # métadonnées des nuages
        "100.64.0.1",                                # NAT des opérateurs
        "0.0.0.0", "224.0.0.1", "255.255.255.255", "240.0.0.1", "192.0.0.1", "198.18.0.1", "192.0.2.1",
        "::1", "::", "fe80::1", "fe80::1%eth0", "fc00::1", "fd12:3456::1", "fec0::1", "ff02::1", "2001:db8::1",
        "::ffff:10.0.0.1", "::ffff:127.0.0.1", "::ffff:169.254.169.254",   # IPv4 privée dans une IPv6
        "64:ff9b::a00:1", "64:ff9b:1::a00:1",        # NAT64 vers 10.0.0.1
        "2002:a00:1::",                              # 6to4 depuis 10.0.0.1
        "2001::1",                                   # Teredo
        "pas une adresse", "",
    ])
    def test_refusee(self, ip: str) -> None:
        assert webfetch._adresse_publique(ip) is False

    @pytest.mark.parametrize("ip", ["93.184.216.34", "1.1.1.1", "8.8.8.8", "2606:4700:4700::1111",
                                    "::ffff:1.1.1.1", "64:ff9b::808:808"])
    def test_acceptee(self, ip: str) -> None:
        assert webfetch._adresse_publique(ip) is True

    def test_une_seule_adresse_privee_suffit_a_refuser(self, reseau: _Reseau) -> None:
        reseau.dns["exemple.fr"] = [PUBLIQUE, "10.0.0.1"]
        with pytest.raises(ToolError, match="10.0.0.1"):
            http_fetch_tool(["exemple.fr"])("https://exemple.fr/")
        assert reseau.ouvertures == []

    def test_aucune_adresse(self, reseau: _Reseau) -> None:
        reseau.dns["exemple.fr"] = []
        with pytest.raises(ToolError):
            http_fetch_tool(["exemple.fr"])("https://exemple.fr/")

    def test_resolution_en_echec(self, reseau: _Reseau) -> None:
        with pytest.raises(ToolError, match="inconnu"):
            http_fetch_tool(["inconnu.fr"])("https://inconnu.fr/")

    def test_une_adresse_injoignable_passe_a_la_suivante(self, reseau: _Reseau, monkeypatch: pytest.MonkeyPatch) -> None:
        """Un site en IPv6 et IPv4, une machine sans IPv6 : la lecture essaie l'adresse suivante (vérifiée elle
        aussi), au lieu d'échouer."""
        reseau.dns["exemple.fr"] = ["2606:4700:4700::1111", PUBLIQUE]
        vrai = reseau.ouvrir

        def ouvrir(schema: str, hote: str, ip: str, port: int, delai: float) -> Any:
            connexion = vrai(schema, hote, ip, port, delai)
            if ":" in ip:
                def injoignable(*a: Any, **k: Any) -> None:
                    raise OSError("Network is unreachable")
                connexion.request = injoignable
            return connexion

        monkeypatch.setattr(webfetch, "_ouvrir", ouvrir)
        assert http_fetch_tool(["exemple.fr"])("https://exemple.fr/")["text"] == "bonjour"
        assert [o[2] for o in reseau.ouvertures] == ["2606:4700:4700::1111", PUBLIQUE]

    def test_toutes_injoignables(self, reseau: _Reseau, monkeypatch: pytest.MonkeyPatch) -> None:
        def ouvrir(*a: Any) -> Any:
            raise OSError("Network is unreachable")
        monkeypatch.setattr(webfetch, "_ouvrir", ouvrir)
        with pytest.raises(ToolError, match="unreachable"):
            http_fetch_tool(["exemple.fr"])("https://exemple.fr/")

    def test_la_connexion_va_sur_l_adresse_verifiee(self, reseau: _Reseau) -> None:
        http_fetch_tool(["exemple.fr"])("https://exemple.fr/")
        assert reseau.resolutions == ["exemple.fr"]                     # une seule résolution
        assert reseau.ouvertures == [("https", "exemple.fr", PUBLIQUE, 443)]


# ---------------------------------------------------------------------------------------------------------
# La requête et la réponse
# ---------------------------------------------------------------------------------------------------------

class TestRequeteEtReponse:
    def test_get_seul_sans_cookie_ni_identifiant(self, reseau: _Reseau) -> None:
        http_fetch_tool(["exemple.fr"])("https://exemple.fr/")
        methode, hote, chemin, entetes = reseau.requetes[0]
        assert (methode, hote, chemin) == ("GET", "exemple.fr", "/")
        noms = {k.lower() for k in entetes}
        assert not noms & {"cookie", "authorization", "proxy-authorization"}
        assert entetes.get("Accept-Encoding") == "identity"

    def test_le_rendu(self, reseau: _Reseau) -> None:
        rendu = http_fetch_tool(["exemple.fr"])("https://exemple.fr/")
        assert rendu == {"url": "https://exemple.fr/", "status": 200, "content_type": "text/html; charset=utf-8",
                         "text": "bonjour", "truncated": False}

    def test_une_erreur_http_est_une_reponse(self, reseau: _Reseau) -> None:
        rendu = http_fetch_tool(["exemple.fr"])("https://exemple.fr/absente")
        assert rendu["status"] == 404 and rendu["text"] == "absent"

    def test_le_jeu_de_caracteres(self, reseau: _Reseau) -> None:
        reseau.pages[("exemple.fr", "/l")] = _Reponse(200, {"Content-Type": "text/plain; charset=ISO-8859-1"},
                                                      "été".encode("latin-1"))
        assert http_fetch_tool(["exemple.fr"])("https://exemple.fr/l")["text"] == "été"

    def test_un_jeu_inconnu_retombe_en_utf8(self, reseau: _Reseau) -> None:
        reseau.pages[("exemple.fr", "/l")] = _Reponse(200, {"Content-Type": "text/plain; charset=bidon"},
                                                      "été".encode("utf-8"))
        assert http_fetch_tool(["exemple.fr"])("https://exemple.fr/l")["text"] == "été"

    @pytest.mark.parametrize("type_", ["image/png", "application/octet-stream", "application/zip", "video/mp4"])
    def test_un_contenu_non_textuel_est_refuse(self, reseau: _Reseau, type_: str) -> None:
        reseau.pages[("exemple.fr", "/b")] = _Reponse(200, {"Content-Type": type_}, b"\x89PNG")
        with pytest.raises(ToolError, match="text"):
            http_fetch_tool(["exemple.fr"])("https://exemple.fr/b")

    @pytest.mark.parametrize("type_", ["text/csv", "application/json", "application/ld+json", "application/xml",
                                       "application/rss+xml", "application/javascript", ""])
    def test_contenus_textuels(self, reseau: _Reseau, type_: str) -> None:
        reseau.pages[("exemple.fr", "/t")] = _Reponse(200, {"Content-Type": type_} if type_ else {}, b"ok")
        assert http_fetch_tool(["exemple.fr"])("https://exemple.fr/t")["text"] == "ok"

    def test_une_compression_non_demandee_est_refusee(self, reseau: _Reseau) -> None:
        reseau.pages[("exemple.fr", "/z")] = _Reponse(200, {"Content-Type": "text/html",
                                                            "Content-Encoding": "gzip"}, b"\x1f\x8b")
        with pytest.raises(ToolError, match="gzip"):
            http_fetch_tool(["exemple.fr"])("https://exemple.fr/z")

    def test_la_taille_est_bornee(self, reseau: _Reseau) -> None:
        corps = _Reponse(200, {"Content-Type": "text/plain"}, b"*infini*")
        reseau.pages[("exemple.fr", "/gros")] = corps
        rendu = http_fetch_tool(["exemple.fr"], max_bytes=1000)("https://exemple.fr/gros")
        assert len(rendu["text"]) == 1000 and rendu["truncated"] is True
        assert corps.octets_lus <= 1001                                 # la lecture s'arrête, elle ne télécharge pas tout

    def test_juste_a_la_limite_n_est_pas_tronque(self, reseau: _Reseau) -> None:
        reseau.pages[("exemple.fr", "/pile")] = _Reponse(200, {"Content-Type": "text/plain"}, b"x" * 1000)
        rendu = http_fetch_tool(["exemple.fr"], max_bytes=1000)("https://exemple.fr/pile")
        assert len(rendu["text"]) == 1000 and rendu["truncated"] is False

    def test_le_temps_total_est_borne(self, reseau: _Reseau) -> None:
        """Un serveur qui envoie au compte-gouttes ne garde pas l'outil indéfiniment."""
        reseau.pages[("exemple.fr", "/lent")] = _Reponse(200, {"Content-Type": "text/plain"}, b"*infini*", lent=0.05)
        debut = time.monotonic()
        with pytest.raises(ToolError, match="timeout|délai"):
            http_fetch_tool(["exemple.fr"], timeout=0.3, max_bytes=10_000_000)("https://exemple.fr/lent")
        assert time.monotonic() - debut < 2.0

    def test_une_panne_de_connexion_est_une_erreur_d_outil(self, reseau: _Reseau, monkeypatch: pytest.MonkeyPatch) -> None:
        def panne(*a: Any) -> Any:
            raise ConnectionRefusedError("refusée")
        monkeypatch.setattr(webfetch, "_ouvrir", panne)
        with pytest.raises(ToolError, match="refusée"):
            http_fetch_tool(["exemple.fr"])("https://exemple.fr/")


# ---------------------------------------------------------------------------------------------------------
# Les redirections : les mêmes règles à chaque saut
# ---------------------------------------------------------------------------------------------------------

class TestRedirections:
    def test_relative_suivie(self, reseau: _Reseau) -> None:
        reseau.pages[("exemple.fr", "/a")] = _redirection("/b")
        reseau.pages[("exemple.fr", "/b")] = _page("arrivé")
        rendu = http_fetch_tool(["exemple.fr"])("https://exemple.fr/a")
        assert rendu["text"] == "arrivé" and rendu["url"] == "https://exemple.fr/b"

    def test_vers_un_sous_domaine(self, reseau: _Reseau) -> None:
        reseau.pages[("exemple.fr", "/a")] = _redirection("https://api.exemple.fr/b", 301)
        reseau.pages[("api.exemple.fr", "/b")] = _page("ok")
        assert http_fetch_tool(["exemple.fr"])("https://exemple.fr/a")["text"] == "ok"

    def test_vers_un_hote_hors_liste(self, reseau: _Reseau) -> None:
        reseau.pages[("exemple.fr", "/a")] = _redirection("https://evil.com/")
        with pytest.raises(ToolError, match="evil.com"):
            http_fetch_tool(["exemple.fr"])("https://exemple.fr/a")
        assert [o[1] for o in reseau.ouvertures] == ["exemple.fr"]

    def test_vers_une_adresse_privee(self, reseau: _Reseau) -> None:
        reseau.dns["interne.exemple.fr"] = ["169.254.169.254"]
        reseau.pages[("exemple.fr", "/a")] = _redirection("https://interne.exemple.fr/latest/meta-data/")
        with pytest.raises(ToolError, match="169.254.169.254"):
            http_fetch_tool(["exemple.fr"])("https://exemple.fr/a")
        assert len(reseau.ouvertures) == 1

    def test_de_https_vers_http(self, reseau: _Reseau) -> None:
        reseau.pages[("exemple.fr", "/a")] = _redirection("http://exemple.fr/b")
        with pytest.raises(ToolError):
            http_fetch_tool(["exemple.fr"])("https://exemple.fr/a")

    def test_trop_de_sauts(self, reseau: _Reseau) -> None:
        for i in range(5):
            reseau.pages[("exemple.fr", f"/{i}")] = _redirection(f"/{i + 1}")
        reseau.pages[("exemple.fr", "/5")] = _page("fin")
        with pytest.raises(ToolError, match="redirect"):
            http_fetch_tool(["exemple.fr"], max_redirects=3)("https://exemple.fr/0")
        assert http_fetch_tool(["exemple.fr"], max_redirects=5)("https://exemple.fr/0")["text"] == "fin"

    def test_sans_location(self, reseau: _Reseau) -> None:
        reseau.pages[("exemple.fr", "/a")] = _Reponse(302, {}, b"")
        with pytest.raises(ToolError, match="Location"):
            http_fetch_tool(["exemple.fr"])("https://exemple.fr/a")

    def test_zero_redirection_permise(self, reseau: _Reseau) -> None:
        reseau.pages[("exemple.fr", "/a")] = _redirection("/b")
        with pytest.raises(ToolError, match="redirect"):
            http_fetch_tool(["exemple.fr"], max_redirects=0)("https://exemple.fr/a")


# ---------------------------------------------------------------------------------------------------------
# Avec un agent : la porte, la teinte, la garde des URL d'appel
# ---------------------------------------------------------------------------------------------------------

class _Scripte(LLMProvider):
    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self.reponses.pop(0) if self.reponses else LLMResponse(content="fini")


def _appel(id_: str, nom: str, **arguments: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=id_, name=nom, arguments=arguments)])


def _resultats(resultat: Any) -> list[str]:
    return [m.content or "" for m in resultat.messages if m.role == "tool"]


class TestAvecUnAgent:
    def test_la_lecture_teinte_le_run(self, reseau: _Reseau) -> None:
        agent = Agent(_Scripte([_appel("c1", "http_fetch", url="https://exemple.fr/")]), max_steps=4)
        agent.add_tool(http_fetch_tool(["exemple.fr"]))
        assert is_tainted(agent.run("lis").messages)

    def test_deux_lectures_dans_un_run_teinte(self, reseau: _Reseau) -> None:
        """Décision 3 : ce n'est pas un envoi. La garde trifecta par défaut laisse passer la seconde lecture."""
        reseau.pages[("exemple.fr", "/2")] = _page("deux")
        agent = Agent(_Scripte([_appel("c1", "http_fetch", url="https://exemple.fr/"),
                                _appel("c2", "http_fetch", url="https://exemple.fr/2")]), max_steps=5)
        agent.add_tool(http_fetch_tool(["exemple.fr"]))
        resultats = _resultats(agent.run("lis"))
        assert "bonjour" in resultats[0] and "deux" in resultats[1]

    def test_url_guard_refuse_une_url_qui_porte_un_secret(self, reseau: _Reseau) -> None:
        """Run teinté + `url_guard="deny"` : une URL qui n'est pas lue mot pour mot dans la conversation est
        refusée AVANT toute connexion ; celle que la page cite passe."""
        reseau.pages[("exemple.fr", "/")] = _page("Voir https://exemple.fr/suite pour la suite.")
        reseau.pages[("exemple.fr", "/suite")] = _page("suite")
        agent = Agent(_Scripte([
            _appel("c1", "http_fetch", url="https://exemple.fr/"),
            _appel("c2", "http_fetch", url="https://exemple.fr/p?d=SECRET-4242"),
            _appel("c3", "http_fetch", url="https://exemple.fr/suite"),
        ]), max_steps=6, url_guard="deny")
        agent.add_tool(http_fetch_tool(["exemple.fr"]))
        resultats = _resultats(agent.run("lis"))
        assert "SECRET-4242" not in str(reseau.requetes)
        assert [r[2] for r in reseau.requetes] == ["/", "/suite"]
        assert "suite" in resultats[2]

    def test_depuis_le_code_du_modele(self, reseau: _Reseau) -> None:
        """Le code comme action l'appelle par `call_host` : par le registre, donc par la porte ; le run est teinté."""
        code = ("def run(args, context):\n"
                "    page = context['call_host']('http_fetch', {'url': 'https://exemple.fr/'})\n"
                "    return len(page['text'])\n")
        agent = Agent(_Scripte([_appel("c1", "run_python", code=code)]), max_steps=4)
        agent.add_tool(http_fetch_tool(["exemple.fr"]))
        agent.enable_code_action()
        resultat = agent.run("compte")
        assert '"result": 7' in _resultats(resultat)[0]
        assert reseau.requetes and is_tainted(resultat.messages)


# ---------------------------------------------------------------------------------------------------------
# Bout en bout, sans doublure de connexion : un vrai serveur HTTP local
# ---------------------------------------------------------------------------------------------------------

class _Gestionnaire(http.server.BaseHTTPRequestHandler):
    vus: ClassVar[list[tuple[str, str]]] = []

    def do_GET(self) -> None:          # nom imposé par http.server
        _Gestionnaire.vus.append((self.command, self.headers.get("Host", "")))
        if self.path == "/depart":
            self.send_response(302)
            self.send_header("Location", "/arrivee")
            self.end_headers()
            return
        corps = "arrivé à bon port".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def log_message(self, *a: Any) -> None:
        pass


@pytest.fixture
def serveur_local(monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    serveur = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Gestionnaire)
    fil = threading.Thread(target=serveur.serve_forever, daemon=True)
    fil.start()
    # Le seul relâchement de ce test : 127.0.0.1 compte comme publique, et `site.test` y mène.
    monkeypatch.setattr(webfetch, "_resoudre", lambda hote, port: ["127.0.0.1"] if hote == "site.test" else [])
    vrai = webfetch._adresse_publique
    monkeypatch.setattr(webfetch, "_adresse_publique", lambda ip: ip == "127.0.0.1" or vrai(ip))
    try:
        yield serveur.server_address[1]
    finally:
        serveur.shutdown()
        serveur.server_close()


def test_bout_en_bout_sur_un_vrai_serveur(serveur_local: int, monkeypatch: pytest.MonkeyPatch) -> None:
    port = serveur_local
    # Le port du serveur de test n'est pas 80 : on autorise ce port-là pour ce seul test.
    monkeypatch.setattr(webfetch, "_PORTS", {"http": port, "https": 443})
    _Gestionnaire.vus.clear()
    rendu = http_fetch_tool(["site.test"], allow_http=True)(f"http://site.test:{port}/depart")
    assert rendu["status"] == 200 and rendu["text"] == "arrivé à bon port"
    assert rendu["url"] == f"http://site.test:{port}/arrivee"
    assert _Gestionnaire.vus == [("GET", f"site.test:{port}"), ("GET", f"site.test:{port}")]
