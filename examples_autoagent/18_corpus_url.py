"""18 — Gros corpus depuis une URL : l'agent télécharge ~1M de tokens et travaille dessus.

LE problème : on ne peut pas (et on ne veut pas) injecter 1 million de tokens
dans une conversation — latence, coût à CHAQUE tour de boucle, attention diluée.
LE pattern : les données restent DEHORS, l'agent reçoit des OUTILS :

    telecharger_corpus(url)   → le host télécharge et INDEXE (jamais injecté)
    chercher(question)        → les 3 passages les plus pertinents (~2k tokens)
    lire_passage(id)          → un passage complet, à la demande

L'agent navigue dans le corpus comme un dev avec grep : quelques appels
d'outils ciblés au lieu d'un contexte d'un million. Le bilan final affiche
le ratio corpus / tokens réellement consommés.

Corpus de démo : « Les Misérables » de Victor Hugo, 5 tomes, Projet Gutenberg
(domaine public, ~3,5 Mo ≈ 1M de tokens). Téléchargements mis en cache local.

Recherche par le SENS incluse : si GEMINI_API_KEY est présente, le corpus est
embarqué une fois (gemini-embedding-001, dimension 256, ~10 requêtes batch,
cache disque) et `chercher` classe par cosinus au lieu du lexical — même
outil, même agent, seul le classement change. À plus grande échelle : base
vectorielle (pgvector, Chroma) ou serveur MCP, pattern identique.

    python examples_autoagent/18_corpus_url.py
"""

import json
import math
import os
import re
import urllib.request
from pathlib import Path

from _common import make_provider

from autoagent import Agent

TOMES = [f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
         for gid in (17489, 17493, 17494, 17518, 17519)]
CACHE = Path(__file__).parent / "corpus_cache"

# ── index côté HOST : le corpus ne touche jamais le contexte du LLM ──
_CHUNKS: list[str] = []
_VECTEURS: list[list[float]] = []   # rempli si GEMINI_API_KEY (recherche par le sens)


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    """Embeddings batch (dimension réduite : suffisant pour du ranking)."""
    corps = json.dumps({"requests": [
        {"model": "models/gemini-embedding-001",
         "content": {"parts": [{"text": t[:6000]}]},
         "outputDimensionality": 256}
        for t in texts
    ]}).encode()
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-embedding-001:batchEmbedContents",
        data=corps,
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    with urllib.request.urlopen(req, timeout=120) as reponse:  # noqa: S310
        return [e["values"] for e in json.loads(reponse.read())["embeddings"]]


def _ensure_vecteurs() -> bool:
    """Embarque les passages pas encore vectorisés (cache disque). True = mode sémantique."""
    if "GEMINI_API_KEY" not in os.environ or not _CHUNKS:
        return False
    if len(_VECTEURS) == len(_CHUNKS):
        return True
    cache = CACHE / f"vecteurs_{len(_CHUNKS)}_d256.json"
    if cache.exists() and not _VECTEURS:
        _VECTEURS.extend(json.loads(cache.read_text(encoding="utf-8")))
        return len(_VECTEURS) == len(_CHUNKS)
    print(f"  (embedding de {len(_CHUNKS) - len(_VECTEURS)} passages — une fois, ensuite cache)")
    for debut in range(len(_VECTEURS), len(_CHUNKS), 100):
        _VECTEURS.extend(_embed_gemini(_CHUNKS[debut : debut + 100]))
    cache.write_text(json.dumps([[round(x, 5) for x in v] for v in _VECTEURS]),
                     encoding="utf-8")
    return True


def _cosinus(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norme = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norme if norme else 0.0


def _indexer(texte: str, mots_par_chunk: int = 500) -> int:
    """Découpe en passages alignés sur les paragraphes ; retourne le nb ajouté."""
    ajoutés = 0
    courant: list[str] = []
    taille = 0
    for para in re.split(r"\n\s*\n", texte):
        mots = para.split()
        if not mots:
            continue
        courant.append(para.strip())
        taille += len(mots)
        if taille >= mots_par_chunk:
            _CHUNKS.append("\n\n".join(courant))
            courant, taille = [], 0
            ajoutés += 1
    if courant:
        _CHUNKS.append("\n\n".join(courant))
        ajoutés += 1
    return ajoutés


def _score(question: str, chunk: str) -> float:
    """Recouvrement lexical pondéré — remplacer par un cosinus d'embeddings en prod."""
    termes = {t for t in re.findall(r"\w+", question.lower()) if len(t) > 3}
    if not termes:
        return 0.0
    mots = re.findall(r"\w+", chunk.lower())
    return sum(mots.count(t) for t in termes) / (len(mots) ** 0.5)


def main() -> None:
    agent = Agent(
        make_provider(),
        max_steps=10,
        system_prompt=(
            "Tu réponds à des questions sur un corpus volumineux. D'abord "
            "telecharger_corpus pour chaque URL fournie, puis chercher(question) "
            "pour localiser les passages utiles (reformule si besoin), et "
            "lire_passage(id) pour le contexte complet. Cite le passage qui "
            "fonde ta réponse. Ne réponds JAMAIS de mémoire sur le corpus."
        ),
    )

    @agent.tool
    def telecharger_corpus(url: str) -> dict:
        """Télécharge un document texte depuis une URL et l'ajoute à l'index de recherche."""
        CACHE.mkdir(exist_ok=True)
        local = CACHE / (re.sub(r"\W+", "_", url)[-60:] + ".txt")
        if not local.exists():
            with urllib.request.urlopen(url, timeout=60) as reponse:  # noqa: S310
                local.write_bytes(reponse.read())
        texte = local.read_text(encoding="utf-8", errors="replace")
        nb = _indexer(texte)
        return {"indexe": True, "passages_ajoutes": nb, "mots": len(texte.split())}

    @agent.tool
    def chercher(question: str) -> dict:
        """Cherche dans le corpus indexé ; renvoie les 3 passages les plus pertinents (tronqués)."""
        try:
            if _ensure_vecteurs():
                # Recherche par le SENS : cosinus sur les embeddings.
                question_vec = _embed_gemini([question])[0]
                scores = sorted(
                    ((_cosinus(question_vec, v), i) for i, v in enumerate(_VECTEURS)),
                    reverse=True,
                )[:3]
                return {"mode": "semantique", "resultats": [
                    {"id": i, "extrait": _CHUNKS[i][:600]} for s, i in scores if s > 0
                ]}
        except Exception as exc:  # embeddings indisponibles → lexical, jamais d'erreur
            print(f"  (embeddings indisponibles ({exc}) — repli lexical)")
        scores = sorted(
            ((_score(question, c), i) for i, c in enumerate(_CHUNKS)),
            reverse=True,
        )[:3]
        return {"mode": "lexical", "resultats": [
            {"id": i, "extrait": _CHUNKS[i][:600]} for s, i in scores if s > 0
        ]}

    @agent.tool
    def lire_passage(passage_id: int) -> dict:
        """Renvoie un passage complet de l'index, par id."""
        if not 0 <= passage_id < len(_CHUNKS):
            return {"erreur": f"id invalide (0..{len(_CHUNKS) - 1})"}
        return {"id": passage_id, "texte": _CHUNKS[passage_id][:4000]}

    urls = "\n".join(TOMES)
    resultat = agent.run(
        "Voici les 5 tomes des Misérables :\n" + urls +
        "\n\nQuestion : que vole Jean Valjean chez l'évêque, et que lui dit "
        "l'évêque quand les gendarmes le ramènent ?"
    )

    print(f"\n=== Réponse ({resultat.steps} tours) ===\n{resultat.output}")
    mots_corpus = sum(len(c.split()) for c in _CHUNKS)
    tokens_corpus = int(mots_corpus * 1.4)
    depense = resultat.usage.total_tokens if resultat.usage else 0
    mode = "sémantique (embeddings, cosinus)" if _VECTEURS else "lexical (recouvrement de mots)"
    print(f"\n📊 Corpus indexé : {len(_CHUNKS)} passages, ~{mots_corpus:,} mots "
          f"(~{tokens_corpus:,} tokens) — recherche : {mode}")
    print(f"📊 Tokens réellement consommés par l'agent : {depense:,} "
          f"(~{100 * depense / max(tokens_corpus, 1):.1f} % du corpus)")


if __name__ == "__main__":
    main()
