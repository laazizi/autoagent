"""La MÊME démo que demo_autoagent.py — SANS autoagent. Tout à la main.

Même modèle, mêmes trois agents (orchestrateur + 2 analystes), même délégation,
même recoupement, même rapport écrit sous garde, même arbre de trace. Tout ce
que la lib faisait gratuitement, c'est toi qui l'écris et le maintiens :

  §1  un traceur JSONL avec span/parent, à la main (SANS la redaction de
      secrets d'autoagent — encore ~40 lignes de plus s'il te la faut)
  §2  le format wire Gemini + les retries HTTP, à la main
      (et il est DIFFÉRENT pour OpenAI/Anthropic/DeepSeek : multiplie ce
      fichier par provider…)
  §3  LA BOUCLE D'AGENT générique : envoyer l'historique → parser les
      functionCalls → exécuter → réinjecter les functionResponse → reboucler,
      avec un plafond d'étapes maison
  §4  chaque schéma d'outil, tapé à la main, synchronisé à la main
  §5  la DÉLÉGATION à la main : « un agent comme outil d'un autre » = relancer
      la boucle récursivement en convertissant ses échecs en tool errors
  §6  la clôture d'écriture à la main : anti-traversée, allowlist d'extension,
      refus des chemins absolus — facile à rater subtilement

~260 lignes. demo_autoagent.py fait la même chose en ~100.

Lancer :  GEMINI_API_KEY=...  python demo_pure_python.py
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ICI = Path(__file__).resolve().parent
MODELE = "gemini-3.5-flash"
API = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={k}"


# ── §1. Un arbre de trace, À LA MAIN ─────────────────────────────────────────
class Traceur:
    """Événements JSONL avec span_id/parent_id. Le TraceEmitter d'autoagent
    fait en plus la redaction des secrets et est thread-safe — pas celui-ci."""

    def __init__(self, chemin: Path) -> None:
        self._fh = open(chemin, "w", encoding="utf-8")

    def emettre(self, type_: str, charge: dict, parent: str | None = None) -> str:
        span = secrets.token_hex(8)
        self._fh.write(json.dumps({"type": type_, "span_id": span,
                                   "parent_id": parent, "ts": time.time(),
                                   "payload": charge}) + "\n")
        self._fh.flush()
        return span


TRACE = Traceur(ICI / "trace_orchestre_pur.jsonl")


# ── §2. Format wire + retries HTTP, À LA MAIN ────────────────────────────────
def appeler_gemini(systeme: str, contents: list[dict], schemas: list[dict]) -> dict:
    corps = json.dumps({
        "systemInstruction": {"parts": [{"text": systeme}]},
        "contents": contents,
        "tools": [{"functionDeclarations": schemas}],
        "generationConfig": {"temperature": 0},
    }).encode()
    url = API.format(m=MODELE, k=os.environ["GEMINI_API_KEY"].strip())
    for essai in range(4):                          # backoff sur 429/5xx
        try:
            req = urllib.request.Request(
                url, data=corps, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and essai < 3:
                time.sleep(2 ** essai)
                continue
            raise
    raise RuntimeError("inatteignable")


# ── §3. LA BOUCLE D'AGENT GÉNÉRIQUE, À LA MAIN ───────────────────────────────
def executer_agent(nom: str, systeme: str, schemas: list[dict], outils: dict,
                   consigne: str, max_etapes: int = 12,
                   span_parent: str | None = None) -> dict:
    """Fait tourner UN agent jusqu'au bout. Renvoie {"sortie", "etapes"} ou lève."""
    span_run = TRACE.emettre("run_start", {"agent": nom, "nb_outils": len(schemas)},
                             parent=span_parent)
    contents: list[dict] = [{"role": "user", "parts": [{"text": consigne}]}]

    for etape in range(1, max_etapes + 1):
        reponse = appeler_gemini(systeme, contents, schemas)
        try:
            parts = reponse["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            raise RuntimeError(f"{nom} : réponse malformée {str(reponse)[:200]}")
        contents.append({"role": "model", "parts": parts})

        appels = [p["functionCall"] for p in parts if "functionCall" in p]
        if not appels:                              # texte seul → cet agent a fini
            sortie = "".join(p.get("text", "") for p in parts)
            TRACE.emettre("run_end", {"agent": nom, "etapes": etape}, parent=span_run)
            return {"sortie": sortie, "etapes": etape}

        retours = []
        for appel in appels:
            cnom, args = appel.get("name", ""), appel.get("args") or {}
            span = TRACE.emettre("tool_call_start",
                                 {"agent": nom, "name": cnom,
                                  "apercu_args": json.dumps(args)[:120]},
                                 parent=span_run)
            impl = outils.get(cnom)
            try:                                    # les erreurs REPARTENT vers le modèle
                resultat = impl(**args, _span=span) if impl else \
                    {"erreur": f"outil inconnu {cnom}"}
            except TypeError:                       # impl sans support de _span
                try:
                    resultat = impl(**args)
                except Exception as exc:
                    resultat = {"erreur": f"{type(exc).__name__}: {exc}"}
            except Exception as exc:
                resultat = {"erreur": f"{type(exc).__name__}: {exc}"}
            if not isinstance(resultat, dict):      # Gemini exige un OBJET
                resultat = {"resultat": resultat}
            TRACE.emettre("tool_call_end",
                          {"agent": nom, "name": cnom,
                           "statut": "erreur" if "erreur" in resultat else "ok"},
                          parent=span)
            retours.append({"functionResponse": {"name": cnom, "response": resultat}})
        contents.append({"role": "user", "parts": retours})

    TRACE.emettre("run_end", {"agent": nom, "etapes": max_etapes, "statut": "max_etapes"},
                  parent=span_run)
    raise RuntimeError(f"{nom} : plafond max_etapes={max_etapes} dépassé")


# ── §4. Les outils + CHAQUE schéma, À LA MAIN ────────────────────────────────
def lire_fichier(nom: str, **_kw) -> dict:
    return {"contenu": (ICI / nom).read_text(encoding="utf-8")[:10000]}


def compter(texte: str, mot: str, **_kw) -> int:
    return texte.lower().count(mot.lower())


DOSSIER_RAPPORTS = ICI / "_out_pur"


def ecrire_rapport(chemin: str, contenu: str, **_kw) -> dict:
    """§6 : la clôture d'écriture, à la main — trois gardes faciles à rater."""
    if Path(chemin).is_absolute():
        return {"erreur": "chemins absolus interdits"}
    if not chemin.endswith(".md"):
        return {"erreur": "seule l'extension .md est autorisée"}
    cible = (DOSSIER_RAPPORTS / chemin).resolve()
    if not str(cible).startswith(str(DOSSIER_RAPPORTS.resolve())):
        return {"erreur": f"le chemin sort de la zone rapports : {chemin}"}
    cible.parent.mkdir(parents=True, exist_ok=True)
    cible.write_text(contenu, encoding="utf-8")
    return {"ecrit": chemin}


SCHEMA_LIRE = {"name": "lire_fichier",
               "description": "Lit un fichier texte du dossier de démo.",
               "parameters": {"type": "OBJECT",
                              "properties": {"nom": {"type": "STRING"}},
                              "required": ["nom"]}}
SCHEMA_COMPTER = {"name": "compter",
                  "description": "Compte les occurrences d'un mot dans un texte "
                                 "(un LLM ne sait pas compter).",
                  "parameters": {"type": "OBJECT",
                                 "properties": {"texte": {"type": "STRING"},
                                                "mot": {"type": "STRING"}},
                                 "required": ["texte", "mot"]}}
SCHEMA_RAPPORT = {"name": "ecrire_rapport",
                  "description": "Enregistre le rapport final (markdown, zone rapports uniquement).",
                  "parameters": {"type": "OBJECT",
                                 "properties": {"chemin": {"type": "STRING"},
                                                "contenu": {"type": "STRING"}},
                                 "required": ["chemin", "contenu"]}}
SCHEMAS_ANALYSTE = [SCHEMA_LIRE, SCHEMA_COMPTER]
OUTILS_ANALYSTE = {"lire_fichier": lire_fichier, "compter": compter}


# ── §5. LA DÉLÉGATION À LA MAIN : un agent enveloppé en outil d'un autre ─────
def creer_outil_analyste(nom_agent: str, mission: str):
    """Ce que `analyste.as_tool(...)` d'autoagent génère gratuitement — y compris
    le contrat d'échec : un analyste qui plante doit devenir une TOOL ERROR pour
    l'orchestrateur, jamais un crash du programme."""
    systeme = (f"Tu es {nom_agent}. {mission} "
               "Tu ne rapportes que des chiffres vérifiés par tes outils.")

    def deleguer(requete: str, _span: str | None = None, **_kw) -> dict:
        try:
            res = executer_agent(nom_agent, systeme, SCHEMAS_ANALYSTE, OUTILS_ANALYSTE,
                                 requete, span_parent=_span)
            return {"sortie": res["sortie"], "etapes": res["etapes"]}
        except Exception as exc:                    # analyste en échec → tool error
            return {"erreur": f"{nom_agent} a échoué : {type(exc).__name__}: {exc}"}

    schema = {"name": f"interroger_{nom_agent}",
              "description": f"Délègue une question à {nom_agent}. {mission}",
              "parameters": {"type": "OBJECT",
                             "properties": {"requete": {"type": "STRING"}},
                             "required": ["requete"]}}
    return schema, deleguer


schema_serveur, deleguer_serveur = creer_outil_analyste(
    "analyste_serveur", "Tu analyses app.log (journal serveur) : erreurs et messages.")
schema_paiements, deleguer_paiements = creer_outil_analyste(
    "analyste_paiements", "Tu analyses payments.log : paiements échoués, raisons, montants.")

SYSTEME_ORCH = (
    "Tu es l'orchestrateur. Délègue l'analyse de chaque fichier au bon spécialiste. "
    "Puis VALIDE leur travail : chaque erreur serveur « payment gateway returned 502 » "
    "doit correspondre à un paiement FAILED avec la raison gateway_502. Si les chiffres "
    "ne collent pas, contre-vérifie toi-même avec tes propres outils avant de conclure. "
    "Termine en ENREGISTRANT un court rapport validé dans « rapport.md » (ecrire_rapport) : "
    "constats, cohérence des deux analyses, montant total non encaissé — puis réponds "
    "par une conclusion en une seule ligne.")
SCHEMAS_ORCH = [schema_serveur, schema_paiements, SCHEMA_LIRE, SCHEMA_COMPTER, SCHEMA_RAPPORT]
OUTILS_ORCH = {"interroger_analyste_serveur": deleguer_serveur,
               "interroger_analyste_paiements": deleguer_paiements,
               "lire_fichier": lire_fichier, "compter": compter,
               "ecrire_rapport": ecrire_rapport}


def principal() -> None:
    res = executer_agent(
        "orchestrateur", SYSTEME_ORCH, SCHEMAS_ORCH, OUTILS_ORCH,
        "Analyse l'incident du jour : combien d'erreurs serveur, combien de paiements "
        "échoués, les deux analyses sont-elles cohérentes, et quel montant n'a pas pu "
        "être encaissé ?")
    print(res["sortie"])
    rapport = DOSSIER_RAPPORTS / "rapport.md"
    print(f"\n📄 rapport enregistré : {rapport.exists()} ({rapport})")
    print(f"({res['etapes']} tours d'orchestrateur — trace dans trace_orchestre_pur.jsonl)")


if __name__ == "__main__":
    principal()
