"""Banc d'EFFICACITÉ : combien de tâches un agent réussit, à quel coût, en combien de temps.

Vingt tâches qui ressemblent aux usages d'une entreprise de données de mobilité, cinq familles de quatre :

  A. extraction      une réponse libre d'enquête → un JSON (commune, heure de départ, modes)
  B. codification    une réponse libre → le code d'une nomenclature de motifs (lue par un outil)
  C. sql             une question sur une base de comptages (outil SELECT, lecture seule)
  D. gros résultat   un journal de capteur (~10 000 caractères) ou un export CSV (~30 000)
  E. plusieurs étapes enchaîner ou combiner des outils

Toutes les données sont SIMULÉES, générées ici avec une graine fixe : rien de réel ne part chez le
fournisseur. Les réponses attendues sont CALCULÉES depuis ces mêmes données (aucune saisie à la main),
et chaque juge est du CODE, audité par `audit_check` avant le premier appel payant (`--verifier`).

Point de départ (étape 2) : la configuration « telle quelle » — `Agent(provider, system_prompt)` et ses
outils, rien d'autre : aucune borne, aucun élagage, pas d'outils en parallèle (les défauts de la lib).

    python evals/eval_efficacite.py --verifier                    # hors ligne, gratuit
    python evals/eval_efficacite.py --provider deepseek --k 3     # point de départ sur un modèle
    python evals/eval_efficacite.py --provider gemini --k 3

Résultats dans evals/resultats_efficacite.json (une entrée par fournisseur/modèle).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import statistics
import sys
import tempfile
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
sys.path.insert(0, str(RACINE / "examples_autoagent"))

from autoagent import Agent  # noqa: E402
from autoagent.eval import Attempt, run_k  # noqa: E402

SORTIE = Path(__file__).resolve().parent / "resultats_efficacite.json"

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Le monde simulé : 4 villes x 5 capteurs, une semaine de comptages horaires, un journal par capteur
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

GRAINE = 2026
VILLES = {"LYO": "Lyon", "GRE": "Grenoble", "VAL": "Valence", "STE": "Saint-Étienne"}
JOURS = [f"2026-09-{d}" for d in range(14, 21)]                 # lundi 14 → dimanche 20 septembre 2026
WEEKEND = {"2026-09-19", "2026-09-20"}
# Statuts posés à la main : ils fixent des réponses uniques (2 pannes à Grenoble, 1 seule à Valence).
STATUTS = {
    "LYO": ["ok", "ok", "maintenance", "ok", "ok"],
    "GRE": ["ok", "panne", "ok", "panne", "ok"],
    "VAL": ["ok", "ok", "panne", "ok", "ok"],
    "STE": ["ok", "ok", "ok", "ok", "ok"],
}
VOLUME_VILLE = {"LYO": 1.5, "GRE": 1.0, "VAL": 0.7, "STE": 0.9}
VELO_CAPTEUR = [1.0, 1.4, 0.8, 1.9, 1.1]                        # le 4e capteur de chaque ville roule le plus à vélo
ERREURS_STE = [4, 9, 15, 7, 11]                                   # erreurs par journal, Saint-Étienne : max unique
PROFIL = [0.08, 0.05, 0.04, 0.04, 0.07, 0.2, 0.55, 0.95, 1.0, 0.7, 0.55, 0.6,
          0.65, 0.6, 0.6, 0.65, 0.8, 0.98, 0.9, 0.6, 0.4, 0.3, 0.2, 0.12]

MOTIFS = {
    "1": "Retour au domicile",
    "11": "Travail sur le lieu habituel",
    "12": "Travail hors du lieu habituel",
    "21": "École, collège, lycée (élève)",
    "22": "Études supérieures",
    "31": "Achats en grande surface",
    "32": "Achats de proximité",
    "41": "Santé (médecin, dentiste, hôpital)",
    "51": "Démarches administratives",
    "61": "Accompagner quelqu'un (le déposer)",
    "62": "Aller chercher quelqu'un",
    "71": "Loisirs sportifs",
    "72": "Loisirs culturels (cinéma, musée, spectacle)",
    "81": "Visite à des proches",
    "91": "Autre motif",
}
MODES = ["MARCHE", "VELO", "VOITURE", "BUS", "TRAM", "METRO", "TRAIN"]


@dataclass
class Monde:
    capteurs: dict[str, dict[str, Any]]
    comptages: list[tuple[str, str, int, int, int]]               # (capteur, jour, heure, vehicules, velos)
    journaux: dict[str, str]
    erreurs: dict[str, list[tuple[str, str]]]                     # capteur → [(horodatage, type d'erreur)]


def construire_monde() -> Monde:
    rng = random.Random(GRAINE)
    capteurs: dict[str, dict[str, Any]] = {}
    comptages: list[tuple[str, str, int, int, int]] = []
    for code, ville in VILLES.items():
        for i in range(5):
            ident = f"CMP-{code}-{i + 1:02d}"
            capteurs[ident] = {
                "ville": ville,
                "type": rng.choice(["boucle", "radar", "caméra"]),
                "statut": STATUTS[code][i],
                "date_installation": f"{rng.randint(2021, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            }
            pointe = rng.randint(250, 700) * VOLUME_VILLE[code]
            for jour in JOURS:
                facteur = 0.55 if jour in WEEKEND else 1.0
                for heure in range(24):
                    vehicules = int(pointe * PROFIL[heure] * facteur * rng.uniform(0.85, 1.15))
                    velos = int(pointe * 0.12 * VELO_CAPTEUR[i] * PROFIL[heure] * facteur * rng.uniform(0.7, 1.3))
                    comptages.append((ident, jour, heure, vehicules, velos))
    journaux: dict[str, str] = {}
    erreurs: dict[str, list[tuple[str, str]]] = {}
    for ident in capteurs:
        code = ident.split("-")[1]
        index = int(ident[-2:]) - 1
        n_erreurs = ERREURS_STE[index] if code == "STE" else rng.randint(3, 22)
        minutes_erreurs = sorted(rng.sample(range(5, 1435, 5), n_erreurs))
        lignes: list[str] = []
        liste: list[tuple[str, str]] = []
        comptage_max = 0
        for minute in range(0, 1440, 10):
            h, m = divmod(minute, 60)
            valeur = int(80 * VOLUME_VILLE[code] * PROFIL[h] * rng.uniform(0.8, 1.2) * 4) + rng.randint(0, 9)
            comptage_max = max(comptage_max, valeur)
            lignes.append((minute, f"2026-09-15T{h:02d}:{m:02d}:{rng.randint(0, 59):02d} INFO {ident} "
                                   f"comptage={valeur} velos={int(valeur * 0.15 * rng.uniform(0.6, 1.4))}"))
            if rng.random() < 0.06:
                lignes.append((minute + 1, f"2026-09-15T{h:02d}:{m + 1:02d}:{rng.randint(0, 59):02d} WARN {ident} "
                                           f"tension_batterie={rng.uniform(10.8, 11.6):.1f}V"))
        for minute in minutes_erreurs:
            h, m = divmod(minute, 60)
            kind = rng.choice(["timeout liaison=4G", "timeout liaison=4G", "checksum trame=invalide"])
            horodatage = f"2026-09-15T{h:02d}:{m:02d}:{rng.randint(0, 59):02d}"
            lignes.append((minute, f"{horodatage} ERROR {ident} {kind}"))
            liste.append((horodatage, kind.split()[0]))
        lignes.sort(key=lambda x: x[0])
        journaux[ident] = "\n".join(t for _, t in lignes) + "\n"
        erreurs[ident] = liste
    return Monde(capteurs, comptages, journaux, erreurs)


def creer_base(monde: Monde, dossier: Path) -> Path:
    chemin = dossier / "comptages.sqlite"
    con = sqlite3.connect(chemin)
    con.execute("CREATE TABLE capteurs (id TEXT PRIMARY KEY, ville TEXT, type TEXT, statut TEXT, date_installation TEXT)")
    con.execute("CREATE TABLE comptages (capteur_id TEXT, jour TEXT, heure INTEGER, vehicules INTEGER, velos INTEGER)")
    con.executemany("INSERT INTO capteurs VALUES (?, ?, ?, ?, ?)",
                    [(i, c["ville"], c["type"], c["statut"], c["date_installation"]) for i, c in monde.capteurs.items()])
    con.executemany("INSERT INTO comptages VALUES (?, ?, ?, ?, ?)", monde.comptages)
    con.commit()
    con.close()
    return chemin


def export_csv(monde: Monde, ville: str) -> str:
    lignes = ["capteur;jour;heure;vehicules;velos"]
    for ident, jour, heure, vehicules, velos in monde.comptages:
        if monde.capteurs[ident]["ville"] == ville:
            lignes.append(f"{ident};{jour};{heure};{vehicules};{velos}")
    return "\n".join(lignes) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Les outils (ce que l'hôte expose) — les mêmes pour toutes les tâches : l'agent choisit
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

MAX_LIGNES_SQL = 200


def fabriquer_outils(monde: Monde, base: Path) -> dict[str, Callable[..., Any]]:
    """Les cinq outils de l'hôte, en fonctions simples : enregistrés comme outils de l'agent, et passés tels
    quels à `PythonRunner(host_functions=...)` pour la configuration « code comme action »."""

    def executer_sql(requete: str) -> dict:
        """Exécute une requête SQL SELECT (lecture seule) sur la base des comptages.
        Tables : capteurs(id, ville, type, statut, date_installation) ;
        comptages(capteur_id, jour, heure, vehicules, velos) — un comptage par capteur, par jour (AAAA-MM-JJ) et par heure (0-23)."""
        if not re.match(r"^\s*(select|with)\b", requete, re.I):
            raise ValueError("seules les requêtes SELECT sont autorisées")
        con = sqlite3.connect(f"file:{base.as_posix()}?mode=ro", uri=True)
        try:
            cur = con.execute(requete)
            colonnes = [d[0] for d in cur.description or []]
            lignes = cur.fetchall()
        finally:
            con.close()
        resultat: dict[str, Any] = {"colonnes": colonnes, "lignes": [list(r) for r in lignes[:MAX_LIGNES_SQL]]}
        if len(lignes) > MAX_LIGNES_SQL:
            resultat["note"] = f"{len(lignes)} lignes au total, {MAX_LIGNES_SQL} affichées"
        return resultat

    def fiche_capteur(capteur_id: str) -> dict:
        """Renvoie la fiche d'un capteur : ville, type, statut, date d'installation."""
        fiche = monde.capteurs.get(capteur_id.strip().upper())
        if fiche is None:
            raise ValueError(f"capteur inconnu : {capteur_id}")
        return {"id": capteur_id.strip().upper(), **fiche}

    def lire_journal(capteur_id: str) -> str:
        """Renvoie le journal technique complet d'un capteur pour la journée du 2026-09-15 (une ligne par événement)."""
        texte = monde.journaux.get(capteur_id.strip().upper())
        if texte is None:
            raise ValueError(f"capteur inconnu : {capteur_id}")
        return texte

    def export_comptages_csv(ville: str) -> str:
        """Renvoie l'export CSV de tous les comptages horaires de la semaine pour une ville
        (colonnes : capteur;jour;heure;vehicules;velos)."""
        cible = _norm(ville)
        for nom in VILLES.values():
            if _norm(nom) == cible:
                return export_csv(monde, nom)
        raise ValueError(f"ville inconnue : {ville}")

    def nomenclature_motifs() -> dict:
        """Renvoie la nomenclature des motifs de déplacement : code → libellé."""
        return dict(MOTIFS)

    return {f.__name__: f for f in (executer_sql, fiche_capteur, lire_journal, export_comptages_csv, nomenclature_motifs)}


def installer_outils(agent: Agent, outils: dict[str, Callable[..., Any]]) -> None:
    for fonction in outils.values():
        agent.tool(fonction)


SYSTEME = (
    "Tu es un assistant d'analyse pour une entreprise qui collecte des données de mobilité. "
    "Utilise les outils disponibles quand la question le demande et n'invente aucune valeur. "
    "Termine TOUJOURS ta réponse par une dernière ligne de la forme : RÉPONSE: <valeur>"
)

# Une phrase de plus pour la configuration « code » : sans elle, rien ne dit au modèle QUAND l'outil vaut le coup.
CONSIGNE_CODE = (
    " Quand la réponse demande de compter, d'additionner, de trier ou de chercher dans un GROS résultat (journal, "
    "export CSV, beaucoup de lignes), écris plutôt un court programme avec l'outil run_python : il peut appeler les "
    "autres outils via context['call_host'](nom, {arguments}) et ne te renvoie que le résultat calculé."
)

# Chaque configuration = UN réglage qui existe déjà dans la lib, le reste identique au point de départ.
CONFIGS: dict[str, dict[str, Any]] = {
    "telle_quelle": {"agent": {}, "code": False, "consigne": "", "description": "défauts de la lib"},
    "code": {"agent": {}, "code": True, "consigne": CONSIGNE_CODE,
             "description": "code comme action : enable_run_python(PythonRunner(host_functions=outils)) + une consigne"},
    "coupe": {"agent": {"max_tool_result_chars": 4000}, "code": False, "consigne": "",
              "description": "max_tool_result_chars=4000 (coupe au milieu des gros résultats)"},
    "elagage": {"agent": {"prune_tool_results_after": 1}, "code": False, "consigne": "",
                "description": "prune_tool_results_after=1 (les vieux résultats ne sont plus renvoyés)"},
    "parallele": {"agent": {"parallel_tool_calls": True}, "code": False, "consigne": "",
                  "description": "parallel_tool_calls=True (les appels d'un même tour en parallèle)"},
    "code_une_ligne": {"agent": {}, "code": "une_ligne", "consigne": "",
                       "description": "code comme action en une ligne : agent.enable_code_action() (description courte, aucune consigne)"},
}


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Les juges — du CODE, sur la dernière ligne « RÉPONSE: … » (ou sur le JSON pour l'extraction)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

def _norm(texte: str) -> str:
    sans_accents = "".join(c for c in unicodedata.normalize("NFKD", texte) if not unicodedata.combining(c))
    return re.sub(r"[\s\-_'\u2019]+", " ", sans_accents).strip().casefold()


def derniere_reponse(sortie: str) -> str | None:
    """La valeur de la DERNIÈRE ligne « RÉPONSE: … » (ou de la ligne suivante si la valeur y est passée)."""
    lignes = [x.strip() for x in (sortie or "").splitlines() if x.strip()]
    for i in range(len(lignes) - 1, -1, -1):
        m = re.match(r"^[*_`>\s]*R[ÉE]PONSE\s*\**\s*:\s*(.*?)$", lignes[i], re.I)
        if m:
            valeur = m.group(1).strip().strip("*_` ").rstrip(".").strip()
            if not valeur and i + 1 < len(lignes):
                valeur = lignes[i + 1].strip().strip("*_` ").rstrip(".").strip()
            return valeur or None
    return None


def _sans_espaces_milliers(texte: str) -> str:
    return re.sub(r"(?<=\d)[\s\u00a0\u202f.](?=\d{3}\b)", "", texte)


# Une réponse qui HÉSITE est fausse, même si la bonne valeur y figure (« 6 ou 7 », « environ 7 », « 6 à 7 »).
# Testé sur le texte AVEC accents : « où » n'est pas « ou ».
_HESITATION = re.compile(r"\b(ou|or|entre|environ|approximativement|probablement|mais)\b|~|≈|±|\?"
                         r"|\d\s*(?:à|\u2013)\s*\d|\d\s+-\s+\d", re.I)


def juge(nature: str, attendu: Any) -> Callable[[Any], bool]:
    """Un juge déterministe : la réponse est la PREMIÈRE valeur de la bonne nature sur la ligne « RÉPONSE: »
    (un modèle ajoute souvent un contexte après — « CMP-LYO-02 (contre 4 904 pour CMP-GRE-05) ») ;
    une ligne qui hésite est refusée."""

    def check(resultat: Any) -> bool:
        valeur = derniere_reponse(getattr(resultat, "output", "") or "")
        if valeur is None or _HESITATION.search(valeur):
            return False
        if nature == "nombre":
            nombres = re.findall(r"(?<![\w-])\d+(?:[.,]\d+)?", _sans_espaces_milliers(valeur))
            return bool(nombres) and nombres[0] == str(attendu)
        if nature == "id":
            ids = re.findall(r"CMP-[A-Z]{3}-\d{2}\b", valeur.upper())
            return bool(ids) and ids[0] == attendu
        if nature == "ville":
            positions = sorted((_norm(valeur).find(_norm(v)), v) for v in VILLES.values() if _norm(v) in _norm(valeur))
            return bool(positions) and positions[0][1] == attendu
        if nature == "heure":
            heures = re.findall(r"\b(\d{1,2})\s*[:h]\s*(\d{2})\b", valeur)
            return bool(heures) and f"{int(heures[0][0]):02d}:{heures[0][1]}" == attendu
        if nature == "date":
            dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", valeur)
            return bool(dates) and dates[0] == attendu
        if nature == "code":
            codes = re.findall(r"(?<![\w-])\d{1,2}\b", valeur)
            return bool(codes) and codes[0] == attendu
        raise ValueError(f"nature inconnue : {nature}")

    check.__name__ = f"juge_{nature}"
    return check


def _premier_json(texte: str) -> dict | None:
    decodeur = json.JSONDecoder()
    for i, c in enumerate(texte or ""):
        if c == "{":
            try:
                objet, _ = decodeur.raw_decode(texte[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(objet, dict):
                return objet
    return None


def juge_extraction(commune: str, heure: str, modes: set[str]) -> Callable[[Any], bool]:
    def check(resultat: Any) -> bool:
        objet = _premier_json(getattr(resultat, "output", "") or "")
        if objet is None:
            return False
        h = re.fullmatch(r"\s*(\d{1,2})\s*[:h]\s*(\d{2})\s*", str(objet.get("heure_depart", "")))
        lus = objet.get("modes")
        return (_norm(str(objet.get("commune_depart", ""))) == _norm(commune)
                and h is not None and f"{int(h.group(1)):02d}:{h.group(2)}" == heure
                and isinstance(lus, list) and {str(x).strip().upper() for x in lus} == modes)

    check.__name__ = "juge_extraction"
    return check


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Les vingt tâches — les réponses attendues sont CALCULÉES depuis le monde
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Tache:
    id: str
    famille: str
    prompt: str
    check: Callable[[Any], bool]
    nature: str
    attendu: Any


def _total(monde: Monde, capteur: str, jour: str, champ: int) -> int:
    return sum(r[champ] for r in monde.comptages if r[0] == capteur and r[1] == jour)


def _unique_max(valeurs: dict[str, int], marge: float = 0.0) -> str:
    tri = sorted(valeurs.items(), key=lambda kv: kv[1], reverse=True)
    assert tri[0][1] > tri[1][1] * (1 + marge), f"maximum non unique ou trop serré : {tri[:2]}"
    return tri[0][0]


def construire_taches(monde: Monde) -> list[Tache]:
    t: list[Tache] = []
    consigne_json = ("Réponds par un objet JSON avec exactement les clés commune_depart (texte), heure_depart (HH:MM) "
                     f"et modes (liste de codes parmi {', '.join(MODES)}, dans n'importe quel ordre).")
    extractions = [
        ("A1", "Ce matin je suis parti de Bron à 7h45, j'ai pris le bus C17 puis le métro A jusqu'à Perrache.",
         "Bron", "07:45", {"BUS", "METRO"}),
        ("A2", "Je suis parti de Villeurbanne à 18h10, à vélo jusqu'à la gare, puis le train pour Ambérieu.",
         "Villeurbanne", "18:10", {"VELO", "TRAIN"}),
        ("A3", "J'ai pris ma voiture à Meyzieu à 7h55, je l'ai laissée au parking relais et j'ai fini en tram.",
         "Meyzieu", "07:55", {"VOITURE", "TRAM"}),
        ("A4", "Je suis sorti de chez moi à Oullins à midi et quart et je suis allé à pied jusqu'au marché.",
         "Oullins", "12:15", {"MARCHE"}),
    ]
    for ident, phrase, commune, heure, modes in extractions:
        t.append(Tache(ident, "A. extraction",
                       f"Voici la réponse libre d'un enquêté sur son dernier déplacement : « {phrase} » {consigne_json}",
                       juge_extraction(commune, heure, modes), "json", {"commune_depart": commune, "heure_depart": heure,
                                                                         "modes": sorted(modes)}))

    codifications = [
        ("B1", "Je suis allé chercher mon fils à son entraînement de foot.", "62"),
        ("B2", "J'avais rendez-vous chez le dentiste.", "41"),
        ("B3", "J'ai fait les grosses courses de la semaine à l'hypermarché.", "31"),
        ("B4", "Je suis allé à la mairie pour refaire ma carte d'identité.", "51"),
    ]
    for ident, phrase, code in codifications:
        t.append(Tache(ident, "B. codification",
                       f"Un enquêté décrit le motif de son déplacement : « {phrase} » Quel est le code de ce motif dans la "
                       "nomenclature officielle des motifs (utilise l'outil) ? La valeur de RÉPONSE est le code seul.",
                       juge("code", code), "code", code))

    gre_pannes = sum(1 for c in monde.capteurs.values() if c["ville"] == "Grenoble" and c["statut"] == "panne")
    velos_lyon = {i: _total(monde, i, "2026-09-16", 4) for i, c in monde.capteurs.items() if c["ville"] == "Lyon"}
    total_val03 = _total(monde, "CMP-VAL-03", "2026-09-18", 3)
    par_ville = {v: sum(r[3] for r in monde.comptages if monde.capteurs[r[0]]["ville"] == v) for v in VILLES.values()}
    t += [
        Tache("C1", "C. sql", "Combien de capteurs de Grenoble sont en panne ?", juge("nombre", gre_pannes), "nombre", gre_pannes),
        Tache("C2", "C. sql", "Quel capteur de Lyon a compté le plus de vélos au total le 2026-09-16 ? Donne son identifiant.",
              juge("id", _unique_max(velos_lyon, 0.03)), "id", _unique_max(velos_lyon, 0.03)),
        Tache("C3", "C. sql", "Quel est le nombre total de véhicules comptés par le capteur CMP-VAL-03 le 2026-09-18 ?",
              juge("nombre", total_val03), "nombre", total_val03),
        Tache("C4", "C. sql", "Quelle ville totalise le plus de véhicules sur toute la semaine, tous ses capteurs confondus ?",
              juge("ville", _unique_max(par_ville, 0.05)), "ville", _unique_max(par_ville, 0.05)),
    ]

    n_err_lyo5 = sum(1 for ligne in monde.journaux["CMP-LYO-05"].splitlines() if " ERROR " in ligne)
    max_gre2 = max(int(m) for m in re.findall(r"comptage=(\d+)", monde.journaux["CMP-GRE-02"]))
    assert sum(1 for m in re.findall(r"comptage=(\d+)", monde.journaux["CMP-GRE-02"]) if int(m) == max_gre2) == 1, "max non unique"
    premier_timeout = next(h for h, kind in monde.erreurs["CMP-STE-04"] if kind == "timeout")[11:16]
    velos_val2 = _total(monde, "CMP-VAL-02", "2026-09-17", 4)
    t += [
        Tache("D1", "D. gros résultat", "Combien de lignes ERROR contient le journal du capteur CMP-LYO-05 ?",
              juge("nombre", n_err_lyo5), "nombre", n_err_lyo5),
        Tache("D2", "D. gros résultat", "Dans le journal du capteur CMP-GRE-02, quelle est la plus grande valeur de comptage ?",
              juge("nombre", max_gre2), "nombre", max_gre2),
        Tache("D3", "D. gros résultat",
              "Dans le journal du capteur CMP-STE-04, à quelle heure (HH:MM) a eu lieu la première erreur de type timeout ?",
              juge("heure", premier_timeout), "heure", premier_timeout),
        Tache("D4", "D. gros résultat",
              "D'après l'export CSV des comptages de Valence, quel est le nombre total de vélos du capteur CMP-VAL-02 "
              "le 2026-09-17 ?", juge("nombre", velos_val2), "nombre", velos_val2),
    ]

    panne_val = [i for i, c in monde.capteurs.items() if c["ville"] == "Valence" and c["statut"] == "panne"]
    assert len(panne_val) == 1
    date_panne_val = monde.capteurs[panne_val[0]]["date_installation"]
    duel = {i: _total(monde, i, "2026-09-15", 3) for i in ("CMP-LYO-02", "CMP-GRE-05")}
    n_err_deux = n_err_lyo5 + sum(1 for ligne in monde.journaux["CMP-GRE-02"].splitlines() if " ERROR " in ligne)
    err_ste = {i: sum(1 for ligne in monde.journaux[i].splitlines() if " ERROR " in ligne)
               for i in monde.capteurs if i.startswith("CMP-STE")}
    t += [
        Tache("E1", "E. plusieurs étapes",
              "Le seul capteur en panne à Valence : quelle est sa date d'installation (format AAAA-MM-JJ) ?",
              juge("date", date_panne_val), "date", date_panne_val),
        Tache("E2", "E. plusieurs étapes",
              "Le 2026-09-15, lequel des deux capteurs CMP-LYO-02 et CMP-GRE-05 a compté le plus de véhicules au total ?",
              juge("id", _unique_max(duel, 0.03)), "id", _unique_max(duel, 0.03)),
        Tache("E3", "E. plusieurs étapes",
              "Combien de lignes ERROR y a-t-il au total dans les journaux des capteurs CMP-LYO-05 et CMP-GRE-02 réunis ?",
              juge("nombre", n_err_deux), "nombre", n_err_deux),
        Tache("E4", "E. plusieurs étapes",
              "Parmi les cinq capteurs de Saint-Étienne (CMP-STE-01 à CMP-STE-05), lequel a le plus de lignes ERROR dans "
              "son journal ?", juge("id", _unique_max(err_ste)), "id", _unique_max(err_ste)),
    ]
    assert len({x.id for x in t}) == 20
    return t


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Vérification hors ligne : cohérence des réponses attendues (SQL ↔ Python) et audit des juges
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

def _exemples(tache: Tache) -> tuple[list[str], list[str]]:
    """Exemples que le juge DOIT accepter / refuser — d'abord les presque-bons."""
    a = tache.attendu
    if tache.nature == "json":
        bon = json.dumps(a, ensure_ascii=False)
        mauvais = [json.dumps({**a, "heure_depart": "08:45" if a["heure_depart"] != "08:45" else "09:45"}, ensure_ascii=False),
                   json.dumps({**a, "modes": a["modes"][:-1] or ["BUS"]}, ensure_ascii=False),
                   json.dumps({**a, "modes": a["modes"] + ["TAXI"]}, ensure_ascii=False),
                   json.dumps({**a, "commune_depart": "Lyon"}, ensure_ascii=False),
                   "commune Bron, 07:45, bus et métro"]
        return [bon, f"Voici le JSON :\n```json\n{bon}\n```", bon.replace(a["heure_depart"], a["heure_depart"].replace(":", "h"))], mauvais
    if tache.nature == "nombre":
        n = int(a)
        return ([f"J'ai compté.\nRÉPONSE: {n}", f"RÉPONSE : {n:,}".replace(",", " "), f"**RÉPONSE: {n}**",
                 f"RÉPONSE: {n} (journal du capteur CMP-LYO-05)", f"RÉPONSE:\n{n}"],
                [f"RÉPONSE: {n * 10}", f"RÉPONSE: {n + 1}", f"RÉPONSE: {n}0", f"RÉPONSE: entre {n - 1} et {n}",
                 f"Le total est {n}.", f"RÉPONSE: {n - 1}", f"RÉPONSE: {n} ou {n + 1}", f"RÉPONSE: environ {n}",
                 f"RÉPONSE: {n - 1} à {n}", f"RÉPONSE: {n + 1} (et non {n})", f"RÉPONSE: {n} ?"])
    if tache.nature == "id":
        autre = a[:-2] + ("01" if a[-2:] != "01" else "02")
        return ([f"RÉPONSE: {a}", f"Le capteur est {a}.\nRÉPONSE: {a}", f"RÉPONSE: {a} (11 269 véhicules contre 4 904 pour {autre})"],
                [f"RÉPONSE: {autre}", f"RÉPONSE: {a} ou {autre}", f"RÉPONSE: {a[:-1]}", f"C'est {a}.",
                 f"RÉPONSE: {autre} (et non {a})"])
    if tache.nature == "ville":
        autre = "Grenoble" if a != "Grenoble" else "Lyon"
        return ([f"RÉPONSE: {a}", f"RÉPONSE: {a.upper()}", f"RÉPONSE: {a} (devant {autre})"],
                [f"RÉPONSE: {autre}", f"RÉPONSE: {a} ou {autre}", f"{a}", f"RÉPONSE: {autre}, devant {a}"])
    if tache.nature == "heure":
        h, m = a.split(":")
        return ([f"RÉPONSE: {a}", f"RÉPONSE: {int(h)}h{m}", f"RÉPONSE: {a} (première erreur timeout)"],
                [f"RÉPONSE: {int(h) + 1:02d}:{m}", f"RÉPONSE: {a} ou {int(h):02d}:{(int(m) + 5) % 60:02d}", f"À {a}."])
    if tache.nature == "date":
        y = int(a[:4])
        return ([f"RÉPONSE: {a}", f"RÉPONSE: {a} (capteur CMP-VAL-03)"],
                [f"RÉPONSE: {y + 1}{a[4:]}", f"RÉPONSE: {a} ou {y - 1}{a[4:]}", f"Installé le {a}."])
    if tache.nature == "code":
        autre = "61" if a != "61" else "62"
        return ([f"RÉPONSE: {a}", f"Motif : {MOTIFS[a]}\nRÉPONSE: {a}", f"RÉPONSE: {a} ({MOTIFS[a]})"],
                [f"RÉPONSE: {autre}", f"RÉPONSE: {a} ou {autre}", f"RÉPONSE: {a}1", f"Le code est {a}.",
                 f"RÉPONSE: {autre} ({MOTIFS[autre]})"])
    raise ValueError(tache.nature)


def verifier(monde: Monde, taches: list[Tache], base: Path) -> bool:
    from autoagent.judge import audit_check

    ok = True
    con = sqlite3.connect(base)
    controles = {
        "C1": con.execute("SELECT COUNT(*) FROM capteurs WHERE ville='Grenoble' AND statut='panne'").fetchone()[0],
        "C2": con.execute("SELECT capteur_id FROM comptages JOIN capteurs ON capteurs.id=capteur_id WHERE ville='Lyon' "
                          "AND jour='2026-09-16' GROUP BY capteur_id ORDER BY SUM(velos) DESC LIMIT 1").fetchone()[0],
        "C3": con.execute("SELECT SUM(vehicules) FROM comptages WHERE capteur_id='CMP-VAL-03' AND jour='2026-09-18'").fetchone()[0],
        "C4": con.execute("SELECT ville FROM comptages JOIN capteurs ON capteurs.id=capteur_id GROUP BY ville "
                          "ORDER BY SUM(vehicules) DESC LIMIT 1").fetchone()[0],
        "E1": con.execute("SELECT date_installation FROM capteurs WHERE ville='Valence' AND statut='panne'").fetchone()[0],
    }
    con.close()
    csv_val = export_csv(monde, "Valence")
    controles["D4"] = sum(int(ligne.split(";")[4]) for ligne in csv_val.splitlines()[1:]
                          if ligne.startswith("CMP-VAL-02;2026-09-17;"))
    par_id = {x.id: x for x in taches}
    for ident, valeur in controles.items():
        if valeur != par_id[ident].attendu:
            print(f"  ✗ {ident} : attendu {par_id[ident].attendu!r}, recalcul indépendant {valeur!r}")
            ok = False
    print(f"  réponses attendues recalculées autrement (SQL, CSV) : {len(controles)} contrôlées, "
          f"{'toutes identiques' if ok else 'ÉCARTS'}")
    tailles = {i: len(monde.journaux[i]) for i in ("CMP-LYO-05", "CMP-GRE-02", "CMP-STE-04")}
    print(f"  tailles : journaux {tailles} caractères ; export CSV Valence {len(csv_val)} caractères")

    for tache in taches:
        bons, mauvais = _exemples(tache)
        rapport = audit_check(tache.check, good=bons, bad=mauvais, name=tache.id)
        if rapport.verdict != "no_defect_found":
            ok = False
            print(f"  ✗ juge {tache.id} : {rapport.verdict}")
            print("    " + rapport.summary().replace("\n", "\n    "))
    print(f"  juges audités (audit_check, presque-bons + négatifs triviaux) : {len(taches)}, "
          f"{'aucun défaut trouvé' if ok else 'DÉFAUT'}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Le point de départ : chaque tâche k fois, configuration « telle quelle »
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

class _AgentMesure:
    """Enveloppe minimale : un run qui LÈVE (plafond d'étapes, budget…) a quand même dépensé des jetons ;
    `run_k` ne les lit pas sur l'exception — on les relève ici pour que le coût des échecs soit compté."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.depense_si_exception: tuple[int, int] | None = None

    def run(self, prompt: str, context: dict[str, Any] | None = None) -> Any:
        try:
            return self.agent.run(prompt, context=context)
        except Exception as exc:
            etat = getattr(exc, "state", None)
            if etat is not None:
                self.depense_si_exception = (getattr(etat, "input_tokens", 0) or 0, getattr(etat, "output_tokens", 0) or 0)
            raise


def fabriquer_agent(provider: Any, outils: dict[str, Callable[..., Any]], config: str) -> Agent:
    """Un agent NEUF dans la configuration demandée — le même pour `mesurer` et `comparer`."""
    reglage = CONFIGS[config]
    agent = Agent(provider, system_prompt=SYSTEME + reglage["consigne"], **reglage["agent"])
    installer_outils(agent, outils)
    if reglage["code"] == "une_ligne":
        agent.enable_code_action()
    elif reglage["code"]:
        from autoagent import PythonRunner
        agent.enable_run_python(PythonRunner(host_functions=outils))
    return agent


def mesurer(provider: Any, monde: Monde, base: Path, taches: list[Tache], k: int,
            config: str = "telle_quelle") -> dict[str, Any]:
    dernier: dict[str, _AgentMesure] = {}
    outils = fabriquer_outils(monde, base)

    def fabrique() -> _AgentMesure:
        dernier["x"] = _AgentMesure(fabriquer_agent(provider, outils, config))
        return dernier["x"]

    resultats: dict[str, Any] = {}
    for tache in taches:
        sorties: list[str] = []

        def suivi(tentative: Attempt, tache: Tache = tache, sorties: list[str] = sorties) -> None:
            enveloppe = dernier.get("x")
            if tentative.total_tokens is None and enveloppe is not None and enveloppe.depense_si_exception:
                entree, sortie = enveloppe.depense_si_exception
                tentative.input_tokens, tentative.output_tokens, tentative.total_tokens = entree, sortie, entree + sortie
            sorties.append((tentative.output or tentative.error or "")[-300:])
            print(f"    {tache.id} essai {tentative.index}/{k} : {'✓' if tentative.ok else '✗'} "
                  f"{tentative.total_tokens or '?':>6} jetons {tentative.seconds or 0:6.1f} s"
                  + (f"  [{tentative.error[:90]}]" if tentative.error else ""), flush=True)

        rapport = run_k(fabrique, tache.prompt, k=k, check=tache.check, on_attempt=suivi)
        resultats[tache.id] = {"famille": tache.famille, "attendu": tache.attendu, **rapport.to_dict(), "sorties": sorties}
    return resultats


def comparer(provider: Any, monde: Monde, base: Path, taches: list[Tache], k: int, config_b: str,
             config_a: str = "telle_quelle") -> dict[str, Any]:
    """La confirmation : `compare_configs` de la lib — les deux bras ALTERNENT à chaque répétition, verdict
    de réussite (test exact + intervalle), coût et durée avec intervalles bootstrap."""
    from autoagent.compare import EvalTask, Variant, compare_configs

    outils = fabriquer_outils(monde, base)
    suite = [EvalTask(name=t.id, prompt=t.prompt, check=t.check) for t in taches]

    def suivi(bras: str, tache: str, tentative: Attempt) -> None:
        print(f"    {bras:15} {tache} essai {tentative.index} : {'✓' if tentative.ok else '✗'} "
              f"{tentative.total_tokens or '?':>6} jetons {tentative.seconds or 0:6.1f} s", flush=True)

    rapport = compare_configs(
        Variant(config_a, lambda: fabriquer_agent(provider, outils, config_a), {"config": config_a}),
        Variant(config_b, lambda: fabriquer_agent(provider, outils, config_b), {"config": config_b}),
        suite, repeats=k, on_attempt=suivi)
    print("\n" + rapport.summary())
    familles: dict[str, Any] = {}
    for t in taches:
        for bras in (config_a, config_b):
            r = rapport.reports[bras][t.id]
            f = familles.setdefault(t.famille, {}).setdefault(bras, {"succes": 0, "tentatives": 0, "jetons": 0})
            f["succes"] += r.successes
            f["tentatives"] += len(r.attempts)
            f["jetons"] += sum(a.total_tokens or 0 for a in r.attempts)
    c, lat = rapport.cost, rapport.latency
    return {
        "verdict": rapport.verdict, "taux_a": rapport.rate_a, "taux_b": rapport.rate_b, "p": rapport.p_value,
        "ecart": [rapport.delta, rapport.low, rapport.high], "detectable": rapport.detectable,
        "jetons_par_tentative": None if c is None else [c.tokens_per_attempt_a, c.tokens_per_attempt_b],
        "jetons_variation": None if c is None else c.relative_change,
        "jetons_intervalle": None if c is None else c.interval,
        "jetons_par_succes": None if c is None else [c.tokens_per_success_a, c.tokens_per_success_b],
        "duree_mediane": None if lat is None else [lat.median_seconds_a, lat.median_seconds_b],
        "duree_variation": None if lat is None else lat.relative_change,
        "duree_intervalle": None if lat is None else lat.interval,
        "familles": familles, "resume": rapport.summary(),
        "taches": {x.name: [x.successes_a, x.successes_b, x.n] for x in rapport.tasks},
    }


def resumer(resultats: dict[str, Any]) -> dict[str, Any]:
    def bloc(ids: list[str]) -> dict[str, Any]:
        tentatives = [a for i in ids for a in resultats[i]["attempts"]]
        jetons = [a["total_tokens"] for a in tentatives if a["total_tokens"] is not None]
        succes = sum(1 for a in tentatives if a["ok"])
        secondes = [a["seconds"] for a in tentatives if a["seconds"] is not None]
        return {
            "taches": len(ids), "tentatives": len(tentatives), "succes": succes,
            "taux": succes / len(tentatives) if tentatives else 0.0,
            "jetons_total": sum(jetons), "jetons_mediane": statistics.median(jetons) if jetons else None,
            "jetons_par_succes": sum(jetons) / succes if succes and jetons else None,
            "tentatives_sans_cout": len(tentatives) - len(jetons),
            "secondes_mediane": statistics.median(secondes) if secondes else None,
            "secondes_max": max(secondes) if secondes else None,
            "taches_toujours_reussies": sum(1 for i in ids if resultats[i]["successes"] == len(resultats[i]["attempts"])),
        }

    familles: dict[str, list[str]] = {}
    for ident, r in resultats.items():
        familles.setdefault(r["famille"], []).append(ident)
    return {"familles": {f: bloc(ids) for f, ids in sorted(familles.items())}, "global": bloc(list(resultats))}


def afficher(resume: dict[str, Any], titre: str) -> None:
    print(f"\n{titre}")
    print(f"  {'famille':22} {'réussite':>12} {'tâches 100 %':>13} {'jetons méd.':>12} {'jetons/succès':>14} {'durée méd.':>11}")
    lignes = [*resume["familles"].items(), ("TOTAL", resume["global"])]
    for nom, b in lignes:
        jps = f"{b['jetons_par_succes']:.0f}" if b["jetons_par_succes"] else "—"
        jm = f"{b['jetons_mediane']:.0f}" if b["jetons_mediane"] is not None else "—"
        sm = f"{b['secondes_mediane']:.1f} s" if b["secondes_mediane"] is not None else "—"
        print(f"  {nom:22} {b['succes']:>4}/{b['tentatives']:<3} {b['taux']:>4.0%} {b['taches_toujours_reussies']:>6}/{b['taches']:<6}"
              f" {jm:>12} {jps:>14} {sm:>11}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verifier", action="store_true", help="hors ligne : réponses attendues et audit des juges")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--taches", default="", help="ids ou lettres de familles, séparés par des virgules (ex. A,C3)")
    parser.add_argument("--config", default="telle_quelle", choices=sorted(CONFIGS), help="le réglage mesuré")
    parser.add_argument("--comparer", default="", choices=["", *sorted(CONFIGS)],
                        help="confirmation : compare_configs (bras alternés) entre « telle_quelle » et ce réglage")
    args, reste = parser.parse_known_args()

    monde = construire_monde()
    taches = construire_taches(monde)
    with tempfile.TemporaryDirectory() as dossier:
        base = creer_base(monde, Path(dossier))
        if args.verifier:
            print("Vérification hors ligne du banc :")
            sys.exit(0 if verifier(monde, taches, base) else 1)
        if args.taches:
            filtres = [x.strip().upper() for x in args.taches.split(",") if x.strip()]
            taches = [x for x in taches if x.id in filtres or x.id[0] in filtres]

        from _common import make_provider
        provider = make_provider(reste)
        if args.comparer:
            nom = f"{provider.config.provider}/{provider.config.model} | comparaison telle_quelle vs {args.comparer}"
            print(f"Confirmation — {nom} — {len(taches)} tâches x {args.k} répétitions par bras (bras alternés)", flush=True)
            debut = time.perf_counter()
            resultat = comparer(provider, monde, base, taches, args.k, args.comparer)
            resultat["secondes_total"] = round(time.perf_counter() - debut)
            archive = json.loads(SORTIE.read_text(encoding="utf-8")) if SORTIE.exists() else {}
            archive[nom] = {"date": time.strftime("%Y-%m-%d %H:%M"), "k": args.k, **resultat}
            SORTIE.write_text(json.dumps(archive, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\nRésultats : {SORTIE}")
            return
        nom = f"{provider.config.provider}/{provider.config.model} | {args.config}"
        print(f"Banc d'efficacité — {nom} — {len(taches)} tâches x k={args.k} — {CONFIGS[args.config]['description']}",
              flush=True)
        debut = time.perf_counter()
        resultats = mesurer(provider, monde, base, taches, args.k, args.config)
        resume = resumer(resultats)
        afficher(resume, f"{nom} ({time.perf_counter() - debut:.0f} s au total)")

    archive = json.loads(SORTIE.read_text(encoding="utf-8")) if SORTIE.exists() else {}
    archive[nom] = {"date": time.strftime("%Y-%m-%d %H:%M"), "configuration": CONFIGS[args.config]["description"],
                    "k": args.k, "resume": resume, "taches": resultats}
    SORTIE.write_text(json.dumps(archive, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRésultats détaillés : {SORTIE}")


if __name__ == "__main__":
    main()
