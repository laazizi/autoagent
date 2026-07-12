"""14 — Base SQL comme source : l'agent répond en INTERROGEANT une base.

La base de données est la SOURCE DE VÉRITÉ — l'agent ne devine rien. Il
inspecte le schéma (outil `schema`), écrit une requête, la lib l'exécute
(outil `requete_sql`, en LECTURE SEULE : toute écriture est refusée PAR LE
CODE, pas par le prompt), puis il répond à partir des lignes réelles.

Zéro dépendance : SQLite (stdlib), base en mémoire seedée au démarrage.

    python examples_autoagent/14_base_sql.py
"""

from __future__ import annotations

import sqlite3

from _common import make_provider

from autoagent import Agent


def _base() -> sqlite3.Connection:
    """Une petite base « comptage routier » en mémoire (la source)."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE capteurs  (id INTEGER PRIMARY KEY, ville TEXT, statut TEXT);
        CREATE TABLE comptages (capteur_id INTEGER, jour TEXT, vehicules INTEGER);
        INSERT INTO capteurs VALUES
          (1,'Lyon','actif'), (2,'Lyon','panne'), (3,'Valence','actif'),
          (4,'Grenoble','actif'), (5,'Grenoble','panne');
        INSERT INTO comptages VALUES
          (1,'2026-07-01',1240), (1,'2026-07-02',1310),
          (3,'2026-07-01', 860), (3,'2026-07-02', 910),
          (4,'2026-07-01',2100), (4,'2026-07-02',1980);
    """)
    return conn


def main() -> None:
    conn = _base()
    agent = Agent(make_provider(), max_steps=6,
                  system_prompt=("Tu réponds UNIQUEMENT à partir de la base SQL. "
                                 "Commence par regarder le schéma, puis écris une "
                                 "requête SELECT. Ne fabrique jamais de chiffres."))

    @agent.tool
    def schema() -> dict:
        """Liste les tables et leurs colonnes (à consulter avant d'écrire une requête)."""
        out = {}
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            out[table] = [f"{c[1]} {c[2]}" for c in conn.execute(f"PRAGMA table_info({table})")]
        return {"tables": out}

    @agent.tool
    def requete_sql(sql: str) -> dict:
        """Exécute une requête SQL en LECTURE SEULE et renvoie les lignes.
        Seul un unique SELECT est autorisé — tout le reste est refusé."""
        clean = sql.strip().rstrip(";").strip()
        if ";" in clean:
            return {"erreur": "une seule requête à la fois (pas de ';' multiples)"}
        if not clean.lower().startswith(("select", "with")):
            return {"erreur": "lecture seule : seules les requêtes SELECT sont autorisées"}
        try:
            cur = conn.execute(clean)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchmany(50)]
            return {"colonnes": cols, "lignes": rows, "n": len(rows)}
        except Exception as exc:  # noqa: BLE001 — l'agent voit l'erreur et corrige sa requête
            return {"erreur": f"{type(exc).__name__}: {exc}"}

    question = ("Quelle ville totalise le plus de véhicules comptés, et combien de "
                "capteurs sont en panne au total ?")
    print(f"Question : {question}\n")
    result = agent.run(question)
    print(result.output)

    # Preuve que l'écriture est bloquée PAR LE CODE (pas par la bonne volonté du LLM) :
    essai = requete_sql("DELETE FROM capteurs")
    print(f"\n[garde-fou] requete_sql('DELETE …') → {essai}")
    if result.usage:
        print(f"[{result.steps} tours | {result.usage.total_tokens} tokens | source : SQLite en mémoire]")


if __name__ == "__main__":
    main()
