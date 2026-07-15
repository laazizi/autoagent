# CLAUDE.md

Lis **AGENTS.md** (à la racine) : c'est le fichier de contexte maître de ce
dépôt — règles dures (zéro dépendance, contrats fail-open/fail-closed,
synchronisation des docs), carte des modules, process de release. Il fait foi.

Rappels spécifiques :
- Le CODE fait foi, pas la doc. `pytest tests -q` avant de conclure.
- Jamais de trailer `Co-Authored-By` dans les commits.
- `.context/` est personnel et gitignoré — ne jamais le committer.
