# Un assistant qui gagne des capacités — vitrine autoagent

Un chat. Au départ il ne sait presque rien faire. Puis il **se souvient de toi**,
**fabrique les outils qui lui manquent**, et **publie des pages** quand un tableau
vaut mieux qu'un paragraphe. Ce qu'il acquiert **survit aux conversations**.

> ⚠️ **Démo locale, pas un service exposé.** Il n'y a **aucune authentification** et
> le CORS est ouvert (`allow_origins=["*"]`) pour que les pages générées, qui
> tournent dans un iframe sandboxé, puissent appeler l'API. N'expose pas ce port
> sur un réseau : quiconque l'atteint lit toutes tes conversations et ta mémoire.
> Écoute sur `127.0.0.1`, et rien d'autre.

> ⚠️ C'est un **consommateur** de la lib : il a des dépendances (FastAPI). La lib
> `autoagent`, elle, reste zéro-dépendance, et ce dossier n'entre jamais dans le
> paquet publié.

## Lancer

```bash
pip install -r showcase/requirements.txt
python -m uvicorn showcase.backend.main:app --reload      # → http://127.0.0.1:8000
```

La clé est lue depuis le `.env` de la racine. **Sans clé**, l'app démarre en mode
démo hors-ligne (provider factice) : tout le pipeline reste testable sans réseau.

## Les trois niveaux de capacité

C'est le cœur du produit, et **c'est toi qui fais monter d'un niveau** :

| Niveau | Ce qui se passe | Qui décide |
|---|---|---|
| **1. Bac à sable** | L'agent écrit un outil. Il tourne isolé (processus jetable, sans réseau, FS en lecture seule avec Docker) | l'agent propose, tu **autorises la création** |
| **2. Natif** | Le **même** outil tourne dans le process, avec accès au contexte hôte. Il peut faire tout ce que ton Python peut faire | **toi**, bouton « Valider → natif » |
| **3. Service** | `EvolutionRuntime` : l'agent écrit le code source d'une vraie application, sous garde et réversible | pas encore branché ici |

La validation porte sur l'**empreinte du code**, pas sur le nom : si l'agent
réécrit l'outil, il **retombe** en bac à sable tout seul. Et l'échelle descend —
« Remettre en bac à sable » retire la confiance sans supprimer l'outil.

## Ce qui s'accumule (et ce qui ne s'accumule pas)

| Chemin | Contenu | Survit à la suppression d'une conversation ? |
|---|---|---|
| `data/outils/` | les outils qu'il s'est écrits + le manifeste de validation | **oui** |
| `data/pages/` | les pages qu'il a publiées | **oui** |
| `data/memoire/faits.json` | sa mémoire factuelle (un seul utilisateur) | **oui** |
| `data/sessions/`, `data/workspace/` | l'historique de chat et la trace, par conversation | non |

Supprimer un échange ne doit pas lui faire perdre une capacité. Pour effacer un
souvenir, demande-lui d'oublier ; pour retirer un outil, supprime son fichier.

## La mesure

Le seul chiffre honnête pour « il devient de plus en plus puissant » : la **part
des demandes résolues sans créer de nouvel outil**, affichée en haut à droite.

* elle **monte** → sa bibliothèque couvre de plus en plus de terrain ;
* elle **stagne** → l'accumulation ne sert à rien, et tu le sais.

L'app distingue le cumul depuis toujours et les **20 dernières demandes** : c'est
la seconde qui compte.

## Ce qui est du CODE, pas du prompt

C'était le défaut de la version précédente (un agent HTML piloté par des consignes
qu'on espérait voir respectées). Ici :

* **publier une page est un OUTIL** — `publier_page(titre, html)`. La signature
  *est* le contrat, il n'y a rien à espérer d'une consigne ;
* les pages sont écrites dans un `ProjectWorkspace` : **allowlist `.html`**,
  anti-traversée, journal réversible. L'hôte possède la frontière ;
* les pages s'affichent dans un **iframe sandboxé** (sans `allow-same-origin`) :
  le HTML généré ne peut pas toucher au stockage de la page hôte ;
* invariants verrouillés à la construction : `trifecta_guard="deny"` (rien ne sort
  si du contenu non fiable est entré), `max_tool_result_chars`,
  `max_repeated_tool_calls`, et `enable_tool_search` dès que la bibliothèque
  grossit ;
* l'outil `meteo` est déclaré `untrusted=True` : ce qui vient d'internet est traité
  comme des **données**, jamais comme des instructions ;
* l'outil d'oubli est en **dry run** : il dit ce qu'il effacerait, c'est l'hôte qui
  confirme.

## Structure

```
showcase/
  backend/
    main.py           # FastAPI : chat SSE, approbation, capacités, API vivante
    agent_factory.py  # UN agent : mémoire + outils + pages + invariants
    capacites.py      # inventaire, promotion/rétrogradation, mesure d'autonomie
    sessions.py       # persistance des conversations
    paths.py          # chemins (source de vérité) : ce qui est partagé vs par conversation
  frontend/
    index.html        # une page : chat · écran de l'agent · capacités + conversations
  data/               # runtime (gitignoré)
```

## Limite connue

**Aucun test automatisé** pour cette app (la lib, elle, en a 755). Les
comportements clés ont été vérifiés à la main contre un vrai modèle — mémoire qui
traverse deux conversations, outil forgé qui atterrit dans le dossier partagé,
passage bac à sable → natif rechargé par une conversation neuve — mais rien ne
protège encore contre une régression.
