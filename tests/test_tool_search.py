"""`enable_tool_search` — divulgation progressive des schémas d'outils (0.18.0).

Le problème : la lib envoie le schéma COMPLET de chaque outil à CHAQUE étape de
CHAQUE run. Deux serveurs MCP montés et ce préfixe domine la requête ; et un
modèle à qui l'on présente 100 outils choisit moins bien qu'avec 6. L'industrie a
convergé en 2026 sur le même remède (tool-search d'Anthropic, code-execution avec
MCP, code mode de Cloudflare) : ne plus expédier les schémas que le modèle n'a pas
demandés.

Invariant de sécurité vérifié ici : la visibilité borne ce qu'on PROPOSE au
modèle, jamais ce que l'hôte peut gouverner (`tool_policy`, taint, exécution
voient toujours le registre complet).
"""

from __future__ import annotations

import json

from .conftest import FakeLLMProvider

from autoagent import Agent
from autoagent.agent import _FIND_TOOLS_NAME, _score_tools
from autoagent.schema import LLMResponse, ToolCall, ToolSpec


# Un jeu d'outils RÉALISTE : des noms/descriptions distinctifs (comme un vrai
# registre ou un serveur MCP), pas des clones qui ne diffèrent que par un numéro
# — sinon on ne teste que le départage d'égalités.
_REALISTES = [
    ("lire_fichier", "Lit le contenu d'un fichier texte sur le disque."),
    ("ecrire_fichier", "Écrit du contenu dans un fichier du dossier de travail."),
    ("envoyer_email", "Envoie un courriel à un destinataire donné."),
    ("chercher_web", "Effectue une recherche sur internet et renvoie des liens."),
    ("requete_sql", "Exécute une requête SELECT sur la base de données."),
    ("convertir_devise", "Convertit un montant d'une monnaie vers une autre."),
    ("redimensionner_image", "Change la taille d'une image en pixels."),
    ("planifier_reunion", "Réserve un créneau dans l'agenda partagé."),
]


def _agent_avec_n_outils(n: int, responses=None, **kwargs) -> tuple[Agent, FakeLLMProvider]:
    """Enregistre les outils réalistes puis complète avec du remplissage jusqu'à n."""
    provider = FakeLLMProvider(responses or ["fini"])
    agent = Agent(provider, max_steps=6, **kwargs)
    for nom, desc in _REALISTES[: min(n, len(_REALISTES))]:
        def _h(**kw) -> dict:
            return {"ok": True}
        agent.tool(_h, name=nom, description=desc)
    for i in range(max(0, n - len(_REALISTES))):
        def _f(i=i, **kw) -> dict:
            return {"i": i}
        agent.tool(_f, name=f"remplissage_{i:03d}",
                   description=f"Outil de remplissage {i} sans rapport.")
    return agent, provider


def _outils_envoyes(provider: FakeLLMProvider, appel: int = 0) -> set[str]:
    return {s.name for s in provider.calls[appel].tools}


class TestScoringLexical:
    def _specs(self):
        return [
            ToolSpec(name="lire_fichier", description="Lit un fichier texte du disque."),
            ToolSpec(name="envoyer_email", description="Envoie un courriel à un destinataire."),
            ToolSpec(name="compter_lignes", description="Compte les lignes d'un texte."),
        ]

    def test_correspondance_sur_la_description(self) -> None:
        res = _score_tools("je veux envoyer un courriel", self._specs())
        assert res and res[0].name == "envoyer_email"

    def test_le_nom_pese_plus_que_la_description(self) -> None:
        res = _score_tools("fichier", self._specs())
        assert res[0].name == "lire_fichier"

    def test_nom_explicite_gagne_toujours(self) -> None:
        res = _score_tools("utilise compter_lignes maintenant", self._specs())
        assert res[0].name == "compter_lignes"

    def test_aucune_correspondance(self) -> None:
        assert _score_tools("quantique astrophysique", self._specs()) == []

    def test_requete_vide(self) -> None:
        assert _score_tools("", self._specs()) == []


class TestSeuil:
    def test_sous_le_seuil_rien_ne_change(self) -> None:
        """Aucun méta-outil, aucun filtrage : mêmes octets sur le fil qu'avant."""
        agent, provider = _agent_avec_n_outils(5)
        agent.enable_tool_search(threshold=15)
        agent.run("vas-y")
        envoyes = _outils_envoyes(provider)
        assert len(envoyes) == 5
        assert _FIND_TOOLS_NAME not in envoyes

    def test_au_dessus_du_seuil_seul_le_meta_outil_est_propose(self) -> None:
        agent, provider = _agent_avec_n_outils(40)
        agent.enable_tool_search(threshold=15)
        agent.run("vas-y")
        assert _outils_envoyes(provider) == {_FIND_TOOLS_NAME}

    def test_desactive_par_defaut(self) -> None:
        """Rétrocompatibilité : sans appel à enable_tool_search, tout est envoyé."""
        agent, provider = _agent_avec_n_outils(40)
        agent.run("vas-y")
        assert len(_outils_envoyes(provider)) == 40


class TestRevelation:
    def test_chercher_puis_appeler(self) -> None:
        """Le cycle complet : find_tools révèle, l'étape suivante peut appeler."""
        agent, provider = _agent_avec_n_outils(40, responses=[
            LLMResponse(tool_calls=[ToolCall(id="c1", name=_FIND_TOOLS_NAME,
                                             arguments={"query": "envoyer un courriel"})]),
            LLMResponse(tool_calls=[ToolCall(id="c2", name="envoyer_email", arguments={})]),
            "fini",
        ])
        agent.enable_tool_search(threshold=15, max_results=3)
        res = agent.run("trouve le bon outil")

        # 1er appel : rien que le méta-outil. 2e : l'outil trouvé est devenu visible.
        assert _outils_envoyes(provider, 0) == {_FIND_TOOLS_NAME}
        assert "envoyer_email" in _outils_envoyes(provider, 1)
        # et il a bien été exécuté
        assert any(m.role == "tool" and m.name == "envoyer_email" for m in res.messages)

    def test_resultat_de_recherche_liste_noms_et_descriptions(self) -> None:
        agent, provider = _agent_avec_n_outils(40, responses=[
            LLMResponse(tool_calls=[ToolCall(id="c1", name=_FIND_TOOLS_NAME,
                                             arguments={"query": "requête sur la base"})]),
            "fini",
        ])
        agent.enable_tool_search(threshold=15, max_results=2)
        res = agent.run("cherche")
        msg = next(m for m in res.messages if m.role == "tool")
        charge = json.loads(msg.content)["result"]
        assert charge["matches"] and "name" in charge["matches"][0]
        assert "description" in charge["matches"][0]

    def test_sans_correspondance_le_catalogue_est_rendu(self) -> None:
        """Le modèle ne doit JAMAIS être coincé : à défaut, on donne les noms."""
        agent, provider = _agent_avec_n_outils(40, responses=[
            LLMResponse(tool_calls=[ToolCall(id="c1", name=_FIND_TOOLS_NAME,
                                             arguments={"query": "astrophysique quantique"})]),
            "fini",
        ])
        agent.enable_tool_search(threshold=15)
        res = agent.run("cherche")
        charge = json.loads(next(m for m in res.messages if m.role == "tool").content)["result"]
        assert charge["matches"] == []
        assert len(charge["available"]) == 40

    def test_always_reste_visible_sans_recherche(self) -> None:
        agent, provider = _agent_avec_n_outils(40)
        agent.enable_tool_search(threshold=15, always=("lire_fichier", "requete_sql"))
        agent.run("vas-y")
        envoyes = _outils_envoyes(provider)
        assert {"lire_fichier", "requete_sql", _FIND_TOOLS_NAME} == envoyes

    def test_reprise_retrouve_les_outils_deja_charges(self) -> None:
        """Après une pause/reprise, un outil déjà appelé reste visible (dérivé du
        transcript, donc aucun champ ajouté à RunState)."""
        agent, provider = _agent_avec_n_outils(40, responses=["fini"])
        agent.enable_tool_search(threshold=15)
        from autoagent.schema import Message
        historique = [
            Message(role="user", content="fais-le"),
            Message(role="assistant", content="",
                    tool_calls=[ToolCall(id="c9", name="chercher_web", arguments={})]),
            Message(role="tool", tool_call_id="c9", name="chercher_web", content='{"ok": true}'),
        ]
        agent.run_messages(historique)
        assert "chercher_web" in _outils_envoyes(provider)


class TestInvariantDeGouvernance:
    def test_la_politique_voit_le_registre_complet(self) -> None:
        """La visibilité borne l'OFFRE au modèle, pas le pouvoir de l'hôte."""
        vus: list[str] = []

        def politique(ctx) -> None:
            vus.append(ctx.call.name)
            assert ctx.spec is not None, "tool_policy doit voir le spec même non révélé"
            return None

        agent, provider = _agent_avec_n_outils(40, responses=[
            LLMResponse(tool_calls=[ToolCall(id="c1", name="planifier_reunion", arguments={})]),
            "fini",
        ], tool_policy=politique)
        agent.enable_tool_search(threshold=15)
        agent.run("vas-y")
        assert vus == ["planifier_reunion"]

    def test_un_outil_non_revele_reste_executable(self) -> None:
        """Si le modèle devine un nom, l'exécution marche : on ne casse rien."""
        agent, provider = _agent_avec_n_outils(40, responses=[
            LLMResponse(tool_calls=[ToolCall(id="c1", name="convertir_devise", arguments={})]),
            "fini",
        ])
        agent.enable_tool_search(threshold=15)
        res = agent.run("vas-y")
        msg = next(m for m in res.messages if m.role == "tool")
        assert '"ok": true' in msg.content
