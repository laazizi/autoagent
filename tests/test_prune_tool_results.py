"""`prune_tool_results_after` — borner la DURÉE DE VIE d'un résultat d'outil (0.19.0).

`max_tool_result_chars` borne la LARGEUR d'un résultat : ce qui entre une fois
dans le transcript. Rien ne bornait sa DURÉE — un résultat de 40 000 caractères
utile à l'étape 3 repart identique à chaque étape jusqu'à tomber de la queue de
la mémoire, et il n'est jamais dans le préfixe mis en cache par le fournisseur
puisque l'historique change à chaque tour.

Ces tests verrouillent les trois invariants qui font toute la difficulté :

  1. La CONVERSATION RESTE BIEN FORMÉE — un message d'outil élagué garde son
     rôle et son `tool_call_id`, sinon les fournisseurs rejettent la requête.
  2. LA TEINTE SURVIT — élaguer un résultat untrusted sans reconduire son cadre
     « laverait » la teinte et désarmerait la garde trifecta. C'est le trou de
     la 0.15, réouvert par la porte de service.
  3. LE REGISTRE N'EST PAS TOUCHÉ — on élague la VUE envoyée au fournisseur,
     pas le transcript rendu à l'hôte. Économiser des jetons en perdant des
     preuves serait un mauvais échange.
"""

from __future__ import annotations

from .conftest import FakeLLMProvider

from autoagent import Agent
from autoagent.agent import _prune_tool_results
from autoagent.schema import (
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    LLMResponse,
    Message,
    ToolCall,
    is_tainted,
)

GROS = "x" * 5000


def _resultat(i: int, contenu: str = GROS, nom: str = "lire") -> Message:
    return Message(role="tool", name=nom, tool_call_id=f"c{i}", content=contenu)


def _transcript(n: int) -> list[Message]:
    """n tours (assistant qui appelle, outil qui répond) après une question."""
    messages: list[Message] = [Message(role="user", content="vas-y")]
    for i in range(n):
        messages.append(Message(
            role="assistant", content="",
            tool_calls=[ToolCall(id=f"c{i}", name="lire", arguments={"n": i})]))
        messages.append(_resultat(i))
    return messages


class TestCeQuiEstElague:
    def test_les_n_derniers_sont_intacts(self) -> None:
        vue, elagues, _ = _prune_tool_results(_transcript(5), keep=2)
        outils = [m for m in vue if m.role == "tool"]
        assert elagues == 3
        assert [m.content for m in outils[-2:]] == [GROS, GROS]
        assert all("PRUNED" in m.content for m in outils[:3])

    def test_sous_le_seuil_rien_ne_bouge(self) -> None:
        transcript = _transcript(2)
        vue, elagues, economie = _prune_tool_results(transcript, keep=3)
        assert (elagues, economie) == (0, 0)
        assert vue is transcript, "aucune copie inutile quand il n'y a rien à faire"

    def test_keep_zero_elague_tout(self) -> None:
        vue, elagues, _ = _prune_tool_results(_transcript(3), keep=0)
        assert elagues == 3
        assert all("PRUNED" in m.content for m in vue if m.role == "tool")

    def test_le_marqueur_nomme_l_outil_et_la_taille(self) -> None:
        """Un modèle à qui on dit « supprimé » sans plus replanifie autour d'un
        échec qui n'a pas eu lieu. Le marqueur dit donc QUOI et COMBIEN, et que
        le résultat était VALIDE."""
        vue, _, _ = _prune_tool_results(_transcript(2), keep=1)
        marqueur = next(m for m in vue if m.role == "tool").content
        assert "`lire`" in marqueur
        assert str(len(GROS)) in marqueur
        assert "VALID" in marqueur


class TestBienFormee:
    def test_role_id_et_nombre_de_messages_preserves(self) -> None:
        transcript = _transcript(4)
        vue, _, _ = _prune_tool_results(transcript, keep=1)
        assert len(vue) == len(transcript)
        for avant, apres in zip(transcript, vue, strict=True):
            assert (apres.role, apres.tool_call_id, apres.name) == (
                avant.role, avant.tool_call_id, avant.name)

    def test_aucun_message_assistant_touche(self) -> None:
        transcript = _transcript(3)
        vue, _, _ = _prune_tool_results(transcript, keep=0)
        for avant, apres in zip(transcript, vue, strict=True):
            if avant.role != "tool":
                assert apres is avant


class TestTeinte:
    def _teinte(self) -> list[Message]:
        sale = f"{UNTRUSTED_OPEN}\n{GROS}\n{UNTRUSTED_CLOSE}"
        return [
            Message(role="user", content="vas-y"),
            Message(role="assistant", content="",
                    tool_calls=[ToolCall(id="c0", name="fetch", arguments={})]),
            _resultat(0, sale, nom="fetch"),
            Message(role="assistant", content="",
                    tool_calls=[ToolCall(id="c1", name="lire", arguments={})]),
            _resultat(1),
        ]

    def test_un_resultat_untrusted_elague_reste_teinte(self) -> None:
        transcript = self._teinte()
        assert is_tainted(transcript)
        vue, elagues, _ = _prune_tool_results(transcript, keep=1)
        assert elagues == 1
        assert is_tainted(vue), "élaguer a lavé la teinte — trifecta désarmée"

    def test_le_cadre_untrusted_est_reconduit(self) -> None:
        vue, _, _ = _prune_tool_results(self._teinte(), keep=1)
        elague = next(m for m in vue if m.role == "tool").content
        assert elague.startswith(UNTRUSTED_OPEN)
        assert elague.rstrip().endswith(UNTRUSTED_CLOSE)
        assert "PRUNED" in elague


class TestNeJamaisGrossir:
    def test_un_resultat_court_est_laisse_tel_quel(self) -> None:
        """Le marqueur fait ~200 caractères : l'appliquer à un résultat de 12
        AJOUTERAIT du contexte. Une borne qui coûte n'est pas une borne."""
        transcript = [
            Message(role="user", content="vas-y"),
            _resultat(0, "ok: 42"),
            _resultat(1, "ok: 43"),
        ]
        vue, elagues, economie = _prune_tool_results(transcript, keep=0)
        assert (elagues, economie) == (0, 0)
        assert [m.content for m in vue if m.role == "tool"] == ["ok: 42", "ok: 43"]

    def test_economie_rapportee_exacte(self) -> None:
        vue, elagues, economie = _prune_tool_results(_transcript(3), keep=1)
        elagues_contenus = [m.content for m in vue if m.role == "tool"][:2]
        attendu = sum(len(GROS) - len(c) for c in elagues_contenus)
        assert economie == attendu > 0

    def test_idempotent(self) -> None:
        """Élaguer deux fois ne doit pas ré-emballer un marqueur dans un autre."""
        une, _, _ = _prune_tool_results(_transcript(3), keep=1)
        deux, elagues, economie = _prune_tool_results(une, keep=1)
        assert (elagues, economie) == (0, 0)
        assert [m.content for m in deux] == [m.content for m in une]


class TestLeRegistreEstIntact:
    def test_la_liste_d_entree_n_est_pas_mutee(self) -> None:
        transcript = _transcript(3)
        avant = [m.content for m in transcript]
        _prune_tool_results(transcript, keep=1)
        assert [m.content for m in transcript] == avant


class TestBoutEnBout:
    def _agent(self, **kwargs) -> tuple[Agent, FakeLLMProvider]:
        reponses = [
            LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name="lire",
                                             arguments={"n": i})])
            for i in range(3)
        ] + ["fini"]
        provider = FakeLLMProvider(reponses)
        agent = Agent(provider, max_steps=6, **kwargs)

        @agent.tool
        def lire(n: int) -> str:
            """Lit un gros journal."""
            return GROS

        return agent, provider

    def test_le_fournisseur_voit_la_vue_elaguee(self) -> None:
        agent, provider = self._agent(prune_tool_results_after=1)
        agent.run("vas-y")
        derniere = provider.calls[-1].messages
        outils = [m for m in derniere if m.role == "tool"]
        assert len(outils) == 3
        assert sum("PRUNED" in m.content for m in outils) == 2
        assert GROS in outils[-1].content, "le plus récent garde sa charge"

    def test_le_transcript_rendu_garde_tout(self) -> None:
        agent, _ = self._agent(prune_tool_results_after=1)
        resultat = agent.run("vas-y")
        outils = [m for m in resultat.messages if m.role == "tool"]
        assert all(GROS in m.content for m in outils), (
            "l'élagage a mangé le registre, pas seulement la vue")

    def test_eteint_par_defaut(self) -> None:
        """Rétrocompatibilité : sans l'option, rien n'est élagué."""
        agent, provider = self._agent()
        agent.run("vas-y")
        outils = [m for m in provider.calls[-1].messages if m.role == "tool"]
        assert all(GROS in m.content for m in outils)

    def test_l_economie_est_reelle(self) -> None:
        """Ce qui compte n'est pas le nombre de messages mais les caractères
        envoyés : c'est ça qui se facture à chaque tour."""
        agent_sans, provider_sans = self._agent()
        agent_avec, provider_avec = self._agent(prune_tool_results_after=1)
        agent_sans.run("vas-y")
        agent_avec.run("vas-y")

        def taille(calls) -> int:
            return sum(len(m.content or "") for m in calls[-1].messages)

        assert taille(provider_avec.calls) < taille(provider_sans.calls)
