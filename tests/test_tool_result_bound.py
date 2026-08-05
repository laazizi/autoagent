"""`max_tool_result_chars` — la borne CODE sur ce qu'un résultat d'outil injecte.

Un seul outil non borné (fetch HTTP, SELECT large, lecture de fichier) suffisait à
faire exploser la fenêtre de contexte, à brûler le `token_budget` et à noyer
l'attention du modèle. Ici la borne est du code testable, pas une consigne de
prompt : le résultat est tronqué PAR LE MILIEU (tête + queue conservées) avec un
marqueur explicite, et le total ne dépasse JAMAIS la borne.
"""

from __future__ import annotations

from .conftest import FakeLLMProvider

from autoagent import Agent
from autoagent.agent import _truncate_tool_result
from autoagent.schema import LLMResponse, ToolCall


def _agent(responses, **kwargs) -> tuple[Agent, FakeLLMProvider]:
    provider = FakeLLMProvider(responses)
    return Agent(provider, max_steps=4, **kwargs), provider


def _tool_msgs(result):
    return [m for m in result.messages if m.role == "tool"]


class TestTroncatureUnitaire:
    def test_sous_la_borne_intact(self) -> None:
        contenu, tronque = _truncate_tool_result("court", 100)
        assert contenu == "court"
        assert tronque is False

    def test_borne_jamais_depassee(self) -> None:
        # Le marqueur compte dans le budget : une borne dépassable n'est pas une borne.
        for limite in (40, 60, 120, 500, 5000):
            contenu, tronque = _truncate_tool_result("x" * 20000, limite)
            assert len(contenu) <= limite, f"borne {limite} dépassée : {len(contenu)}"
            assert tronque is True

    def test_tete_et_queue_conservees(self) -> None:
        source = "DEBUT" + ("m" * 5000) + "FIN"
        contenu, _ = _truncate_tool_result(source, 400)
        assert contenu.startswith("DEBUT")      # la forme du payload
        assert contenu.endswith("FIN")          # les totaux / le curseur de page
        assert "TRUNCATED" in contenu           # le modèle SAIT qu'il manque des données

    def test_marqueur_annonce_la_taille_reelle(self) -> None:
        contenu, _ = _truncate_tool_result("z" * 10000, 300)
        assert "10000 characters" in contenu

    def test_budget_plus_petit_que_le_marqueur_coupe_net(self) -> None:
        contenu, tronque = _truncate_tool_result("y" * 900, 12)
        assert len(contenu) == 12 and tronque is True


class TestDansLaBoucle:
    def test_defaut_inchange_aucune_troncature(self) -> None:
        """Rétrocompatibilité : sans l'option, le résultat passe verbatim."""
        gros = "A" * 30000
        agent, _ = _agent([
            LLMResponse(tool_calls=[ToolCall(id="c1", name="lire", arguments={})]),
            "fini",
        ])

        @agent.tool
        def lire() -> str:
            """Renvoie un gros payload."""
            return gros

        res = agent.run("vas-y")
        assert gros in _tool_msgs(res)[0].content

    def test_borne_appliquee_dans_le_transcript(self) -> None:
        agent, _ = _agent([
            LLMResponse(tool_calls=[ToolCall(id="c1", name="lire", arguments={})]),
            "fini",
        ], max_tool_result_chars=500)

        @agent.tool
        def lire() -> str:
            """Renvoie un gros payload."""
            return "B" * 40000

        res = agent.run("vas-y")
        contenu = _tool_msgs(res)[0].content
        assert len(contenu) <= 500
        assert "TRUNCATED" in contenu

    def test_chemin_parallele_aussi_borne(self) -> None:
        """Les deux chemins (séquentiel ET ThreadPoolExecutor) passent par la borne."""
        agent, _ = _agent([
            LLMResponse(tool_calls=[
                ToolCall(id="c1", name="lire", arguments={}),
                ToolCall(id="c2", name="lire", arguments={}),
            ]),
            "fini",
        ], max_tool_result_chars=300, parallel_tool_calls=True)

        @agent.tool
        def lire() -> str:
            """Renvoie un gros payload."""
            return "C" * 20000

        res = agent.run("vas-y")
        msgs = _tool_msgs(res)
        assert len(msgs) == 2
        assert all(len(m.content) <= 300 for m in msgs)

    def test_marqueurs_untrusted_jamais_coupes(self) -> None:
        """La borne s'applique au RÉSULTAT ; le cadre de sécurité reste entier."""
        from autoagent.schema import UNTRUSTED_CLOSE, UNTRUSTED_OPEN

        agent, _ = _agent([
            LLMResponse(tool_calls=[ToolCall(id="c1", name="fetch", arguments={})]),
            "fini",
        ], max_tool_result_chars=200)

        @agent.tool(untrusted=True)
        def fetch() -> str:
            """Contenu externe non fiable."""
            return "D" * 9000

        res = agent.run("vas-y")
        contenu = _tool_msgs(res)[0].content
        assert contenu.startswith(UNTRUSTED_OPEN)
        assert contenu.endswith(UNTRUSTED_CLOSE)

    def test_erreur_d_outil_reste_lisible(self) -> None:
        """Un message d'erreur court n'est jamais amputé (il doit rester actionnable)."""
        agent, _ = _agent([
            LLMResponse(tool_calls=[ToolCall(id="c1", name="casse", arguments={})]),
            "fini",
        ], max_tool_result_chars=400)

        @agent.tool
        def casse() -> str:
            """Lève une erreur."""
            raise ValueError("chemin introuvable : /data/x.csv")

        res = agent.run("vas-y")
        assert "chemin introuvable" in _tool_msgs(res)[0].content
