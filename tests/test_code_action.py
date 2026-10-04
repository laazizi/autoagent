"""`Agent.enable_code_action()` (0.25.0) : le code comme action en UNE ligne.

Le modèle écrit un programme qui appelle les outils de l'agent par `context["call_host"]` ; seul ce que le
programme rend revient dans la conversation. Ce qui est garanti ici, et que le recâblage à la main
(`enable_run_python(PythonRunner(host_functions=...))`) ne garantissait pas :

* chaque outil passe par le REGISTRE : validation des arguments, contexte de l'hôte, et le même `ToolSpec`
  pour la porte — un outil `egress` reste `egress`, un outil `untrusted` teinte le run ;
* les outils sont résolus à chaque programme (un outil enregistré après est disponible) ;
* jamais exposés : `run_python` lui-même, les sous-agents (leur dépense échapperait à `token_budget`) ;
* la description de l'outil est courte (elle ne relit pas la liste des outils, déjà dans la requête).
"""

from __future__ import annotations

from typing import Any

import pytest

from autoagent import Agent, PythonRunner
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, ToolCall


class _Scripte(LLMProvider):
    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.reponses = list(reponses)
        self.requetes: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requetes.append(request)
        return self.reponses.pop(0) if self.reponses else LLMResponse(content="fini")


def _programme(code: str, **args: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id="c-prog", name="run_python", arguments={"code": code, "args": args})])


def _resultats_outils(resultat: Any) -> str:
    return "\n".join(m.content or "" for m in resultat.messages if m.role == "tool")


JOURNAL = "\n".join(["INFO ok"] * 300 + ["ERROR timeout"] * 7) + "\n"
COMPTER = ("def run(args, context):\n"
           "    texte = context['call_host']('lire_journal', {'capteur_id': args['capteur']})\n"
           "    return {'erreurs': sum(1 for l in texte.splitlines() if 'ERROR' in l)}\n")


def _agent(*reponses: LLMResponse, **kw: Any) -> Agent:
    agent = Agent(_Scripte([*reponses, LLMResponse(content="fini")]), max_steps=6, **kw)

    @agent.tool
    def lire_journal(capteur_id: str) -> str:
        """Le journal complet d'un capteur."""
        return JOURNAL

    return agent


class TestUneLigne:
    def test_le_programme_appelle_l_outil_et_seul_le_compte_revient(self) -> None:
        agent = _agent(_programme(COMPTER, capteur="CMP-1"))
        agent.enable_code_action()
        resultat = agent.run("combien d'erreurs ?")
        rendu = _resultats_outils(resultat)
        assert '"erreurs": 7' in rendu
        assert "INFO ok" not in rendu                       # le journal est resté dans le programme

    def test_un_outil_enregistre_apres_est_disponible(self) -> None:
        code = "def run(args, context):\n    return context['call_host']('double', {'n': 21})\n"
        agent = _agent(_programme(code))
        agent.enable_code_action()

        @agent.tool
        def double(n: int) -> int:
            """Le double de n."""
            return 2 * n

        assert '"result": 42' in _resultats_outils(agent.run("go"))

    def test_les_arguments_sont_valides_comme_pour_un_appel_direct(self) -> None:
        executes: list[Any] = []
        code = "def run(args, context):\n    return context['call_host']('compter', {'n': 'pas un nombre'})\n"
        agent = _agent(_programme(code))

        @agent.tool
        def compter(n: int) -> int:
            """Compte jusqu'à n."""
            executes.append(n)
            return n

        agent.enable_code_action()
        rendu = _resultats_outils(agent.run("go"))
        assert executes == []                               # l'outil n'a PAS tourné
        assert "integer" in rendu                           # l'erreur de validation est rendue au modèle

    def test_le_contexte_de_l_hote_arrive_a_l_outil(self) -> None:
        vus: list[Any] = []
        code = "def run(args, context):\n    return context['call_host']('qui', {})\n"
        agent = _agent(_programme(code))

        @agent.tool
        def qui(context: dict) -> str:
            """L'utilisateur courant."""
            vus.append(context.get("utilisateur"))
            return "ok"

        agent.enable_code_action()
        agent.run("go", context={"utilisateur": "u-42"})
        assert vus == ["u-42"]


class TestLaPorteVoitLesMemesDrapeaux:
    """Le recâblage à la main perdait `egress` : la porte lit le drapeau sur la FONCTION, `agent.tool(f,
    egress=True)` le garde dans le registre. Par `enable_code_action`, le drapeau suit."""

    LIRE_PUIS_ENVOYER = ("def run(args, context):\n"
                         "    page = context['call_host']('lire_page', {'url': 'https://exemple.test'})\n"
                         "    return context['call_host']('envoyer', {'texte': page})\n")

    def _agent_trifecta(self, **kw: Any) -> tuple[Agent, list[str]]:
        envois: list[str] = []
        agent = _agent(_programme(self.LIRE_PUIS_ENVOYER), **kw)

        def lire_page(url: str) -> str:
            """Lit une page web."""
            return "contenu externe"

        def envoyer(texte: str) -> str:
            """Envoie un message à l'extérieur."""
            envois.append(texte)
            return "envoyé"

        agent.tool(lire_page, untrusted=True)
        agent.tool(envoyer, egress=True)
        return agent, envois

    def test_un_envoi_apres_une_lecture_non_fiable_est_refuse(self) -> None:
        agent, envois = self._agent_trifecta()
        agent.enable_code_action()
        rendu = _resultats_outils(agent.run("go"))
        assert envois == []                                 # l'effet n'a PAS eu lieu
        assert "EgressBlocked" in rendu

    def test_le_recablage_a_la_main_perdait_le_drapeau(self) -> None:
        """Le défaut que ce chemin corrige, reproduit : mêmes outils, passés à la main."""
        agent, envois = self._agent_trifecta()
        fonctions = {nom: agent.registry.handler_for(nom) for nom in ("lire_page", "envoyer")}
        agent.enable_run_python(PythonRunner(host_functions=fonctions))
        agent.run("go")
        assert envois == ["contenu externe"]                # parti : la porte ne voyait pas `egress`

    def test_la_politique_voit_la_source_et_le_spec_du_registre(self) -> None:
        vus: list[Any] = []

        def politique(ctx: Any) -> str | None:
            vus.append(ctx)
            return None

        code = "def run(args, context):\n    return context['call_host']('lire_journal', {'capteur_id': 'X'})\n"
        agent = _agent(_programme(code), tool_policy=politique)
        agent.enable_code_action()
        agent.run("go")
        par_le_code = [c for c in vus if c.source == "host_function"]
        assert [c.call.name for c in par_le_code] == ["lire_journal"]
        assert par_le_code[0].spec is not None and par_le_code[0].spec.description == "Le journal complet d'un capteur."

    def test_une_politique_qui_refuse_bloque_aussi_le_chemin_du_code(self) -> None:
        def politique(ctx: Any) -> str | None:
            return "lecture interdite" if ctx.call.name == "lire_journal" else None

        agent = _agent(_programme(COMPTER, capteur="CMP-1"), tool_policy=politique)
        agent.enable_code_action()
        assert "ToolPolicyDenied: lecture interdite" in _resultats_outils(agent.run("go"))


class TestCeQuiNEstPasExpose:
    def test_run_python_ne_s_appelle_pas_lui_meme(self) -> None:
        code = "def run(args, context):\n    return context['call_host']('run_python', {'code': 'x'})\n"
        agent = _agent(_programme(code))
        agent.enable_code_action()
        assert "host function not allowed: run_python" in _resultats_outils(agent.run("go"))

    def test_un_sous_agent_n_est_pas_expose(self) -> None:
        specialiste = Agent(_Scripte([LLMResponse(content="avis")]))
        code = "def run(args, context):\n    return context['call_host']('expert', {'task': 'x'})\n"
        agent = _agent(_programme(code))
        agent.add_tool(specialiste.as_tool(name="expert", description="Un expert."))
        assert "expert" in agent.registry                   # il EST un outil de l'agent…
        assert "expert" not in agent._outils_pour_le_code(None)    # …mais pas une fonction pour le code
        agent.enable_code_action()
        assert "host function not allowed: expert" in _resultats_outils(agent.run("go"))

    def test_tools_restreint_la_liste(self) -> None:
        code = "def run(args, context):\n    return context['call_host']('autre', {})\n"
        agent = _agent(_programme(code))

        @agent.tool
        def autre() -> str:
            """Un autre outil."""
            return "non"

        agent.enable_code_action(tools=["lire_journal"])
        assert "host function not allowed: autre" in _resultats_outils(agent.run("go"))


class TestLaDescription:
    def test_elle_est_courte_et_ne_reliste_pas_les_outils(self) -> None:
        agent = _agent()

        @agent.tool
        def outil_avec_une_longue_docstring(x: str) -> str:
            """Une docstring qui n'a rien à faire dans la description de run_python."""
            return x

        agent.enable_code_action()
        spec = next(s for s in agent.registry.specs() if s.name == "run_python")
        assert "call_host" in spec.description and "LARGE tool results" in spec.description
        assert "outil_avec_une_longue_docstring" not in spec.description
        assert "lire_journal" not in spec.description
        assert len(spec.description) < 520

    def test_le_budget_de_programmes_par_run_tient(self) -> None:
        code = "def run(args, context):\n    return 1\n"
        agent = _agent(_programme(code), _programme(code))
        agent.enable_code_action(max_runs_per_run=1)
        rendu = _resultats_outils(agent.run("go"))
        assert "run_python budget exhausted" in rendu


class TestLaConsigne:
    """Mesuré : la description de l'outil ne suffit pas à dire QUAND écrire un programme — `hint=True` ajoute
    une phrase au prompt système, au rendu, sans jamais toucher au `system_prompt` de l'hôte."""

    def test_elle_s_ajoute_au_rendu_sans_modifier_le_prompt_de_l_hote(self) -> None:
        from autoagent.agent import CODE_ACTION_HINT

        agent = _agent(system_prompt="Tu es l'assistant de l'hôte.")
        agent.enable_code_action()
        agent.run("go")
        systeme = agent.provider.requetes[0].messages[0]  # type: ignore[attr-defined]
        assert systeme.role == "system"
        assert systeme.content == f"Tu es l'assistant de l'hôte.\n\n{CODE_ACTION_HINT}"
        assert agent.system_prompt == "Tu es l'assistant de l'hôte."

    def test_hint_false_n_ajoute_rien(self) -> None:
        agent = _agent(system_prompt="P")
        agent.enable_code_action(hint=False)
        assert agent.render_system_prompt() == "P"

    def test_un_prompt_callable_la_recoit_aussi(self) -> None:
        from autoagent.agent import CODE_ACTION_HINT

        agent = _agent(system_prompt=lambda: "dynamique")
        agent.enable_code_action()
        assert agent.render_system_prompt() == f"dynamique\n\n{CODE_ACTION_HINT}"

    def test_sans_code_action_le_rendu_ne_change_pas(self) -> None:
        def casse() -> str:
            raise RuntimeError("prompt cassé")

        from autoagent.agent import DEFAULT_SYSTEM_PROMPT

        assert _agent(system_prompt="P").render_system_prompt() == "P"
        assert _agent(system_prompt=casse).render_system_prompt() == DEFAULT_SYSTEM_PROMPT


@pytest.mark.parametrize("nom", ["create_python_tool", "find_tools"])
def test_les_meta_outils_ne_sont_jamais_exposes(nom: str) -> None:
    agent = _agent()
    agent.registry.register(lambda: "x", name=nom, description="méta")
    assert nom not in agent._outils_pour_le_code(None)
    assert "lire_journal" in agent._outils_pour_le_code(None)
