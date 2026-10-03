"""La porte de décision unique (D1) : tout ce qui AGIT passe par la même décision.

Avant, la boucle décidait pour un appel d'outil DIRECT (politique de l'hôte, garde
trifecta, teinte). Le modèle agit aussi par d'autres chemins — du code qu'il écrit qui
appelle des fonctions de l'hôte (`run_python`, outil généré), le dispatcher
`call_host_function`, un sous-agent — et ceux-là n'étaient pas vus : une politique qui
refusait `envoyer_mail` ne refusait rien quand le MÊME envoi partait d'un programme.

Ces tests prouvent deux choses. 1) Comportement : chaque chemin obéit à la même
politique, à la même garde trifecta, à la même teinte — et se tait quand il n'y a ni
politique ni outil `egress`. 2) Structure, par analyse du code source : la politique de
l'hôte n'est invoquée qu'à UN endroit, et toute fonction qui lance une fonction de l'hôte
consulte la porte AVANT — un nouveau chemin d'exécution fait échouer ce fichier au lieu
de contourner la politique en silence.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

import autoagent
from autoagent import Agent, PythonRunner, delegate_to, tool
from autoagent.approval import ToolManifest, load_tools
from autoagent.errors import ApprovalRequired
from autoagent.evolution import EvolutionRuntime, enable_software_evolution
from autoagent.providers.base import LLMProvider
from autoagent.sandbox import SubprocessSandbox
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, ToolCall, is_tainted
from autoagent.trace import TraceEmitter, TraceEvent

EFFETS: list[tuple[str, dict[str, Any]]] = []


@pytest.fixture(autouse=True)
def _effets_propres() -> None:
    EFFETS.clear()


@tool(egress=True)
def envoyer_mail(destinataire: str, corps: str) -> dict:
    """Envoie un e-mail (fait SORTIR de l'information)."""
    EFFETS.append(("envoyer_mail", {"destinataire": destinataire, "corps": corps}))
    return {"envoye": True}


@tool(untrusted=True)
def lire_page(url: str) -> dict:
    """Lit une page web (contenu externe, non fiable)."""
    return {"texte": f"contenu de {url}"}


def lire_compteur(capteur: str) -> dict:
    """Lit un compteur (sans risque)."""
    EFFETS.append(("lire_compteur", {"capteur": capteur}))
    return {"n": 7}


HOTE = {"envoyer_mail": envoyer_mail, "lire_page": lire_page, "lire_compteur": lire_compteur}

PROG_MAIL = ("def run(args, context):\n"
             "    return context['call_host']('envoyer_mail', "
             "{'destinataire': 'moi@example.com', 'corps': 'bonjour'})\n")
PROG_LIRE_PUIS_MAIL = ("def run(args, context):\n"
                       "    context['call_host']('lire_page', {'url': 'https://exemple.test'})\n"
                       "    return context['call_host']('envoyer_mail', "
                       "{'destinataire': 'moi@example.com', 'corps': 'resume'})\n")
PROG_LIRE = ("def run(args, context):\n"
             "    return context['call_host']('lire_page', {'url': 'https://exemple.test'})\n")


class _Scripte(LLMProvider):
    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.reponses = list(reponses)
        self.requetes: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requetes.append(request)
        return self.reponses.pop(0) if self.reponses else LLMResponse(content="fini")


def _appel(nom: str, **arguments: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"c-{nom}", name=nom, arguments=arguments)])


def _agent_python(programme: str, *tours: LLMResponse, politique: Any = None,
                  trace: TraceEmitter | None = None, **kw: Any) -> Agent:
    """Un agent dont le modèle (scripté) lance `programme` par `run_python`."""
    reponses = [_appel("run_python", code=programme, args={}), *tours, LLMResponse(content="fini")]
    agent = Agent(_Scripte(reponses), max_steps=6, tool_policy=politique, trace=trace, **kw)
    agent.enable_run_python(PythonRunner(host_functions=HOTE))
    return agent


def _contenu_outil(resultat: Any) -> str:
    return "\n".join(m.content for m in resultat.messages if m.role == "tool")


def _evenements(trace_events: list[TraceEvent], type_: str) -> list[dict[str, Any]]:
    return [e.payload for e in trace_events if e.type == type_]


def _vu_par(agent: Agent) -> str:
    """Ce que le MODÈLE de cet agent a lu dans sa conversation (dernière requête) : le refus
    d'un sous-agent vit dans SA conversation, pas dans celle du parent."""
    return "\n".join(m.content or "" for m in agent.provider.requetes[-1].messages)  # type: ignore[attr-defined]


# ── Le pont de `run_python` ──────────────────────────────────────────────────


class TestPontRunPython:
    def test_sans_politique_ni_outil_marque_rien_ne_change(self) -> None:
        resultat = _agent_python(PROG_MAIL).run("go")
        assert EFFETS == [("envoyer_mail", {"destinataire": "moi@example.com", "corps": "bonjour"})]
        assert "ToolPolicyDenied" not in _contenu_outil(resultat)

    def test_la_politique_de_l_hote_refuse_une_fonction_appelee_par_un_programme(self) -> None:
        """LE trou : cette politique refusait `envoyer_mail` appelé directement, et rien
        quand le même envoi partait d'un programme écrit par le modèle."""
        def politique(ctx: Any) -> str | None:
            return "envoi interdit" if ctx.call.name == "envoyer_mail" else None

        resultat = _agent_python(PROG_MAIL, politique=politique).run("go")
        assert EFFETS == []                                            # l'effet n'a PAS eu lieu
        contenu = _contenu_outil(resultat)
        assert "ToolPolicyDenied: envoi interdit" in contenu

    def test_la_politique_voit_la_source_les_arguments_et_les_drapeaux(self) -> None:
        vus: list[Any] = []

        def politique(ctx: Any) -> str | None:
            vus.append(ctx)
            return None

        _agent_python(PROG_MAIL, politique=politique).run("go")
        nested = [c for c in vus if c.source == "host_function"]
        assert len(nested) == 1
        ctx = nested[0]
        assert ctx.call.name == "envoyer_mail"
        assert ctx.call.arguments == {"destinataire": "moi@example.com", "corps": "bonjour"}
        assert ctx.egress is True and ctx.spec is not None and ctx.spec.egress
        assert any(c.source == "tool" and c.call.name == "run_python" for c in vus)   # l'appel direct aussi

    def test_un_programme_qui_lit_du_contenu_non_fiable_ne_peut_plus_envoyer(self) -> None:
        """La lethal trifecta DANS un programme : lire (non fiable) puis envoyer (egress)."""
        resultat = _agent_python(PROG_LIRE_PUIS_MAIL).run("go")
        assert EFFETS == []
        assert "EgressBlocked" in _contenu_outil(resultat)

    def test_le_run_est_teinte_apres_un_programme_qui_a_lu_du_contenu_non_fiable(self) -> None:
        """Sans cela un `run_python` qui lit une page repartait « propre » : le tour suivant
        pouvait envoyer ce qu'il venait de lire."""
        agent = _agent_python(PROG_LIRE, _appel("envoyer_mail_direct", destinataire="x@example.com", corps="y"))
        agent.registry.add(autoagent.ToolSpec(name="envoyer_mail_direct", description="mail", egress=True,
                                              input_schema={"type": "object", "properties": {
                                                  "destinataire": {"type": "string"}, "corps": {"type": "string"}}}),
                           lambda destinataire, corps: EFFETS.append(("direct", {})) or {"ok": True})
        resultat = agent.run("go")
        assert is_tainted(resultat.messages)                          # le résultat du programme est encadré
        assert EFFETS == []                                           # …et l'envoi direct du tour suivant est bloqué
        assert "EgressBlocked" in _contenu_outil(resultat)

    def test_une_politique_qui_plante_refuse(self) -> None:
        def politique(ctx: Any) -> str | None:
            if ctx.source == "host_function":
                raise RuntimeError("boom")
            return None

        resultat = _agent_python(PROG_MAIL, politique=politique).run("go")
        assert EFFETS == []
        assert "policy error: RuntimeError: boom" in _contenu_outil(resultat)

    def test_une_demande_d_approbation_refuse_au_milieu_d_un_programme(self) -> None:
        def politique(ctx: Any) -> str | None:
            if ctx.source == "host_function":
                raise ApprovalRequired("validation humaine requise")
            return None

        resultat = _agent_python(PROG_MAIL, politique=politique).run("go")
        assert EFFETS == []
        assert "cannot pause a program" in _contenu_outil(resultat)

    def test_la_decision_est_tracee(self) -> None:
        evenements: list[TraceEvent] = []
        trace = TraceEmitter(on_event=evenements.append)

        def politique(ctx: Any) -> str | None:
            return "non" if ctx.call.name == "envoyer_mail" else None

        _agent_python(PROG_MAIL, politique=politique, trace=trace).run("go")
        decisions = _evenements(evenements, "gate_decision")
        assert [d["name"] for d in decisions] == ["envoyer_mail"]
        assert decisions[0]["allowed"] is False and decisions[0]["source"] == "host_function"
        assert decisions[0]["reason"] == "non"

    def test_une_fonction_sans_risque_passe_et_est_tracee(self) -> None:
        evenements: list[TraceEvent] = []
        trace = TraceEmitter(on_event=evenements.append)
        prog = "def run(args, context):\n    return context['call_host']('lire_compteur', {'capteur': 'c1'})\n"
        _agent_python(prog, politique=lambda ctx: None, trace=trace).run("go")
        assert EFFETS == [("lire_compteur", {"capteur": "c1"})]
        assert _evenements(evenements, "gate_decision")[0]["allowed"] is True

    def test_la_politique_en_donnees_peut_viser_la_source(self) -> None:
        """`{"when": {"source": "host_function"}}` : « aucun envoi depuis un programme », en
        DONNÉES — sans interdire l'appel direct du même outil."""
        from autoagent import ToolPolicySpec

        spec = ToolPolicySpec.from_dict({"default": "allow", "rules": [
            {"tool": "envoyer_mail", "action": "deny", "when": {"source": "host_function"},
             "reason": "pas d'envoi depuis un programme"}]})
        resultat = _agent_python(PROG_MAIL, politique=spec.compile()).run("go")
        assert EFFETS == []
        assert "pas d'envoi depuis un programme" in _contenu_outil(resultat)
        # …et l'appel direct du même outil, lui, passe (source == "tool").
        agent = Agent(_Scripte([_appel("envoyer_mail", destinataire="moi@example.com", corps="x"),
                                LLMResponse(content="fini")]), tool_policy=spec.compile())
        agent.add_tool(envoyer_mail)
        agent.run("go")
        assert len(EFFETS) == 1

    def test_summarize_trace_compte_les_refus_de_la_porte(self) -> None:
        from autoagent import summarize_trace

        evenements: list[TraceEvent] = []
        trace = TraceEmitter(on_event=evenements.append)
        _agent_python(PROG_MAIL, politique=lambda ctx: "non" if ctx.call.name == "envoyer_mail" else None,
                      trace=trace).run("go")
        metriques = summarize_trace(evenements)
        assert metriques.blocked_by_guard == {"gate_decision:host_function": 1}

    def test_le_mode_temoin_trace_sans_appliquer_la_garde_trifecta(self) -> None:
        evenements: list[TraceEvent] = []
        trace = TraceEmitter(on_event=evenements.append)
        _agent_python(PROG_LIRE_PUIS_MAIL, trace=trace, shadow_guards=True).run("go")
        assert len(EFFETS) == 1                                       # témoin : l'envoi PART
        decisions = _evenements(evenements, "gate_decision")
        assert any(d["name"] == "envoyer_mail" and d["would_block"] for d in decisions)

    def test_l_option_de_retour_au_comportement_precedent(self) -> None:
        agent = _agent_python(PROG_MAIL, politique=lambda ctx: "non" if ctx.call.name == "envoyer_mail" else None)
        agent.govern_host_calls = False
        agent.run("go")
        assert len(EFFETS) == 1                                       # pont non gouverné, comme en 0.22.0

    def test_hors_d_un_run_d_agent_rien_ne_change(self) -> None:
        sortie = PythonRunner(host_functions=HOTE)(PROG_MAIL)
        assert sortie["result"] == {"envoye": True}
        assert len(EFFETS) == 1


# ── Les autres chemins : outil généré, outil promu, dispatcher ───────────────


def _ecrire_outil(dossier: Path, nom: str, corps: str) -> None:
    lignes = "\n".join("    " + ligne for ligne in corps.strip().splitlines())
    (dossier / f"{nom}.py").write_text(
        "TOOL = {\n"
        f'    "name": "{nom}",\n    "description": "outil de test",\n'
        '    "input_schema": {"type": "object", "properties": {}},\n    "permissions": [],\n}\n\n'
        f"def run(args, context):\n{lignes}\n", encoding="utf-8")


class TestAutresChemins:
    def test_un_outil_genere_en_bac_a_sable_est_gouverne(self, tmp_path: Path) -> None:
        _ecrire_outil(tmp_path, "expedier",
                      "return context['call_host']('envoyer_mail', "
                      "{'destinataire': 'moi@example.com', 'corps': 'x'})")
        agent = Agent(_Scripte([_appel("expedier"), LLMResponse(content="fini")]),
                      tool_policy=lambda ctx: "non" if ctx.call.name == "envoyer_mail" else None)
        modes = dict(load_tools(agent, tmp_path, ToolManifest.load(tmp_path / "m.json"),
                                sandbox=SubprocessSandbox(timeout=15), sandbox_host_functions=HOTE))
        assert modes["expedier"] == "sandbox"
        resultat = agent.run("go")
        assert EFFETS == []
        assert "ToolPolicyDenied: non" in _contenu_outil(resultat)

    def test_un_outil_promu_en_natif_est_gouverne_aussi(self, tmp_path: Path) -> None:
        _ecrire_outil(tmp_path, "expedier",
                      "return context['call_host']('envoyer_mail', "
                      "{'destinataire': 'moi@example.com', 'corps': 'x'})")
        manifest = ToolManifest.load(tmp_path / "m.json")
        from autoagent.approval import approve_tool
        approve_tool(tmp_path / "expedier.py", manifest, approved_by="test")
        agent = Agent(_Scripte([_appel("expedier"), LLMResponse(content="fini")]),
                      tool_policy=lambda ctx: "non" if ctx.call.name == "envoyer_mail" else None)
        modes = dict(load_tools(agent, tmp_path, manifest, sandbox=SubprocessSandbox(timeout=15),
                                sandbox_host_functions=HOTE))
        assert modes["expedier"] == "native"
        resultat = agent.run("go")
        assert EFFETS == []
        assert "ToolPolicyDenied: non" in _contenu_outil(resultat)

    def test_le_dispatcher_call_host_function_decide_sur_le_nom_de_la_fonction(self, tmp_path: Path) -> None:
        """La politique voyait `call_host_function`, pas la fonction qu'il lance."""
        runtime = EvolutionRuntime(tmp_path)
        runtime.register_host_function("envoyer_mail", envoyer_mail)
        agent = Agent(_Scripte([_appel("call_host_function", name="envoyer_mail",
                                       arguments={"destinataire": "moi@example.com", "corps": "x"}),
                                LLMResponse(content="fini")]),
                      tool_policy=lambda ctx: "non" if ctx.call.name == "envoyer_mail" else None)
        enable_software_evolution(agent, runtime, capabilities={"host_call"})
        resultat = agent.run("go")
        assert EFFETS == []
        assert "ToolPolicyDenied: non" in _contenu_outil(resultat)


# ── Un sous-agent hérite de la politique du parent — sur demande ─────────────


def _parent_et_enfant(*, heriter: bool, deleguer: bool = False) -> tuple[Agent, Agent]:
    enfant = Agent(_Scripte([_appel("envoyer_mail", destinataire="moi@example.com", corps="x"),
                             LLMResponse(content="envoye")]), max_steps=4)
    enfant.add_tool(envoyer_mail)
    if deleguer:
        outil = delegate_to({"expert": enfant}, name="deleguer", inherit_policy=heriter)
        appel = _appel("deleguer", requests=[{"specialist": "expert", "request": "envoie"}])
    else:
        outil = enfant.as_tool(name="deleguer", description="délègue", inherit_policy=heriter)
        appel = _appel("deleguer", request="envoie")
    parent = Agent(_Scripte([appel, LLMResponse(content="fini")]), max_steps=4,
                   tool_policy=lambda ctx: "non" if ctx.call.name == "envoyer_mail" else None)
    parent.add_tool(outil)
    return parent, enfant


class TestSousAgent:
    @pytest.mark.parametrize("deleguer", [False, True])
    def test_par_defaut_le_sous_agent_agit_sous_sa_propre_politique(self, deleguer: bool) -> None:
        parent, _ = _parent_et_enfant(heriter=False, deleguer=deleguer)
        parent.run("go")
        assert len(EFFETS) == 1                                       # comportement historique inchangé

    @pytest.mark.parametrize("deleguer", [False, True])
    def test_avec_inherit_policy_la_politique_du_parent_couvre_le_sous_agent(self, deleguer: bool) -> None:
        parent, enfant = _parent_et_enfant(heriter=True, deleguer=deleguer)
        parent.run("go")
        assert EFFETS == []                                           # refusé par la politique du PARENT
        assert "ToolPolicyDenied: non" in _vu_par(enfant)             # c'est le sous-agent qui lit le refus
        assert enfant._parent_gate is None                            # la porte est rendue après le run

    def test_la_politique_du_parent_voit_la_source_subagent(self) -> None:
        vus: list[Any] = []

        def politique(ctx: Any) -> str | None:
            vus.append((ctx.source, ctx.call.name))
            return None

        parent, _ = _parent_et_enfant(heriter=True)
        parent.tool_policy = politique
        parent.run("go")
        assert ("subagent", "envoyer_mail") in vus and ("tool", "deleguer") in vus

    def test_la_teinte_du_parent_s_applique_au_sous_agent(self) -> None:
        """Un parent teinté qui délègue ne débloque pas l'envoi chez son spécialiste."""
        enfant = Agent(_Scripte([_appel("envoyer_mail", destinataire="moi@example.com", corps="x"),
                                 LLMResponse(content="envoye")]), max_steps=4)
        enfant.add_tool(envoyer_mail)
        parent = Agent(_Scripte([_appel("lire_page", url="https://exemple.test"),
                                 _appel("deleguer", request="envoie"), LLMResponse(content="fini")]),
                       max_steps=5)
        parent.add_tool(lire_page)
        parent.add_tool(enfant.as_tool(name="deleguer", description="délègue", inherit_policy=True))
        parent.run("go")
        assert EFFETS == []
        assert "EgressBlocked" in _vu_par(enfant)

    def test_l_heritage_est_transitif(self) -> None:
        petit_enfant = Agent(_Scripte([_appel("envoyer_mail", destinataire="moi@example.com", corps="x"),
                                       LLMResponse(content="envoye")]), max_steps=4)
        petit_enfant.add_tool(envoyer_mail)
        enfant = Agent(_Scripte([_appel("sous_deleguer", request="envoie"), LLMResponse(content="ok")]),
                       max_steps=4)
        enfant.add_tool(petit_enfant.as_tool(name="sous_deleguer", description="d", inherit_policy=True))
        parent = Agent(_Scripte([_appel("deleguer", request="envoie"), LLMResponse(content="fini")]),
                       max_steps=4, tool_policy=lambda ctx: "non" if ctx.call.name == "envoyer_mail" else None)
        parent.add_tool(enfant.as_tool(name="deleguer", description="d", inherit_policy=True))
        parent.run("go")
        assert EFFETS == []                                           # deux niveaux plus bas, toujours refusé


# ── La structure : un nouveau chemin d'exécution doit faire échouer ce fichier ──

PAQUET = Path(autoagent.__file__).parent


def _sources() -> dict[str, ast.Module]:
    return {p.name: ast.parse(p.read_text(encoding="utf-8")) for p in sorted(PAQUET.glob("*.py"))}


def _fonctions_englobantes(arbre: ast.Module) -> dict[int, str]:
    """ligne -> nom de la fonction qui la contient (la plus interne)."""
    englobante: dict[int, str] = {}
    for noeud in ast.walk(arbre):
        if isinstance(noeud, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sous in ast.walk(noeud):
                if hasattr(sous, "lineno"):
                    englobante[sous.lineno] = noeud.name
    return englobante


def _appels(arbre: ast.Module, nom: str) -> list[ast.Call]:
    """Appels `x.nom(...)` ou `nom(...)`."""
    trouves = []
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Call):
            f = noeud.func
            if (isinstance(f, ast.Attribute) and f.attr == nom) or (isinstance(f, ast.Name) and f.id == nom):
                trouves.append(noeud)
    return trouves


class TestStructure:
    def test_la_politique_de_l_hote_n_est_invoquee_qu_a_un_seul_endroit(self) -> None:
        sites = {nom for nom, arbre in _sources().items() if _appels(arbre, "tool_policy")}
        assert sites == {"gate.py"}, (
            f"`tool_policy(...)` est appelé hors de gate.evaluate_policy : {sorted(sites)}. "
            "Une décision de plus = un chemin que la porte ne voit pas.")

    def test_toute_fonction_qui_lance_une_fonction_de_l_hote_consulte_la_porte_avant(self) -> None:
        """Les trois endroits où une fonction de l'hôte est exécutée pour le compte d'un
        modèle. Un quatrième fait échouer ce test : il faut le passer par la porte, puis
        l'inscrire ici."""
        attendus = {("sandbox.py", "_drive_bridge"), ("approval.py", "_call_host"),
                    ("evolution.py", "call_host_function")}
        trouves: set[tuple[str, str]] = set()
        for fichier, arbre in _sources().items():
            englobante = _fonctions_englobantes(arbre)
            for appel in _appels(arbre, "get"):
                cible = appel.func.value if isinstance(appel.func, ast.Attribute) else None
                nom = (cible.attr if isinstance(cible, ast.Attribute)
                       else cible.id if isinstance(cible, ast.Name) else None)
                if nom == "host_functions":
                    trouves.add((fichier, englobante[appel.lineno]))
        assert trouves == attendus, f"nouveau chemin d'exécution de fonctions de l'hôte : {sorted(trouves ^ attendus)}"

        for fichier, fonction in attendus:
            arbre = _sources()[fichier]
            corps = next(n for n in ast.walk(arbre)
                         if isinstance(n, ast.FunctionDef) and n.name == fonction)
            decisions = [a.lineno for a in _appels(corps, "decide")]
            lancements = [a.lineno for a in ast.walk(corps)
                          if isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id in {"fn", "func"}]
            assert decisions and lancements, f"{fichier}::{fonction} : décision ou lancement introuvable"
            assert min(decisions) < min(lancements), f"{fichier}::{fonction} lance AVANT de décider"

    def test_la_boucle_n_execute_un_outil_que_dans_la_porte(self) -> None:
        """Tout appel à `registry.execute` du paquet : la boucle (dans `_executer`, qui pose la
        porte) et les deux enveloppes de record/replay. Un nouvel appelant fait échouer ce test."""
        attendus = {("agent.py", "_executer"), ("registry.py", "execute"), ("replay.py", "execute")}
        trouves: set[tuple[str, str]] = set()
        for fichier, arbre in _sources().items():
            englobante = _fonctions_englobantes(arbre)
            for appel in _appels(arbre, "execute"):
                trouves.add((fichier, englobante[appel.lineno]))
        assert trouves == attendus, f"nouveau chemin d'exécution d'outils : {sorted(trouves ^ attendus)}"
        corps = next(n for n in ast.walk(_sources()["agent.py"])
                     if isinstance(n, ast.FunctionDef) and n.name == "_executer")
        assert _appels(corps, "gate_scope"), "`_executer` doit poser la porte autour de l'exécution"
