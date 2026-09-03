"""`summarize_trace` — l'efficacité lue dans la trace, après coup (0.21.0).

Ce que ces tests verrouillent :

  1. LES RUNS SONT RECONSTITUÉS PAR LA PARENTÉ DES SPANS, pas par l'ordre des
     lignes — deux runs entrelacés ne se contaminent pas.
  2. UN APPEL REDONDANT = même run, même outil, mêmes arguments, déjà vu. Le même
     appel dans un AUTRE run n'est pas redondant.
  3. AUCUN CHIFFRE INVENTÉ : pas de jetons → None ; zéro succès → None.
  4. Ça marche sur une vraie trace du dépôt (`trace_demo.jsonl`) et sur des
     événements produits par un vrai run avec gardes et élagage.
"""

from __future__ import annotations

from pathlib import Path

from autoagent import Agent, TraceEmitter, summarize_trace
from autoagent.schema import LLMResponse, ToolCall

from .conftest import FakeLLMProvider

RACINE = Path(__file__).resolve().parent.parent
TRACE_DEMO = RACINE / "examples_autoagent" / "trace_demo.jsonl"


def _ev(type_: str, span: str, parent: str | None, **payload):  # type: ignore[no-untyped-def]
    return {"type": type_, "span_id": span, "parent_id": parent, "ts": 0.0, "payload": payload}


class TestSurLaTraceDuDepot:
    def test_trace_demo_jsonl(self) -> None:
        m = summarize_trace(TRACE_DEMO)
        assert m.runs == 2 and m.llm_calls == 3 and m.tool_calls == 2
        assert m.runs_by_status.get("ok") == 1
        assert m.total_tokens is not None and m.total_tokens > 0
        assert m.tokens_per_success == m.total_tokens / 1
        assert m.redundant_tool_calls == 0
        d = m.to_dict()
        assert d["runs"] == 2 and "tokens_per_success" in d
        assert "runs 2 (ok 1)" in m.summary()


class TestRedondanceParRun:
    def test_meme_appel_deux_fois_dans_un_run_est_redondant(self) -> None:
        events = [
            _ev("run_start", "r1", None, max_steps=3),
            _ev("llm_request", "q1", "r1", step=1),
            _ev("tool_call_start", "t1", "q1", name="lire", call_id="a", arguments_preview='{"n": 1}'),
            _ev("tool_call_end", "e1", "t1", name="lire", call_id="a", status="ok"),
            _ev("llm_request", "q2", "r1", step=2),
            _ev("tool_call_start", "t2", "q2", name="lire", call_id="b", arguments_preview='{"n": 1}'),
            _ev("tool_call_end", "e2", "t2", name="lire", call_id="b", status="error"),
            _ev("run_end", "f1", "r1", status="ok", steps=2),
        ]
        m = summarize_trace(events)
        assert m.tool_calls == 2 and m.redundant_tool_calls == 1
        assert m.redundant_by_tool == {"lire": 1}
        assert m.tool_errors == 1 and m.tool_error_ratio == 0.5
        assert m.redundant_ratio == 0.5

    def test_le_meme_appel_dans_deux_runs_entrelaces_n_est_pas_redondant(self) -> None:
        """L'ordre des lignes mélange deux runs ; la parenté les sépare."""
        events = [
            _ev("run_start", "rA", None), _ev("run_start", "rB", None),
            _ev("llm_request", "qA", "rA", step=1), _ev("llm_request", "qB", "rB", step=1),
            _ev("tool_call_start", "tA", "qA", name="lire", call_id="1", arguments_preview='{"n": 1}'),
            _ev("tool_call_start", "tB", "qB", name="lire", call_id="2", arguments_preview='{"n": 1}'),
            _ev("run_end", "fA", "rA", status="ok", steps=1), _ev("run_end", "fB", "rB", status="ok", steps=1),
        ]
        m = summarize_trace(events)
        assert m.runs == 2 and m.tool_calls == 2 and m.redundant_tool_calls == 0


class TestAucunChiffreInvente:
    def test_sans_jetons_none(self) -> None:
        m = summarize_trace([_ev("run_start", "r", None), _ev("run_end", "f", "r", status="ok", steps=1)])
        assert m.total_tokens is None and m.tokens_per_success is None
        assert "jetons" not in m.summary()

    def test_zero_succes_none(self) -> None:
        m = summarize_trace([
            _ev("run_start", "r", None),
            _ev("llm_response", "p", "r", step=1, input_tokens=100, output_tokens=10),
            _ev("run_end", "f", "r", status="max_steps", steps=3),
        ])
        assert m.total_tokens == 110 and m.runs_ok == 0 and m.tokens_per_success is None

    def test_trace_vide(self) -> None:
        m = summarize_trace([])
        assert m.runs == 0 and m.redundant_ratio is None and m.tool_error_ratio is None


class TestSurUnVraiRun:
    def _run(self, **kw):  # type: ignore[no-untyped-def]
        reps = [LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name="lire", arguments={"n": 1})])
                for i in range(4)] + ["fini"]
        events = []
        with TraceEmitter(on_event=events.append) as tr:
            agent = Agent(FakeLLMProvider(reps), max_steps=8, trace=tr, **kw)

            @agent.tool
            def lire(n: int) -> str:
                """Lit."""
                return "x" * 3000

            agent.run("vas-y")
        return summarize_trace(events)

    def test_redondance_et_garde_active(self) -> None:
        m = self._run(max_repeated_tool_calls=2)
        # 4 appels identiques demandés : 3 redondants ; la garde en refuse 2 (le 3e et le 4e).
        assert m.tool_calls == 4 and m.redundant_tool_calls == 3
        assert m.blocked_by_guard.get("loop_guard_block") == 2
        assert m.runs_ok == 1 and m.runs == 1

    def test_mode_temoin_et_elagage_dans_la_trace(self) -> None:
        m = self._run(max_repeated_tool_calls=2, shadow_guards=True, prune_tool_results_after=1)
        assert m.would_block.get("loop_guard_would_block") == 2
        assert not m.blocked_by_guard
        assert m.pruned_chars > 0
        assert "témoin" in m.summary() and "élagué" in m.summary()

    def test_accepte_des_objets_trace_event(self) -> None:
        events = []
        with TraceEmitter(on_event=events.append) as tr:
            Agent(FakeLLMProvider(["ok"]), trace=tr).run("x")
        assert summarize_trace(events).runs == 1
        assert summarize_trace(e.to_dict() for e in events).runs == 1
