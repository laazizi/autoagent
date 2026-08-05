"""Convention sémantique GenAI d'OpenTelemetry (`semconv="gen_ai"`, 0.18.0).

Constat avant : l'exporteur aplatissait TOUT sous le préfixe `autoagent.*` et
nommait ses spans `agent.run` / `llm` / `tool.<nom>`. Résultat concret — les
traces exportées n'étaient PAS reconnues par Langfuse, Phoenix, Grafana ou tout
backend GenAI : on obtenait des spans anonymes, sans modèle ni coût en jetons.

Le mapping est purement ADDITIF : `gen_ai.*` s'ajoute, `autoagent.*` reste.
"""

from __future__ import annotations

import pytest

from autoagent.otel import OTelTraceExporter
from autoagent.trace import TraceEmitter

from .test_otel import FakeAPI, FakeTracer


def _exporteur(semconv: str):
    tracer = FakeTracer()
    return OTelTraceExporter(tracer, semconv=semconv, _api=FakeAPI()), tracer


def _run(exporteur) -> None:
    """Un run minimal : agent → llm → outil."""
    with TraceEmitter(on_event=exporteur) as trace:
        run = trace.emit("run_start", {"model": "gemini-3.5-flash", "max_steps": 8})
        llm = trace.emit("llm_request", {"step": 1, "model": "gemini-3.5-flash"},
                         parent_id=run)
        trace.emit("llm_response", {"step": 1, "input_tokens": 1200,
                                    "output_tokens": 64}, parent_id=llm)
        tool = trace.emit("tool_call_start", {"name": "lire_fichier", "call_id": "c1"},
                          parent_id=run)
        trace.emit("tool_call_end", {"name": "lire_fichier", "call_id": "c1",
                                     "status": "ok", "duration_ms": 12}, parent_id=tool)
        trace.emit("run_end", {"status": "ok", "steps": 1}, parent_id=run)


class TestNomsDeSpans:
    def test_noms_standard_genai(self) -> None:
        exp, tracer = _exporteur("gen_ai")
        _run(exp)
        noms = [s.name for s in tracer.spans]
        assert "invoke_agent" in noms
        assert "chat" in noms
        assert "execute_tool lire_fichier" in noms

    def test_noms_historiques_par_defaut(self) -> None:
        """Rétrocompatibilité : sans `semconv`, rien ne change."""
        exp, tracer = _exporteur("autoagent")
        _run(exp)
        noms = [s.name for s in tracer.spans]
        assert "agent.run" in noms and "llm" in noms
        assert "tool.lire_fichier" in noms


class TestAttributs:
    def test_modele_et_jetons_lisibles_par_un_backend_genai(self) -> None:
        exp, tracer = _exporteur("gen_ai")
        _run(exp)
        fusion: dict = {}
        for span in tracer.spans:
            fusion.update(span.attributes)
        assert fusion["gen_ai.request.model"] == "gemini-3.5-flash"
        assert fusion["gen_ai.usage.input_tokens"] == 1200
        assert fusion["gen_ai.usage.output_tokens"] == 64
        assert fusion["gen_ai.tool.name"] == "lire_fichier"
        assert fusion["gen_ai.tool.call.id"] == "c1"

    def test_operation_name_renseigne(self) -> None:
        exp, tracer = _exporteur("gen_ai")
        _run(exp)
        chat = next(s for s in tracer.spans if s.name == "chat")
        assert chat.attributes["gen_ai.operation.name"] == "chat"
        outil = next(s for s in tracer.spans if s.name.startswith("execute_tool"))
        assert outil.attributes["gen_ai.operation.name"] == "execute_tool"

    def test_additif_les_attributs_autoagent_restent(self) -> None:
        """Aucune rupture : ce que consommaient les tableaux de bord existants
        est toujours là."""
        exp, tracer = _exporteur("gen_ai")
        _run(exp)
        fusion: dict = {}
        for span in tracer.spans:
            fusion.update(span.attributes)
        assert fusion["autoagent.model"] == "gemini-3.5-flash"
        assert fusion["autoagent.status"] == "ok"

    def test_aucun_attribut_genai_en_mode_historique(self) -> None:
        exp, tracer = _exporteur("autoagent")
        _run(exp)
        for span in tracer.spans:
            assert not any(k.startswith("gen_ai.") for k in span.attributes)


class TestGarde:
    def test_semconv_invalide_refuse(self) -> None:
        with pytest.raises(ValueError, match="semconv"):
            OTelTraceExporter(FakeTracer(), semconv="openinference", _api=FakeAPI())
