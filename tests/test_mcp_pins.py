"""Épingler les définitions des outils MCP par empreinte (0.26.0).

Un serveur MCP peut changer la description ou le schéma d'un outil APRÈS qu'on l'a relu et accepté
(« rug pull », OWASP Top 10 for Agentic Applications 2026, ASI04 ; MITRE ATLAS, empoisonnement d'outil) :
le modèle lit alors une consigne que personne n'a vue. `pin_tools()` écrit l'empreinte sha256 de chaque
définition relue ; `mount(pins=…)` refuse de monter si une définition a changé ou si un outil non épinglé
est apparu. Rien n'est monté en cas de refus (fail-closed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from autoagent.errors import MCPError
from autoagent.mcp import MCPClient, tool_fingerprint
from autoagent.registry import ToolRegistry
from autoagent.schema import ToolCall

FAKE_SERVER = Path(__file__).with_name("fake_mcp_server.py")


def _client(**variantes: str) -> MCPClient:
    return MCPClient([sys.executable, str(FAKE_SERVER)], timeout=15.0, inherit_env=True,
                     env=variantes or None)


def _noms(registre: ToolRegistry) -> list[str]:
    return sorted(s.name for s in registre.specs())


class TestEmpreinte:
    def test_stable_quel_que_soit_l_ordre_des_cles(self) -> None:
        a = {"name": "t", "description": "d", "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}}
        b = {"inputSchema": {"properties": {"x": {"type": "string"}}, "type": "object"}, "description": "d", "name": "t"}
        assert tool_fingerprint(a) == tool_fingerprint(b)
        assert len(tool_fingerprint(a)) == 64

    def test_change_avec_la_description_le_schema_ou_les_annotations(self) -> None:
        base = {"name": "t", "description": "d", "inputSchema": {"type": "object"}}
        empreinte = tool_fingerprint(base)
        assert tool_fingerprint({**base, "description": "d!"}) != empreinte
        assert tool_fingerprint({**base, "inputSchema": {"type": "object", "properties": {"y": {}}}}) != empreinte
        assert tool_fingerprint({**base, "annotations": {"readOnlyHint": False}}) != empreinte

    def test_ignore_meta(self) -> None:
        base = {"name": "t", "description": "d"}
        assert tool_fingerprint({**base, "_meta": {"x": 1}}) == tool_fingerprint(base)


class TestEpinglage:
    def test_pin_puis_mount_sans_changement(self, tmp_path: Path) -> None:
        fichier = tmp_path / "pins.json"
        with _client() as mcp:
            ecrites = mcp.pin_tools(fichier)
            assert sorted(ecrites) == ["add", "boom", "echo"]
        contenu = json.loads(fichier.read_text(encoding="utf-8"))
        assert contenu["server"]["name"] == "fake-mcp" and set(contenu["tools"]) == {"add", "boom", "echo"}
        registre = ToolRegistry()
        with _client() as mcp:
            assert sorted(mcp.mount(registre, pins=fichier)) == ["add", "boom", "echo"]
            assert registre.execute(ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})).ok

    def test_description_changee_refus_et_rien_n_est_monte(self, tmp_path: Path) -> None:
        fichier = tmp_path / "pins.json"
        with _client() as mcp:
            mcp.pin_tools(fichier)
        registre = ToolRegistry()
        with _client(FAKE_MCP_RUGPULL="1") as mcp, pytest.raises(MCPError, match="echo") as erreur:
            mcp.mount(registre, pins=fichier)
        assert "changed" in str(erreur.value)
        assert _noms(registre) == [], "fail-closed : aucun outil monté, même ceux qui n'ont pas changé"

    def test_outil_nouveau_non_epingle_refuse(self, tmp_path: Path) -> None:
        fichier = tmp_path / "pins.json"
        with _client() as mcp:
            mcp.pin_tools(fichier)
        with _client(FAKE_MCP_NEW_TOOL="1") as mcp, pytest.raises(MCPError, match="exfil"):
            mcp.mount(ToolRegistry(), pins=fichier)

    def test_le_filtre_include_ne_verifie_que_ce_qu_il_monte(self, tmp_path: Path) -> None:
        fichier = tmp_path / "pins.json"
        with _client() as mcp:
            mcp.pin_tools(fichier, include={"add"})
        registre = ToolRegistry()
        with _client(FAKE_MCP_RUGPULL="1", FAKE_MCP_NEW_TOOL="1") as mcp:
            assert mcp.mount(registre, include={"add"}, pins=fichier) == ["add"]

    def test_outil_epingle_disparu_n_empeche_pas_de_monter(self, tmp_path: Path) -> None:
        with _client() as mcp:
            empreintes = mcp.fingerprints()
        empreintes["retire_depuis"] = "0" * 64
        with _client() as mcp:
            assert sorted(mcp.mount(ToolRegistry(), pins=empreintes)) == ["add", "boom", "echo"]

    def test_meta_volatile_ne_declenche_rien(self, tmp_path: Path) -> None:
        fichier = tmp_path / "pins.json"
        with _client() as mcp:
            mcp.pin_tools(fichier)
        with _client(FAKE_MCP_META="1") as mcp:
            assert len(mcp.mount(ToolRegistry(), pins=fichier)) == 3

    def test_dictionnaire_en_memoire(self) -> None:
        with _client() as mcp:
            empreintes = mcp.fingerprints()
        with _client(FAKE_MCP_RUGPULL="1") as mcp, pytest.raises(MCPError, match="echo"):
            mcp.tools(pins=empreintes)


class TestFermeSurLeDoute:
    def test_fichier_absent(self, tmp_path: Path) -> None:
        with _client() as mcp, pytest.raises(MCPError, match="pins"):
            mcp.mount(ToolRegistry(), pins=tmp_path / "absent.json")

    @pytest.mark.parametrize("contenu", ["{pas du json", '{"tools": []}', '{"tools": {"echo": 42}}', "[]"])
    def test_fichier_illisible_ou_mal_forme(self, tmp_path: Path, contenu: str) -> None:
        fichier = tmp_path / "pins.json"
        fichier.write_text(contenu, encoding="utf-8")
        with _client() as mcp, pytest.raises(MCPError, match="pins"):
            mcp.mount(ToolRegistry(), pins=fichier)

    def test_sans_pins_le_comportement_ne_change_pas(self) -> None:
        with _client(FAKE_MCP_RUGPULL="1") as mcp:
            assert len(mcp.mount(ToolRegistry())) == 3

    def test_la_description_vue_par_le_modele_est_celle_du_montage(self, tmp_path: Path) -> None:
        """Ce que le modèle lit est figé au montage : un changement ULTÉRIEUR côté serveur ne lui
        parvient pas (le handler garde la spec capturée)."""
        with _client() as mcp:
            outils = mcp.tools(pins=mcp.fingerprints())
            spec = next(h for h in outils if h.__name__ == "echo").__autoagent_tool_spec__
        assert spec.description == "Echo text back."
