"""`ToolPolicySpec` — politique d'outils déclarative (0.18.0).

`tool_policy` est une fonction Python : puissante, mais elle ne se versionne pas en
revue, ne se relit pas en diff, ne se transporte pas dans un snapshot. Ici la même
chose en JSON, `compile()` rendant un callable de la signature EXISTANTE — le code
de production ne bouge pas.

Trois invariants vérifiés :
  * précédence par ACTION (deny > approve > allow), donc aucun comportement caché
    dépendant de l'ordre des lignes ;
  * fail-CLOSED (structure invalide refusée au démarrage ; évaluation qui lève ⇒
    refus) ;
  * confinement monotone (`narrow` libre, `expand` sous approbation).
"""

from __future__ import annotations

import pytest

from autoagent import Agent, ApprovalRequired
from autoagent.agent import ToolPolicyContext
from autoagent.policy import ToolPolicySpec
from autoagent.schema import LLMResponse, ToolCall, ToolSpec

from .conftest import FakeLLMProvider


def _ctx(nom: str, args: dict | None = None, *, tainted: bool = False,
         egress: bool = False, step: int = 1, permissions: list[str] | None = None):
    return ToolPolicyContext(
        call=ToolCall(id="c1", name=nom, arguments=args or {}),
        spec=ToolSpec(name=nom, description="d", permissions=permissions or [], egress=egress),
        step=step, messages=[], context={}, tainted=tainted, egress=egress,
    )


class TestValidation:
    def test_politique_minimale(self) -> None:
        spec = ToolPolicySpec.from_dict({"rules": []})
        assert spec.default == "allow"

    def test_default_invalide(self) -> None:
        with pytest.raises(ValueError, match="default"):
            ToolPolicySpec.from_dict({"default": "peut-être"})

    def test_action_invalide_signale_la_position(self) -> None:
        with pytest.raises(ValueError, match=r"rules\[1\]\.action"):
            ToolPolicySpec.from_dict({"rules": [
                {"tool": "a", "action": "deny"},
                {"tool": "b", "action": "autorise-le"},
            ]})

    def test_cle_de_condition_inconnue(self) -> None:
        with pytest.raises(ValueError, match="when.humeur inconnu"):
            ToolPolicySpec.from_dict({"rules": [
                {"tool": "a", "action": "deny", "when": {"humeur": "mauvaise"}}]})

    def test_operateur_inconnu_refuse_au_demarrage(self) -> None:
        """Une faute de frappe dans une politique de SÉCURITÉ doit exploser au
        boot, pas laisser passer un appel parce que la règle ne matche jamais."""
        with pytest.raises(ValueError, match="opérateur.*inconnu"):
            ToolPolicySpec.from_dict({"rules": [
                {"tool": "write_file", "action": "allow",
                 "when": {"args": {"path": {"commence_par": "x"}}}}]})

    def test_round_trip_json(self) -> None:
        import json
        data = {"default": "deny", "rules": [
            {"tool": "lire", "action": "allow", "when": {"args": {"n": {"le": 10}}},
             "reason": "borné"}]}
        spec = ToolPolicySpec.from_dict(data)
        assert ToolPolicySpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


class TestPrecedence:
    def test_deny_gagne_sur_allow_quel_que_soit_l_ordre(self) -> None:
        for ordre in ([{"tool": "x", "action": "allow"}, {"tool": "x", "action": "deny"}],
                      [{"tool": "x", "action": "deny"}, {"tool": "x", "action": "allow"}]):
            spec = ToolPolicySpec.from_dict({"rules": ordre})
            assert spec.decide(_ctx("x"))[0] == "deny"

    def test_approve_gagne_sur_allow(self) -> None:
        spec = ToolPolicySpec.from_dict({"rules": [
            {"tool": "x", "action": "allow"}, {"tool": "x", "action": "approve"}]})
        assert spec.decide(_ctx("x"))[0] == "approve"

    def test_deny_gagne_sur_approve(self) -> None:
        spec = ToolPolicySpec.from_dict({"rules": [
            {"tool": "x", "action": "approve"}, {"tool": "x", "action": "deny"}]})
        assert spec.decide(_ctx("x"))[0] == "deny"

    def test_defaut_quand_rien_ne_matche(self) -> None:
        spec = ToolPolicySpec.from_dict({"default": "deny", "rules": [
            {"tool": "autre", "action": "allow"}]})
        assert spec.decide(_ctx("x"))[0] == "deny"

    def test_joker(self) -> None:
        spec = ToolPolicySpec.from_dict({"rules": [{"tool": "*", "action": "deny"}]})
        assert spec.decide(_ctx("n_importe_quoi"))[0] == "deny"


class TestConditions:
    def test_sur_un_argument(self) -> None:
        spec = ToolPolicySpec.from_dict({"default": "deny", "rules": [
            {"tool": "write_file", "action": "allow",
             "when": {"args": {"path": {"starts_with": "rapports/"}}}}]})
        assert spec.decide(_ctx("write_file", {"path": "rapports/x.md"}))[0] == "allow"
        assert spec.decide(_ctx("write_file", {"path": "/etc/passwd"}))[0] == "deny"

    def test_valeur_brute_vaut_egalite(self) -> None:
        spec = ToolPolicySpec.from_dict({"rules": [
            {"tool": "t", "action": "deny", "when": {"args": {"mode": "destructif"}}}]})
        assert spec.decide(_ctx("t", {"mode": "destructif"}))[0] == "deny"
        assert spec.decide(_ctx("t", {"mode": "lecture"}))[0] == "allow"

    def test_conditions_cumulees_ET(self) -> None:
        """LA règle trifecta écrite en données."""
        spec = ToolPolicySpec.from_dict({"rules": [
            {"tool": "*", "action": "deny", "when": {"tainted": True, "egress": True},
             "reason": "exfiltration"}]})
        assert spec.decide(_ctx("mail", tainted=True, egress=True)) == ("deny", "exfiltration")
        assert spec.decide(_ctx("mail", tainted=True, egress=False))[0] == "allow"
        assert spec.decide(_ctx("mail", tainted=False, egress=True))[0] == "allow"

    def test_operateurs_numeriques(self) -> None:
        spec = ToolPolicySpec.from_dict({"default": "deny", "rules": [
            {"tool": "t", "action": "allow", "when": {"args": {"montant": {"le": 100}}}}]})
        assert spec.decide(_ctx("t", {"montant": 50}))[0] == "allow"
        assert spec.decide(_ctx("t", {"montant": 500}))[0] == "deny"

    def test_regex(self) -> None:
        spec = ToolPolicySpec.from_dict({"rules": [
            {"tool": "t", "action": "deny",
             "when": {"args": {"cible": {"matches": r"^https?://(?!interne\.)"}}}}]})
        assert spec.decide(_ctx("t", {"cible": "http://dehors.tld"}))[0] == "deny"
        assert spec.decide(_ctx("t", {"cible": "http://interne.tld"}))[0] == "allow"

    def test_permissions_de_la_spec(self) -> None:
        spec = ToolPolicySpec.from_dict({"rules": [
            {"tool": "*", "action": "approve", "when": {"permissions": {"contains": "write"}}}]})
        assert spec.decide(_ctx("t", permissions=["write"]))[0] == "approve"
        assert spec.decide(_ctx("t", permissions=["read"]))[0] == "allow"

    def test_etape(self) -> None:
        spec = ToolPolicySpec.from_dict({"rules": [
            {"tool": "t", "action": "deny", "when": {"step": {"gt": 3}}}]})
        assert spec.decide(_ctx("t", step=2))[0] == "allow"
        assert spec.decide(_ctx("t", step=9))[0] == "deny"

    def test_in_et_not_in(self) -> None:
        spec = ToolPolicySpec.from_dict({"default": "deny", "rules": [
            {"tool": "t", "action": "allow",
             "when": {"args": {"env": {"in": ["dev", "recette"]}}}}]})
        assert spec.decide(_ctx("t", {"env": "dev"}))[0] == "allow"
        assert spec.decide(_ctx("t", {"env": "prod"}))[0] == "deny"

    def test_max_length(self) -> None:
        spec = ToolPolicySpec.from_dict({"default": "deny", "rules": [
            {"tool": "t", "action": "allow", "when": {"args": {"txt": {"max_length": 5}}}}]})
        assert spec.decide(_ctx("t", {"txt": "court"}))[0] == "allow"
        assert spec.decide(_ctx("t", {"txt": "beaucoup trop long"}))[0] == "deny"

    def test_argument_absent_ne_matche_pas(self) -> None:
        spec = ToolPolicySpec.from_dict({"rules": [
            {"tool": "t", "action": "deny", "when": {"args": {"path": {"starts_with": "/"}}}}]})
        assert spec.decide(_ctx("t", {}))[0] == "allow"


class TestConfinementMonotone:
    def _base(self) -> ToolPolicySpec:
        return ToolPolicySpec.from_dict({"default": "allow", "rules": [
            {"tool": "lire", "action": "allow"}]})

    def test_narrow_applique_librement(self) -> None:
        durci = self._base().narrow([{"tool": "supprimer", "action": "deny"}])
        assert durci.decide(_ctx("supprimer"))[0] == "deny"
        assert durci.decide(_ctx("lire"))[0] == "allow"

    def test_narrow_refuse_une_regle_elargissante(self) -> None:
        with pytest.raises(ValueError, match="narrow"):
            self._base().narrow([{"tool": "tout", "action": "allow"}])

    def test_expand_exige_une_approbation(self) -> None:
        with pytest.raises(ApprovalRequired, match="élargissement"):
            self._base().expand([{"tool": "supprimer", "action": "allow"}])

    def test_expand_approuve_passe(self) -> None:
        large = self._base().expand([{"tool": "supprimer", "action": "allow"}], approved=True)
        assert large.decide(_ctx("supprimer"))[0] == "allow"

    def test_changer_le_defaut_vers_allow_exige_une_approbation(self) -> None:
        strict = ToolPolicySpec.from_dict({"default": "deny"})
        with pytest.raises(ApprovalRequired):
            strict.expand(default="allow")

    def test_la_politique_de_depart_est_immuable(self) -> None:
        base = self._base()
        base.narrow([{"tool": "x", "action": "deny"}])
        assert base.decide(_ctx("x"))[0] == "allow"      # l'original n'a pas bougé


class TestCompileEtIntegration:
    def test_allow_rend_none(self) -> None:
        politique = ToolPolicySpec.from_dict({"rules": []}).compile()
        assert politique(_ctx("t")) is None

    def test_deny_rend_le_motif(self) -> None:
        politique = ToolPolicySpec.from_dict({"rules": [
            {"tool": "t", "action": "deny", "reason": "hors périmètre"}]}).compile()
        assert politique(_ctx("t")) == "hors périmètre"

    def test_approve_leve_approval_required(self) -> None:
        politique = ToolPolicySpec.from_dict({"rules": [
            {"tool": "t", "action": "approve"}]}).compile()
        with pytest.raises(ApprovalRequired):
            politique(_ctx("t"))

    def test_branchee_sur_un_vrai_agent(self) -> None:
        """Le point de la feature : `compile()` se branche sur le hook EXISTANT."""
        spec = ToolPolicySpec.from_dict({"default": "allow", "rules": [
            {"tool": "supprimer_tout", "action": "deny", "reason": "jamais en prod"}]})
        agent = Agent(FakeLLMProvider([
            LLMResponse(tool_calls=[ToolCall(id="c1", name="supprimer_tout", arguments={})]),
            "compris",
        ]), max_steps=4, tool_policy=spec.compile())
        appels = {"n": 0}

        @agent.tool
        def supprimer_tout() -> dict:
            """Ne doit JAMAIS tourner."""
            appels["n"] += 1
            return {"fait": True}

        res = agent.run("nettoie")
        assert appels["n"] == 0
        assert any("jamais en prod" in (m.content or "") for m in res.messages)

    def test_fail_closed_si_l_evaluation_leve(self) -> None:
        """Contrat de sécurité : une politique qui ne sait pas conclure REFUSE."""
        spec = ToolPolicySpec.from_dict({"rules": [{"tool": "*", "action": "allow"}]})
        # on casse volontairement une règle APRÈS validation (objet gelé → tuple)
        object.__setattr__(spec, "rules", ({"tool": "*", "action": "allow",
                                            "when": {"args": None}},))
        verdict = spec.compile()(_ctx("t"))
        assert verdict is not None and "policy error" in verdict
