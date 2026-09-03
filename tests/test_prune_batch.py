"""`prune_batch` — élaguer PAR LOTS pour ne pas casser le cache de préfixe (0.21.0).

Le cache de prompt d'un fournisseur ne sert que si le préfixe est IDENTIQUE à
l'octet d'un appel au suivant. Élaguer à chaque étape réécrit la vue à chaque
étape : le cache repart de zéro à chaque tour. TokenPilot (arXiv 2606.17016)
mesure l'effet — jetons hors cache 5,9 M → 1,6 M — quand la compaction se fait
par lots, à des frontières stables.

Ces tests verrouillent :

  1. LE NOMBRE ÉLAGUÉ EST UN MULTIPLE DE K — donc la vue ne change qu'une fois
     tous les K résultats, et est IDENTIQUE À L'OCTET entre deux lots.
  2. `batch=1` = comportement 0.19.0, inchangé.
  3. Les trois invariants de l'élagage (teinte, jamais grossir, registre intact)
     tiennent toujours par lots.
  4. La trace porte `batch`.
"""

from __future__ import annotations

from .conftest import FakeLLMProvider

from autoagent import Agent, TraceEmitter
from autoagent.agent import _prune_tool_results
from autoagent.schema import UNTRUSTED_CLOSE, UNTRUSTED_OPEN, LLMResponse, Message, ToolCall, is_tainted

GROS = "x" * 3000


def _transcript(n: int) -> list[Message]:
    msgs: list[Message] = [Message(role="user", content="vas-y")]
    for i in range(n):
        msgs.append(Message(role="assistant", content="",
                            tool_calls=[ToolCall(id=f"c{i}", name="lire", arguments={"n": i})]))
        msgs.append(Message(role="tool", name="lire", tool_call_id=f"c{i}", content=GROS))
    return msgs


def _vue(n: int, keep: int, batch: int) -> str:
    view, _, _ = _prune_tool_results(_transcript(n), keep, batch)
    return "\n".join(m.content for m in view)


class TestMultiplesDeK:
    def test_le_compte_elague_est_un_multiple_du_lot(self) -> None:
        for n in range(1, 14):
            _, elagues, _ = _prune_tool_results(_transcript(n), keep=2, batch=3)
            assert elagues % 3 == 0, f"n={n} : {elagues} élagués, pas un multiple de 3"
            assert elagues <= max(0, n - 2)

    def test_rien_tant_que_le_lot_n_est_pas_plein(self) -> None:
        # keep=2, batch=3 : il faut 5 résultats pour que 3 vieux soient élagués.
        for n in (3, 4):
            _, elagues, _ = _prune_tool_results(_transcript(n), keep=2, batch=3)
            assert elagues == 0
        _, elagues, _ = _prune_tool_results(_transcript(5), keep=2, batch=3)
        assert elagues == 3


class TestStabiliteDuPrefixe:
    def test_la_vue_est_identique_a_l_octet_entre_deux_lots(self) -> None:
        """Le point qui justifie tout : entre deux lots, le préfixe envoyé au
        fournisseur ne bouge pas d'un octet — seuls les NOUVEAUX messages
        s'ajoutent à la fin."""
        keep, batch = 2, 4
        vues = {n: _vue(n, keep, batch) for n in range(1, 15)}
        # Le nombre de messages élagués, par n : doit changer une fois tous les 4.
        elagues = {n: _prune_tool_results(_transcript(n), keep, batch)[1] for n in vues}
        changements = sum(1 for n in range(2, 15) if elagues[n] != elagues[n - 1])
        assert changements == 3, f"{changements} ruptures de préfixe pour 14 étapes (attendu 3)"
        # Et entre deux ruptures, la vue de n est un PRÉFIXE strict de celle de n+1.
        for n in range(1, 14):
            if elagues[n] == elagues[n + 1]:
                assert vues[n + 1].startswith(vues[n]), f"préfixe cassé entre {n} et {n + 1}"

    def test_batch_1_rompt_le_prefixe_a_chaque_etape(self) -> None:
        """Le contraste : en 0.19.0 la vue changeait à CHAQUE étape passé `keep`."""
        keep = 2
        elagues = {n: _prune_tool_results(_transcript(n), keep, 1)[1] for n in range(1, 15)}
        changements = sum(1 for n in range(2, 15) if elagues[n] != elagues[n - 1])
        assert changements == 12


class TestCompatibilite:
    def test_batch_1_est_le_comportement_historique(self) -> None:
        for n in range(1, 10):
            a = _prune_tool_results(_transcript(n), keep=2, batch=1)
            b = _prune_tool_results(_transcript(n), keep=2)
            assert [m.content for m in a[0]] == [m.content for m in b[0]]
            assert a[1:] == b[1:]

    def test_teinte_et_registre_tiennent_par_lots(self) -> None:
        transcript = _transcript(6)
        transcript[2] = Message(role="tool", name="lire", tool_call_id="c0",
                                content=f"{UNTRUSTED_OPEN}\n{GROS}\n{UNTRUSTED_CLOSE}")
        avant = [m.content for m in transcript]
        view, elagues, _ = _prune_tool_results(transcript, keep=1, batch=5)
        assert elagues == 5 and is_tainted(view)
        assert [m.content for m in transcript] == avant, "le registre a été muté"

    def test_ne_grossit_jamais_par_lots(self) -> None:
        petits = [Message(role="user", content="v")] + [
            Message(role="tool", name="t", tool_call_id=f"c{i}", content="ok") for i in range(6)]
        _, elagues, economie = _prune_tool_results(petits, keep=0, batch=3)
        assert (elagues, economie) == (0, 0)


class TestDansLaBoucle:
    def test_le_parametre_et_la_trace(self) -> None:
        reps = [LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name="lire", arguments={"n": i})])
                for i in range(6)] + ["fini"]
        vus: list[dict] = []
        with TraceEmitter(on_event=lambda e: vus.append(e.payload) if e.type == "context_pruned" else None) as tr:
            agent = Agent(FakeLLMProvider(reps), max_steps=10, trace=tr,
                          prune_tool_results_after=1, prune_batch=2)

            @agent.tool
            def lire(n: int) -> str:
                """Lit."""
                return GROS

            agent.run("vas-y")
        assert agent.prune_batch == 2
        assert vus and all(p["batch"] == 2 and p["pruned"] % 2 == 0 for p in vus)

    def test_batch_invalide_ramene_a_1(self) -> None:
        assert Agent(FakeLLMProvider([]), prune_batch=0).prune_batch == 1
