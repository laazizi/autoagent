"""07 — Sandbox & sécurité : le bornement est du CODE, pas du prompt.

SANS clé API (100 % déterministe). Trois démonstrations :
  1. la validation AST refuse le code dangereux (os, eval, dunders d'évasion) ;
  2. un outil sain s'exécute en sandbox isolée (Docker si dispo, sinon
     subprocess durci `-I -S`, env vide) ;
  3. le PONT host-function : l'outil sandboxé (SANS réseau) appelle une
     fonction du host explicitement whitelistée — et RIEN d'autre.

    python examples_autoagent/07_sandbox_securite.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _common  # noqa: F401  (force la console en UTF-8)

from autoagent.errors import ToolValidationError
from autoagent.sandbox import docker_available, make_sandbox, validate_generated_tool_code

DANGEREUX = [
    ("import os",              "import os\ndef run(args, context):\n    return os.listdir('/')"),
    ("eval()",                 "def run(args, context):\n    return eval('2+2')"),
    ("évasion par dunders",    "def run(args, context):\n    return ().__class__.__bases__"),
]

SAIN = (
    'TOOL = {"name": "stats", "description": "moyenne", '
    '"input_schema": {"type": "object", "properties": {"xs": {"type": "array"}}}}\n'
    "def run(args, context):\n"
    "    xs = args['xs']\n"
    "    return {'moyenne': sum(xs) / len(xs), 'max': max(xs)}\n"
)

# Outil sandboxé qui a besoin d'une donnée du host : il la demande par le PONT.
AVEC_PONT = (
    'TOOL = {"name": "compter", "description": "compte via host", '
    '"input_schema": {"type": "object", "properties": {"capteur": {"type": "string"}}}}\n'
    "def run(args, context):\n"
    "    releves = context['call_host']('releves_capteur', {'id': args['capteur']})\n"
    "    return {'nb_releves': len(releves), 'total': sum(releves)}\n"
)

# Même outil, mais qui tente d'appeler une fonction NON whitelistée.
PONT_INTERDIT = (
    "def run(args, context):\n"
    "    return context['call_host']('lire_secrets', {})\n"
)


def _write(code: str) -> str:
    path = Path(tempfile.mkdtemp()) / "tool.py"
    path.write_text(code, encoding="utf-8")
    return str(path)


def main() -> None:
    print("1) VALIDATION AST — le code dangereux est refusé AVANT toute exécution\n")
    for label, code in DANGEREUX:
        try:
            validate_generated_tool_code(code, permissions=[])
            print(f"   ✗ {label:22} : ACCEPTÉ (ne devrait pas !)")
        except ToolValidationError as exc:
            print(f"   ✓ {label:22} : refusé — {str(exc)[:60]}")

    print(f"\n2) EXÉCUTION SANDBOX (Docker dispo: {docker_available()})\n")
    sandbox = make_sandbox(timeout=15)
    res = sandbox.run_python_tool(_write(SAIN), {"xs": [10, 20, 30, 40]})
    print(f"   outil sain → ok={res['ok']} résultat={res.get('result')}")

    print("\n3) PONT HOST-FUNCTION — accès contrôlé, whitelist stricte\n")
    # Le host expose UNE fonction ; l'outil sandboxé (sans réseau) l'appelle.
    host_functions = {"releves_capteur": lambda id: [12, 8, 15, 9]}
    res = sandbox.run_python_tool(
        _write(AVEC_PONT), {"capteur": "CMP-42"}, host_functions=host_functions
    )
    print(f"   appel whitelisté  → ok={res['ok']} résultat={res.get('result')}")

    res = sandbox.run_python_tool(
        _write(PONT_INTERDIT), {}, host_functions=host_functions
    )
    print(f"   appel NON whitelisté → ok={res['ok']} erreur={str(res.get('error'))[:55]}")
    print("\n   → l'outil ne peut atteindre QUE ce que le host a explicitement exposé.")


if __name__ == "__main__":
    main()
