"""Un outil réécrit avec la MÊME taille dans la MÊME seconde doit exécuter le
NOUVEAU code.

Le runner du `SubprocessSandbox` chargeait le fichier par `importlib`, donc via
`__pycache__` : Python juge un `.pyc` valide sur (mtime en secondes, taille) de
la source. Or `synthesize_tool` jette l'essai 1 et écrit l'essai 2 sous le même
nom — sur un runner CI rapide, dans la même seconde, et `x * 3` a la taille de
`x * 2`. L'essai 2 exécutait alors le bytecode périmé de l'essai 1 : la CI
était rouge sur les 8 jobs, le poste local (plus lent) restait vert.

Le test force les deux conditions (`os.utime` remet le mtime) : il échouait
avant la correction, il passe après.
"""

from __future__ import annotations

import os
from pathlib import Path

from autoagent.sandbox import SubprocessSandbox
from autoagent.synthesis import _discard


def test_reecrire_un_outil_de_meme_taille_dans_la_meme_seconde_execute_le_nouveau_code(tmp_path: Path) -> None:
    f = tmp_path / "doubler.py"
    v1 = "def run(args, context):\n    return args['x'] * 3\n"
    v2 = 'def run(args, context):\n    return args["x"] * 2\n'
    assert len(v1) == len(v2), "le piège exige la même taille"
    f.write_text(v1, encoding="utf-8")
    t0 = os.stat(f).st_mtime
    bac = SubprocessSandbox(timeout=60)
    assert bac.run_python_tool(f, {"x": 5})["result"] == 15
    f.write_text(v2, encoding="utf-8")
    os.utime(f, (t0, t0))                       # même seconde : le pire cas
    assert bac.run_python_tool(f, {"x": 5})["result"] == 10
    cache = tmp_path / "__pycache__"
    assert not (cache.exists() and list(cache.glob("*.pyc"))), "le bac à sable ne doit plus écrire de bytecode"


def test_jeter_un_outil_retire_aussi_son_bytecode(tmp_path: Path) -> None:
    """Défense en profondeur : même si un `.pyc` existait (autre chargeur), un
    outil rejeté n'en laisse pas derrière lui."""
    from autoagent.sandbox import GeneratedPythonTool
    from autoagent.schema import ToolSpec

    f = tmp_path / "outil.py"
    f.write_text("def run(args, context):\n    return 1\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "outil.cpython-311.pyc").write_bytes(b"perime")
    (cache / "autre.cpython-311.pyc").write_bytes(b"garde")
    spec = ToolSpec(name="outil", description="x", input_schema={"type": "object", "properties": {}})
    _discard(GeneratedPythonTool(spec=spec, file_path=f, sandbox=SubprocessSandbox()))
    assert not f.exists()
    assert not (cache / "outil.cpython-311.pyc").exists()
    assert (cache / "autre.cpython-311.pyc").exists()
