"""Red tests for sandbox bypass attempts.

The current `validate_generated_tool_code` walks the AST and matches
import names and call names directly. An attacker (a malicious or
hallucinating LLM) can usually evade pattern matching by hiding the
dangerous call behind one level of indirection:

    * `getattr(__builtins__, 'eval')('payload')` instead of `eval(...)`
    * `importlib.import_module('subprocess')` instead of `import subprocess`
    * `__builtins__['__import__']('os')` via subscript
    * `globals()['eval']('...')` via dict lookup
    * String-built imports: `__import__('s' + 'ubprocess')`

Each test here triggers one bypass. They are RED until
`validate_generated_tool_code` is strengthened to deny these patterns
or until we adopt a deny-by-default model (block any reference to
__builtins__, __import__, getattr-on-modules, importlib, globals,
locals, vars).
"""

from __future__ import annotations

import pytest

from autoagent.errors import ToolValidationError
from autoagent.sandbox import validate_generated_tool_code


def _wrap(body: str) -> str:
    indented = "\n".join("    " + line for line in body.strip().splitlines())
    return f"def run(args, context):\n{indented}\n"


class TestKnownBypasses:
    def test_getattr_builtins_eval_blocked(self) -> None:
        code = _wrap("getattr(__builtins__, 'eval')('1+1')")
        with pytest.raises(ToolValidationError):
            validate_generated_tool_code(code)

    def test_builtins_subscript_blocked(self) -> None:
        code = _wrap("__builtins__['__import__']('os')")
        with pytest.raises(ToolValidationError):
            validate_generated_tool_code(code)

    def test_dunder_import_via_name_blocked(self) -> None:
        code = _wrap("x = __import__('subprocess')")
        with pytest.raises(ToolValidationError):
            validate_generated_tool_code(code)

    def test_importlib_import_module_blocked(self) -> None:
        code = "import importlib\n" + _wrap("importlib.import_module('subprocess')")
        with pytest.raises(ToolValidationError):
            validate_generated_tool_code(code)

    def test_globals_lookup_to_call_eval_blocked(self) -> None:
        code = _wrap("globals()['eval']('1+1')")
        with pytest.raises(ToolValidationError):
            validate_generated_tool_code(code)

    def test_dunder_builtins_reference_blocked(self) -> None:
        # Even just referencing __builtins__ as a name is suspicious;
        # generated tools have no legitimate reason to touch it.
        code = _wrap("b = __builtins__")
        with pytest.raises(ToolValidationError):
            validate_generated_tool_code(code)

    def test_vars_or_locals_lookup_blocked(self) -> None:
        code = _wrap("vars()['eval']('1+1')")
        with pytest.raises(ToolValidationError):
            validate_generated_tool_code(code)

    def test_string_concat_import_blocked(self) -> None:
        # Even if we cannot fully prove the runtime target, __import__ as a
        # name MUST be banned (it has no legitimate use in a tool).
        code = _wrap("__import__('s' + 'ubprocess')")
        with pytest.raises(ToolValidationError):
            validate_generated_tool_code(code)


class TestOsAndIntrospectionEscapes:
    """Hardening pass: close the `os`/`sys` module escape and the classic
    object-introspection escape `().__class__.__subclasses__()`."""

    @pytest.mark.parametrize("module", ["os", "sys", "posix", "nt", "pty"])
    def test_os_family_import_blocked(self, module: str) -> None:
        code = "import " + module + "\n" + _wrap("pass")
        with pytest.raises(ToolValidationError, match=f"not allowed: {module}"):
            validate_generated_tool_code(code)

    def test_os_file_op_blocked_even_with_filesystem_permission(self) -> None:
        # `os` is banned outright (not gated by filesystem.*) because it ALSO
        # exposes exec/spawn/fork. Tools must use pathlib for file access.
        code = "import os\n" + _wrap("os.remove('/etc/passwd')")
        with pytest.raises(ToolValidationError, match="not allowed: os"):
            validate_generated_tool_code(code, permissions=["filesystem.write"])

    @pytest.mark.parametrize("call_name", ["execv", "execve", "spawnv", "posix_spawn", "startfile", "fork"])
    def test_process_spawn_call_blocked(self, call_name: str) -> None:
        # Caught by call-NAME even when reached via an alias/attribute, so a
        # smuggled `os` reference can't launch a program.
        code = _wrap(f"o.{call_name}('x')")
        with pytest.raises(ToolValidationError, match="Process-spawning call is not allowed"):
            validate_generated_tool_code(code)

    def test_subclasses_introspection_escape_blocked(self) -> None:
        # The classic escape: walk the object graph to a dangerous class with
        # NO import at all.
        code = _wrap("cls = ().__class__.__bases__[0].__subclasses__()")
        with pytest.raises(ToolValidationError):
            validate_generated_tool_code(code)

    @pytest.mark.parametrize(
        "attr", ["__class__", "__bases__", "__subclasses__", "__mro__", "__globals__", "__dict__"]
    )
    def test_introspection_dunder_attr_blocked(self, attr: str) -> None:
        code = _wrap(f"x = (1).{attr}")
        with pytest.raises(ToolValidationError, match="dangerous attribute"):
            validate_generated_tool_code(code)

    def test_function_globals_escape_blocked(self) -> None:
        code = _wrap("g = run.__globals__")
        with pytest.raises(ToolValidationError, match="dangerous attribute"):
            validate_generated_tool_code(code)

    def test_legit_pure_tool_still_accepted(self) -> None:
        # Regression guard: the hardening must NOT reject ordinary tool code.
        code = _wrap("total = sum(args.get('xs', []))\nreturn {'total': total, 'n': len(args)}")
        validate_generated_tool_code(code)  # must not raise
