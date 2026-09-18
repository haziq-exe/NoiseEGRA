#!/usr/bin/env python
"""No function may import a name the module already imports at the top.

Python decides a name is local to a function if the function assigns to it
*anywhere*, and an import is an assignment. So

    from transformers import LogitsProcessorList   # module level, line 6
    ...
    def generate_with_orthogonal_steering(...):
        if probes:
            from transformers import LogitsProcessorList   # <- makes it local
            processors = LogitsProcessorList(probes)
        ...
        processors = LogitsProcessorList([probe])          # <- UnboundLocalError

raises UnboundLocalError on the last line whenever `probes` is empty, even
though the module-level import is right there. That is not hypothetical: it
reached a Kaggle run, failed both shards on the first story, and cost five
minutes of GPU on three accounts' worth of queued work. The offline tests could
not catch it because they never enter the generation path.

This catches the whole class statically, in milliseconds, without importing
anything or touching a GPU.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def shadowing_imports(path: Path):
    """(function, name) for every function-local import of a module-level name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = module_level_names(tree)
    if not top:
        return []

    found = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    if name in top:
                        found.append((func.name, name))
    return found


def test_no_function_reimports_a_module_level_name() -> None:
    offenders = []
    for path in sorted((ROOT / "noiseegra").rglob("*.py")):
        for func, name in shadowing_imports(path):
            offenders.append(f"{path.relative_to(ROOT)}: {func}() re-imports {name!r}")

    assert not offenders, (
        "a function imports a name the module already imports, which makes every "
        "reference to it in that function local and can raise UnboundLocalError:\n  "
        + "\n  ".join(offenders)
    )
    print(f"  [PASS] no function re-imports a module-level name "
          f"({len(list((ROOT / 'noiseegra').rglob('*.py')))} files)")


def test_the_check_would_have_caught_the_real_one() -> None:
    """The pattern that failed on Kaggle is detected."""
    import tempfile

    source = (
        "from transformers import LogitsProcessorList\n"
        "\n"
        "def generate(probes):\n"
        "    if probes:\n"
        "        from transformers import LogitsProcessorList\n"
        "        return LogitsProcessorList(probes)\n"
        "    return LogitsProcessorList([])\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(source)
        tmp = Path(fh.name)
    try:
        found = shadowing_imports(tmp)
        assert found == [("generate", "LogitsProcessorList")], found
    finally:
        tmp.unlink()
    print("  [PASS] the check detects the pattern that failed on Kaggle")


if __name__ == "__main__":
    test_the_check_would_have_caught_the_real_one()
    test_no_function_reimports_a_module_level_name()
    print("\nok")
