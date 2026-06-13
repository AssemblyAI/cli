from __future__ import annotations

import ast
import sys
from pathlib import Path

# Docstring-coverage ratchet for the shipped package's public API, replacing `interrogate`
# (which can't parse this codebase's PEP 695 generics, e.g. `def emit[T](...)`). Public =
# the module plus every non-underscore class/function/method. The FLOOR is set at the
# current level and only ever ratchets up: a change may not drop public-API documentation
# below it, but nobody is forced to backfill the existing gap in one go. Raising FLOOR as
# coverage climbs is a deliberate, reviewed edit here — the same model as a coverage gate.
FLOOR = 79.0

PACKAGE = Path(__file__).resolve().parent.parent / "aai_cli"

_Def = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _public_nodes(tree: ast.Module) -> list[ast.AST]:
    """The module plus its public API surface: top-level functions/classes and the
    methods of public classes. Functions nested inside functions (command-body
    closures, callback handlers) are implementation detail, not public API, so they
    are not counted — only definitions reachable through the module or a class body."""
    nodes: list[ast.AST] = [tree]
    for top in tree.body:
        if isinstance(top, _Def) and not top.name.startswith("_"):
            nodes.append(top)
            if isinstance(top, ast.ClassDef):
                nodes.extend(
                    m for m in top.body if isinstance(m, _Def) and not m.name.startswith("_")
                )
    return nodes


def _coverage() -> tuple[int, int, list[str]]:
    total = documented = 0
    missing: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "templates" in path.parts or path.name == "_version.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _public_nodes(tree):
            total += 1
            if ast.get_docstring(node):
                documented += 1
            else:
                name = getattr(node, "name", "<module>")
                missing.append(f"{path.relative_to(PACKAGE.parent)}:{name}")
    return documented, total, missing


def main() -> int:
    documented, total, missing = _coverage()
    pct = 100.0 * documented / total if total else 100.0
    if pct + 1e-9 >= FLOOR:
        sys.stdout.write(f"Public docstring coverage {pct:.1f}% (>= floor {FLOOR}%).\n")
        return 0
    sys.stdout.write(
        f"Public docstring coverage {pct:.1f}% fell below the {FLOOR}% floor "
        f"({documented}/{total}). Add docstrings to public APIs you touched:\n"
    )
    for item in missing[:20]:
        sys.stdout.write(f"  {item}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
