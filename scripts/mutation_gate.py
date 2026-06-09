"""Diff-scoped mutation testing gate.

Coverage proves a changed line *ran*; it can't prove a test would *fail* if that
line broke. This gate mutates only the lines changed versus the compare branch
(default origin/main), reruns just the tests that cover each mutant (read from
per-test coverage contexts in ``.coverage``), and fails if any mutant survives —
i.e. the suite still passed with the code deliberately broken.

Run after the pytest step that wrote ``.coverage`` with ``--cov-context=test``
(see scripts/check.sh). Mark a genuinely-equivalent or intentionally-unasserted
line ``# pragma: no mutate`` to exclude it.

Usage: python scripts/mutation_gate.py [compare-branch]
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import coverage

_PKG = "aai_cli"
_TEMPLATES = "aai_cli/init/templates"
_DEFAULT_MARKERS = "not e2e and not install and not install_script"
_TEST_TIMEOUT = 120  # seconds; a mutant that hangs (e.g. a flipped loop guard) counts killed
_SUPPRESS = "pragma: no mutate"

_FILE_HEADER = re.compile(r"^\+\+\+ b/(?P<path>.+)$")
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")

# Each operator maps to its strongest single mutation (negation / boundary flip),
# which rarely produces an equivalent mutant. `not`-removal and statement deletion
# are deliberately omitted (they need parent rewiring and breed equivalents).
_COMPARE_SWAP: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.Gt: ast.LtE,
    ast.LtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}
_BOOL_SWAP: dict[type[ast.boolop], type[ast.boolop]] = {ast.And: ast.Or, ast.Or: ast.And}
_BINOP_SWAP: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
}


@dataclass
class _Mutant:
    label: str
    linenos: frozenset[int]
    apply: Callable[[], None]
    undo: Callable[[], None]


def _git(*args: str) -> tuple[int, str]:
    proc = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout


def _merge_base(compare: str) -> str | None:
    code, out = _git("merge-base", compare, "HEAD")
    return out.strip() if code == 0 else None


def _changed_lines(base: str) -> dict[Path, set[int]]:
    """Map each changed aai_cli/*.py file to the set of added line numbers."""
    _, out = _git("diff", "-U0", base, "--", _PKG)
    result: dict[Path, set[int]] = {}
    target: set[int] | None = None
    for line in out.splitlines():
        header = _FILE_HEADER.match(line)
        if header:
            path = header.group("path")
            keep = path.endswith(".py") and not path.startswith(_TEMPLATES)
            target = result.setdefault(Path(path), set()) if keep else None
            continue
        hunk = _HUNK_HEADER.match(line)
        if hunk and target is not None:
            start = int(hunk.group("start"))
            count = int(hunk.group("count") or "1")
            target.update(range(start, start + count))
    return {path: lines for path, lines in result.items() if lines}


def _swap_in_list(
    ops: list[ast.cmpop], index: int, old: ast.cmpop, new: type[ast.cmpop]
) -> tuple[str, Callable[[], None], Callable[[], None]]:
    def apply() -> None:
        ops[index] = new()

    def undo() -> None:
        ops[index] = old

    return f"{type(old).__name__} -> {new.__name__}", apply, undo


def _swap_boolop(
    node: ast.BoolOp, new: type[ast.boolop]
) -> tuple[str, Callable[[], None], Callable[[], None]]:
    old = node.op

    def apply() -> None:
        node.op = new()

    def undo() -> None:
        node.op = old

    return f"{type(old).__name__} -> {new.__name__}", apply, undo


def _swap_arith(
    node: ast.BinOp | ast.AugAssign, new: type[ast.operator]
) -> tuple[str, Callable[[], None], Callable[[], None]]:
    old = node.op

    def apply() -> None:
        node.op = new()

    def undo() -> None:
        node.op = old

    return f"{type(old).__name__} -> {new.__name__}", apply, undo


def _set_const(
    node: ast.Constant, new: object
) -> tuple[str, Callable[[], None], Callable[[], None]]:
    old = node.value

    def apply() -> None:
        node.value = new

    def undo() -> None:
        node.value = old

    return f"{old!r} -> {new!r}", apply, undo


_Mutation = tuple[str, Callable[[], None], Callable[[], None]]


def _compare_mutations(node: ast.Compare) -> Iterator[_Mutation]:
    for index, op in enumerate(node.ops):
        new_cmp = _COMPARE_SWAP.get(type(op))
        if new_cmp is not None:
            yield _swap_in_list(node.ops, index, op, new_cmp)


def _constant_mutations(node: ast.Constant) -> Iterator[_Mutation]:
    if isinstance(node.value, bool):
        yield _set_const(node, not node.value)
    elif isinstance(node.value, int | float):
        yield _set_const(node, node.value + 1)


def _node_mutations(node: ast.AST) -> Iterator[_Mutation]:
    if isinstance(node, ast.Compare):
        yield from _compare_mutations(node)
    elif isinstance(node, ast.BoolOp):
        new_bool = _BOOL_SWAP.get(type(node.op))
        if new_bool is not None:
            yield _swap_boolop(node, new_bool)
    elif isinstance(node, ast.BinOp | ast.AugAssign):
        new_bin = _BINOP_SWAP.get(type(node.op))
        if new_bin is not None:
            yield _swap_arith(node, new_bin)
    elif isinstance(node, ast.Constant):
        yield from _constant_mutations(node)


def _collect(path: Path, changed: set[int]) -> tuple[ast.Module, str, list[_Mutant]]:
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    mutants: list[_Mutant] = []
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", 0)
        if lineno not in changed:
            continue
        end = getattr(node, "end_lineno", lineno) or lineno
        span = frozenset(range(lineno, end + 1))
        # Scan the whole statement span, not just its first line: a `# pragma: no
        # mutate` can land on any line the formatter wrapped the statement across.
        if any(_SUPPRESS in lines[ln - 1] for ln in span):
            continue
        for desc, apply, undo in _node_mutations(node):
            mutants.append(_Mutant(f"{path}:{lineno}: {desc}", span, apply, undo))
    return tree, src, mutants


def _covering_tests(data: coverage.CoverageData, path: Path, linenos: frozenset[int]) -> list[str]:
    by_line = data.contexts_by_lineno(str(path.resolve()))
    nodeids: set[str] = set()
    for lineno in linenos:
        for context in by_line.get(lineno, ()):
            nodeid = context.split("|", 1)[0]
            if nodeid:
                nodeids.add(nodeid)
    return sorted(nodeids)


def _run_tests(nodeids: list[str]) -> bool:
    """True if the selected tests fail (the mutant is killed)."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:randomly",
        "--no-cov",
        "-x",
        "--no-header",
    ]
    cmd += nodeids if nodeids else ["-m", _DEFAULT_MARKERS, "tests"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=_TEST_TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        return True
    return proc.returncode != 0


def _invalidate_bytecode(path: Path) -> None:
    """Drop the module's cached ``.pyc`` so the test subprocess recompiles from the
    source we just wrote.

    Consecutive mutants ``ast.unparse`` to files that differ by a single token, so
    they're usually byte-for-byte the same length and can be written within the same
    mtime-second. CPython's default timestamp-based cache validates a ``.pyc`` by
    exact (mtime, size) match, so without this it can serve the previous mutant's
    (or the original's) bytecode and run *unmutated* code — a false survivor.
    """
    cached = importlib.util.cache_from_source(str(path))
    with contextlib.suppress(OSError):
        Path(cached).unlink()


def _survives(
    path: Path, tree: ast.Module, src: str, mutant: _Mutant, data: coverage.CoverageData
) -> bool:
    # Safety: only ever rewrite files inside the package under test. The file list
    # comes from `git diff`, so this can't normally escape, but guard against a path
    # that resolves outside aai_cli/ before we write to it.
    if not path.resolve().is_relative_to(Path(_PKG).resolve()):
        raise ValueError(f"refusing to mutate a file outside {_PKG}/: {path}")
    mutant.apply()
    try:
        path.write_text(ast.unparse(tree), encoding="utf-8")
        _invalidate_bytecode(path)
        killed = _run_tests(_covering_tests(data, path, mutant.linenos))
    finally:
        path.write_text(src, encoding="utf-8")
        _invalidate_bytecode(path)
        mutant.undo()
    return not killed


def _report(total: int, survivors: list[str]) -> int:
    sys.stdout.write(f"   tested {total} mutant(s) on changed lines\n")
    if not survivors:
        return 0
    sys.stdout.write("Surviving mutants (no test fails when this line is broken):\n")
    for label in survivors:
        sys.stdout.write(f"  - {label}\n")
    sys.stdout.write("Add an assertion that kills it, or mark the line `# pragma: no mutate`.\n")
    return 1


def main() -> int:
    compare = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    base = _merge_base(compare)
    if base is None:
        sys.stdout.write(f"   {compare} not found; skipping mutation gate (CI provides it)\n")
        return 0
    changed = _changed_lines(base)
    if not changed:
        sys.stdout.write("   no changed aai_cli lines to mutate\n")
        return 0
    data = coverage.CoverageData()
    data.read()
    survivors: list[str] = []
    total = 0
    for path, lines in sorted(changed.items()):
        tree, src, mutants = _collect(path, lines)
        for mutant in mutants:
            total += 1
            if _survives(path, tree, src, mutant, data):
                survivors.append(mutant.label)
    return _report(total, survivors)


if __name__ == "__main__":
    raise SystemExit(main())
