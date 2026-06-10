"""Whole-file mutation sweep — the diff-scoped gate's repo-wide companion.

``scripts/mutation_gate.py`` only mutates lines changed versus a branch, so code
that predates the gate is never held to its bar. This sweeps EVERY eligible line
of the given files (or the whole package) and reports the mutants that survive —
i.e. the suite still passes with the line deliberately broken — so you can add an
assertion that kills each one (or mark a genuinely-equivalent line
``# pragma: no mutate``). It reuses the gate's own mutation/kill engine, so a
survivor here is a survivor there.

Usage::

    # 1. Refresh per-test coverage contexts the sweep reads from .coverage:
    uv run pytest -q -n auto --timeout=60 --cov=aai_cli --cov-branch \
        --cov-context=test --cov-report=
    # 2. Sweep specific files (or omit paths to sweep the whole package):
    uv run python scripts/mutation_sweep.py aai_cli/config.py
    uv run python scripts/mutation_sweep.py

Pass ``--timeout`` to the pytest step above: the default suite has no per-test
timeout (it is opt-in; see pyproject), and a deadlocked test would otherwise wedge
the whole run instead of failing fast.

Exit status is 1 if any real survivor is found, else 0. Lines whose mutants have
no covering test are reported separately as UNCOVERED (not failed): coverage
attributes import-time evaluated defaults to no test, so that bucket needs a
manual look rather than blind action.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import coverage

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent / "aai_cli"
_TEMPLATES = _PKG / "init" / "templates"


def _load_gate() -> ModuleType:
    # ModuleType attribute access is dynamic, so reusing the gate's private helpers
    # (_collect/_covering_tests/_survives) below needs no type-checker escape hatch.
    spec = importlib.util.spec_from_file_location("mutation_gate", _HERE / "mutation_gate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/mutation_gate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mutation_gate"] = module
    spec.loader.exec_module(module)
    return module


def _package_files() -> list[Path]:
    return sorted(p for p in _PKG.rglob("*.py") if _TEMPLATES not in p.parents)


def _sweep_file(
    mg: ModuleType, path: Path, data: coverage.CoverageData
) -> tuple[int, list[str], list[str]]:
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    tree, src, mutants = mg._collect(path, set(range(1, line_count + 1)))
    survivors: list[str] = []
    uncovered: list[str] = []
    for mutant in mutants:
        if not mg._covering_tests(data, path, mutant.linenos):
            uncovered.append(mutant.label)
        elif mg._survives(path, tree, src, mutant, data):
            survivors.append(mutant.label)
    return len(mutants), survivors, uncovered


def main() -> int:
    mg = _load_gate()
    args = [Path(a) for a in sys.argv[1:]] or _package_files()
    data = coverage.CoverageData()
    data.read()
    total = 0
    all_survivors: list[str] = []
    for path in args:
        tested, survivors, uncovered = _sweep_file(mg, path, data)
        total += tested
        all_survivors += survivors
        sys.stdout.write(f"\n=== {path} : {tested} mutants ===\n")
        for label in survivors:
            sys.stdout.write(f"  SURVIVES  {label}\n")
        for label in uncovered:
            sys.stdout.write(f"  uncovered {label}\n")
        if not survivors and not uncovered:
            sys.stdout.write("  clean\n")
        sys.stdout.flush()
    sys.stdout.write(f"\nTOTAL {total} mutant(s); {len(all_survivors)} surviving\n")
    return 1 if all_survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
