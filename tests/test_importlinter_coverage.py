"""Guard: the import-linter contracts must cover every module in the package.

Contract 1 ("core modules do not import command modules") enumerates its source
modules by name, so a newly added module would be silently *uncovered* — worse
than a merge conflict, because nothing fails until the architecture has already
drifted (this is exactly how `onboard` once grew imports of command modules
unnoticed). This test compares the enumerated list against the filesystem so a
new top-level module fails loudly until it is added to `.importlinter` (or to
the deliberate exemption list below).

Contract 2 needs no guard: it wildcards over ``aai_cli.commands.*``.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import aai_cli

# Modules that legitimately import aai_cli.commands and so are deliberately
# outside contract 1: main registers the discovered command apps,
# command_registry performs that discovery, and commands is the layer itself.
EXEMPT = {"aai_cli.main", "aai_cli.command_registry", "aai_cli.commands"}

_REPO_ROOT = Path(aai_cli.__file__).resolve().parent.parent


def _top_level_modules() -> set[str]:
    package_dir = Path(aai_cli.__file__).resolve().parent
    modules: set[str] = set()
    for path in package_dir.iterdir():
        if path.name.startswith("_"):
            continue
        if path.is_dir() and (path / "__init__.py").exists():
            modules.add(f"aai_cli.{path.name}")
        elif path.suffix == ".py":
            modules.add(f"aai_cli.{path.stem}")
    return modules


def _contract_one_sources() -> set[str]:
    parser = configparser.ConfigParser()
    parser.read(_REPO_ROOT / ".importlinter")
    return set(parser["importlinter:contract:1"]["source_modules"].split())


def test_every_core_module_is_covered_by_contract_one():
    listed = _contract_one_sources()
    actual = _top_level_modules()
    missing = sorted(actual - listed - EXEMPT)
    assert missing == [], (
        f"new top-level module(s) {missing} are not covered by .importlinter contract 1; "
        "add them to source_modules (or to EXEMPT here if they may import commands)"
    )
    stale = sorted(listed - actual)
    assert stale == [], f".importlinter contract 1 lists module(s) that no longer exist: {stale}"
