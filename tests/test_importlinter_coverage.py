"""Guard: every top-level module is covered by an architecture contract.

The architecture is enforced by `.importlinter` as a layered stack
(``commands > app > ui > core``, contract 1) plus the vertical feature slices
that sit beside it (contract 2 forbids them from importing the command layer).
A newly added top-level module that belongs to neither would be silently
*uncovered* — nothing would fail until the architecture had already drifted
(this is exactly how `onboard` once grew imports of command modules unnoticed).

This test partitions the filesystem against the contracts: every top-level
entry under ``aai_cli/`` must be either a declared layer (contract 1), a
declared feature slice (contract 2), or one of the framework-glue modules that
legitimately assemble the command layer from above. A stray module fails
loudly until it is placed.

Contract 3 needs no guard: it wildcards over ``aai_cli.commands.*``.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import aai_cli

# The CLI framework glue lives at the package root, *above* the command layer:
# main builds the app, command_registry discovers/registers the command apps,
# help_panels/options are the shared command-definition support they pull in.
# They legitimately import aai_cli.commands, so they sit outside the layered
# stack and the feature-slice list.
EXEMPT = {
    "aai_cli.main",
    "aai_cli.command_registry",
    "aai_cli.help_panels",
    "aai_cli.options",
}

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


def _parser() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(_REPO_ROOT / ".importlinter")
    return parser


def _layer_modules(parser: configparser.ConfigParser) -> set[str]:
    container = parser["importlinter:contract:1"]["containers"].split()[0]
    layers = parser["importlinter:contract:1"]["layers"].split()
    return {f"{container}.{layer}" for layer in layers}


def _feature_slices(parser: configparser.ConfigParser) -> set[str]:
    return set(parser["importlinter:contract:2"]["source_modules"].split())


def test_every_top_level_module_is_placed_by_a_contract():
    parser = _parser()
    layers = _layer_modules(parser)
    features = _feature_slices(parser)
    covered = layers | features | EXEMPT
    actual = _top_level_modules()

    missing = sorted(actual - covered)
    assert missing == [], (
        f"top-level module(s) {missing} are not placed by any .importlinter contract; "
        "move them into a layer package (commands/app/ui/core), add them to contract 2's "
        "feature slices, or to EXEMPT here if they are framework glue that may import commands"
    )

    stale = sorted((layers | features) - actual)
    assert stale == [], f".importlinter names module(s) that no longer exist: {stale}"
