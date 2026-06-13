"""Convention-based registration for command modules.

Every module under ``aai_cli.commands`` declares a module-level
``SPEC = CommandModuleSpec(...)`` describing how it plugs into the root app
(which help panel it renders under, its rank within that panel, the top-level
command names it contributes, and — for named sub-groups like ``transcripts`` —
the ``add_typer`` name). ``main.py`` discovers and registers every module via
:func:`discover`, so adding a command is purely additive: drop a new module in
``aai_cli/commands/`` with a ``SPEC`` and it is imported, registered, ordered,
and covered by the help-snapshot partition without editing any shared file.

A module that forgets (or misdeclares) its ``SPEC`` fails loudly at import time
rather than silently dropping out of the CLI.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

import typer

from aai_cli import help_panels


@dataclass(frozen=True)
class CommandModuleSpec:
    """How a module under ``aai_cli.commands`` plugs into the root app."""

    # One of the help_panels.PANEL_ORDER headings its commands render under.
    panel: str
    # Rank within the panel. Sparse by convention (10, 20, 30, …) so a new command
    # slots between two existing ones without renumbering its neighbors.
    order: int
    # Top-level command names this module contributes, in display order. Most
    # modules contribute one; merged multi-command modules (e.g. login/logout/whoami)
    # list each name so `assembly --help` ordering stays fully derived.
    commands: tuple[str, ...]
    # ``app.add_typer(name=...)`` for named sub-groups (``assembly keys list`` style);
    # None for merged modules whose commands sit directly on the root app.
    group_name: str | None = None


@dataclass(frozen=True)
class RegisteredModule:
    """A discovered command module: its declared spec plus its Typer sub-app."""

    spec: CommandModuleSpec
    app: typer.Typer


def _load(module_name: str) -> RegisteredModule:
    """Import one command module and validate its registration convention."""
    module = importlib.import_module(module_name)
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, CommandModuleSpec):
        raise TypeError(
            f"{module_name} must declare a module-level SPEC = CommandModuleSpec(...) "
            "so it can be registered (see aai_cli/command_registry.py)."
        )
    if spec.panel not in help_panels.PANEL_ORDER:
        raise RuntimeError(
            f"{module_name} declares unknown help panel {spec.panel!r}; "
            "use one of help_panels.PANEL_ORDER."
        )
    sub_app = getattr(module, "app", None)
    if not isinstance(sub_app, typer.Typer):
        raise TypeError(f"{module_name} must expose a module-level `app = typer.Typer(...)`.")
    return RegisteredModule(spec=spec, app=sub_app)


def discover() -> tuple[RegisteredModule, ...]:
    """Every command module under ``aai_cli.commands``, in display order.

    Display order is (panel rank, module's order, command names): panels render in
    ``help_panels.PANEL_ORDER`` and stay contiguous by construction.
    """
    from aai_cli import commands as commands_pkg

    panel_rank = {panel: rank for rank, panel in enumerate(help_panels.PANEL_ORDER)}
    registered = [
        _load(f"{commands_pkg.__name__}.{info.name}")
        for info in pkgutil.iter_modules(commands_pkg.__path__)
        if not info.name.startswith("_")
    ]
    registered.sort(key=lambda reg: (panel_rank[reg.spec.panel], reg.spec.order, reg.spec.commands))
    return tuple(registered)


def command_order(registered: tuple[RegisteredModule, ...]) -> tuple[str, ...]:
    """Top-level command names in display order (drives `assembly --help`)."""
    return tuple(name for module in registered for name in module.spec.commands)
