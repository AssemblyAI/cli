"""The convention-based command registry (aai_cli/command_registry.py).

Discovery is exercised against the real ``aai_cli.commands`` package; the
rejection paths use fake modules injected into ``sys.modules`` so a module that
forgets (or misdeclares) its ``SPEC`` is proven to fail loudly at import time
rather than silently dropping out of the CLI.
"""

from __future__ import annotations

import sys
import types

import pytest
import typer

from aai_cli import command_registry, help_panels


def _fake_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


_VALID_SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION, order=10, commands=("fake",)
)


def test_module_without_spec_is_rejected(monkeypatch):
    name = "aai_cli.commands._fake_missing_spec"
    monkeypatch.setitem(sys.modules, name, _fake_module(name, app=typer.Typer()))
    with pytest.raises(TypeError, match="SPEC = CommandModuleSpec"):
        command_registry._load(name)


def test_module_with_unknown_panel_is_rejected(monkeypatch):
    bad_spec = command_registry.CommandModuleSpec(
        panel="No Such Panel", order=10, commands=("fake",)
    )
    name = "aai_cli.commands._fake_bad_panel"
    monkeypatch.setitem(sys.modules, name, _fake_module(name, SPEC=bad_spec, app=typer.Typer()))
    with pytest.raises(RuntimeError, match="unknown help panel 'No Such Panel'"):
        command_registry._load(name)


def test_module_without_typer_app_is_rejected(monkeypatch):
    name = "aai_cli.commands._fake_no_app"
    monkeypatch.setitem(sys.modules, name, _fake_module(name, SPEC=_VALID_SPEC))
    with pytest.raises(TypeError, match=r"app = typer\.Typer"):
        command_registry._load(name)


def test_load_returns_spec_and_app(monkeypatch):
    name = "aai_cli.commands._fake_valid"
    sub_app = typer.Typer()
    monkeypatch.setitem(sys.modules, name, _fake_module(name, SPEC=_VALID_SPEC, app=sub_app))
    registered = command_registry._load(name)
    assert registered.spec is _VALID_SPEC
    assert registered.app is sub_app


def test_discovery_renders_panels_contiguously_in_panel_order():
    registered = command_registry.discover()
    ranks = [help_panels.PANEL_ORDER.index(reg.spec.panel) for reg in registered]
    assert ranks == sorted(ranks)  # panels stay contiguous, in PANEL_ORDER order
    assert {reg.spec.panel for reg in registered} == set(help_panels.PANEL_ORDER)


def test_command_order_lists_every_declared_command_exactly_once():
    registered = command_registry.discover()
    order = command_registry.command_order(registered)
    assert len(order) == len(set(order))  # no module may claim another's command name
    assert set(order) == {name for reg in registered for name in reg.spec.commands}
    assert order[0] == "onboard"  # Quick Start renders first
