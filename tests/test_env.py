"""`aai_cli.core.env` — the single chokepoint for raw environment access."""

from __future__ import annotations

import os

from aai_cli.core import env


def test_get_returns_value_or_default(monkeypatch):
    monkeypatch.setenv("AAI_TEST_VAR", "present")
    monkeypatch.delenv("AAI_TEST_MISSING", raising=False)
    assert env.get("AAI_TEST_VAR") == "present"
    assert env.get("AAI_TEST_MISSING") is None
    assert env.get("AAI_TEST_MISSING", "fallback") == "fallback"


def test_child_env_overlays_without_mutating(monkeypatch):
    monkeypatch.setattr(os, "environ", {"KEEP": "1", "PORT": "old"})
    child = env.child_env(PORT="9999")
    # Overrides win over the inherited value, and inherited keys survive.
    assert child == {"KEEP": "1", "PORT": "9999"}
    # The parent environment is untouched (it's a copy, not a mutation).
    assert os.environ == {"KEEP": "1", "PORT": "old"}


def test_force_color_sets_force_clears_no_color(monkeypatch):
    monkeypatch.setattr(os, "environ", {"NO_COLOR": "1"})
    env.force_color()
    assert os.environ == {"FORCE_COLOR": "1"}


def test_force_color_is_safe_when_no_color_absent(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    env.force_color()  # pop must tolerate a missing NO_COLOR
    assert os.environ == {"FORCE_COLOR": "1"}


def test_disable_color_sets_no_color_clears_force(monkeypatch):
    monkeypatch.setattr(os, "environ", {"FORCE_COLOR": "1"})
    env.disable_color()
    assert os.environ == {"NO_COLOR": "1"}


def test_disable_color_is_safe_when_force_absent(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    env.disable_color()  # pop must tolerate a missing FORCE_COLOR
    assert os.environ == {"NO_COLOR": "1"}
