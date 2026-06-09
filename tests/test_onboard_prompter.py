from __future__ import annotations

import pytest

from aai_cli.errors import UsageError
from aai_cli.onboard.prompter import InteractivePrompter, NonInteractivePrompter, WizardCancelled


def test_noninteractive_section_and_note() -> None:
    p = NonInteractivePrompter()
    p.section("Setup")  # exercises NonInteractivePrompter.section()
    p.note("a hint")  # exercises NonInteractivePrompter.note()


def test_noninteractive_confirm_returns_default() -> None:
    p = NonInteractivePrompter()
    assert p.confirm("Run setup?", default=True) is True
    assert p.confirm("Run setup?", default=False) is False


def test_noninteractive_select_returns_default_or_first() -> None:
    p = NonInteractivePrompter()
    options = [("a", "Option A"), ("b", "Option B")]
    assert p.select("Pick", options) == "a"
    assert p.select("Pick", options, default="b") == "b"


def test_noninteractive_text_requires_default() -> None:
    p = NonInteractivePrompter()
    assert p.text("Name?", default="x") == "x"
    with pytest.raises(UsageError):
        p.text("Name?")


def test_interactive_section_and_note() -> None:
    p = InteractivePrompter()
    p.section("Heading")  # exercises section()
    p.note("a hint")  # exercises note()


def test_interactive_confirm_delegates_to_typer(monkeypatch: pytest.MonkeyPatch) -> None:
    import aai_cli.onboard.prompter as pm

    monkeypatch.setattr(pm.typer, "confirm", lambda *a, **k: True)
    assert InteractivePrompter().confirm("ok?", default=True) is True


def test_interactive_text_delegates_to_typer(monkeypatch: pytest.MonkeyPatch) -> None:
    import aai_cli.onboard.prompter as pm

    monkeypatch.setattr(pm.typer, "prompt", lambda *a, **k: "typed")
    assert InteractivePrompter().text("name", default="d") == "typed"


def test_interactive_select_returns_chosen_value(monkeypatch: pytest.MonkeyPatch) -> None:
    import questionary

    class _Q:
        def ask(self) -> str:
            return "b"

    monkeypatch.setattr(questionary, "select", lambda *a, **k: _Q())
    result = InteractivePrompter().select("Pick", [("a", "A"), ("b", "B")], default="a")
    assert result == "b"


def test_interactive_select_cancel_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import questionary

    class _QNone:
        def ask(self) -> None:
            return None

    monkeypatch.setattr(questionary, "select", lambda *a, **k: _QNone())
    with pytest.raises(WizardCancelled):
        InteractivePrompter().select("Pick", [("a", "A")])
