from __future__ import annotations

import pytest

from aai_cli.errors import UsageError
from aai_cli.onboard.prompter import NonInteractivePrompter


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
