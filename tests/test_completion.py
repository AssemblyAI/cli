from __future__ import annotations

import typer

from aai_cli.agent.voices import VOICES, complete_voice
from aai_cli.llm import KNOWN_MODELS, complete_model
from aai_cli.main import app


def test_shell_completion_is_enabled():
    # add_completion=True registers Typer's --install-completion on the root command.
    # Introspect the Click command rather than rendered --help text, which wraps at
    # narrow terminal widths (and rendered differently under CI, hiding the flag).
    command = typer.main.get_command(app)
    option_names = {opt for param in command.params for opt in param.opts}
    assert "--install-completion" in option_names


def test_complete_model_filters_by_prefix():
    suggestions = complete_model("gpt")
    assert suggestions  # at least one gpt-* model is known
    assert all(m.startswith("gpt") for m in suggestions)


def test_complete_model_empty_prefix_returns_all_known():
    assert complete_model("") == list(KNOWN_MODELS)


def test_complete_model_unknown_prefix_returns_nothing():
    assert complete_model("no-such-model") == []


def test_complete_voice_filters_by_prefix():
    prefix = VOICES[0][:2]
    suggestions = complete_voice(prefix)
    assert suggestions
    assert all(v.startswith(prefix) for v in suggestions)


def test_complete_voice_empty_prefix_returns_all():
    assert complete_voice("") == VOICES
