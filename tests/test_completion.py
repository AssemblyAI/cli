from __future__ import annotations

import typer

from aai_cli.agent.voices import VOICE_NAMES, complete_voice
from aai_cli.core import choices
from aai_cli.core.llm import KNOWN_MODELS, complete_model
from aai_cli.main import app


def test_complete_prefix_keeps_only_matching_options():
    # The shared completion body that complete_voice/complete_model delegate to: it
    # filters to the prefix (dropping "banana") rather than echoing every option back.
    assert choices.complete_prefix(["apple", "apricot", "banana"], "ap") == ["apple", "apricot"]


def test_shell_completion_is_enabled():
    # add_completion=True registers Typer's --install-completion on the root command.
    # Introspect the Click command rather than rendered --help text, which wraps at
    # narrow terminal widths (and rendered differently under CI, hiding the flag).
    command = typer.main.get_command(app)
    option_names = {opt for param in command.params for opt in param.opts}
    assert "--install-completion" in option_names


def test_show_completion_help_is_trimmed_and_scoped():
    # main.py trims the long built-in --show-completion help to one line; that trim must
    # actually fire (the `for … or ()` loop) and target only --show-completion, not
    # --install-completion (the `isinstance(...) and startswith(...)` guard).
    command = typer.main.get_command(app)
    help_by_opt = {
        opt: getattr(param, "help", None) for param in command.params for opt in param.opts
    }
    assert help_by_opt.get("--show-completion") == "Show completion for the current shell."
    assert help_by_opt.get("--install-completion") != "Show completion for the current shell."


def test_complete_model_filters_by_prefix():
    suggestions = complete_model("gpt")
    assert suggestions  # at least one gpt-* model is known
    assert all(m.startswith("gpt") for m in suggestions)


def test_complete_model_empty_prefix_returns_all_known():
    assert complete_model("") == list(KNOWN_MODELS)


def test_complete_model_unknown_prefix_returns_nothing():
    assert complete_model("no-such-model") == []


def test_complete_voice_filters_by_prefix():
    prefix = VOICE_NAMES[0][:2]
    suggestions = complete_voice(prefix)
    assert suggestions
    assert all(v.startswith(prefix) for v in suggestions)


def test_complete_voice_empty_prefix_returns_all():
    assert complete_voice("") == VOICE_NAMES
