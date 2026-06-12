from __future__ import annotations

from abc import abstractmethod
from typing import Protocol

import typer

from aai_cli import output
from aai_cli.errors import UsageError


class WizardCancelled(Exception):
    """Raised when the user aborts the wizard (Ctrl-C / empty selection)."""


class Prompter(Protocol):
    """How the wizard asks for input — one interface, interactive or not."""

    # True only when a human can answer prompts (and complete a browser sign-in);
    # the wizard reads this to skip steps that would otherwise hang a headless run.
    interactive: bool

    def section(self, title: str) -> None:
        """Print a step heading."""

    def note(self, message: str) -> None:
        """Print an informational line."""

    @abstractmethod
    def confirm(self, title: str, *, default: bool = True) -> bool:  # pragma: no mutate
        """Ask a yes/no question."""

    @abstractmethod
    def select(
        self, title: str, options: list[tuple[str, str]], *, default: str | None = None
    ) -> str:
        """Pick one value from `options` (label, value) pairs."""

    @abstractmethod
    def text(self, title: str, *, default: str | None = None) -> str:
        """Ask for a free-form line of text."""


class InteractivePrompter:
    """Drives real terminal prompts (questionary for select, Typer for the rest)."""

    interactive = True

    def section(self, title: str) -> None:
        output.console.print("\n" + output.heading(title))

    def note(self, message: str) -> None:
        output.console.print(output.hint(message))

    def confirm(self, title: str, *, default: bool = True) -> bool:
        return typer.confirm(title, default=default)

    def select(
        self, title: str, options: list[tuple[str, str]], *, default: str | None = None
    ) -> str:
        import questionary

        choice = questionary.select(
            title,
            choices=[questionary.Choice(title=label, value=value) for value, label in options],
            default=default,
        ).ask()
        if choice is None:  # Ctrl-C
            raise WizardCancelled
        return str(choice)

    def text(self, title: str, *, default: str | None = None) -> str:
        return str(typer.prompt(title, default=default))


class NonInteractivePrompter:
    """Never blocks for input: returns defaults, logs choices, refuses when no default.

    Keeps the CLI pipeline-safe — `--json`, a piped stdin, or an agent run can call
    the wizard without it hanging on a prompt no human will answer.
    """

    interactive = False

    def section(self, title: str) -> None:
        output.error_console.print(output.heading(title))

    def note(self, message: str) -> None:
        output.error_console.print(output.hint(message))

    def confirm(self, title: str, *, default: bool = True) -> bool:
        output.error_console.print(output.hint(f"{title} → {default} (non-interactive)"))
        return default

    def select(
        self, title: str, options: list[tuple[str, str]], *, default: str | None = None
    ) -> str:
        chosen = default if default is not None else options[0][0]
        output.error_console.print(output.hint(f"{title} → {chosen} (non-interactive)"))
        return chosen

    def text(self, title: str, *, default: str | None = None) -> str:
        if default is None:
            raise UsageError(
                f"'{title}' needs a value, but this is a non-interactive session.",
                suggestion="Re-run `assembly onboard` in an interactive terminal.",
            )
        return default
