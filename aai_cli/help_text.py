from __future__ import annotations

from collections.abc import Sequence

from rich.markup import escape

# An (description, command) pair shown under a command's `--help`.
Example = tuple[str, str]


def examples_epilog(examples: Sequence[Example]) -> str:
    """Build a Typer ``epilog`` that renders each example on its own line.

    The app runs with ``rich_markup_mode="rich"``, which reflows single newlines
    into one paragraph but treats a blank line as a paragraph break. We join every
    line with a blank line so each renders on its own row, dim the descriptions so
    the commands stand out, and escape both so brackets in example commands (e.g.
    ``jq '.x[]'``) are not parsed as rich markup tags.
    """
    blocks = ["[bold]Examples[/bold]"]
    for description, command in examples:
        blocks.extend((f"[dim]{escape(description)}[/dim]", f"$ {escape(command)}"))
    return "\n\n".join(blocks)
