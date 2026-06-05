"""Shared helpers for walking the live Typer command tree in tests.

Both the help snapshot test and the examples-coverage guard derive their command
list from the running app so new commands are picked up automatically; this keeps
the single tree-walk in one place.
"""

from __future__ import annotations

import typer

from aai_cli.main import app


def leaf_commands(click_cmd, prefix=()):
    """Yield (path_tuple, command) for every non-group (leaf) command in the tree."""
    sub = getattr(click_cmd, "commands", None)
    if not sub:
        yield prefix, click_cmd
        return
    for name, child in sub.items():
        yield from leaf_commands(child, (*prefix, name))


def leaf_command_items():
    """(path_tuple, command) for every leaf command in the live app tree."""
    return list(leaf_commands(typer.main.get_command(app)))


def leaf_command_argvs():
    """Sorted argv paths of every leaf command (the empty root path is dropped)."""
    return sorted((list(path) for path, _ in leaf_command_items() if path), key=" ".join)
