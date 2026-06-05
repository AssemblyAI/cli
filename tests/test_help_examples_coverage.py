# tests/test_help_examples_coverage.py
import typer

from aai_cli.main import app

# `version` is a trivial command with no flags; examples would be noise.
_EXEMPT = {"version"}


def _leaf_commands(click_cmd, prefix=()):
    """Yield (path_tuple, command) for every non-group command in the tree."""
    sub = getattr(click_cmd, "commands", None)
    if not sub:
        yield prefix, click_cmd
        return
    for name, child in sub.items():
        yield from _leaf_commands(child, (*prefix, name))


def test_every_leaf_command_has_examples_epilog():
    root = typer.main.get_command(app)
    missing = []
    for path, cmd in _leaf_commands(root):
        name = path[-1] if path else cmd.name
        if name in _EXEMPT:
            continue
        epilog = getattr(cmd, "epilog", None)
        if not (epilog and "Examples" in epilog):
            missing.append(" ".join(path))
    assert not missing, f"commands missing --help examples: {missing}"
