"""Guard that every ``--help`` example still parses against the live CLI tree.

``test_help_examples_coverage`` proves each leaf command *has* an ``Examples``
epilog, but not that the example commands are real. The examples are the snippets
users copy-paste, so a flag rename (the ``login --api-key`` → ``--with-api-key``
deprecation is the canonical case) or a removed subcommand silently rots them.
``docs_consistency_gate.py`` keeps REFERENCE.md/README in sync with the code, but
nothing checked the in-``--help`` examples — this closes that gap.

The check is deliberately scoped to *flags and subcommand paths* (the parts that
break on a rename), not full argument validation: examples carry placeholders
(``<file>``, ``TRANSCRIPT_ID``) that aren't real paths, so parsing them with
Click would false-positive. Pipelines (``a | assembly … | assembly …``) are split
into per-``assembly`` segments; a segment whose ``assembly`` token is glued to
other shell syntax (``$(assembly …``) is skipped rather than mis-parsed —
conservative by design, since those same commands appear unglued elsewhere.
"""

from __future__ import annotations

import shlex

import typer

from aai_cli.main import app
from tests._cli_tree import leaf_command_items

# Shell tokens that end one command and start another; an example may chain several
# `assembly` invocations through a pipe, so each segment is validated independently.
_BOUNDARIES = frozenset({"|", ">", ">>", "<", ";", "&&", "||", "&"})


def _option_names(command):
    """Every flag spelling a Click command accepts (long, short, and --no- forms)."""
    names = {"--help"}
    for param in command.params:
        if param.param_type_name == "option":
            names.update(param.opts)
            names.update(param.secondary_opts)
    return names


def _example_commands(command):
    """The ``$ …`` command lines from a leaf command's rendered examples epilog."""
    epilog = getattr(command, "epilog", None) or ""
    return [line.strip()[2:] for line in epilog.splitlines() if line.strip().startswith("$ ")]


def _assembly_segments(tokens):
    """Split a token stream into the argv of each literal ``assembly`` invocation.

    Each ``assembly`` token opens a fresh segment (appended up front, then grown in
    place), and a shell boundary closes the current one — so tokens belonging to a
    non-``assembly`` command (``ls``, ``jq``, ``$(assembly …``) are dropped.
    """
    segments: list[list[str]] = []
    current: list[str] | None = None
    for token in tokens:
        if token == "assembly":
            current = []
            segments.append(current)
        elif token in _BOUNDARIES:
            current = None
        elif current is not None:
            current.append(token)
    return segments


def _unknown_flags(argv, root):
    """Flags in one ``assembly`` argv that no command at their position accepts.

    Walks the tree token by token: a token matching a subcommand descends, and a
    flag is checked against whatever command is current (so a root flag like
    ``--sandbox`` is validated against the root, a leaf flag against the leaf).
    """
    command = root
    bad = []
    for token in argv:
        sub = getattr(command, "commands", None)
        if sub and token in sub:
            command = sub[token]
            continue
        if token.startswith("-") and token not in ("-", "--"):
            flag = token.split("=", 1)[0]  # --model=x → --model
            if flag not in _option_names(command):
                bad.append(flag)
    return bad


def _stale_examples(items, root):
    """Map each command path to the (example, unknown-flags) pairs it ships, if any."""
    stale: dict[str, list[tuple[str, list[str]]]] = {}
    for path, command in items:
        for example in _example_commands(command):
            for segment in _assembly_segments(shlex.split(example)):
                bad = _unknown_flags(segment, root)
                if bad:
                    stale.setdefault(" ".join(path), []).append((example, bad))
    return stale


def test_help_examples_reference_only_real_flags():
    root = typer.main.get_command(app)
    stale = _stale_examples(leaf_command_items(), root)
    assert stale == {}, f"--help examples reference flags the CLI no longer accepts: {stale}"


class _FakeLeaf:
    def __init__(self, epilog):
        self.epilog = epilog


def test_stale_examples_detects_renamed_and_removed_flags():
    # Drives the detection path the real examples (correctly) never trigger: a stale
    # flag is reported under its command, and a command with no epilog contributes
    # nothing — proving the guard would actually fail on drift, not just pass vacuously.
    root = typer.main.get_command(app)
    items = [
        (("renamed",), _FakeLeaf("[bold]Examples[/bold]\n\n$ assembly transcribe x --gone-flag")),
        (("blank",), _FakeLeaf(None)),
    ]
    assert _stale_examples(items, root) == {
        "renamed": [("assembly transcribe x --gone-flag", ["--gone-flag"])]
    }


def test_assembly_segments_splits_pipelines_and_drops_foreign_commands():
    # The parser splits a chained pipeline into per-`assembly` argv and drops tokens
    # owned by a non-`assembly` command (the leading `ls`).
    tokens = shlex.split("ls *.wav | assembly stream --from-stdin | assembly llm -f")
    assert _assembly_segments(tokens) == [["stream", "--from-stdin"], ["llm", "-f"]]
