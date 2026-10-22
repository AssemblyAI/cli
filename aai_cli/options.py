"""Shared Typer option factories for flags every command repeats.

Centralizing them keeps the flag name, default, and help text uniform across
the ~26 command signatures instead of copy-pasting (and drifting) per command.
"""

from __future__ import annotations

import typer

from aai_cli import help_panels
from aai_cli.app import tts_voices
from aai_cli.app.context import run_command
from aai_cli.core import argscan

DEFAULT_BATCH_CONCURRENCY = 4


def json_option(help_text: str = "Output raw JSON") -> bool:
    """The standard ``--json``/``-j`` flag; pass ``help_text`` where the shape differs."""
    flag: bool = typer.Option(False, "--json", "-j", help=help_text)
    return flag


def fields_option() -> str | None:
    """The ``-o/--output`` field projection shared by the list/account read commands.

    Lets ``assembly transcripts list -o id`` or ``assembly sessions list -o
    session_id,status`` replace a ``--json | jq`` column grab: it projects the named
    fields out of the same JSON the command would emit, one tab-separated line per
    record. Comma-separated for multiple fields; dotted paths reach nested objects.
    """
    value: str | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Project fields from the JSON result (comma-separated, e.g. id,status)",
        metavar="FIELDS",
    )
    return value


def chars_per_caption_option() -> int | None:
    """The ``--chars-per-caption`` knob for the ``-o srt``/``-o vtt`` subtitle exports."""
    value: int | None = typer.Option(
        None,
        "--chars-per-caption",
        min=1,
        help="Max characters per caption line (only with -o srt or -o vtt)",
    )
    return value


# Batch-mode flags for `transcribe` (see transcribe_batch.py). Defined here because
# this module owns the FBT003 carve-out for Typer's boolean positional defaults.


def batch_from_stdin_option(
    help_text: str = "Batch mode: read audio paths/URLs from stdin, one per line "
    "(composes with find/ls/yt-dlp output)",
) -> bool:
    """The ``--from-stdin`` flag: batch mode fed one path/URL per stdin line.

    ``help_text`` lets the media commands (clip/dub/caption) reword "audio" for
    their own source kind; the default carries ``transcribe``'s wording.
    """
    flag: bool = typer.Option(
        False,
        "--from-stdin",
        help=help_text,
        rich_help_panel=help_panels.OPT_BATCH,
    )
    return flag


def batch_concurrency_option(
    help_text: str = "How many sources to transcribe at once in batch mode",
) -> int:
    """The ``--concurrency`` option: how many sources run at once in batch mode."""
    value: int = typer.Option(
        DEFAULT_BATCH_CONCURRENCY,
        "--concurrency",
        min=1,
        help=help_text,
        rich_help_panel=help_panels.OPT_BATCH,
    )
    return value


def batch_force_option(
    help_text: str = "Batch mode: re-transcribe sources whose sidecar already records a completed run",
) -> bool:
    """The ``--force`` flag: reprocess a source even when its output already exists."""
    flag: bool = typer.Option(
        False,
        "--force",
        help=help_text,
        rich_help_panel=help_panels.OPT_BATCH,
    )
    return flag


def list_voices_option(command: str) -> bool:
    """The ``--list-voices`` flag of the streaming-TTS commands (speak, dub, live).

    One factory so the three can't drift. The flag does its work in its own option
    callback and exits, so it needs none of the command's other input (``assembly
    --sandbox dub --list-voices`` runs without the otherwise-required ``--lang``);
    it is *eager* so that, like ``--help``, it wins even when an option given
    earlier on the line is invalid. ``command`` names the command in the sandbox hint.
    """

    def _callback(ctx: typer.Context, *, value: bool) -> bool:
        if value:
            # The subcommand's own --json isn't parsed yet when an eager option fires,
            # so honor it the way the root callback does: off the raw token list.
            raw_args: list[str] = ctx.meta.get(argscan.RAW_ARGS_META_KEY, [])
            run_command(
                ctx,
                lambda state, json_mode: tts_voices.emit_voice_list(
                    command, state, json_mode=json_mode
                ),
                json=argscan.requests_json(raw_args),
            )
            raise typer.Exit()
        return value

    flag: bool = typer.Option(
        False,
        "--list-voices",
        help="List the voices the TTS service offers and exit",
        is_eager=True,
        callback=_callback,
    )
    return flag
