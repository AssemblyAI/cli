from __future__ import annotations

from pathlib import Path

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import run_with_options
from aai_cli.commands.caption import _exec as caption_exec
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=90,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("caption",),
)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Burn captions into a video", "assembly caption talk.mp4"),
            (
                "Caption a YouTube video (downloaded via yt-dlp)",
                'assembly caption "https://youtube.com/watch?v=ID"',
            ),
            (
                "Reuse a finished transcript instead of re-transcribing",
                "assembly caption talk.mp4 -t TRANSCRIPT_ID",
            ),
            (
                "Shorter caption lines in a bigger font",
                "assembly caption talk.mp4 --chars-per-caption 32 --font-size 28",
            ),
            ("Choose the output file", "assembly caption talk.mp4 --out talk-captioned.mp4"),
            (
                "Caption a whole folder of videos",
                'find clips -name "*.mp4" | assembly caption --from-stdin',
            ),
        ]
    ),
)
def caption(
    ctx: typer.Context,
    media: str | None = typer.Argument(
        None,
        help="Video to caption: a local file, or a YouTube/media-page URL "
        "(the full video is downloaded via yt-dlp). Omit with --from-stdin",
    ),
    from_stdin: bool = options.batch_from_stdin_option(
        "Batch mode: read video paths/URLs from stdin, one per line (composes with find/ls output)"
    ),
    concurrency: int = options.batch_concurrency_option(
        "How many sources to caption at once in batch mode"
    ),
    force: bool = options.batch_force_option(
        "Batch mode: re-caption sources whose output file already exists"
    ),
    transcript_id: str | None = typer.Option(
        None,
        "--transcript-id",
        "-t",
        help="Reuse an existing transcript of this media instead of transcribing it again",
    ),
    chars_per_caption: int | None = typer.Option(
        None,
        "--chars-per-caption",
        min=1,
        help="Max characters per caption line",
    ),
    font_size: int | None = typer.Option(
        None,
        "--font-size",
        min=1,
        help="Font size of the burned-in captions (ffmpeg's default styling when omitted)",
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Output file (default: <name>.captioned<ext> next to the input)"
    ),
    json_out: bool = options.json_option("Emit JSON describing the captioned file"),
) -> None:
    """Burn always-visible captions into a video

    The video is transcribed (or an existing transcript is reused with
    --transcript-id), the transcript's SRT captions are fetched, and ffmpeg
    (which must be installed) burns them into the picture as open captions —
    the audio stream is copied untouched. A YouTube/media-page URL is
    downloaded first (always the full video); its output lands in --out or
    the current directory.

    Batch mode: pipe one path/URL per line with --from-stdin to caption many
    sources concurrently (--concurrency). Each writes its own <name>.captioned<ext>;
    a re-run skips sources whose output already exists (--force re-does them).
    """
    opts = caption_exec.CaptionOptions(
        media=media or "",
        transcript_id=transcript_id,
        chars_per_caption=chars_per_caption,
        font_size=font_size,
        out=out,
        from_stdin=from_stdin,
        concurrency=concurrency,
        force=force,
    )
    run_with_options(ctx, caption_exec.run_caption, opts, json=json_out)
