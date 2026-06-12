from __future__ import annotations

from pathlib import Path

import typer

from aai_cli import clip_exec, help_panels, llm, options
from aai_cli.context import run_command
from aai_cli.help_text import examples_epilog

app = typer.Typer()


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Clip everything speaker A says", "assembly clip meeting.mp4 --speaker A"),
            (
                "Clip the sentences that mention a topic",
                'assembly clip call.mp3 --search "pricing"',
            ),
            ("Cut an explicit time range", "assembly clip talk.mp4 --range 1:30-2:45"),
            (
                "Let an LLM pick the moments worth clipping",
                'assembly clip meeting.mp4 --llm "the three strongest customer objections"',
            ),
            (
                "Clip a YouTube video's audio with an LLM",
                'assembly clip "https://youtube.com/watch?v=ID" --llm "the best quote"',
            ),
            (
                "Cut video clips from the full YouTube video",
                'assembly clip "https://youtube.com/watch?v=ID" --video --llm "the best quote"',
            ),
            (
                "Reuse a finished transcript instead of re-transcribing",
                "assembly clip meeting.mp4 -t TRANSCRIPT_ID --speaker B",
            ),
            (
                "Pipe transcribe straight into clip",
                "assembly transcribe meeting.mp4 --speaker-labels --json"
                ' | assembly clip meeting.mp4 -t - --llm "the funniest exchange"',
            ),
            (
                "Pad each clip and collect them in a directory",
                "assembly clip meeting.mp4 --speaker A --padding 0.5 --out-dir clips",
            ),
        ]
    ),
)
def clip(
    ctx: typer.Context,
    media: str = typer.Argument(
        ...,
        help="Audio/video to cut clips from: a local file, or a YouTube/media-page "
        "URL (audio downloaded via yt-dlp).",
    ),
    transcript_id: str | None = typer.Option(
        None,
        "--transcript-id",
        "-t",
        help="Reuse an existing transcript of this media instead of transcribing it again: "
        "an id, or '-' to read an id or 'transcribe --json' output from stdin.",
    ),
    speaker: list[str] = typer.Option(
        [],
        "--speaker",
        help="Keep segments spoken by this diarized speaker label (repeatable, e.g. --speaker A).",
    ),
    search: str | None = typer.Option(
        None, "--search", help="Keep segments whose text contains this (case-insensitive)."
    ),
    llm_prompt: str | None = typer.Option(
        None,
        "--llm",
        help="Let an LLM Gateway model pick the windows to clip from the timestamped "
        'transcript (e.g. --llm "the funniest moments"). Composes with --speaker/--search.',
        rich_help_panel=help_panels.OPT_LLM,
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model for --llm.",
        rich_help_panel=help_panels.OPT_LLM,
        autocompletion=llm.complete_model,
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max tokens for the --llm selection reply.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    ranges: list[str] = typer.Option(
        [],
        "--range",
        help="Keep an explicit START-END window (seconds or [HH:]MM:SS; repeatable).",
    ),
    padding: float = typer.Option(
        0.0, "--padding", min=0.0, help="Seconds of padding to add around each clip."
    ),
    snap: bool = typer.Option(
        True,
        "--snap/--no-snap",
        help="Snap clip boundaries into nearby silence (detected with ffmpeg) so cuts "
        "don't land mid-word; --no-snap cuts at the exact selected times.",
    ),
    out_dir: Path | None = typer.Option(
        None, "--out-dir", help="Directory for the clip files (default: next to the input)."
    ),
    video: bool = typer.Option(
        False,
        "--video",
        help="Download the full video (not just the audio track) for a URL source, "
        "so the clips are cut from the video. Local files keep their video already.",
    ),
    json_out: bool = options.json_option("Emit JSON describing the clips written."),
) -> None:
    """Cut clips out of a media file by speaker, text match, LLM pick, or time range.

    --speaker and --search select from a diarized transcript (made on the fly,
    or reused with --transcript-id); --llm has an LLM Gateway model pick the
    windows; --range adds explicit ones. Overlapping selections merge, clip
    boundaries snap into nearby silence so cuts don't land mid-word (--no-snap
    disables), and each surviving segment is written as <name>.clipNN<ext>
    using ffmpeg (which must be installed). A YouTube/media-page source is
    downloaded first (audio only, or the full video with --video); its clips
    land in --out-dir or the current directory.
    """
    opts = clip_exec.ClipOptions(
        media=media,
        transcript_id=transcript_id,
        speakers=speaker,
        search=search,
        llm_prompt=llm_prompt,
        model=model,
        max_tokens=max_tokens,
        ranges=ranges,
        padding=padding,
        snap=snap,
        out_dir=out_dir,
        video=video,
    )
    run_command(
        ctx,
        lambda state, json_mode: clip_exec.run_clip(opts, state, json_mode=json_mode),
        json=json_out,
    )
