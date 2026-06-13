from __future__ import annotations

from pathlib import Path

import typer

from aai_cli import command_registry, dub_exec, help_panels, llm, options
from aai_cli.context import run_command
from aai_cli.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=80,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("dub",),
)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    # --sandbox is a root flag, so it must come before the subcommand in every example.
    epilog=examples_epilog(
        [
            ("Dub a talk into German (sandbox only)", "assembly --sandbox dub talk.mp4 --lang de"),
            ("Use a language name instead of a code", "assembly --sandbox dub talk.mp4 -l Spanish"),
            (
                "Dub the full video from YouTube",
                'assembly --sandbox dub "https://youtube.com/watch?v=ID" -l de --video',
            ),
            (
                "Dub only the first 15 minutes of a YouTube video",
                'assembly --sandbox dub "https://youtube.com/watch?v=ID" -l de --video '
                '--download-sections "*0:00-15:00"',
            ),
            (
                "Dub every speaker with one voice",
                "assembly --sandbox dub talk.mp4 -l fr --voice paul",
            ),
            (
                "Pin a voice per diarized speaker",
                "assembly --sandbox dub panel.mp4 -l de --voice A=jane --voice B=paul",
            ),
            (
                "Reuse a finished transcript instead of re-transcribing",
                "assembly --sandbox dub talk.mp4 -l de -t TRANSCRIPT_ID",
            ),
            (
                "Choose the output file",
                "assembly --sandbox dub talk.mp4 -l de --out talk-german.mp4",
            ),
        ]
    ),
)
def dub(
    ctx: typer.Context,
    media: str = typer.Argument(
        ...,
        help="Audio/video to dub: a local file (the video stream is copied untouched), "
        "or a YouTube/media-page URL (downloaded via yt-dlp).",
    ),
    lang: str = typer.Option(
        ...,
        "--lang",
        "-l",
        help="Target language: an ISO code (de, fr, es, …) or a language name (German).",
    ),
    source_lang: str | None = typer.Option(
        None,
        "--source-lang",
        help="ISO code of the source audio (e.g. de). Default: auto-detect the language.",
    ),
    transcript_id: str | None = typer.Option(
        None,
        "--transcript-id",
        "-t",
        help="Reuse an existing diarized transcript of this media instead of "
        "transcribing it again.",
    ),
    voice: list[str] = typer.Option(
        [],
        "--voice",
        help="Voice id for every speaker (e.g. jane, michael, paul), or SPEAKER=VOICE "
        "to pin a diarized speaker (repeatable, e.g. --voice A=jane). Default: the "
        "target language's native voice(s).",
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model that translates the utterances.",
        rich_help_panel=help_panels.OPT_LLM,
        autocompletion=llm.complete_model,
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max tokens per utterance translation.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Output file (default: <name>.dub.<lang><ext> next to the input)."
    ),
    video: bool = typer.Option(
        False,
        "--video",
        help="Download the full video (not just the audio track) for a URL source, "
        "so the dub keeps the picture. Local files keep their video already.",
    ),
    download_sections: list[str] = typer.Option(
        [],
        "--download-sections",
        help="For a URL source, download (and dub) only part of it (yt-dlp "
        '"--download-sections" syntax, e.g. "*0:00-15:00" for the first fifteen '
        "minutes; repeatable).",
    ),
    json_out: bool = options.json_option("Emit JSON describing the dubbed file."),
) -> None:
    """[sandbox] Dub a video or audio file into another language.

    The whole platform in one command: the media is transcribed with diarized
    utterance timestamps, each utterance is translated by an LLM Gateway model,
    the translations are synthesized with streaming TTS (one voice per
    speaker), and ffmpeg lays the new audio over the original — video copied
    untouched. A YouTube/media-page URL is downloaded first (audio only, or
    the full video with --video; --download-sections fetches and dubs only a
    time slice of it). Streaming TTS only exists in the sandbox
    today — run it as 'assembly --sandbox dub' (--sandbox goes before the
    subcommand). Requires ffmpeg.
    """
    opts = dub_exec.DubOptions(
        media=media,
        language=lang,
        source_language=source_lang,
        transcript_id=transcript_id,
        voice=voice,
        model=model,
        max_tokens=max_tokens,
        out=out,
        video=video,
        download_sections=download_sections,
    )
    run_command(
        ctx,
        lambda state, json_mode: dub_exec.run_dub(opts, state, json_mode=json_mode),
        json=json_out,
    )
