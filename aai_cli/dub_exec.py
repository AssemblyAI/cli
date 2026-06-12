"""Run logic for `assembly dub`: transcribe → translate → synthesize → ffmpeg track-swap.

The command module (aai_cli/commands/dub.py) only parses argv — it builds a
``DubOptions`` and hands it to ``run_dub`` via ``context.run_command`` (the
options/run split, see AGENTS.md), so tests drive the whole pipeline by
constructing options directly.

The pipeline runs the platform end to end in one command: the media is
transcribed with diarized utterance timestamps, each utterance is translated to
the target language by an LLM Gateway model, each translation is synthesized
with streaming TTS (one voice per speaker), the segments are laid out on a
silence timeline at their original start times, and ffmpeg swaps the new track
over the original media (video stream copied untouched). A YouTube/media-page
URL is downloaded first — audio only, or the full video with ``--video`` so the
dub keeps the picture. Streaming TTS only exists in the sandbox today, so —
like `assembly speak` — the command is sandbox-only.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import assemblyai as aai
from rich.markup import escape

from aai_cli import client, environments, jsonshape, output, youtube
from aai_cli import llm as gateway
from aai_cli.context import AppState
from aai_cli.errors import APIError, CLIError, UsageError
from aai_cli.tts import audio, dialogue, session
from aai_cli.tts.session import SpeakConfig

# ISO-639-1 codes accepted by --lang, mapped to the language *name* both the
# translation prompt and the streaming-TTS `language` param expect. A value not
# listed passes through as typed, so a full name ("German") — or an unlisted
# language the gateway can translate to — still works.
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

# System prompt for the per-utterance translation calls. Length matters: the dub
# replaces speech that occupied a fixed window, so the model is told to keep the
# spoken length close to the original.
TRANSLATION_SYSTEM_TEMPLATE = (
    "You translate dialogue for dubbing. Translate the user's text to {language}. "
    "Keep the meaning and register, and stay close to the original spoken length so "
    "the dub fits the original timing. Reply with only the translated text — no "
    "quotes, notes, or extra commentary."
)


@dataclass(frozen=True)
class DubOptions:
    """Every `assembly dub` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    media: str
    language: str
    transcript_id: str | None
    voice: list[str]
    model: str
    max_tokens: int
    out: Path | None
    video: bool


def resolve_language(value: str) -> str:
    """The target language name: an ISO code maps to its name, anything else
    passes through as typed (the gateway accepts more languages than the map)."""
    cleaned = value.strip()
    if not cleaned:
        raise UsageError(
            "--lang needs a language.",
            suggestion="Pass an ISO code (--lang de) or a language name (--lang German).",
        )
    return LANGUAGE_NAMES.get(cleaned.casefold(), cleaned)


def default_out_path(media: Path, language: str) -> Path:
    """The default output file: ``<stem>.dub.<lang><ext>`` next to the input."""
    slug = re.sub(r"[^a-z0-9]+", "-", language.casefold()).strip("-")
    return media.parent / f"{media.stem}.dub.{slug}{media.suffix}"


def assemble_timeline(
    placed: list[tuple[int, bytes]],
    sample_rate: int,
    total_seconds: float | None,
) -> bytes:
    """Lay each ``(start_ms, pcm)`` segment onto a silence timeline.

    Gaps before a segment's start are filled with silence; a segment whose
    predecessor overran its start time is appended immediately (the dub drifts
    rather than dropping speech). The tail is padded out to ``total_seconds``
    (the source duration) so the dubbed track never ends early.
    """
    pcm = bytearray()
    for start_ms, segment in placed:
        gap = start_ms / 1000 - _pcm_seconds(pcm, sample_rate)
        if gap > 0:
            pcm.extend(audio.silence(sample_rate, gap))
        pcm.extend(segment)
    if total_seconds is not None:
        tail = total_seconds - _pcm_seconds(pcm, sample_rate)
        if tail > 0:
            pcm.extend(audio.silence(sample_rate, tail))
    return bytes(pcm)


def _pcm_seconds(pcm: bytes | bytearray, sample_rate: int) -> float:
    """Seconds of audio in 16-bit mono PCM: two bytes per sample."""
    return len(pcm) / 2 / sample_rate


def _require_sandbox() -> None:
    """`assembly dub` synthesizes with streaming TTS, which is sandbox-only today."""
    if not session.is_available():
        raise CLIError(
            "assembly dub is only available in the sandbox (it uses streaming TTS).",
            error_type="unsupported_environment",
            exit_code=2,
            suggestion="Re-run as: assembly --sandbox dub … "
            f"(--sandbox goes before the command; or use --env {environments.SANDBOX_ENV}).",
        )


def _validate_media(media: Path) -> None:
    """Reject a missing local source before credential resolution, so a typo'd
    path reads as "file not found", never as a login prompt or an ffmpeg error."""
    if not media.exists():
        raise CLIError(
            f"File not found: {media}",
            error_type="file_not_found",
            exit_code=2,
            suggestion="Check the path. assembly dub needs a local audio/video file.",
        )
    if not media.is_file():
        raise CLIError(
            f"Not a file: {media}",
            error_type="not_a_file",
            exit_code=2,
            suggestion="Pass a media file, not a directory.",
        )


def _validate_out(out: Path, media: Path) -> None:
    """The dub must never overwrite its own input: ffmpeg would read and write the
    same file concurrently, corrupting it."""
    if out.resolve() == media.resolve():
        raise UsageError(
            "--out would overwrite the input file.",
            suggestion="Pick a different output path.",
        )


def _require_ffmpeg() -> str:
    """The ffmpeg executable; checked before any (billed) transcription work."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise CLIError(
            "ffmpeg is required to write the dubbed file, but it isn't on PATH.",
            error_type="missing_dependency",
            suggestion="Install it (brew install ffmpeg / apt install ffmpeg) and re-run.",
        )
    return path


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Boundary seam for tests: one ffmpeg invocation, output captured."""
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _mux(ffmpeg: str, media: Path, track: Path, out: Path) -> None:
    """Swap ``track`` in as the audio of ``media``, writing ``out``.

    ``-map 0:v?`` carries the video stream over untouched (``-c:v copy``) when
    there is one, and maps nothing for audio-only input, so the same invocation
    dubs both a video and a plain audio file. ``-y`` makes a re-run overwrite
    its own earlier output instead of stalling on ffmpeg's prompt.
    """
    result = _run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media),
            "-i",
            str(track),
            "-map",
            "0:v?",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            str(out),
        ]
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        reason = detail[-1] if detail else f"ffmpeg exited with code {result.returncode}"
        raise CLIError(
            f"Could not write {out.name}: {reason}",
            error_type="dub_failed",
            suggestion="Check that the input is a readable audio/video file.",
        )


@dataclass(frozen=True)
class _Utterance:
    """One diarized utterance reduced to the fields the dub pipeline needs."""

    start_ms: int
    speaker: str
    text: str


def _resolve_transcript(
    opts: DubOptions, media: Path, state: AppState, *, json_mode: bool
) -> object:
    """The diarized transcript driving the dub: fetched by id, or made fresh from
    the (already local) media file — always with speaker labels, so each speaker
    can keep a distinct voice in the dub."""
    if opts.transcript_id is not None:
        return client.get_transcript(state.resolve_api_key(), opts.transcript_id)
    config = aai.TranscriptionConfig(speaker_labels=True)
    api_key = state.resolve_api_key()
    with output.status("Transcribing for dubbing…", json_mode=json_mode, quiet=state.quiet):
        return client.transcribe(api_key, str(media), config=config)


def _utterances_of(transcript: object) -> list[_Utterance]:
    """The transcript's spoken utterances, with empty-text ones dropped."""
    utterances = [
        _Utterance(
            start_ms=jsonshape.as_int(getattr(item, "start", 0)),
            speaker=str(getattr(item, "speaker", None) or "A"),
            text=str(getattr(item, "text", "") or "").strip(),
        )
        for item in jsonshape.object_list(getattr(transcript, "utterances", None))
    ]
    spoken = [utterance for utterance in utterances if utterance.text]
    if not spoken:
        transcript_id = str(getattr(transcript, "id", ""))
        raise CLIError(
            f"Transcript {transcript_id} has no utterances to dub.",
            error_type="no_utterances",
            exit_code=2,
            suggestion=(
                "Dubbing needs a diarized transcript. Pass a --transcript-id created "
                "with --speaker-labels, or drop -t to let dub transcribe the file."
            ),
        )
    return spoken


def _total_seconds(transcript: object) -> float | None:
    """The source duration in seconds (used to pad the dubbed track's tail)."""
    duration = getattr(transcript, "audio_duration", None)
    if isinstance(duration, int | float) and not isinstance(duration, bool):
        return float(duration)
    return None


def _translate(
    api_key: str,
    utterances: list[_Utterance],
    language: str,
    opts: DubOptions,
    *,
    json_mode: bool,
    quiet: bool,
) -> list[str]:
    """Translate each utterance to ``language`` with the LLM Gateway, in order.

    One call per utterance keeps the translation↔timestamp alignment exact —
    no reply-parsing step that could shift a line against its window.
    """
    system = TRANSLATION_SYSTEM_TEMPLATE.format(language=language)
    translating = f"Translating {len(utterances)} utterance(s) to {language} with {opts.model}…"
    translations: list[str] = []
    with output.status(translating, json_mode=json_mode, quiet=quiet):
        for index, utterance in enumerate(utterances, 1):
            messages = gateway.build_messages(utterance.text, system=system)
            response = gateway.complete(
                api_key, model=opts.model, messages=messages, max_tokens=opts.max_tokens
            )
            translated = gateway.content_of(response).strip()
            if not translated:
                raise APIError(
                    f"The model returned an empty translation for utterance {index} "
                    f"({utterance.text[:50]!r})."
                )
            translations.append(translated)
    return translations


def _synthesize(
    api_key: str,
    segments: list[tuple[str, str]],
    language: str,
    *,
    json_mode: bool,
    quiet: bool,
) -> tuple[list[bytes], int]:
    """Synthesize each ``(voice, text)`` segment; returns the PCM list + sample rate.

    Every segment must come back at one rate — the timeline math places segments
    by sample position, so a mid-run rate change would silently shift timing.
    """
    synthesizing = f"Synthesizing {len(segments)} segment(s)…"
    with output.status(synthesizing, json_mode=json_mode, quiet=quiet):
        results = [
            session.synthesize(
                api_key,
                SpeakConfig(text=text, voice=voice, language=language),
                on_warning=lambda m: output.emit_warning(m, json_mode=json_mode),
            )
            for voice, text in segments
        ]
    rates = {result.sample_rate for result in results}
    if len(rates) > 1:
        raise APIError(f"TTS service returned mixed sample rates ({sorted(rates)}).")
    # `segments` is never empty (_utterances_of raised otherwise), so results[0] exists.
    return [result.pcm for result in results], results[0].sample_rate


def _assign_voices(
    utterances: list[_Utterance],
    translations: list[str],
    voice_values: list[str],
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Resolve each translated utterance to ``(voice, text)`` plus the speaker→voice map.

    A bare ``--voice`` dubs every speaker with that one voice; ``SPEAKER=VOICE``
    mappings pin individual speakers; everyone else takes the rotation in
    first-appearance order (the same rules as `assembly speak`).
    """
    bare_voice, overrides = dialogue.parse_voice_overrides(voice_values)
    rotation = (bare_voice,) if bare_voice is not None else dialogue.DEFAULT_VOICE_ROTATION
    segments = [
        dialogue.Segment(utterance.speaker, translated)
        # strict=True is an invariant guard only: _translate returns exactly one
        # translation per utterance, so the lengths can never differ.
        for utterance, translated in zip(utterances, translations, strict=True)  # pragma: no mutate
    ]
    return dialogue.assign_voices(segments, rotation, overrides)


def run_dub(opts: DubOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly dub` invocation from already-parsed flags."""
    language = resolve_language(opts.language)
    _require_sandbox()
    youtube.validate_video_flag(opts.media, video=opts.video)
    if youtube.is_downloadable_url(opts.media):
        # A media-page URL (YouTube, podcast page, …) is downloaded once — the
        # audio track by default, the full video with --video so the dub keeps
        # the picture — and dubbed locally. ffmpeg is checked before the
        # download so a missing dependency fails before any fetch.
        ffmpeg = _require_ffmpeg()
        downloading = "Downloading video…" if opts.video else "Downloading audio…"
        with tempfile.TemporaryDirectory(prefix="aai-dub-src-") as td:
            with output.status(downloading, json_mode=json_mode, quiet=state.quiet):
                local = youtube.download_media(opts.media, Path(td), video=opts.video)
            # The download dir is temporary, so the default output lands in the
            # current directory — never next to the temp file.
            out = (
                opts.out
                if opts.out is not None
                else Path.cwd() / default_out_path(local, language).name
            )
            _validate_out(out, local)
            _dub_and_emit(opts, local, out, language, ffmpeg, state, json_mode=json_mode)
        return
    if opts.media.startswith(("http://", "https://")):
        raise UsageError(
            "assembly dub can't fetch this URL; it dubs a local file or a "
            "media-page URL yt-dlp can download (YouTube, podcasts, …).",
            suggestion="Download the media first, then dub the local copy.",
        )
    media = Path(opts.media)
    _validate_media(media)
    out = opts.out if opts.out is not None else default_out_path(media, language)
    _validate_out(out, media)
    ffmpeg = _require_ffmpeg()
    _dub_and_emit(opts, media, out, language, ffmpeg, state, json_mode=json_mode)


def _dub_and_emit(
    opts: DubOptions,
    media: Path,
    out: Path,
    language: str,
    ffmpeg: str,
    state: AppState,
    *,
    json_mode: bool,
) -> None:
    """Dub an already-local media file into ``out`` and report the result."""
    transcript = _resolve_transcript(opts, media, state, json_mode=json_mode)
    transcript_id = str(getattr(transcript, "id", ""))
    utterances = _utterances_of(transcript)
    api_key = state.resolve_api_key()
    translations = _translate(
        api_key, utterances, language, opts, json_mode=json_mode, quiet=state.quiet
    )
    resolved, speakers = _assign_voices(utterances, translations, opts.voice)
    pcm_segments, sample_rate = _synthesize(
        api_key, resolved, language, json_mode=json_mode, quiet=state.quiet
    )

    # strict=True is an invariant guard only: _synthesize returns one PCM per segment.
    starts = (u.start_ms for u in utterances)
    placed = list(zip(starts, pcm_segments, strict=True))  # pragma: no mutate
    track = assemble_timeline(placed, sample_rate, _total_seconds(transcript))
    with tempfile.TemporaryDirectory(prefix="aai-dub-") as tmp:
        wav = Path(tmp) / "dub.wav"
        audio.write_wav(wav, track, sample_rate)
        with output.status("Writing the dubbed file…", json_mode=json_mode, quiet=state.quiet):
            _mux(ffmpeg, media, wav, out)

    duration = round(_pcm_seconds(track, sample_rate), 3)
    voices = ", ".join(f"{speaker}={voice}" for speaker, voice in speakers.items())
    payload: dict[str, object] = {
        "source": opts.media,
        "out": str(out),
        "language": language,
        "transcript_id": transcript_id,
        "utterances": len(utterances),
        "speakers": speakers,
        "sample_rate": sample_rate,
        "audio_duration_seconds": duration,
    }
    output.emit(
        payload,
        lambda _: output.success(
            f"{escape(str(out))}  dubbed to {language} ({len(utterances)} utterances, {voices})"
        ),
        json_mode=json_mode,
    )
