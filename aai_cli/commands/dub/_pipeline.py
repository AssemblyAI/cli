"""The `assembly dub` audio pipeline: translate → synthesize → timeline → mux.

The orchestration (argv resolution, source download, result reporting) lives in
``_exec``; the per-utterance transforms that turn a diarized transcript into a
dubbed audio track are gathered here so each stage stays unit-testable on its own
(see tests/test_dub_exec.py for the pure helpers, tests/test_dub_pipeline.py for
the faked end-to-end runs). ``_exec`` imports this module as ``pipeline`` and the
names below are its public surface; ``_pcm_seconds``-style internals stay private.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aai_cli.app import mediafile
from aai_cli.core import jsonshape
from aai_cli.core import llm as gateway
from aai_cli.core.errors import APIError, CLIError
from aai_cli.tts import audio, dialogue, session, voices
from aai_cli.tts.session import SpeakConfig
from aai_cli.ui import output

if TYPE_CHECKING:
    from aai_cli.commands.dub._exec import DubOptions

# System prompt for the per-utterance translation calls. Length matters: the dub
# replaces speech that occupied a fixed window, so the model is told to keep the
# spoken length close to the original.
TRANSLATION_SYSTEM_TEMPLATE = (
    "You translate dialogue for dubbing. Translate the user's text to {language}. "
    "Keep the meaning and register, and stay close to the original spoken length so "
    "the dub fits the original timing. Reply with only the translated text — no "
    "quotes, notes, or extra commentary."
)


def assemble_timeline(
    placed: list[tuple[int, bytes]],
    sample_rate: int,
    total_seconds: float | None,
) -> bytearray:
    """Lay each ``(start_ms, pcm)`` segment onto a silence timeline.

    Gaps before a segment's start are filled with silence; a segment whose
    predecessor overran its start time is appended immediately (the dub drifts
    rather than dropping speech). The tail is padded out to ``total_seconds``
    (the source duration) so the dubbed track never ends early.
    """
    pcm = bytearray()
    for start_ms, segment in placed:
        gap = start_ms / 1000 - pcm_seconds(pcm, sample_rate)
        if gap > 0:
            pcm.extend(audio.silence(sample_rate, gap))
        pcm.extend(segment)
    if total_seconds is not None:
        tail = total_seconds - pcm_seconds(pcm, sample_rate)
        if tail > 0:
            pcm.extend(audio.silence(sample_rate, tail))
    return pcm


def pcm_seconds(pcm: bytes | bytearray, sample_rate: int) -> float:
    """Seconds of audio in 16-bit mono PCM: two bytes per sample."""
    return len(pcm) / 2 / sample_rate


def mux(ffmpeg: str, media: Path, track: Path, out: Path) -> None:
    """Swap ``track`` in as the audio of ``media``, writing ``out``.

    ``-map 0:v?`` carries the video stream over untouched (``-c:v copy``) when
    there is one, and maps nothing for audio-only input, so the same invocation
    dubs both a video and a plain audio file. ``-y`` makes a re-run overwrite
    its own earlier output instead of stalling on ffmpeg's prompt.
    """
    result = mediafile.run_ffmpeg(
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
            mediafile.path_arg(out),
        ]
    )
    if result.returncode != 0:
        raise mediafile.ffmpeg_failure(result, "write", out, error_type="dub_failed")


@dataclass(frozen=True)
class Utterance:
    """One diarized utterance reduced to the fields the dub pipeline needs."""

    start_ms: int
    speaker: str
    text: str


def utterances_of(transcript: object, transcript_id: str) -> list[Utterance]:
    """The transcript's spoken utterances, with empty-text ones dropped."""
    utterances = [
        Utterance(
            start_ms=jsonshape.as_int(getattr(item, "start", 0)),
            speaker=str(getattr(item, "speaker", None) or "A"),
            text=str(getattr(item, "text", "") or "").strip(),
        )
        for item in jsonshape.object_list(getattr(transcript, "utterances", None))
    ]
    spoken = [utterance for utterance in utterances if utterance.text]
    if not spoken:
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


def total_seconds(transcript: object) -> float | None:
    """The source duration in seconds (used to pad the dubbed track's tail)."""
    duration = getattr(transcript, "audio_duration", None)
    if isinstance(duration, int | float) and not isinstance(duration, bool):
        return float(duration)
    return None


def translate(
    api_key: str,
    utterances: list[Utterance],
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
            # "length" is OpenAI's truncation marker; the gateway's Anthropic-flavored
            # responses use "max_tokens". A clipped translation must never be dubbed.
            if getattr(response.choices[0], "finish_reason", None) in {"length", "max_tokens"}:
                raise APIError(
                    f"The translation of utterance {index} was cut off at --max-tokens "
                    f"({opts.max_tokens}).",
                    suggestion="Re-run with a higher --max-tokens.",
                )
            if not translated:
                raise APIError(
                    f"The model returned an empty translation for utterance {index} "
                    f"({utterance.text[:50]!r})."
                )
            translations.append(translated)
    return translations


def synthesize(
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
    # `segments` is never empty (utterances_of raised otherwise), so results[0] exists.
    return [result.pcm for result in results], results[0].sample_rate


def warn_ignored_voice_pins(
    overrides: dict[str, str], speakers: dict[str, str], *, json_mode: bool
) -> None:
    """Mirror `assembly speak`: a requested --voice mapping is never dropped
    silently, so a pin for a speaker the diarization didn't produce is called out."""
    present = {speaker.casefold() for speaker in speakers}
    ignored = [speaker for speaker in overrides if speaker not in present]
    if ignored:
        output.emit_warning(
            "Ignoring --voice mapping(s) for speaker(s) not in the transcript: "
            f"{', '.join(ignored)}.",
            json_mode=json_mode,
        )


@dataclass(frozen=True)
class VoicePlan:
    """The parsed --voice flags: the bare voice (if any) plus SPEAKER=VOICE pins.

    Parsed in run_dub — before the billed pipeline, so a malformed mapping
    fails fast — and carried as one value through _dub_and_emit."""

    bare: str | None
    overrides: dict[str, str]


def assign_voices(
    utterances: list[Utterance],
    translations: list[str],
    plan: VoicePlan,
    language: str,
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Resolve each translated utterance to ``(voice, text)`` plus the speaker→voice map.

    A bare ``--voice`` dubs every speaker with that one voice; ``SPEAKER=VOICE``
    mappings pin individual speakers; everyone else takes the target language's
    rotation in first-appearance order (the same rules as `assembly speak`) —
    each voice speaks one language, so a non-English dub switches to that
    language's native voice(s).
    """
    rotation = (plan.bare,) if plan.bare is not None else voices.rotation_for(language)
    segments = [
        dialogue.Segment(utterance.speaker, translated)
        # strict=True is an invariant guard only: translate returns exactly one
        # translation per utterance, so the lengths can never differ.
        for utterance, translated in zip(utterances, translations, strict=True)  # pragma: no mutate
    ]
    return dialogue.assign_voices(segments, rotation, plan.overrides)
