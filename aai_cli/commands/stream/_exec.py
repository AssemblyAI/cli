"""Run logic for `assembly stream`: a gh-style options/run split.

The command module (aai_cli/commands/stream/__init__.py) only parses argv — it builds a
``StreamOptions`` and hands it to ``run_stream`` via ``context.run_command``. Keeping
the run path a module-level function of plain data (instead of a closure over the
Typer locals) lets tests drive validation, --show-code, and session wiring by
constructing a ``StreamOptions`` directly, with no CliRunner argv round-trip.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from assemblyai import PIISubstitutionPolicy
from assemblyai.streaming.v3 import Encoding, NoiseSuppressionModel, SpeechModel

from aai_cli import code_gen
from aai_cli.app.context import AppState
from aai_cli.core import choices, client, config_builder, youtube
from aai_cli.core.errors import UsageError
from aai_cli.core.microphone import MicrophoneSource
from aai_cli.streaming.macos import MacSystemAudioSource
from aai_cli.streaming.render import StreamRenderer
from aai_cli.streaming.session import (
    SourceOptions,
    StreamSession,
    resolve_output_modes,
    validate_sources,
)
from aai_cli.streaming.sources import TARGET_RATE, FileSource, StdinSource
from aai_cli.ui import output
from aai_cli.ui.follow import FollowRenderer


@dataclass(frozen=True)
class StreamOptions:
    """Every `assembly stream` flag as plain data.

    One field per CLI flag (``--json`` excluded: run_command resolves it into the
    ``json_mode`` argument), so a test can describe an invocation without argv.
    """

    source: str | None
    sample: bool
    sample_rate: int | None
    device: int | None
    system_audio: bool
    system_audio_only: bool
    speech_model: SpeechModel
    encoding: Encoding | None
    language_detection: bool | None
    domain: str | None
    prompt: str | None
    keyterms_prompt: list[str] | None
    end_of_turn_confidence_threshold: float | None
    min_turn_silence: int | None
    max_turn_silence: int | None
    vad_threshold: float | None
    format_turns: bool | None
    include_partial_turns: bool | None
    speaker_labels: bool | None
    max_speakers: int | None
    voice_focus: NoiseSuppressionModel | None
    voice_focus_threshold: float | None
    inactivity_timeout: int | None
    filter_profanity: bool | None
    redact_pii: bool | None
    redact_pii_policy: str | None
    redact_pii_sub: PIISubstitutionPolicy | None
    webhook_url: str | None
    webhook_auth_header: str | None
    llm_prompt: list[str] | None
    llm_interval: float
    model: str
    max_tokens: int
    config_kv: list[str] | None
    config_file: Path | None
    output_field: choices.TextOrJson | None
    show_code: bool

    def source_options(self) -> SourceOptions:
        """The audio-input subset, in the shape the validation/dispatch helpers read."""
        return SourceOptions(
            source=self.source,
            sample=self.sample,
            sample_rate=self.sample_rate,
            device=self.device,
            system_audio=self.system_audio,
            system_audio_only=self.system_audio_only,
        )

    def base_flags(self) -> dict[str, object]:
        """Every streaming flag except sample_rate, which is set per source at stream time."""
        flags: dict[str, object] = {
            "speech_model": config_builder.enum_value(self.speech_model),
            "format_turns": self.format_turns if self.format_turns is not None else True,
            "encoding": config_builder.enum_value(self.encoding),
            "language_detection": self.language_detection,
            "domain": self.domain,
            "end_of_turn_confidence_threshold": self.end_of_turn_confidence_threshold,
            "min_turn_silence": self.min_turn_silence,
            "max_turn_silence": self.max_turn_silence,
            "vad_threshold": self.vad_threshold,
            "include_partial_turns": self.include_partial_turns,
            "keyterms_prompt": list(self.keyterms_prompt) if self.keyterms_prompt else None,
            "filter_profanity": self.filter_profanity,
            "speaker_labels": self.speaker_labels,
            "max_speakers": self.max_speakers,
            "voice_focus": config_builder.enum_value(self.voice_focus),
            "voice_focus_threshold": self.voice_focus_threshold,
            "redact_pii": self.redact_pii,
            "redact_pii_policies": config_builder.split_csv(self.redact_pii_policy),
            "redact_pii_sub": config_builder.enum_value(self.redact_pii_sub),
            "inactivity_timeout": self.inactivity_timeout,
            "webhook_url": self.webhook_url,
            "prompt": self.prompt,
        }
        flags.update(config_builder.auth_header_flags(self.webhook_auth_header))
        return flags


def _print_show_code(
    opts: StreamOptions,
    sources: SourceOptions,
    base_flags: dict[str, object],
    *,
    text_mode: bool,
) -> None:
    """Print the equivalent SDK script without opening audio or authenticating.

    Emits a script faithful to the requested source — mic (default), stdin (-), or a
    file/URL — on raw stdout, so `--show-code > script.py` is runnable. Applies the
    same source validation as a real run, so e.g. a file + --sample-rate conflict
    errors here too instead of silently generating mic code.
    """
    validate_sources(sources, has_llm=bool(opts.llm_prompt), text_mode=text_mode)
    if sources.from_system_audio:
        raise UsageError("--show-code does not support macOS system audio capture yet.")
    if sources.source and youtube.is_downloadable_url(sources.source):
        raise UsageError(
            "--show-code does not support downloaded sources (YouTube, podcast pages) yet.",
            suggestion="Download the audio first (e.g. yt-dlp) and pass the local file.",
        )
    code_source: str | None = None
    if sources.from_stdin:
        code_source = "-"
    elif sources.from_file:
        # check_local=False: generating code for a file you don't have yet is fine.
        code_source = client.resolve_audio_source(
            sources.source, sample=sources.sample, check_local=False
        )
    merged = config_builder.merge_streaming_params(
        # sample_rate precedence: --sample-rate (None is dropped by the merge)
        # beats --config/--config-file, which beat the 16 kHz default below —
        # so an explicit `--config sample_rate=…` is honored, not overridden.
        flags=base_flags | {"sample_rate": sources.sample_rate},
        overrides=opts.config_kv,
        config_file=opts.config_file,
    )
    merged.setdefault("sample_rate", TARGET_RATE)
    gateway = code_gen.gateway_options(
        list(opts.llm_prompt or []), opts.model, opts.max_tokens, interval=opts.llm_interval
    )
    output.print_code(code_gen.stream(merged, llm=gateway, source=code_source))


def _dispatch(session: StreamSession, opts: SourceOptions) -> None:
    """Open the right audio source(s) for the flags and stream them."""
    if opts.from_system_audio:
        system = MacSystemAudioSource(on_open=session.on_open)
        if opts.system_audio_only:
            session.run(system, system.sample_rate, source_label="system")
        else:
            mic = MicrophoneSource(
                target_rate=TARGET_RATE,
                device=opts.device,
                capture_rate=opts.sample_rate,
                on_open=session.on_open,
            )
            session.run_parallel(
                [("system", system, system.sample_rate), ("you", mic, mic.sample_rate)]
            )
    elif opts.from_stdin:
        # Raw PCM16 mono piped on stdin (e.g. `ffmpeg … -f s16le - | assembly stream -`).
        stdin_src = StdinSource(sample_rate=opts.sample_rate or TARGET_RATE)
        session.run(stdin_src, stdin_src.sample_rate)
    elif opts.source and youtube.is_downloadable_url(opts.source):
        # Fetch the audio first, then stream the local file in real time.
        with tempfile.TemporaryDirectory(prefix="aai-yt-") as td:
            local = youtube.download_media(opts.source, Path(td))
            session.run(FileSource(str(local)), TARGET_RATE)
    elif opts.from_file:
        file_audio = FileSource(client.resolve_audio_source(opts.source, sample=opts.sample))
        session.run(file_audio, file_audio.sample_rate)
    else:
        # Capture at the device's native rate (or --sample-rate override) and tell the
        # streaming API that rate, rather than forcing one the device may reject.
        # "Listening…" is announced once the device is open (see StreamSession.on_open),
        # not when the session opens — so early speech isn't lost in the gap.
        mic = MicrophoneSource(
            device=opts.device, capture_rate=opts.sample_rate, on_open=session.on_open
        )
        session.run(mic, mic.sample_rate)


def run_stream(opts: StreamOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly stream` invocation from already-parsed flags."""
    text_mode, json_mode = resolve_output_modes(opts.output_field, json_mode=json_mode)
    sources = opts.source_options()
    base_flags = opts.base_flags()

    if opts.show_code:
        _print_show_code(opts, sources, base_flags, text_mode=text_mode)
        return

    # Validate the requested sources (including that a local file exists) before
    # credentials, so a typo'd path reads as "file not found" — not as a login.
    validate_sources(sources, has_llm=bool(opts.llm_prompt), text_mode=text_mode)
    if sources.from_file and not sources.from_stdin:
        client.resolve_audio_source(sources.source, sample=sources.sample)
    api_key = state.resolve_api_key()

    llm_prompts = list(opts.llm_prompt or [])
    session = StreamSession(
        api_key=api_key,
        base_flags=base_flags,
        overrides=opts.config_kv,
        config_file=opts.config_file,
        renderer=StreamRenderer(json_mode=json_mode, text_mode=text_mode),
        follow=FollowRenderer(json_mode=json_mode) if llm_prompts else None,
        llm_prompts=llm_prompts,
        model=opts.model,
        max_tokens=opts.max_tokens,
        llm_interval=opts.llm_interval,
    )
    _dispatch(session, sources)
