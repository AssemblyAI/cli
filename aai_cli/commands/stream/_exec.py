"""Run logic for `assembly stream`: a gh-style options/run split.

The command module (aai_cli/commands/stream/__init__.py) only parses argv — it builds a
``StreamOptions`` and hands it to ``run_stream`` via ``context.run_command``. Keeping
the run path a module-level function of plain data (instead of a closure over the
Typer locals) lets tests drive validation, --show-code, and session wiring by
constructing a ``StreamOptions`` directly, with no CliRunner argv round-trip.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from assemblyai import PIISubstitutionPolicy
from assemblyai.streaming.v3 import Encoding, NoiseSuppressionModel, SpeechModel

from aai_cli import code_gen
from aai_cli.app.context import AppState
from aai_cli.core import choices, client, config_builder, signals, stdio, youtube
from aai_cli.core.errors import UsageError, mutually_exclusive
from aai_cli.core.microphone import MicrophoneSource
from aai_cli.streaming import naming, record, savedir, transcript, turn_presets
from aai_cli.streaming.batch import stream_batch_sources
from aai_cli.streaming.macos import MacSystemAudioSource
from aai_cli.streaming.render import StreamRenderer
from aai_cli.streaming.session import StreamSession
from aai_cli.streaming.sources import TARGET_RATE, FileSource, StdinSource
from aai_cli.streaming.turn_presets import TurnDetectionPreset
from aai_cli.streaming.validate import SourceOptions, resolve_output_modes, validate_sources
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
    from_stdin: bool
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
    turn_detection: TurnDetectionPreset | None
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
    save_audio: Path | None
    save_transcript: Path | None
    save_dir: Path | None
    name: str | None
    auto_name: bool
    no_save_audio: bool

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
        end_of_turn_confidence_threshold, min_turn_silence, max_turn_silence = turn_presets.resolve(
            self.turn_detection,
            self.end_of_turn_confidence_threshold,
            self.min_turn_silence,
            self.max_turn_silence,
        )
        flags: dict[str, object] = {
            "speech_model": config_builder.enum_value(self.speech_model),
            "format_turns": self.format_turns if self.format_turns is not None else True,
            "encoding": config_builder.enum_value(self.encoding),
            "language_detection": self.language_detection,
            "domain": self.domain,
            "end_of_turn_confidence_threshold": end_of_turn_confidence_threshold,
            "min_turn_silence": min_turn_silence,
            "max_turn_silence": max_turn_silence,
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


def _reject_save_with_show_code(opts: StreamOptions) -> None:
    """Reject any save flag combined with --show-code: the generated SDK code never
    writes audio or a transcript to disk, so silently dropping the save would mislead."""
    for flag, given in (
        ("--save-audio", opts.save_audio is not None),
        ("--save-transcript", opts.save_transcript is not None),
        ("--save-dir", opts.save_dir is not None),
    ):
        if given:
            raise UsageError(
                f"{flag} cannot be combined with --show-code; the generated SDK code "
                "does not save to disk."
            )


@dataclass(frozen=True)
class SaveTargets:
    """Resolved save destinations for one streaming run.

    ``audio`` tees a single source to one WAV; ``audio_by_label`` instead maps each
    parallel ``--system-audio`` channel ("you", "system") to its own WAV when the two
    streams can't share a file. At most one of the two is set; ``transcript`` is the
    single shared transcript either way. ``plan`` is set only under ``--save-dir`` and
    carries the post-stream finalization (auto-name rename, ``--llm`` note, sidecar).
    """

    transcript: Path | None = None
    audio: Path | None = None
    audio_by_label: dict[str, Path] | None = None
    plan: savedir.SaveDirPlan | None = None


def _save_dir_targets(opts: StreamOptions, sources: SourceOptions, save_dir: Path) -> SaveTargets:
    """Resolve ``--save-dir`` into auto-named targets plus the finalization plan.

    ``--save-dir`` owns filename assembly, so it rejects the explicit
    ``--save-audio``/``--save-transcript`` paths and the conflicting ``--name``/
    ``--auto-name`` title pair. Two parallel ``--system-audio`` streams can't tee to one
    WAV, so each channel gets its own ``<stem>-{you,system}.wav`` (one shared transcript);
    ``--no-save-audio`` drops the WAV(s) entirely.
    """
    mutually_exclusive(
        ("--save-dir", True),
        ("--save-audio", opts.save_audio is not None),
        ("--save-transcript", opts.save_transcript is not None),
        suggestion="--save-dir names the files for you; drop the explicit path.",
    )
    mutually_exclusive(
        ("--name", opts.name is not None),
        ("--auto-name", opts.auto_name),
        suggestion="Both set the title — pass --name for an explicit one or "
        "--auto-name to derive it from the transcript.",
    )
    # Local wall-clock time (what a meeting filename wants); the explicit utc-then-
    # astimezone keeps the now() call timezone-aware for the linter.
    now = datetime.now(UTC).astimezone()
    plan = savedir.SaveDirPlan(
        save_dir=save_dir,
        now=now,
        name=opts.name,
        auto_name=opts.auto_name,
        write_note=bool(opts.llm_prompt),
    )
    paths = plan.paths
    naming.ensure_dir(paths.directory)
    if opts.no_save_audio:
        # Transcript + sidecar (+ note) only; no WAV teed for any source.
        return SaveTargets(transcript=paths.transcript, plan=plan)
    if sources.system_audio:
        # Parallel mic + system: one WAV per channel beside the shared transcript.
        return SaveTargets(
            transcript=paths.transcript,
            audio_by_label={
                "you": naming.channel_audio(paths.audio, "you"),
                "system": naming.channel_audio(paths.audio, "system"),
            },
            plan=plan,
        )
    if sources.system_audio_only:
        # A lone system-audio stream; label its single WAV so it reads like the pair.
        return SaveTargets(
            transcript=paths.transcript,
            audio=naming.channel_audio(paths.audio, "system"),
            plan=plan,
        )
    return SaveTargets(transcript=paths.transcript, audio=paths.audio, plan=plan)


def _resolve_save_targets(opts: StreamOptions, sources: SourceOptions) -> SaveTargets:
    """Resolve the save flags into the destinations the session writes.

    ``--save-dir`` owns filename assembly (see ``_save_dir_targets``); the explicit
    ``--save-audio``/``--save-transcript`` paths are the fallback, with the save-dir-only
    ``--name``/``--auto-name``/``--no-save-audio`` flags rejected outside it.
    """
    if opts.save_dir is not None:
        return _save_dir_targets(opts, sources, opts.save_dir)
    if opts.name is not None:
        raise UsageError(
            "--name applies only with --save-dir.",
            suggestion="Pass --save-dir DIR to auto-name the files, "
            "or --save-transcript PATH for an explicit path.",
        )
    if opts.auto_name:
        raise UsageError(
            "--auto-name applies only with --save-dir.",
            suggestion="Pass --save-dir DIR so there's an auto-named file to title.",
        )
    if opts.no_save_audio:
        raise UsageError(
            "--no-save-audio applies only with --save-dir.",
            suggestion="Omit --save-audio to skip the WAV, or pass --save-dir DIR.",
        )
    if opts.save_audio is not None:
        if sources.system_audio:
            raise UsageError(
                "--save-audio cannot be combined with --system-audio; the mic and system "
                "streams can't share one file.",
                suggestion="Pass --save-dir DIR to save one WAV per channel, "
                "or record a single source.",
            )
        record.validate_target(opts.save_audio)
    if opts.save_transcript is not None:
        transcript.validate_target(opts.save_transcript)
    return SaveTargets(transcript=opts.save_transcript, audio=opts.save_audio)


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


def _collect_batch_sources(opts: StreamOptions, *, text_mode: bool) -> list[str]:
    """The newline-delimited source list for ``--from-stdin``, with the flag combos it
    can't honor rejected first.

    ``--from-stdin`` reinterprets stdin as a list of file paths/URLs (one per line),
    each streamed as its own realtime session — distinct from ``-`` (raw PCM bytes). It
    therefore can't also take a positional source, ``--sample``, the mic/system-audio
    inputs, the mic-only capture flags, or ``--show-code`` (which renders one source).
    """
    mutually_exclusive(
        ("--from-stdin", True),
        ("a source argument", opts.source is not None),
        ("--sample", opts.sample),
        suggestion="--from-stdin reads the source list from stdin; don't also pass one.",
    )
    mutually_exclusive(
        ("--from-stdin", True),
        ("--system-audio", opts.system_audio),
        ("--system-audio-only", opts.system_audio_only),
        suggestion="--from-stdin streams files/URLs, not live capture.",
    )
    if opts.device is not None or opts.sample_rate is not None:
        raise UsageError("--device and --sample-rate apply only to microphone input.")
    mutually_exclusive(
        ("--from-stdin", True),
        ("--show-code", opts.show_code),
        suggestion="--show-code renders one source; pass a single file or URL.",
    )
    mutually_exclusive(
        ("--from-stdin", True),
        ("--save-audio", opts.save_audio is not None),
        ("--save-transcript", opts.save_transcript is not None),
        ("--save-dir", opts.save_dir is not None),
        ("--name", opts.name is not None),
        ("--auto-name", opts.auto_name),
        ("--no-save-audio", opts.no_save_audio),
        suggestion="--from-stdin streams many sources; saving applies to a single run.",
    )
    mutually_exclusive(
        ("--llm", bool(opts.llm_prompt)),
        ("-o text", text_mode),
        suggestion="--llm renders a live panel (or NDJSON when piped).",
    )
    sources = list(dict.fromkeys(stdio.iter_piped_stdin_lines()))  # dedupe, keep order
    if not sources:
        raise UsageError(
            "No sources received on stdin.",
            suggestion="Pipe one path or URL per line, e.g. "
            "ls *.wav | assembly stream --from-stdin.",
        )
    return sources


def _run_batch(opts: StreamOptions, state: AppState, *, json_mode: bool, text_mode: bool) -> None:
    """Stream a ``--from-stdin`` list of sources, one realtime session each, in turn."""
    sources = _collect_batch_sources(opts, text_mode=text_mode)
    api_key = state.resolve_api_key()
    base_flags = opts.base_flags()
    llm_prompts = list(opts.llm_prompt or [])
    renderer = StreamRenderer(json_mode=json_mode, text_mode=text_mode)

    def make_session() -> StreamSession:
        return StreamSession(
            api_key=api_key,
            base_flags=base_flags,
            overrides=opts.config_kv,
            config_file=opts.config_file,
            renderer=renderer,
            follow=FollowRenderer(json_mode=json_mode) if llm_prompts else None,
            llm_prompts=llm_prompts,
            model=opts.model,
            max_tokens=opts.max_tokens,
            llm_interval=opts.llm_interval,
        )

    def open_source(source: str) -> tuple[Iterable[bytes], int]:
        file_audio = FileSource(client.resolve_audio_source(source, sample=False))
        return file_audio, file_audio.sample_rate

    stream_batch_sources(
        sources,
        make_session=make_session,
        open_source=open_source,
        renderer=renderer,
        json_mode=json_mode,
    )


def run_stream(opts: StreamOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly stream` invocation from already-parsed flags."""
    text_mode, json_mode = resolve_output_modes(opts.output_field, json_mode=json_mode)
    if opts.from_stdin:
        # SIGTERM stops the stream as cleanly as Ctrl-C, so an external supervisor
        # (Hammerspoon, a service manager, a wrapper's `kill`) can end a recording.
        with signals.terminate_as_interrupt():
            _run_batch(opts, state, json_mode=json_mode, text_mode=text_mode)
        return
    sources = opts.source_options()
    base_flags = opts.base_flags()

    if opts.show_code:
        _reject_save_with_show_code(opts)
        _print_show_code(opts, sources, base_flags, text_mode=text_mode)
        return

    # Validate the requested sources (including that a local file exists) before
    # credentials, so a typo'd path reads as "file not found" — not as a login.
    validate_sources(sources, has_llm=bool(opts.llm_prompt), text_mode=text_mode)
    targets = _resolve_save_targets(opts, sources)
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
        save_audio=targets.audio,
        save_audio_by_label=targets.audio_by_label,
        save_transcript=targets.transcript,
        save_plan=targets.plan,
        llm_interval=opts.llm_interval,
    )
    with signals.terminate_as_interrupt():
        _dispatch(session, sources)
