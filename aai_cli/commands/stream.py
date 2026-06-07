from __future__ import annotations

import queue
import tempfile
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import typer
from assemblyai.streaming.v3 import Encoding, NoiseSuppressionModel, SpeechModel

from aai_cli import (
    choices,
    client,
    code_gen,
    config,
    config_builder,
    help_panels,
    llm,
    output,
    youtube,
)
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError, UsageError
from aai_cli.follow import FollowRenderer
from aai_cli.help_text import examples_epilog
from aai_cli.microphone import MicrophoneSource
from aai_cli.streaming.macos import MacSystemAudioSource
from aai_cli.streaming.render import StreamRenderer
from aai_cli.streaming.sources import TARGET_RATE, FileSource, StdinSource

app = typer.Typer()

DEFAULT_SPEECH_MODEL = SpeechModel.u3_rt_pro

# Sources that can be transcribed in parallel sessions: (label, audio chunks, sample rate).
_ParallelStreams = list[tuple[str, Iterable[bytes], int]]


@dataclass(frozen=True)
class _SourceOptions:
    """Where the audio comes from, distilled from the CLI flags.

    Centralizes the "which input?" predicates so the validation and dispatch helpers
    below read off one object instead of re-deriving the same booleans.
    """

    source: str | None
    sample: bool
    sample_rate: int | None
    device: int | None
    system_audio: bool
    system_audio_only: bool

    @property
    def from_stdin(self) -> bool:
        return self.source == "-"

    @property
    def from_file(self) -> bool:
        return bool(self.source) or self.sample

    @property
    def from_system_audio(self) -> bool:
        return self.system_audio or self.system_audio_only

    @property
    def has_capture_overrides(self) -> bool:
        """Whether a microphone-only flag (--sample-rate or --device) was given."""
        return self.sample_rate is not None or self.device is not None


def _validate_sources(opts: _SourceOptions, *, has_llm: bool, text_mode: bool) -> None:
    """Reject flag combinations that can't be honored, before any audio is opened."""
    if opts.system_audio and opts.system_audio_only:
        raise UsageError("Use either --system-audio or --system-audio-only, not both.")
    _validate_input_source(opts)
    if has_llm and text_mode:
        raise UsageError(
            "--llm renders a live panel (or NDJSON when piped); it can't be combined with -o text."
        )


def _validate_input_source(opts: _SourceOptions) -> None:
    """Reject --sample-rate/--device/source combinations the chosen input can't accept."""
    if opts.from_system_audio:
        if opts.from_file:
            raise UsageError("--system-audio cannot be combined with an audio source or --sample.")
        if opts.system_audio_only and opts.has_capture_overrides:
            raise UsageError(
                "--sample-rate and --device require microphone input; use --system-audio."
            )
    elif opts.from_stdin:
        if opts.device is not None:
            raise UsageError("--device applies only to microphone input.")
    elif opts.from_file and opts.has_capture_overrides:
        raise UsageError("--sample-rate and --device apply only to microphone input.")


@dataclass
class _StreamSession:
    """Owns one streaming run: the renderers, the LLM-chain state, and the audio
    plumbing shared across single- and parallel-source streaming.

    Holding this as an object (rather than a nest of closures inside the command body)
    keeps each step a small, independently readable method, and collapses the ~25
    per-call flags into one ``base_flags`` dict that only varies by sample rate.
    """

    api_key: str
    base_flags: dict[str, object]
    overrides: list[str] | None
    config_file: str | None
    renderer: StreamRenderer
    follow: FollowRenderer | None
    llm_prompts: list[str]
    model: str
    max_tokens: int
    transcript: list[str] = field(default_factory=list[str])
    _callback_lock: threading.RLock = field(default_factory=threading.RLock)
    _listening_lock: threading.Lock = field(default_factory=threading.Lock)
    _listening_started: bool = False

    @property
    def on_open(self) -> Callable[[], None]:
        """First-audio callback: announce "Listening…" once — unless the FollowRenderer
        owns the screen in --llm mode, where the notice would clutter the live panel."""
        return (lambda: None) if self.follow is not None else self._listening_once

    def _listening_once(self) -> None:
        with self._listening_lock:
            if self._listening_started:
                return
            self._listening_started = True
        self.renderer.listening()

    def on_turn(self, event: object, *, source_label: str | None = None) -> None:
        with self._callback_lock:
            if self.follow is None:
                self.renderer.turn(event, source=source_label)
            else:
                self._refresh_answer(event, source_label)

    def _refresh_answer(self, event: object, source_label: str | None) -> None:
        """Live --llm mode: re-run the prompt chain over the growing transcript on every
        finalized turn, refreshing one evolving answer (partials are ignored)."""
        follow = self.follow
        if follow is None or not getattr(event, "end_of_turn", False):
            return
        text = getattr(event, "transcript", "") or ""
        if not text:
            return
        if source_label is not None:
            display_source = {"system": "System", "you": "You"}.get(source_label, source_label)
            text = f"{display_source}: {text}"
        self.transcript.append(text)
        answer = llm.run_chain(
            self.api_key,
            self.llm_prompts,
            transcript_text=" ".join(self.transcript),
            model=self.model,
            max_tokens=self.max_tokens,
        )
        follow(answer, len(self.transcript))

    def stream_one(
        self, audio: Iterable[bytes], rate: int, *, source_label: str | None = None
    ) -> None:
        merged = config_builder.merge_streaming_params(
            flags=self.base_flags | {"sample_rate": rate},
            overrides=self.overrides,
            config_file=self.config_file,
        )
        params = config_builder.construct_streaming_params(merged)
        client.stream_audio(
            self.api_key,
            audio,
            params=params,
            on_begin=(
                None
                if self.follow is not None
                else lambda event: self.renderer.begin(event, source=source_label)
            ),
            on_turn=lambda event: self.on_turn(event, source_label=source_label),
            on_termination=(
                None
                if self.follow is not None
                else lambda event: self.renderer.termination(event, source=source_label)
            ),
        )

    def _guarded(self, work: Callable[[], None]) -> None:
        """Run a streaming body with the shared lifecycle handling: enter the
        FollowRenderer's live panel if present, treat Ctrl-C as a clean stop, exit 0 on
        a closed downstream pipe, and always close the renderer."""
        try:
            if self.follow is not None:
                with self.follow:
                    work()
            else:
                work()
        except KeyboardInterrupt:
            # Ctrl-C is a normal "user stopped" signal -> exit 0.
            if self.follow is None:
                self.renderer.close()
                self.renderer.stopped()
        except BrokenPipeError:
            # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
            raise typer.Exit(code=0) from None
        finally:
            if self.follow is None:
                self.renderer.close()

    def run(self, audio: Iterable[bytes], rate: int, *, source_label: str | None = None) -> None:
        self._guarded(lambda: self.stream_one(audio, rate, source_label=source_label))

    def run_parallel(self, streams: _ParallelStreams) -> None:
        self._guarded(lambda: self._drive(streams))

    def _drive(self, streams: _ParallelStreams) -> None:
        """Stream every source concurrently, surfacing the first worker error."""
        errors: queue.Queue[Exception] = queue.Queue()

        def worker(source_label: str, audio: Iterable[bytes], rate: int) -> None:
            try:
                self.stream_one(audio, rate, source_label=source_label)
            except (CLIError, BrokenPipeError) as exc:
                errors.put(exc)

        threads = [
            threading.Thread(target=worker, args=(label, audio, rate), daemon=True)
            for label, audio, rate in streams
        ]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.1)
            if not errors.empty():
                raise errors.get()
        if not errors.empty():
            raise errors.get()


def _dispatch(session: _StreamSession, opts: _SourceOptions) -> None:
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
        # Raw PCM16 mono piped on stdin (e.g. `ffmpeg … -f s16le - | aai stream -`).
        stdin_src = StdinSource(sample_rate=opts.sample_rate or TARGET_RATE)
        session.run(stdin_src, stdin_src.sample_rate)
    elif opts.source and youtube.is_youtube_url(opts.source):
        # Fetch the audio first, then stream the local file in real time.
        with tempfile.TemporaryDirectory(prefix="aai-yt-") as td:
            local = youtube.download_audio(opts.source, Path(td))
            session.run(FileSource(str(local)), TARGET_RATE)
    elif opts.from_file:
        file_audio = FileSource(client.resolve_audio_source(opts.source, sample=opts.sample))
        session.run(file_audio, file_audio.sample_rate)
    else:
        # Capture at the device's native rate (or --sample-rate override) and tell the
        # streaming API that rate, rather than forcing one the device may reject.
        # "Listening…" is announced once the device is open (see _StreamSession.on_open),
        # not when the session opens — so early speech isn't lost in the gap.
        mic = MicrophoneSource(
            device=opts.device, capture_rate=opts.sample_rate, on_open=session.on_open
        )
        session.run(mic, mic.sample_rate)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Stream from your microphone", "aai stream"),
            ("Stream mic + system audio on macOS", "aai stream --system-audio"),
            ("Stream the hosted sample", "aai stream --sample"),
            (
                "Summarize action items live as you talk",
                'aai stream --llm "summarize action items"',
            ),
            ("Print equivalent Python instead of running", "aai stream --show-code"),
        ]
    ),
)
def stream(
    ctx: typer.Context,
    source: str | None = typer.Argument(
        None,
        help="Audio file path, URL, or YouTube URL to stream. Omit to use the microphone.",
    ),
    sample: bool = typer.Option(False, "--sample", help="Stream the hosted wildfires.mp3 sample."),
    # audio capture
    sample_rate: int | None = typer.Option(
        None,
        "--sample-rate",
        help="Force a microphone capture rate in Hz (default: device native).",
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    device: int | None = typer.Option(
        None, "--device", help="Microphone device index.", rich_help_panel=help_panels.OPT_CAPTURE
    ),
    system_audio: bool = typer.Option(
        False,
        "--system-audio",
        help="macOS only: stream system/app audio and microphone as separate sessions.",
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    system_audio_only: bool = typer.Option(
        False,
        "--system-audio-only",
        help="macOS only: stream system/app audio without the microphone.",
        rich_help_panel=help_panels.OPT_CAPTURE,
    ),
    # model & input
    speech_model: SpeechModel = typer.Option(
        DEFAULT_SPEECH_MODEL,
        "--speech-model",
        help="Streaming speech model.",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    encoding: Encoding | None = typer.Option(
        None,
        "--encoding",
        help="Audio encoding.",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    language_detection: bool | None = typer.Option(
        None,
        "--language-detection",
        help="Auto-detect the spoken language.",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    domain: str | None = typer.Option(
        None,
        "--domain",
        help="Domain preset (e.g. medical).",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Prompt to bias the speech model (u3-pro).",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    keyterms_prompt: list[str] | None = typer.Option(
        None,
        "--keyterms-prompt",
        help="Boost a key term (repeatable).",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    # turn detection
    end_of_turn_confidence_threshold: float | None = typer.Option(
        None,
        "--end-of-turn-confidence-threshold",
        help="End-of-turn confidence (0-1).",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    min_turn_silence: int | None = typer.Option(
        None,
        "--min-turn-silence",
        help="Min silence to end a turn (ms).",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    max_turn_silence: int | None = typer.Option(
        None,
        "--max-turn-silence",
        help="Max silence before ending a turn (ms).",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    vad_threshold: float | None = typer.Option(
        None,
        "--vad-threshold",
        help="Voice-activity threshold.",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    format_turns: bool | None = typer.Option(
        None,
        "--format-turns/--no-format-turns",
        help="Punctuate/format finalized turns.",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    include_partial_turns: bool | None = typer.Option(
        None,
        "--include-partial-turns",
        help="Emit partial turns.",
        rich_help_panel=help_panels.OPT_TURNS,
    ),
    # speakers
    speaker_labels: bool | None = typer.Option(
        None, "--speaker-labels", help="Label speakers.", rich_help_panel=help_panels.OPT_SPEAKERS
    ),
    max_speakers: int | None = typer.Option(
        None, "--max-speakers", help="Max speakers.", rich_help_panel=help_panels.OPT_SPEAKERS
    ),
    # features
    voice_focus: NoiseSuppressionModel | None = typer.Option(
        None,
        "--voice-focus",
        help="Voice focus (noise suppression model).",
        rich_help_panel=help_panels.OPT_FEATURES,
    ),
    voice_focus_threshold: float | None = typer.Option(
        None,
        "--voice-focus-threshold",
        help="Voice-focus threshold.",
        rich_help_panel=help_panels.OPT_FEATURES,
    ),
    inactivity_timeout: int | None = typer.Option(
        None,
        "--inactivity-timeout",
        help="Auto-close after N seconds idle.",
        rich_help_panel=help_panels.OPT_FEATURES,
    ),
    # guardrails
    filter_profanity: bool | None = typer.Option(
        None,
        "--filter-profanity",
        help="Mask profanity.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii: bool | None = typer.Option(
        None,
        "--redact-pii",
        help="Redact PII from turns.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii_policy: str | None = typer.Option(
        None,
        "--redact-pii-policy",
        help="Comma-separated PII policies.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii_sub: str | None = typer.Option(
        None,
        "--redact-pii-sub",
        help="Replace redacted PII with: hash or entity_name.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    # webhooks
    webhook_url: str | None = typer.Option(
        None, "--webhook-url", help="Webhook URL.", rich_help_panel=help_panels.OPT_WEBHOOKS
    ),
    webhook_auth_header: str | None = typer.Option(
        None,
        "--webhook-auth-header",
        help="Webhook auth header as NAME:VALUE.",
        rich_help_panel=help_panels.OPT_WEBHOOKS,
    ),
    # llm transform
    llm_prompt: list[str] | None = typer.Option(
        None,
        "--llm",
        help="Run a prompt over the live transcript through LLM Gateway, refreshing the "
        "answer on every finalized turn. Repeatable: each prompt runs on the previous "
        "one's response (a chain).",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL, "--model", help="LLM Gateway model.", rich_help_panel=help_panels.OPT_LLM
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max tokens.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    # escape hatch
    config_kv: list[str] | None = typer.Option(
        None,
        "--config",
        help="Set any StreamingParameters field as KEY=VALUE (repeatable).",
        rich_help_panel=help_panels.OPT_ADVANCED,
    ),
    config_file: str | None = typer.Option(
        None,
        "--config-file",
        help="JSON file of streaming fields.",
        rich_help_panel=help_panels.OPT_ADVANCED,
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
    output_field: choices.TextOrJson | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output mode: text (finalized turns as plain lines, pipe-friendly) or json.",
    ),
    show_code: bool = typer.Option(
        False,
        "--show-code",
        help="Print the equivalent Python SDK code and exit (does not stream).",
    ),
) -> None:
    """Transcribe live audio in real time — from your mic, a file, or a URL.

    --prompt biases the speech model. --llm runs a prompt over the live transcript
    through LLM Gateway, refreshing the answer on every finalized turn (e.g.
    "summarize action items").
    """

    def body(state: AppState, json_mode: bool) -> None:
        text_mode, json_mode = output.stream_output_modes(output_field, json_mode=json_mode)
        opts = _SourceOptions(
            source=source,
            sample=sample,
            sample_rate=sample_rate,
            device=device,
            system_audio=system_audio,
            system_audio_only=system_audio_only,
        )
        # Every streaming flag except sample_rate, which is set per source at stream time.
        base_flags: dict[str, object] = {
            "speech_model": config_builder.enum_value(speech_model),
            "format_turns": format_turns if format_turns is not None else True,
            "encoding": config_builder.enum_value(encoding),
            "language_detection": language_detection,
            "domain": domain,
            "end_of_turn_confidence_threshold": end_of_turn_confidence_threshold,
            "min_turn_silence": min_turn_silence,
            "max_turn_silence": max_turn_silence,
            "vad_threshold": vad_threshold,
            "include_partial_turns": include_partial_turns,
            "keyterms_prompt": list(keyterms_prompt) if keyterms_prompt else None,
            "filter_profanity": filter_profanity,
            "speaker_labels": speaker_labels,
            "max_speakers": max_speakers,
            "voice_focus": config_builder.enum_value(voice_focus),
            "voice_focus_threshold": voice_focus_threshold,
            "redact_pii": redact_pii,
            "redact_pii_policies": config_builder.split_csv(redact_pii_policy),
            "redact_pii_sub": redact_pii_sub,
            "inactivity_timeout": inactivity_timeout,
            "webhook_url": webhook_url,
            "prompt": prompt,
        }
        base_flags.update(config_builder.auth_header_flags(webhook_auth_header))

        if show_code:
            # Print-only: emit the canonical microphone-streaming script (16 kHz) from
            # the flags and exit without opening audio or authenticating. Raw stdout so
            # `--show-code > script.py` yields a runnable file.
            if opts.from_system_audio:
                raise UsageError("--show-code does not support macOS system audio capture yet.")
            merged = config_builder.merge_streaming_params(
                flags=base_flags | {"sample_rate": TARGET_RATE},
                overrides=config_kv,
                config_file=config_file,
            )
            gateway = code_gen.gateway_options(list(llm_prompt or []), model, max_tokens)
            output.print_code(code_gen.stream(merged, llm=gateway))
            return

        api_key = config.resolve_api_key(profile=state.profile)
        _validate_sources(opts, has_llm=bool(llm_prompt), text_mode=text_mode)

        llm_prompts = list(llm_prompt or [])
        session = _StreamSession(
            api_key=api_key,
            base_flags=base_flags,
            overrides=config_kv,
            config_file=config_file,
            renderer=StreamRenderer(json_mode=json_mode, text_mode=text_mode),
            follow=FollowRenderer(json_mode=json_mode) if llm_prompts else None,
            llm_prompts=llm_prompts,
            model=model,
            max_tokens=max_tokens,
        )
        _dispatch(session, opts)

    run_command(ctx, body, json=json_out)
