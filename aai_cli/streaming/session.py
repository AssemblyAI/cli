from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import typer

from aai_cli import client, config_builder, llm
from aai_cli.errors import CLIError, UsageError
from aai_cli.follow import FollowRenderer
from aai_cli.streaming.render import StreamRenderer

# Sources that can be transcribed in parallel sessions: (label, audio chunks, sample rate).
_ParallelStreams = list[tuple[str, Iterable[bytes], int]]


@dataclass(frozen=True)
class SourceOptions:
    """Where the audio comes from, distilled from the CLI flags.

    Centralizes the "which input?" predicates so the validation and dispatch helpers
    in the command module read off one object instead of re-deriving the same booleans.
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


def validate_sources(opts: SourceOptions, *, has_llm: bool, text_mode: bool) -> None:
    """Reject flag combinations that can't be honored, before any audio is opened."""
    if opts.system_audio and opts.system_audio_only:
        raise UsageError("Use either --system-audio or --system-audio-only, not both.")
    _validate_input_source(opts)
    if has_llm and text_mode:
        raise UsageError(
            "--llm renders a live panel (or NDJSON when piped); it can't be combined with -o text."
        )


def _validate_input_source(opts: SourceOptions) -> None:
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
class StreamSession:
    """Owns one streaming run: the renderers, the LLM-chain state, and the audio
    plumbing shared across single- and parallel-source streaming.

    Holding this as an object (rather than a nest of closures inside the command body)
    keeps each step a small, independently readable method, and collapses the ~25
    per-call flags into one ``base_flags`` dict that only varies by sample rate.
    """

    api_key: str
    base_flags: dict[str, object]
    overrides: list[str] | None
    config_file: str | Path | None
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
