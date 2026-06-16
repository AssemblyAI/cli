from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import typer

from aai_cli.core import choices, client, config_builder, errors, llm
from aai_cli.core.errors import (
    APIError,
    CLIError,
    NotAuthenticated,
    UsageError,
    mutually_exclusive,
)
from aai_cli.streaming import record
from aai_cli.streaming.render import StreamRenderer, speaker_prefix
from aai_cli.streaming.transcript import TranscriptWriter
from aai_cli.ui import output
from aai_cli.ui.follow import FollowRenderer

# Sources that can be transcribed in parallel sessions: (label, audio chunks, sample rate).
_ParallelStreams = list[tuple[str, Iterable[bytes], int]]


def _finalized_turn_line(event: object, source_label: str | None) -> str | None:
    """The transcript line for a finalized, non-empty turn, or None for a partial/empty one.

    Prefixes the speaker/source label exactly as the text renderer does, so the saved
    file and the ``--llm`` transcript record both read like the on-screen turns.
    """
    if not getattr(event, "end_of_turn", False):
        return None  # partials don't belong in the transcript
    text = getattr(event, "transcript", "") or ""
    if not text:
        return None
    prefix = speaker_prefix(source_label, getattr(event, "speaker_label", None))
    return f"{prefix[0]}: {text}" if prefix is not None else text


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


def validate_output_flags(*, json_mode: bool, output_field: choices.TextOrJson | None) -> None:
    """Reject --json combined with -o text, shared by `stream` and `agent`.

    Same precedent as --llm + -o text: contradictory output shapes are a clean
    usage error, not a silent coin-flip between plain text and NDJSON.
    """
    mutually_exclusive(
        ("--json", json_mode),
        ("-o text", output_field is choices.TextOrJson.text),
        suggestion="Pick one output format.",
    )


def resolve_output_modes(
    output_field: choices.TextOrJson | None, *, json_mode: bool
) -> tuple[bool, bool]:
    """Validate the -o/--json combination, then fold it into (text_mode, json_mode).

    The two steps always run together for the realtime commands (`stream`, `agent`),
    so pairing them here keeps a caller from resolving the modes without first
    rejecting a contradictory pair.
    """
    validate_output_flags(json_mode=json_mode, output_field=output_field)
    return output.stream_output_modes(output_field, json_mode=json_mode)


def validate_sources(opts: SourceOptions, *, has_llm: bool, text_mode: bool) -> None:
    """Reject flag combinations that can't be honored, before any audio is opened."""
    mutually_exclusive(
        ("--system-audio", opts.system_audio),
        ("--system-audio-only", opts.system_audio_only),
    )
    _validate_input_source(opts)
    mutually_exclusive(
        ("--llm", has_llm),
        ("-o text", text_mode),
        suggestion="--llm renders a live panel (or NDJSON when piped).",
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
        if opts.sample:
            # The stdin branch wins dispatch over --sample, so without this the
            # hosted clip would be silently ignored in favor of the pipe.
            raise UsageError("- (stdin) cannot be combined with --sample.")
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
    # When set, tee the streamed PCM to this path as a WAV (see record.tee_wav). The
    # single-source path sets it; the batch (--from-stdin) caller rejects --save-audio.
    save_audio: Path | None = None
    # --system-audio runs two parallel streams that can't share one WAV, so each is teed
    # to its own file keyed by source label ("you", "system"). Set instead of save_audio
    # for that path; a label with no entry (or no save flag) tees nowhere.
    save_audio_by_label: dict[str, Path] | None = None
    # When set, write each finalized turn to this path as plain text (see TranscriptWriter).
    # One shared transcript even under --system-audio (both channels); batch rejects it.
    save_transcript: Path | None = None
    # Seconds between --llm summary refreshes; <=0 re-runs the chain on every turn.
    llm_interval: float = 0.0
    # Monotonic clock, injectable so the interval throttle is deterministic in tests.
    clock: Callable[[], float] = time.monotonic
    transcript: list[str] = field(default_factory=list[str])
    # The open transcript-file writer for a single run; created/closed in _guarded so a
    # save target is opened once per session and a Ctrl-C still leaves a flushed file.
    _transcript_writer: TranscriptWriter | None = None
    _callback_lock: threading.RLock = field(default_factory=threading.RLock)
    _listening_lock: threading.Lock = field(default_factory=threading.Lock)
    _listening_started: bool = False
    # How many turns the last refresh covered, and when it ran (monotonic seconds).
    # -inf so the very first finalized turn always produces an immediate summary.
    _summarized_len: int = 0
    _last_summary_at: float = float("-inf")
    # First CLIError raised by an --llm refresh. The chain runs on the SDK reader
    # thread, where raising would dump a thread traceback instead of reaching
    # run_command, so the error is recorded (and warned once) there and re-raised
    # from the final main-thread flush to fail the command cleanly.
    _llm_error: CLIError | None = None

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
        if self.follow is None:
            with self._callback_lock:
                self.renderer.turn(event, source=source_label)
                line = _finalized_turn_line(event, source_label)
                if line is not None:
                    self._save_line(line)
        else:
            # --llm mode locks only to record the turn; the chain re-runs (network) are
            # left unlocked so the other source's turns keep flowing during a refresh.
            self._record_turn(event, source_label)

    def _save_line(self, line: str) -> None:
        """Append one finalized turn to the open --save-transcript/--save-dir file, if any.

        Always called under ``_callback_lock`` so parallel source threads can't interleave
        a partial write into the file.
        """
        if self._transcript_writer is not None:
            self._transcript_writer.write_turn(line)

    def _record_turn(self, event: object, source_label: str | None) -> None:
        """Append a finalized turn to the running transcript (and the saved file), then
        refresh the --llm answer if a refresh is due (every turn, or once per
        ``llm_interval`` seconds)."""
        line = _finalized_turn_line(event, source_label)
        if line is None:
            return
        with self._callback_lock:
            self.transcript.append(line)
            self._save_line(line)
        self._maybe_summarize()

    def _maybe_summarize(self, *, final: bool = False) -> None:
        """Re-run the prompt chain over the transcript so far and refresh the answer.

        Claims the work under the lock — bumping ``_summarized_len``/``_last_summary_at``
        before releasing — so concurrent source threads never double-run the chain or
        race the throttle. No-op when nothing new has been transcribed, or (unless
        ``final``) when fewer than ``llm_interval`` seconds have elapsed since the last
        refresh. ``final`` forces the closing flush so the tail turns aren't lost."""
        follow = self.follow
        if follow is None:
            return
        if self._llm_error is not None:
            # The chain already failed; don't keep hammering a failing gateway. The
            # final flush runs on the main thread, so re-raise there to fail cleanly.
            if final:
                raise self._llm_error
            return
        with self._callback_lock:
            turns = len(self.transcript)
            if turns <= self._summarized_len:
                return  # nothing new since the last refresh
            now = self.clock()
            throttled = self.llm_interval > 0 and now - self._last_summary_at < self.llm_interval
            if throttled and not final:
                return
            transcript_text = " ".join(self.transcript)
            self._summarized_len = turns
            self._last_summary_at = now
        try:
            answer = llm.run_chain(
                self.api_key,
                self.llm_prompts,
                transcript_text=transcript_text,
                model=self.model,
                max_tokens=self.max_tokens,
            )
        except CLIError as exc:
            self._llm_error = exc
            if final:
                raise
            # Reader thread: raising would print a thread traceback, so warn on
            # stderr now and let the final flush surface the error + exit code.
            output.error_console.print(
                f"[aai.muted]--llm refresh failed: {exc.message}[/aai.muted]"
            )
            return
        follow(answer, turns)

    def _audio_target(self, source_label: str | None) -> Path | None:
        """The WAV path this source tees to, if any.

        ``--system-audio`` records two channels to two files (``save_audio_by_label``);
        every other run tees its single source to ``save_audio``.
        """
        if self.save_audio_by_label is not None:
            return self.save_audio_by_label.get(source_label or "")
        return self.save_audio

    def stream_one(
        self, audio: Iterable[bytes], rate: int, *, source_label: str | None = None
    ) -> None:
        target = self._audio_target(source_label)
        if target is not None:
            # Tee verbatim to disk at the source's true rate before it hits the wire.
            audio = record.tee_wav(audio, target, rate=rate)
        flags = self.base_flags | {"sample_rate": rate}
        if source_label == "you":
            # The microphone captures you alone, so never diarize it into separate
            # speakers — force speaker_labels off so the mic stays labeled "You" even
            # when --speaker-labels splits the system audio into speakers.
            flags = flags | {"speaker_labels": False, "max_speakers": None}
        merged = config_builder.merge_streaming_params(
            flags=flags,
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

    def _guarded(self, work: Callable[[], None], *, handle_interrupt: bool = True) -> None:
        """Run a streaming body with the shared lifecycle handling: enter the
        FollowRenderer's live panel if present, treat Ctrl-C as a clean stop (exit 130),
        exit 0 on a closed downstream pipe, and always close the renderer.

        ``handle_interrupt=False`` lets a Ctrl-C or a closed pipe propagate instead of
        being swallowed here — the batch driver owns those signals across the whole
        ``--from-stdin`` sequence, so one Ctrl-C stops the batch rather than just
        advancing to the next source."""
        if self.save_transcript is not None:
            # Open before streaming so a bad path fails up front; the finally closes it
            # (flushing the turns so far) even on Ctrl-C or a worker error.
            self._transcript_writer = TranscriptWriter(self.save_transcript)
        try:
            if self.follow is not None:
                with self.follow:
                    try:
                        work()
                    finally:
                        # Flush a closing summary (incl. on Ctrl-C) so turns since the
                        # last interval tick are reflected, while the panel's still live.
                        self._maybe_summarize(final=True)
            else:
                work()
        except KeyboardInterrupt:
            if not handle_interrupt:
                raise
            # Ctrl-C is a clean "user stopped" signal: flush the closing state, print
            # "Stopped.", then exit 130 (the cancel code) so a `stream && next` chain
            # doesn't treat the interrupt as success.
            if self.follow is None:
                self.renderer.close()
                self.renderer.stopped()
            raise typer.Exit(code=errors.CANCELLED_EXIT_CODE) from None
        except BrokenPipeError:
            if not handle_interrupt:
                raise
            # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
            raise typer.Exit(code=0) from None
        finally:
            if self._transcript_writer is not None:
                self._transcript_writer.close()
            if self.follow is None:
                self.renderer.close()

    def run(
        self,
        audio: Iterable[bytes],
        rate: int,
        *,
        source_label: str | None = None,
        handle_interrupt: bool = True,
    ) -> None:
        self._guarded(
            lambda: self.stream_one(audio, rate, source_label=source_label),
            handle_interrupt=handle_interrupt,
        )

    def run_parallel(self, streams: _ParallelStreams) -> None:
        self._guarded(lambda: self._drive(streams))

    def _drive(self, streams: _ParallelStreams) -> None:
        """Stream every source concurrently, surfacing the first worker error."""
        errors: queue.Queue[Exception] = queue.Queue()

        def worker(source_label: str, audio: Iterable[bytes], rate: int) -> None:
            try:
                try:
                    self.stream_one(audio, rate, source_label=source_label)
                except (CLIError, BrokenPipeError):
                    raise
                except Exception as exc:
                    # A non-CLIError here is a bug, but it must still fail the run:
                    # uncaught, it dies with this daemon thread and the command
                    # exits 0 for a stream that actually failed.
                    raise APIError(f"Streaming worker ({source_label}) failed: {exc}") from exc
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
                # Poll interval: a responsiveness/CPU tradeoff, not behavior -- the loop
                # surfaces a worker error within ~0.1s. Exact value isn't assertable.
                thread.join(timeout=0.1)  # pragma: no mutate
            if not errors.empty():
                raise errors.get()
        if not errors.empty():
            raise errors.get()


# A batch source string resolved to its real-time audio chunks and declared rate.
_OpenedSource = tuple[Iterable[bytes], int]


def stream_batch_sources(
    sources: list[str],
    *,
    make_session: Callable[[], StreamSession],
    open_source: Callable[[str], _OpenedSource],
    renderer: StreamRenderer,
    json_mode: bool,
) -> None:
    """Stream each source in ``sources`` in turn — the ``assembly stream --from-stdin``
    batch mode.

    The realtime API is one session at a time, so a list of files/URLs streams
    sequentially: each source gets a fresh ``StreamSession`` from ``make_session`` (its
    own transcript and ``--llm`` chain state) and is announced via ``renderer.source``
    before its turns. ``open_source`` resolves a source string to ``(audio, rate)`` and
    may raise ``CLIError`` (bad path, missing ffmpeg, decode failure), which is recorded
    as a per-source failure so the batch carries on — except ``NotAuthenticated``, which
    re-raises to abort the whole batch (one rejected key fails every source identically).

    A Ctrl-C stops the batch with the cancel code (exit 130); a closed downstream pipe
    stops it quietly (exit 0). When any source failed, raises a ``CLIError`` at the end
    so a script can trust the exit code.
    """
    total = len(sources)
    failures: list[str] = []
    try:
        for index, source in enumerate(sources, start=1):
            renderer.source(source, index=index, total=total)
            try:
                audio, rate = open_source(source)
                make_session().run(audio, rate, handle_interrupt=False)
            except NotAuthenticated:
                raise
            except CLIError as exc:
                failures.append(source)
                output.emit_warning(f"{source}: {exc.message}", json_mode=json_mode)
    except KeyboardInterrupt:
        # One Ctrl-C stops the whole batch, not just the current source. Exit 130
        # (cancel) so the interrupt isn't mistaken for a clean run of every source.
        renderer.stopped()
        raise typer.Exit(code=errors.CANCELLED_EXIT_CODE) from None
    except BrokenPipeError:
        # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
        raise typer.Exit(code=0) from None
    finally:
        renderer.close()
    if failures:
        raise CLIError(
            f"{len(failures)} of {total} sources failed.",
            error_type="batch_failed",
            suggestion="Check each failed path or URL, then re-run.",
        )
