"""Input + output-mode validation for the realtime commands (`stream`, `agent`, …).

The ``StreamSession`` execution engine (``session.py``) doesn't use any of this — only
the command layer does, ahead of opening audio or credentials. Keeping the "which input,
which output shape, is this flag combo legal?" logic here (the frozen ``SourceOptions``
carrier plus the validate/resolve helpers) leaves ``session.py`` to the run machinery,
mirroring the ``app/transcribe`` run/validate split.
"""

from __future__ import annotations

from dataclasses import dataclass

from aai_cli.core import choices
from aai_cli.core.errors import UsageError, mutually_exclusive
from aai_cli.ui import output


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
