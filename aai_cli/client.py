from __future__ import annotations

import contextlib
import json
import re
from abc import abstractmethod
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any, Literal, Protocol

import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)

from aai_cli import environments, jsonshape, stdio
from aai_cli.errors import APIError, CLIError, UsageError, auth_failure, is_auth_failure
from aai_cli.streaming.diagnostics import classify_error, silence_streaming_logging

SAMPLE_AUDIO_URL = "https://assembly.ai/wildfires.mp3"
_StreamHandler = Callable[[Any, Any], object]


class _StreamingClientLike(Protocol):
    def on(self, event: StreamingEvents, handler: _StreamHandler) -> None:
        """Register a handler for a streaming event."""

    def connect(self, params: StreamingParameters) -> None:
        """Open the realtime session."""

    def stream(self, data: bytes | Generator[bytes, None, None] | Iterable[bytes]) -> None:
        """Send audio to the session."""

    def disconnect(self, *, terminate: Literal[False, True] = False) -> None:  # pragma: no mutate
        """Close the session."""


def _make_streaming_client(api_key: str) -> _StreamingClientLike:
    client: _StreamingClientLike = StreamingClient(
        StreamingClientOptions(api_key=api_key, api_host=environments.active().streaming_host)
    )
    return client


def resolve_audio_source(source: str | None, *, sample: bool, check_local: bool = True) -> str:
    """The audio reference to use: the hosted --sample clip, else the given path/URL.

    Shared by `transcribe` and `stream` so both accept a file or URL and `--sample`.
    A local path that doesn't exist fails here — before any credential resolution or
    request — so a typo'd filename reads as "file not found", not as an API failure.
    ``--show-code`` paths pass ``check_local=False``: generating code for a file you
    don't have yet is legitimate.
    """
    if sample:
        if source:
            # Never silently prefer one over the other: the user asked for both.
            raise UsageError(
                "An audio source and --sample cannot be combined.",
                suggestion="Pass the file/URL or --sample, not both.",
            )
        return SAMPLE_AUDIO_URL
    if not source:
        raise UsageError(
            "Provide an audio path or URL.",
            suggestion="Or pass --sample to use the hosted demo file.",
        )
    if check_local and not source.startswith(("http://", "https://")):
        path = Path(source)
        if not path.exists():
            raise CLIError(
                f"File not found: {source}",
                error_type="file_not_found",
                exit_code=2,
                suggestion="Check the path. For remote audio, pass an http(s):// URL.",
            )
        if not path.is_file():
            # A directory (or socket/FIFO) would otherwise fall through to credential
            # resolution and fail much later as an opaque upload error.
            raise CLIError(
                f"Not a file: {source}",
                error_type="not_a_file",
                exit_code=2,
                suggestion="Pass an audio file, not a directory.",
            )
    return source


def _configure(api_key: str) -> None:
    aai.settings.api_key = api_key
    aai.settings.base_url = environments.active().api_base


@contextlib.contextmanager
def _sdk_errors(message: str) -> Generator[None]:
    """Normalize SDK exceptions for one call: a rejected key becomes a single clean
    auth_failure(); any other error becomes APIError(f"{message}: {exc}").

    CLIErrors (including an APIError the SDK call raised itself) and a closed
    downstream pipe pass through untouched; KeyboardInterrupt propagates as a
    BaseException. This is the one shape every wrapper around the assemblyai SDK
    shares — keeping it here means auth/error classification lives in one place.
    """
    try:
        yield
    except (CLIError, BrokenPipeError):
        raise
    except Exception as exc:
        if is_auth_failure(exc):
            raise auth_failure() from exc
        # Compact the reason (httpx-backed SDK errors embed a multi-line
        # `Request: <…>` repr) and drop the SDK's own "failed to …" preamble when
        # `message` already says the same thing, so the error reads once, cleanly.
        reason = _SDK_PREAMBLE_RE.sub("", _compact_reason(exc))
        raise APIError(
            f"{message}: {reason}",
            suggestion="Check your network and try again.",
        ) from exc


def _list_transcript_params(limit: int) -> aai.ListTranscriptParameters:
    """List-transcripts params that serialize without the spurious ``model_config`` key.

    assemblyai==0.64.4 under pydantic==2.13.4: the SDK's pydantic-v1-shim request model
    picks up the v2-style ``model_config`` class attribute as a regular field, so the
    ``.dict(exclude_none=True)`` the SDK puts on the query string ships a junk
    ``?model_config=...`` param on every request. Null the bogus field out so
    ``exclude_none`` drops it from the wire.
    """
    params = aai.ListTranscriptParameters(limit=limit)
    object.__setattr__(params, "model_config", None)
    return params


# httpx-backed SDK errors embed a multi-line repr ("…\nReason: …\nRequest: <Request(…)>").
_REQUEST_REPR_RE = re.compile(r"Request: <[^>]*>")

# The SDK prefixes transcript-history errors with its own "failed to retrieve
# transcript(s) …:" preamble, doubling up with the wrapper's message ("Could not
# list transcripts: failed to retrieve transcripts: …"). Strip just that known
# preamble; transcript ids are url-safe tokens, so the optional id segment can
# never swallow a colon-bearing reason (e.g. a URL).
_SDK_PREAMBLE_RE = re.compile(r"^failed to retrieve transcripts?(?: [A-Za-z0-9_-]+)?: ")


def _compact_reason(exc: object) -> str:
    """``str(exc)`` as a single clean line: drop the trailing ``Request: <…>`` repr and
    collapse all whitespace/newlines, keeping the informative reason text."""
    text = _REQUEST_REPR_RE.sub("", str(exc))
    return re.sub(r"\s+", " ", text).strip()


def validate_key(api_key: str) -> bool:
    """True if the key authenticates, False on an auth failure. Raises APIError otherwise."""
    _configure(api_key)
    try:
        aai.Transcriber().list_transcripts(_list_transcript_params(1))
    except aai.types.AssemblyAIError as exc:
        if is_auth_failure(exc):
            return False
        raise APIError(f"Could not validate key: {_compact_reason(exc)}") from exc
    except Exception as exc:
        raise APIError(f"Network error contacting AssemblyAI: {_compact_reason(exc)}") from exc
    return True


def _item_to_dict(item: Any) -> dict[str, Any]:
    """JSON-safe dict for an SDK model across pydantic v2 (model_dump) and v1 (.json)."""
    if hasattr(item, "model_dump"):
        return dict(item.model_dump(mode="json"))
    return dict(json.loads(item.json()))  # pydantic v1 (assemblyai transcription models)


def list_transcripts(api_key: str, *, limit: int = 10) -> list[dict[str, object]]:
    _configure(api_key)
    with _sdk_errors("Could not list transcripts"):
        resp = aai.Transcriber().list_transcripts(_list_transcript_params(limit))
    return [_item_to_dict(item) for item in resp.transcripts]


def transcribe(api_key: str, audio: str, *, config: aai.TranscriptionConfig) -> aai.Transcript:
    _configure(api_key)
    with _sdk_errors("Transcription request failed"):
        transcript = aai.Transcriber().transcribe(audio, config=config)
    if transcript.status == aai.TranscriptStatus.error:
        raise APIError(transcript.error or "Transcription failed.", transcript_id=transcript.id)
    return transcript


def status_str(transcript: aai.Transcript) -> str:
    """The transcript's status as a plain string (SDK enum `.value` or raw value)."""
    status = transcript.status
    return str(getattr(status, "value", status))


def transcript_summary(transcript: Any) -> dict[str, object]:
    """The compact ``{id, status, text}`` dict the commands emit for a transcript."""
    return {
        "id": transcript.id,
        "status": status_str(transcript),
        "text": transcript.text,
    }


def transcript_json_payload(transcript: Any) -> dict[str, object]:
    """The transcript's full ``json_response`` if present, else the compact summary."""
    return getattr(transcript, "json_response", None) or transcript_summary(transcript)


def _transcript_text(transcript: Any) -> str:
    return str(getattr(transcript, "text", "") or "")


def _render_utterances(transcript: Any) -> str:
    utterances = jsonshape.object_list(getattr(transcript, "utterances", None))
    if not utterances:
        return _transcript_text(transcript)
    return "\n".join(
        f"Speaker {getattr(utterance, 'speaker', '')}: {getattr(utterance, 'text', '')}"
        for utterance in utterances
    )


class _SubtitleTranscript(Protocol):
    """The slice of ``aai.Transcript`` the subtitle renderers touch."""

    @abstractmethod
    def export_subtitles_srt(self, chars_per_caption: int | None) -> str:
        """Fetch the transcript's SRT captions."""

    @abstractmethod
    def export_subtitles_vtt(self, chars_per_caption: int | None) -> str:
        """Fetch the transcript's VTT captions."""


def _export_srt(transcript: _SubtitleTranscript, chars_per_caption: int | None) -> str:
    # The SDK fetches SRT from the `/srt` export endpoint, so this hits the network.
    with _sdk_errors("Could not export SRT subtitles"):
        return str(transcript.export_subtitles_srt(chars_per_caption=chars_per_caption))


def _export_vtt(transcript: _SubtitleTranscript, chars_per_caption: int | None) -> str:
    # The SDK fetches VTT from the `/vtt` export endpoint, so this hits the network.
    with _sdk_errors("Could not export VTT subtitles"):
        return str(transcript.export_subtitles_vtt(chars_per_caption=chars_per_caption))


# Subtitle fields hit an export endpoint and take the --chars-per-caption knob.
_SUBTITLE_RENDERERS: dict[str, Callable[[_SubtitleTranscript, int | None], str]] = {
    "srt": _export_srt,
    "vtt": _export_vtt,
}

# Output field -> renderer. Fields absent here fall back to the plain transcript text.
_FIELD_RENDERERS: dict[str, Callable[[Any], str]] = {
    "id": lambda t: str(getattr(t, "id", "") or ""),
    "status": status_str,
    "utterances": _render_utterances,
    "json": lambda t: json.dumps(transcript_json_payload(t), default=str),
}


def validate_chars_per_caption(chars_per_caption: int | None, field: str | None) -> None:
    """``--chars-per-caption`` only shapes subtitle exports; any other ``-o`` contradicts it."""
    if chars_per_caption is not None and field not in _SUBTITLE_RENDERERS:
        raise UsageError(
            "--chars-per-caption only applies to subtitle output.",
            suggestion="Add -o srt or -o vtt.",
        )


def select_transcript_field(
    transcript: Any, field: str, *, chars_per_caption: int | None = None
) -> str:
    """Render a single transcript field for ``-o/--output``."""
    subtitles = _SUBTITLE_RENDERERS.get(field)
    if subtitles is not None:
        return subtitles(transcript, chars_per_caption)
    return _FIELD_RENDERERS.get(field, _transcript_text)(transcript)


# Transcript ids are opaque url-safe tokens. Reject anything else before it is
# interpolated into the request path: an "id" like ../v2/other would otherwise steer
# an authenticated GET at an arbitrary API route.
_TRANSCRIPT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_transcript_id(transcript_id: str) -> str:
    if not _TRANSCRIPT_ID_RE.match(transcript_id):
        raise UsageError(
            f"{transcript_id!r} doesn't look like a transcript id.",
            suggestion="Ids are letters/digits/dashes; find them with 'assembly transcripts list'.",
        )
    return transcript_id


def get_transcript(api_key: str, transcript_id: str) -> aai.Transcript:
    validate_transcript_id(transcript_id)
    _configure(api_key)
    with _sdk_errors(f"Could not fetch transcript {transcript_id}"):
        return aai.Transcript.get_by_id(transcript_id)


def _streaming_run_error(error: object) -> CLIError:
    """Classify a recorded streaming Error event into the CLIError to raise.

    A rejected handshake (HTTP 401/403) gets an actionable suggestion: bare
    "Streaming error: WebSocket handshake rejected (HTTP 403)" left users cold.
    """
    return classify_error(error, "Streaming error", host=environments.active().streaming_host)


def stream_audio(
    api_key: str,
    source: Iterable[bytes],
    *,
    params: StreamingParameters,
    on_begin: Callable[[Any], Any] | None = None,
    on_turn: Callable[[Any], Any] | None = None,
    on_termination: Callable[[Any], Any] | None = None,
) -> None:
    """Stream `source` (an iterable of PCM bytes) through the v3 realtime API.

    Forwards Begin/Turn/Termination events to the callbacks; raises APIError on a stream error.
    `params` is a fully-built StreamingParameters (sample_rate/speech_model/etc).
    """
    # The SDK's reader thread and websockets both log connection failures at ERROR;
    # with no logging configured those lines would duplicate the CLIError on stderr.
    silence_streaming_logging()
    sc = _make_streaming_client(api_key)

    def _guard(cb: Callable[[Any], Any]) -> _StreamHandler:
        # Event callbacks run on the SDK's reader thread. If the downstream pipe is
        # gone (e.g. a Ctrl-C'd `| assembly llm`, or `| head`), writing a turn raises
        # BrokenPipeError there with no handler -> an ugly thread traceback. Swallow
        # it and point stdout at /dev/null so the interpreter's exit-flush can't
        # re-raise either; the main thread still stops via Ctrl-C / source EOF.
        def handler(_client: Any, event: Any) -> None:
            try:
                cb(event)
            except BrokenPipeError:
                stdio.silence_stdout()

        return handler

    errors: list[object] = []

    def _record_error(_client: object, error: object) -> None:
        errors.append(error)

    if on_begin is not None:
        sc.on(StreamingEvents.Begin, _guard(on_begin))
    if on_turn is not None:
        sc.on(StreamingEvents.Turn, _guard(on_turn))
    if on_termination is not None:
        sc.on(StreamingEvents.Termination, _guard(on_termination))
    sc.on(StreamingEvents.Error, _record_error)

    with _sdk_errors("Could not start streaming session"):
        sc.connect(params)

    try:
        with _sdk_errors("Streaming failed"):
            sc.stream(source)
    finally:
        sc.disconnect(terminate=True)

    if errors:
        raise _streaming_run_error(errors[0])
