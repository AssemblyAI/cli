from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)

from aai_cli import environments, stdio
from aai_cli.errors import APIError, CLIError, UsageError, auth_failure, is_auth_failure

SAMPLE_AUDIO_URL = "https://assembly.ai/wildfires.mp3"


def resolve_audio_source(source: str | None, *, sample: bool) -> str:
    """The audio reference to use: the hosted --sample clip, else the given path/URL.

    Shared by `transcribe` and `stream` so both accept a file or URL and `--sample`.
    """
    if sample:
        return SAMPLE_AUDIO_URL
    if not source:
        raise UsageError(
            "Provide an audio path or URL.",
            suggestion="Or pass --sample to use the hosted demo file.",
        )
    return source


def _configure(api_key: str) -> None:
    aai.settings.api_key = api_key
    aai.settings.base_url = environments.active().api_base


@contextlib.contextmanager
def _sdk_errors(message: str) -> Iterator[None]:
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
        raise APIError(f"{message}: {exc}") from exc


def validate_key(api_key: str) -> bool:
    """True if the key authenticates, False on an auth failure. Raises APIError otherwise."""
    _configure(api_key)
    try:
        aai.Transcriber().list_transcripts(aai.ListTranscriptParameters(limit=1))
        return True
    except aai.types.AssemblyAIError as exc:
        if is_auth_failure(exc):
            return False
        raise APIError(f"Could not validate key: {exc}") from exc
    except Exception as exc:
        raise APIError(f"Network error contacting AssemblyAI: {exc}") from exc


def _item_to_dict(item: Any) -> dict[str, Any]:
    """JSON-safe dict for an SDK model across pydantic v2 (model_dump) and v1 (.json)."""
    if hasattr(item, "model_dump"):
        return dict(item.model_dump(mode="json"))
    return dict(json.loads(item.json()))  # pydantic v1 (assemblyai transcription models)


def list_transcripts(api_key: str, *, limit: int = 10) -> list[dict[str, object]]:
    _configure(api_key)
    with _sdk_errors("Could not list transcripts"):
        resp = aai.Transcriber().list_transcripts(aai.ListTranscriptParameters(limit=limit))
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


# Fields `transcribe` and `transcripts get` expose via `-o/--output` (raw, pipe-friendly).
TRANSCRIPT_OUTPUT_FIELDS = ("text", "id", "status", "utterances", "srt", "json")


def _transcript_text(transcript: Any) -> str:
    return str(getattr(transcript, "text", "") or "")


def _render_utterances(transcript: Any) -> str:
    utterances = getattr(transcript, "utterances", None) or []
    if not utterances:
        return _transcript_text(transcript)
    return "\n".join(f"Speaker {u.speaker}: {u.text}" for u in utterances)


def _export_srt(transcript: Any) -> str:
    # The SDK fetches SRT from the `/srt` export endpoint, so this hits the network.
    with _sdk_errors("Could not export SRT subtitles"):
        return str(transcript.export_subtitles_srt())


# Output field -> renderer. Fields absent here fall back to the plain transcript text.
_FIELD_RENDERERS: dict[str, Callable[[Any], str]] = {
    "id": lambda t: str(getattr(t, "id", "") or ""),
    "status": status_str,
    "utterances": _render_utterances,
    "srt": _export_srt,
    "json": lambda t: json.dumps(transcript_json_payload(t), default=str),
}


def select_transcript_field(transcript: Any, field: str) -> str:
    """Render a single transcript field for ``-o/--output``."""
    return _FIELD_RENDERERS.get(field, _transcript_text)(transcript)


def get_transcript(api_key: str, transcript_id: str) -> aai.Transcript:
    _configure(api_key)
    with _sdk_errors(f"Could not fetch transcript {transcript_id}"):
        return aai.Transcript.get_by_id(transcript_id)


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
    sc = StreamingClient(
        StreamingClientOptions(api_key=api_key, api_host=environments.active().streaming_host)
    )

    def _guard(cb: Callable[[Any], Any]) -> Callable[[Any, Any], None]:
        # Event callbacks run on the SDK's reader thread. If the downstream pipe is
        # gone (e.g. a Ctrl-C'd `| aai llm`, or `| head`), writing a turn raises
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
    if on_begin is not None:
        sc.on(StreamingEvents.Begin, _guard(on_begin))
    if on_turn is not None:
        sc.on(StreamingEvents.Turn, _guard(on_turn))
    if on_termination is not None:
        sc.on(StreamingEvents.Termination, _guard(on_termination))
    sc.on(StreamingEvents.Error, lambda _client, error: errors.append(error))

    with _sdk_errors("Could not start streaming session"):
        sc.connect(params)

    try:
        with _sdk_errors("Streaming failed"):
            sc.stream(source)
    finally:
        sc.disconnect(terminate=True)

    if errors:
        if is_auth_failure(errors[0]):
            raise auth_failure()
        raise APIError(f"Streaming error: {errors[0]}")
