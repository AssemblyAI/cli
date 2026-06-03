from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import assemblyai as aai
from assemblyai.streaming.v3 import (
    SpeechModel,
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)

from assemblyai_cli.errors import APIError, CLIError, UsageError, auth_failure, is_auth_failure

SAMPLE_AUDIO_URL = "https://assembly.ai/wildfires.mp3"


def resolve_audio_source(source: str | None, *, sample: bool) -> str:
    """The audio reference to use: the hosted --sample clip, else the given path/URL.

    Shared by `transcribe` and `stream` so both accept a file or URL and `--sample`.
    """
    if sample:
        return SAMPLE_AUDIO_URL
    if not source:
        raise UsageError("Provide an audio path/URL or use --sample.")
    return source


def _configure(api_key: str) -> None:
    aai.settings.api_key = api_key


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


def list_transcripts(api_key: str, *, limit: int = 10) -> list[dict[str, object]]:
    _configure(api_key)
    try:
        resp = aai.Transcriber().list_transcripts(aai.ListTranscriptParameters(limit=limit))
    except aai.types.AssemblyAIError as exc:
        if is_auth_failure(exc):
            raise auth_failure() from exc
        raise APIError(f"Could not list transcripts: {exc}") from exc
    except Exception as exc:
        raise APIError(f"Network error contacting AssemblyAI: {exc}") from exc
    return [item.model_dump(mode="json") for item in resp.transcripts]


def transcribe(
    api_key: str, audio: str, *, speaker_labels: bool, prompt: str | None = None
) -> aai.Transcript:
    _configure(api_key)
    config = aai.TranscriptionConfig(speaker_labels=speaker_labels, prompt=prompt)
    try:
        transcript = aai.Transcriber().transcribe(audio, config=config)
    except APIError:
        raise
    except Exception as exc:
        if is_auth_failure(exc):
            raise auth_failure() from exc
        raise APIError(f"Transcription request failed: {exc}") from exc
    if transcript.status == aai.TranscriptStatus.error:
        raise APIError(transcript.error or "Transcription failed.", transcript_id=transcript.id)
    return transcript


def status_str(transcript: aai.Transcript) -> str:
    """The transcript's status as a plain string (SDK enum `.value` or raw value)."""
    status = transcript.status
    return str(getattr(status, "value", status))


def get_transcript(api_key: str, transcript_id: str) -> aai.Transcript:
    _configure(api_key)
    try:
        return aai.Transcript.get_by_id(transcript_id)
    except Exception as exc:
        if is_auth_failure(exc):
            raise auth_failure() from exc
        raise APIError(f"Could not fetch transcript {transcript_id}: {exc}") from exc


def stream_audio(
    api_key: str,
    source: Iterable[bytes],
    *,
    sample_rate: int,
    on_begin: Callable[[Any], Any] | None = None,
    on_turn: Callable[[Any], Any] | None = None,
    on_termination: Callable[[Any], Any] | None = None,
    prompt: str | None = None,
    speech_model: SpeechModel = SpeechModel.universal_streaming_multilingual,
) -> None:
    """Stream `source` (an iterable of PCM bytes) through the v3 realtime API.

    Forwards Begin/Turn/Termination events to the callbacks; raises APIError on a stream error.
    `prompt` biases the speech model (the realtime `prompt` parameter).
    """
    sc = StreamingClient(
        StreamingClientOptions(api_key=api_key, api_host="streaming.assemblyai.com")
    )
    errors: list[object] = []
    if on_begin is not None:
        sc.on(StreamingEvents.Begin, lambda _client, event: on_begin(event))
    if on_turn is not None:
        sc.on(StreamingEvents.Turn, lambda _client, event: on_turn(event))
    if on_termination is not None:
        sc.on(StreamingEvents.Termination, lambda _client, event: on_termination(event))
    sc.on(StreamingEvents.Error, lambda _client, error: errors.append(error))

    try:
        sc.connect(
            StreamingParameters(
                sample_rate=sample_rate,
                format_turns=True,
                speech_model=speech_model,
                prompt=prompt,
            )
        )
    except CLIError:
        raise
    except Exception as exc:
        if is_auth_failure(exc):
            raise auth_failure() from exc
        raise APIError(f"Could not start streaming session: {exc}") from exc

    try:
        sc.stream(source)
    except (CLIError, KeyboardInterrupt, BrokenPipeError):
        raise  # clean CLI errors, user Ctrl-C, and closed-pipe are handled upstream
    except Exception as exc:
        if is_auth_failure(exc):
            raise auth_failure() from exc
        raise APIError(f"Streaming failed: {exc}") from exc
    finally:
        sc.disconnect(terminate=True)

    if errors:
        if is_auth_failure(errors[0]):
            raise auth_failure()
        raise APIError(f"Streaming error: {errors[0]}")
