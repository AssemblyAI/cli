"""Load real API responses recorded by ``scripts/record_fixtures.py`` and rebuild the
objects the CLI's boundary functions (``client.* / llm.*``) return, so a replay test can
drive a command end-to-end against a real payload without touching the network.

The fixtures under ``tests/fixtures/api/`` are scrubbed snapshots; refresh them by
re-running the recorder. See ``tests/test_replay_e2e.py`` for the replay tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import assemblyai as aai
from assemblyai import types
from openai.types.chat import ChatCompletion

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "api"


def load_object(name: str) -> dict[str, object]:
    """Parse a fixture recorded from a JSON *object* response."""
    data = json.loads((FIXTURE_DIR / f"{name}.json").read_text())
    assert isinstance(data, dict)
    return data


def load_list(name: str) -> list[dict[str, object]]:
    """Parse a fixture recorded from a JSON *array* response."""
    data = json.loads((FIXTURE_DIR / f"{name}.json").read_text())
    assert isinstance(data, list)
    return data


def transcript(name: str) -> aai.Transcript:
    """Rebuild a real SDK Transcript from a recorded ``json_response``, offline.

    Mirrors what ``Transcript.get_by_id`` returns, so the command's real render path
    (``transcribe_render`` / ``select_transcript_field``) runs against the recorded
    payload. Setting a placeholder key lets ``get_default()`` build a client without a
    request; ``from_response`` parses the payload locally.
    """
    aai.settings.api_key = "replay-key"
    response = types.TranscriptResponse(**load_object(name))
    return aai.Transcript.from_response(client=aai.Client.get_default(), response=response)


def completion(name: str) -> ChatCompletion:
    """Rebuild an OpenAI ChatCompletion the way the SDK parses a wire response.

    The gateway returns Anthropic-flavored fields (``finish_reason='end_turn'``, token
    counts under ``input_tokens``/``output_tokens``) that strict validation rejects; the
    OpenAI SDK builds responses with the lenient, recursive ``model_construct``, so the
    replay rebuild does too — matching exactly what the live command receives.
    """
    return ChatCompletion.model_construct(None, **load_object(name))
