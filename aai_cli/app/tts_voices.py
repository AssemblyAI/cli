"""The ``--list-voices`` body shared by the streaming-TTS commands (speak, dub, live).

The listing is the built-in voices plus the account's own from the service's
``GET /v1/voices`` (``tts.voice_list``), so it needs an API key like the synthesis
it informs. The Typer flag that invokes it is ``options.list_voices_option``.
"""

from __future__ import annotations

from aai_cli.app.context import AppState
from aai_cli.tts import session, voice_list
from aai_cli.ui import output


def emit_voice_list(command: str, state: AppState, *, json_mode: bool) -> None:
    """Print the voices: the raw JSON array, or one line per voice."""
    session.require_available(command)
    listing = voice_list.list_voices(state.resolve_api_key())
    output.emit(listing, voice_list.format_voice_list, json_mode=json_mode)
