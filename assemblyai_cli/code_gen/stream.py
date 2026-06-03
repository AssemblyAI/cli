from __future__ import annotations

from assemblyai_cli.code_gen import serialize

_HEADER = """import os

import assemblyai as aai
from assemblyai.streaming.v3 import (
    SpeechModel,
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
)

# Export your key first:  export ASSEMBLYAI_API_KEY="<your key>"
API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
aai.settings.api_key = API_KEY


def on_turn(client: StreamingClient, event: TurnEvent) -> None:
    print(event.transcript, end="\\r", flush=True)
    if event.end_of_turn:
        print()


client = StreamingClient(
    StreamingClientOptions(api_key=API_KEY, api_host="streaming.assemblyai.com")
)
client.on(StreamingEvents.Turn, on_turn)
"""

_FOOTER = """
print("Listening… press Ctrl-C to stop.")
try:
    client.stream(aai.extras.MicrophoneStream(sample_rate=16000))
finally:
    client.disconnect(terminate=True)
"""


def render(merged: dict[str, object]) -> str:
    """Generate a runnable microphone-streaming script with the given params."""
    if merged:
        kwargs = "\n".join(serialize.config_kwarg_lines(merged, indent=8))
        connect = f"client.connect(\n    StreamingParameters(\n{kwargs}\n    )\n)"
    else:
        connect = "client.connect(StreamingParameters())"
    return _HEADER + "\n" + connect + "\n" + _FOOTER
