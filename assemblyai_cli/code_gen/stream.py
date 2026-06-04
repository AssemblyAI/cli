from __future__ import annotations

from assemblyai_cli.code_gen import serialize

# Streaming-class imports always used by the generated scaffold. SpeechModel is added
# only when a speech_model kwarg is emitted, so the generated script stays lint-clean.
_BASE_IMPORTS = [
    "StreamingClient",
    "StreamingClientOptions",
    "StreamingEvents",
    "StreamingParameters",
    "TurnEvent",
]

_PREAMBLE = """import os

import assemblyai as aai
from assemblyai.streaming.v3 import (
{imports}
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
    client.stream(aai.extras.MicrophoneStream(sample_rate={rate}))
finally:
    client.disconnect(terminate=True)
"""


def render(merged: dict[str, object]) -> str:
    """Generate a runnable microphone-streaming script with the given params."""
    names = list(_BASE_IMPORTS)
    if "speech_model" in merged:
        names.append("SpeechModel")
    imports = "\n".join(f"    {name}," for name in sorted(names))
    preamble = _PREAMBLE.format(imports=imports)

    # Mic capture rate must match StreamingParameters.sample_rate, else audio is corrupt.
    rate = merged.get("sample_rate", 16000)

    if merged:
        # indent=8: 4 for connect(), 4 more for the StreamingParameters() args.
        kwargs = "\n".join(serialize.config_kwarg_lines(merged, indent=8))
        connect = f"client.connect(\n    StreamingParameters(\n{kwargs}\n    )\n)"
    else:
        connect = "client.connect(StreamingParameters())"

    return preamble + "\n" + connect + "\n" + _FOOTER.format(rate=rate)
