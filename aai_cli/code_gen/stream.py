from __future__ import annotations

from typing import cast

from aai_cli import llm as gateway
from aai_cli.code_gen import serialize

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

_LLM_PREAMBLE = """import os

import assemblyai as aai
from assemblyai.streaming.v3 import (
{imports}
)
from openai import OpenAI

# Export your key first:  export ASSEMBLYAI_API_KEY="<your key>"
API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
aai.settings.api_key = API_KEY

# Run the prompts over the LLM Gateway (OpenAI-compatible). Each prompt runs on the
# previous response; the first runs on the transcript accumulated so far.
gateway = OpenAI(api_key=API_KEY, base_url={base_url!r})
PROMPTS = [
{prompts}
]
transcript: list[str] = []


def run_chain(text: str) -> str:
    result = text
    for i, prompt in enumerate(PROMPTS):
        source = text if i == 0 else result
        response = gateway.chat.completions.create(
            model={model!r},
            messages=[{{"role": "user", "content": prompt + "\\n\\nTranscript:\\n" + source}}],
            max_tokens={max_tokens},
        )
        result = response.choices[0].message.content
    return result


def on_turn(client: StreamingClient, event: TurnEvent) -> None:
    # Refresh the answer on every finalized turn, over the growing transcript.
    if not event.end_of_turn or not event.transcript:
        return
    transcript.append(event.transcript)
    print(run_chain(" ".join(transcript)))


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


def render(merged: dict[str, object], *, llm: dict[str, object] | None = None) -> str:
    """Generate a runnable microphone-streaming script with the given params.

    With `llm`, the script transforms the live transcript through the LLM Gateway,
    refreshing a prompt chain on every finalized turn (the live sibling of
    `transcribe --llm`).
    """
    names = list(_BASE_IMPORTS)
    if "speech_model" in merged:
        names.append("SpeechModel")
    imports = "\n".join(f"    {name}," for name in sorted(names))

    if llm:
        prompts = "\n".join(f"    {p!r}," for p in cast("list[str]", llm["prompts"]))
        preamble = _LLM_PREAMBLE.format(
            imports=imports,
            base_url=gateway.GATEWAY_BASE_URL,
            prompts=prompts,
            model=llm["model"],
            max_tokens=llm["max_tokens"],
        )
    else:
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
