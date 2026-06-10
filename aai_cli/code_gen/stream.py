from __future__ import annotations

from typing import cast

from aai_cli import environments
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
    StreamingClientOptions(api_key=API_KEY, api_host={api_host!r})
)
client.on(StreamingEvents.Turn, on_turn)
"""

_LLM_PREAMBLE = """import os
import time

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
# Turns accumulate continuously; the prompt chain re-runs at most once every
# LLM_INTERVAL seconds (0 = on every finalized turn).
LLM_INTERVAL = {interval}
transcript: list[str] = []
_summarized = 0
_last_summary = float("-inf")


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


def summarize(*, final: bool = False) -> None:
    # Refresh the answer over the growing transcript, throttled to LLM_INTERVAL. `final`
    # forces a closing refresh so turns since the last tick aren't lost on stop.
    global _summarized, _last_summary
    turns = len(transcript)
    if turns <= _summarized:
        return
    now = time.monotonic()
    if not final and LLM_INTERVAL > 0 and now - _last_summary < LLM_INTERVAL:
        return
    _summarized = turns
    _last_summary = now
    print(run_chain(" ".join(transcript)))


def on_turn(client: StreamingClient, event: TurnEvent) -> None:
    if not event.end_of_turn or not event.transcript:
        return
    transcript.append(event.transcript)
    summarize()


client = StreamingClient(
    StreamingClientOptions(api_key=API_KEY, api_host={api_host!r})
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

# Same as _FOOTER, but flushes a closing summary (incl. on Ctrl-C) so the turns since the
# last interval tick are reflected before disconnecting.
_LLM_FOOTER = """
print("Listening… press Ctrl-C to stop.")
try:
    client.stream(aai.extras.MicrophoneStream(sample_rate={rate}))
finally:
    summarize(final=True)
    client.disconnect(terminate=True)
"""


def _imports_block(merged: dict[str, object]) -> str:
    """Sorted streaming-class import lines; SpeechModel only when a model kwarg is emitted."""
    names = list(_BASE_IMPORTS)
    if "speech_model" in merged:
        names.append("SpeechModel")
    return "\n".join(f"    {name}," for name in sorted(names))


def _build_preamble(imports: str, llm: dict[str, object] | None) -> str:
    """Pick and fill the plain vs. LLM-Gateway preamble for the given imports.

    Hosts come from the active environment, so a sandbox run generates a script
    that targets the sandbox its key was minted for, not production.
    """
    env = environments.active()
    if llm:
        prompts = "\n".join(f"    {p!r}," for p in cast("list[str]", llm["prompts"]))
        return _LLM_PREAMBLE.format(
            imports=imports,
            api_host=env.streaming_host,
            base_url=env.llm_gateway_base,
            prompts=prompts,
            model=llm["model"],
            max_tokens=llm["max_tokens"],
            interval=llm.get("interval", 0.0),
        )
    return _PREAMBLE.format(imports=imports, api_host=env.streaming_host)


def _build_connect(merged: dict[str, object]) -> str:
    """The `client.connect(StreamingParameters(...))` call for the given params."""
    if not merged:
        return "client.connect(StreamingParameters())"
    # indent=8: 4 for connect(), 4 more for the StreamingParameters() args.
    kwargs = "\n".join(serialize.config_kwarg_lines(merged, indent=8))
    return f"client.connect(\n    StreamingParameters(\n{kwargs}\n    )\n)"


def render(merged: dict[str, object], *, llm: dict[str, object] | None = None) -> str:
    """Generate a runnable microphone-streaming script with the given params.

    With `llm`, the script transforms the live transcript through the LLM Gateway,
    refreshing a prompt chain on every finalized turn (the live sibling of
    `transcribe --llm`).
    """
    preamble = _build_preamble(_imports_block(merged), llm)
    # Mic capture rate must match StreamingParameters.sample_rate, else audio is corrupt.
    rate = merged.get("sample_rate", 16000)
    connect = _build_connect(merged)
    footer = _LLM_FOOTER if llm else _FOOTER
    return preamble + "\n" + connect + "\n" + footer.format(rate=rate)
