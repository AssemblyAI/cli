from __future__ import annotations

import json

from aai_cli.agent import session

# Injected fields ({ws_url}/{voice}/{system_prompt}/{greeting}) are substituted with
# json.dumps values via str.format in a single pass, so prompt text can't collide with
# the template's own braces. Every other literal brace below is doubled for str.format.
_TEMPLATE = """# Live two-way voice conversation with an AssemblyAI voice agent.
# Requires audio support:  pip install sounddevice websockets
# Tip: use headphones — the mic stays open while the agent speaks.
import base64
import json
import os
import queue
import threading

import sounddevice as sd
from websockets.sync.client import connect

API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
WS_URL = {ws_url}
RATE = 24000  # Voice Agent native PCM16 mono sample rate

# ONE full-duplex stream (mic + speaker together) at the agent's native 24 kHz. Opening
# two separate input/output streams on one device fails on macOS CoreAudio, which
# silently kills capture; a single sd.RawStream callback handles both directions.
mic_queue: queue.Queue = queue.Queue()
play_buffer = bytearray()
buffer_lock = threading.Lock()
ready = threading.Event()


def on_audio(indata, outdata, _frames, _time, _status):
    mic_queue.put_nowait(bytes(indata))  # capture -> queue for the agent
    # Playback: drain the agent's audio into the output, zero-filling any shortfall.
    needed = len(outdata)
    with buffer_lock:
        take = bytes(play_buffer[:needed])
        del play_buffer[:needed]
    outdata[: len(take)] = take
    if len(take) < needed:
        outdata[len(take):] = b"\\x00" * (needed - len(take))


def send_mic(ws):
    while True:
        chunk = mic_queue.get()
        if ready.is_set():
            ws.send(json.dumps({{"type": "input.audio", "audio": base64.b64encode(chunk).decode()}}))


stream = sd.RawStream(
    samplerate=RATE, channels=1, dtype="int16", blocksize=RATE // 10, callback=on_audio
)
stream.start()

with connect(WS_URL, additional_headers={{"Authorization": f"Bearer {{API_KEY}}"}}) as ws:
    ws.send(json.dumps({{
        "type": "session.update",
        "session": {{
            "system_prompt": {system_prompt},
            "greeting": {greeting},
            "output": {{"voice": {voice}}},
        }},
    }}))
    threading.Thread(target=send_mic, args=(ws,), daemon=True).start()
    print("Connected — start talking. (Ctrl-C to stop)")
    try:
        for raw in ws:
            event = json.loads(raw)
            etype = event.get("type")
            if etype == "session.ready":
                ready.set()
            elif etype == "reply.audio" and event.get("data"):
                with buffer_lock:
                    play_buffer += base64.b64decode(event["data"])
            elif etype == "transcript.user":
                print("you:  ", event.get("text", ""))
            elif etype == "transcript.agent":
                print("agent:", event.get("text", ""))
    except KeyboardInterrupt:
        print("\\nStopped.")
    finally:
        stream.stop()
        stream.close()
"""


def render(voice: str, system_prompt: str, greeting: str) -> str:
    """Generate a runnable voice-agent script with the given session settings.

    The socket URL comes from the active environment, so a sandbox run generates
    a script that targets the sandbox its key was minted for, not production.
    """
    return _TEMPLATE.format(
        ws_url=json.dumps(session.ws_url()),
        voice=json.dumps(voice),
        system_prompt=json.dumps(system_prompt),
        greeting=json.dumps(greeting),
    )
