from __future__ import annotations

import json

_TEMPLATE = """# Live two-way voice conversation with an AssemblyAI voice agent.
# Requires audio support:  pip install sounddevice websockets
# Tip: use headphones — the mic stays open while the agent speaks.
import base64
import json
import os
import threading

import sounddevice as sd
from websockets.sync.client import connect

# Export your key first:  export ASSEMBLYAI_API_KEY="<your key>"
API_KEY = os.environ["ASSEMBLYAI_API_KEY"]
WS_URL = "wss://agents.assemblyai.com/v1/ws"
RATE = 24000  # Voice Agent native PCM16 mono sample rate

speaker = sd.RawOutputStream(samplerate=RATE, channels=1, dtype="int16")
speaker.start()
mic = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=1024)
mic.start()

ready = threading.Event()


def send_mic(ws):
    while True:
        try:
            data, _overflowed = mic.read(1024)
            chunk = bytes(data)
            if ready.is_set():
                ws.send(json.dumps({{"type": "input.audio", "audio": base64.b64encode(chunk).decode()}}))
        except Exception:
            return


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
    print("Connected \\u2014 start talking. (Ctrl-C to stop)")
    try:
        for raw in ws:
            event = json.loads(raw)
            etype = event.get("type")
            if etype == "session.ready":
                ready.set()
            elif etype == "reply.audio" and event.get("data"):
                speaker.write(base64.b64decode(event["data"]))
            elif etype == "transcript.user":
                print("you:  ", event.get("text", ""))
            elif etype == "transcript.agent":
                print("agent:", event.get("text", ""))
    except KeyboardInterrupt:
        print("\\nStopped.")
    finally:
        speaker.stop()
        speaker.close()
        mic.stop()
        mic.close()
"""


def render(voice: str, system_prompt: str, greeting: str) -> str:
    """Generate a runnable voice-agent script with the given session settings."""
    return _TEMPLATE.format(
        voice=json.dumps(voice),
        system_prompt=json.dumps(system_prompt),
        greeting=json.dumps(greeting),
    )
