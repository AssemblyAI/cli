"""The static body of the generated agent-cascade script.

Kept separate from the header so the orchestration's many literal braces (dict/set
literals, the STT/TTS protocol loops) stay verbatim — this string is concatenated
onto the formatted header, never passed through str.format itself.
"""

from __future__ import annotations

# The constants (API_KEY, STT_URL, TTS_URL, GATEWAY_URL, MODEL, …) and is_reply_cue()
# are defined by the header above this body; everything here references them.
BODY = """

gateway = OpenAI(api_key=API_KEY, base_url=GATEWAY_URL)
history = []  # alternating user/assistant turns — the sliding LLM-context window
stop_reply = threading.Event()  # set on barge-in to cut a reply short
reply_thread = None

# ONE full-duplex stream (mic + speaker together) at 24 kHz. Opening two separate
# input/output streams on one device fails on macOS CoreAudio, which silently kills
# capture; a single sd.RawStream callback handles both directions.
mic_queue: queue.Queue = queue.Queue()
play_buffer = bytearray()
buffer_lock = threading.Lock()


def on_audio(indata, outdata, _frames, _time, _status):
    mic_queue.put_nowait(bytes(indata))  # capture -> queue for STT
    # Playback: drain the agent's audio into the output, zero-filling any shortfall.
    needed = len(outdata)
    with buffer_lock:
        take = bytes(play_buffer[:needed])
        del play_buffer[:needed]
    outdata[: len(take)] = take
    if len(take) < needed:
        outdata[len(take):] = b"\\x00" * (needed - len(take))


def enqueue_audio(pcm):
    with buffer_lock:
        play_buffer.extend(pcm)


def flush_audio():  # drop queued-but-unplayed audio (used on barge-in)
    with buffer_lock:
        play_buffer.clear()


def trim_history():  # cap the running history to the most recent MAX_HISTORY messages
    if len(history) > MAX_HISTORY:
        del history[: len(history) - MAX_HISTORY]


def split_sentences(text):
    # Split a reply into sentences (each ending in . ! ?) so the first audio can play
    # before the whole answer is synthesized; a trailing fragment is kept too.
    sentences, start = [], 0
    for i, ch in enumerate(text):
        if ch in ".!?":
            piece = text[start: i + 1].strip()
            if piece:
                sentences.append(piece)
            start = i + 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def synthesize(text):
    # Open a fresh streaming-TTS socket (the voice is fixed at connect time), drive the
    # Begin -> Generate -> Flush -> Audio protocol, and return the concatenated PCM. TTS
    # authenticates with the raw API key, not a Bearer token (the streaming convention).
    pcm = bytearray()
    with connect(TTS_URL, additional_headers={"Authorization": API_KEY}, max_size=None) as ws:
        if json.loads(ws.recv()).get("type") != "Begin":
            return b""
        ws.send(json.dumps({"type": "Generate", "text": text}))
        ws.send(json.dumps({"type": "Flush"}))
        for raw in ws:
            frame = json.loads(raw)
            kind = frame.get("type")
            if kind == "Audio":
                pcm += base64.b64decode(frame.get("audio", ""))
                if frame.get("is_final"):
                    break
            elif kind in ("FlushDone", "Error"):
                break
        ws.send(json.dumps({"type": "Terminate"}))
    return bytes(pcm)


def speak(text):  # show + synthesize one chunk of agent speech, honoring a barge-in
    print("agent:", text)
    if not stop_reply.is_set():
        enqueue_audio(synthesize(text))


def generate_reply():
    # One LLM completion over the running history, spoken sentence-by-sentence. Record
    # what was actually spoken so a barge-in still leaves the history alternating.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
    reply = gateway.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=MAX_TOKENS
    ).choices[0].message.content or ""
    spoken = []
    for sentence in split_sentences(reply):
        if stop_reply.is_set():
            break
        speak(sentence)
        spoken.append(sentence)
    said = " ".join(spoken).strip()
    if said:
        history.append({"role": "assistant", "content": said})
        trim_history()


def barge_in():
    # A new user turn cuts off any reply still playing: stop the worker and drop the
    # queued audio (the flush is what silences the already-buffered speech).
    if reply_thread is not None and reply_thread.is_alive():
        stop_reply.set()
        flush_audio()
        reply_thread.join()


def send_mic(stt):
    while True:
        chunk = mic_queue.get()
        try:
            stt.send(chunk)
        except Exception:
            return  # socket closed (session over): end the mic thread quietly


stream = sd.RawStream(
    samplerate=RATE, channels=1, dtype="int16", blocksize=RATE // 10, callback=on_audio
)
stream.start()

# Greet first, seeding the opening line into the history so the model has a record of it.
if GREETING:
    history.append({"role": "assistant", "content": GREETING})
    speak(GREETING)

with connect(STT_URL, additional_headers={"Authorization": API_KEY}) as stt:
    threading.Thread(target=send_mic, args=(stt,), daemon=True).start()
    print("Connected — start talking. (Ctrl-C to stop)")
    try:
        for raw in stt:
            event = json.loads(raw)
            if event.get("type") != "Turn":
                continue
            text = (event.get("transcript") or "").strip()
            if not text:
                continue
            if is_reply_cue(event):
                print("you:  ", text)
                barge_in()
                history.append({"role": "user", "content": text})
                trim_history()
                stop_reply.clear()
                reply_thread = threading.Thread(target=generate_reply, daemon=True)
                reply_thread.start()
            else:
                barge_in()  # an interim turn only interrupts a playing reply
    except KeyboardInterrupt:
        print("\\nStopped.")
    finally:
        stop_reply.set()
        stream.stop()
        stream.close()
"""
