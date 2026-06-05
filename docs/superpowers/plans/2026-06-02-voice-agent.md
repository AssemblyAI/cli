# Voice Agent (`aai agent`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `aai agent` — a live two-way voice conversation against AssemblyAI's Voice Agent API (mic in, agent speech out, transcript on screen).

**Architecture:** A new `agent/` package mirroring `streaming/`: `voices.py` (static voice list), `render.py` (`AgentRenderer` — human lines / NDJSON), `audio.py` (`Player` + `MicCapture`, both with injectable stream factories so they test without PyAudio), and `session.py` (`VoiceAgentSession` with a pure `dispatch(event)` router, plus a `run_session(...)` transport that opens the raw WebSocket and runs capture/playback threads). A thin `commands/agent.py` wires it up, patched in command tests exactly like `commands/stream.py` patches `client.stream_audio`.

**Tech Stack:** Python 3.10+, Typer, `websockets` (sync client, already transitive via the SDK), PyAudio (existing `[mic]` extra), pytest.

---

## File Structure

- Create `assemblyai_cli/agent/__init__.py` — empty package marker.
- Create `assemblyai_cli/agent/voices.py` — `VOICES` list + `format_voice_list()`.
- Create `assemblyai_cli/agent/render.py` — `AgentRenderer`.
- Create `assemblyai_cli/agent/audio.py` — `Player`, `MicCapture`.
- Create `assemblyai_cli/agent/session.py` — `VoiceAgentSession`, `run_session(...)`, defaults.
- Create `assemblyai_cli/commands/agent.py` — the `aai agent` Typer command.
- Modify `assemblyai_cli/main.py` — register the agent command.
- Modify `pyproject.toml` — add `websockets>=13` to base dependencies.
- Modify `README.md` — document `aai agent`.
- Create `tests/test_agent_voices.py`, `tests/test_agent_render.py`, `tests/test_agent_audio.py`, `tests/test_agent_session.py`, `tests/test_agent_command.py`.

Conventions to match (from the existing code): `from __future__ import annotations` at the top of every module; `CLIError`/`APIError` from `assemblyai_cli.errors`; `config.resolve_api_key(profile=...)`; `output.resolve_json(explicit=...)`; `run_command(ctx, body, json=...)`; the `\r\x1b[K` in-place-line trick from `streaming/render.py`; the `[mic]`-missing message string used in `streaming/sources.py`.

---

## Task 1: Voice list + `voices.py`

**Files:**
- Create: `assemblyai_cli/agent/__init__.py`
- Create: `assemblyai_cli/agent/voices.py`
- Test: `tests/test_agent_voices.py`

- [ ] **Step 1: Create the empty package marker**

Create `assemblyai_cli/agent/__init__.py` with no content (empty file).

- [ ] **Step 2: Write the failing test**

Create `tests/test_agent_voices.py`:

```python
from assemblyai_cli.agent import voices


def test_voices_includes_default():
    assert "ivy" in voices.VOICES


def test_voices_are_unique_and_nonempty():
    assert voices.VOICES
    assert len(voices.VOICES) == len(set(voices.VOICES))


def test_format_voice_list_mentions_voices():
    out = voices.format_voice_list()
    assert "ivy" in out
    assert "james" in out
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_voices.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'assemblyai_cli.agent.voices'`.

- [ ] **Step 4: Write the implementation**

Create `assemblyai_cli/agent/voices.py`:

```python
from __future__ import annotations

# Known Voice Agent voice IDs (from the Voice Agent quickstart). The server is
# the source of truth; this list backs --list-voices and catches obvious typos.
VOICES: list[str] = [
    # English
    "ivy", "james", "tyler", "winter", "sam", "mia", "bella", "david",
    "jack", "kyle", "helen", "martha", "river", "emma", "victor", "eleanor",
    "sophie", "oliver",
    # Multilingual
    "arjun", "ethan", "dmitri", "lukas", "lena", "pierre", "mina", "ren",
    "mei", "joon", "giulia", "luca", "lucia", "hana", "mateo", "diego",
]

DEFAULT_VOICE = "ivy"


def format_voice_list() -> str:
    """Human-readable, newline-separated voice IDs for --list-voices."""
    return "\n".join(VOICES)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_voices.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/agent/__init__.py assemblyai_cli/agent/voices.py tests/test_agent_voices.py
git commit -m "feat(cli): add voice-agent voice list"
```

---

## Task 2: `AgentRenderer`

Renders server events as human transcript lines or NDJSON. Mirrors `StreamRenderer` (writes to `self.out`, uses `\r\x1b[K` for in-place partials). Audio bytes are never emitted.

**Files:**
- Create: `assemblyai_cli/agent/render.py`
- Test: `tests/test_agent_render.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_render.py`:

```python
import io
import json

from assemblyai_cli.agent.render import AgentRenderer


def _json_lines(buf: io.StringIO):
    return [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]


def test_json_emits_user_and_agent_events():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=True, out=buf)
    r.connected()
    r.user_final("hello there")
    r.agent_transcript("hi back", interrupted=False)
    lines = _json_lines(buf)
    assert {"type": "session.ready"} in lines
    assert {"type": "transcript.user", "text": "hello there"} in lines
    assert {
        "type": "transcript.agent",
        "text": "hi back",
        "interrupted": False,
    } in lines


def test_json_never_emits_audio_bytes():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=True, out=buf)
    r.reply_started()
    r.reply_done(interrupted=True)
    text = buf.getvalue()
    assert "data" not in text  # no base64 audio leaks
    lines = _json_lines(buf)
    assert {"type": "reply.started"} in lines
    assert {"type": "reply.done", "interrupted": True} in lines


def test_human_partial_updates_in_place_then_finalizes():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=False, out=buf)
    r.user_partial("what is")
    r.user_final("what is the time")
    out = buf.getvalue()
    assert "\r\x1b[K" in out          # cleared the line for the partial
    assert "what is the time" in out  # finalized text present
    assert out.endswith("\n")         # finalized line ends clean


def test_human_agent_line_labeled():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=False, out=buf)
    r.agent_transcript("the time is noon", interrupted=False)
    assert "the time is noon" in buf.getvalue()


def test_close_finalizes_open_partial_line():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=False, out=buf)
    r.user_partial("half a sen")
    r.close()
    assert buf.getvalue().endswith("\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'assemblyai_cli.agent.render'`.

- [ ] **Step 3: Write the implementation**

Create `assemblyai_cli/agent/render.py`:

```python
from __future__ import annotations

import json
import sys


class AgentRenderer:
    """Renders Voice Agent events: human transcript lines, or NDJSON for agents.

    Audio payloads are never written; only text/state events are surfaced.
    """

    def __init__(self, *, json_mode: bool, out=None) -> None:
        self.json_mode = json_mode
        self.out = out if out is not None else sys.stdout
        self._partial_open = False

    # --- lifecycle ---------------------------------------------------------
    def connected(self) -> None:
        if self.json_mode:
            self._emit({"type": "session.ready"})
        else:
            self._write("Connected — start talking. (Ctrl-C to stop)\n")

    def stopped(self) -> None:
        if not self.json_mode:
            self._write("Stopped.\n")

    def error(self, message: str) -> None:
        if self.json_mode:
            self._emit({"type": "session.error", "message": message})
        else:
            self._write(f"Error: {message}\n")

    # --- user --------------------------------------------------------------
    def user_partial(self, text: str) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.user.delta", "text": text})
            return
        self._write("\r\x1b[Kyou: " + text)
        self._partial_open = True

    def user_final(self, text: str) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.user", "text": text})
            return
        self._write("\r\x1b[Kyou: " + text + "\n")
        self._partial_open = False

    # --- agent -------------------------------------------------------------
    def reply_started(self) -> None:
        if self.json_mode:
            self._emit({"type": "reply.started"})

    def agent_transcript(self, text: str, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit(
                {"type": "transcript.agent", "text": text, "interrupted": interrupted}
            )
            return
        self._finish_partial()
        self._write("agent: " + text + "\n")

    def reply_done(self, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit({"type": "reply.done", "interrupted": interrupted})

    # --- teardown ----------------------------------------------------------
    def close(self) -> None:
        if self.json_mode:
            return
        self._finish_partial()

    # --- internals ---------------------------------------------------------
    def _finish_partial(self) -> None:
        if self._partial_open:
            self._partial_open = False
            self._write("\n")

    def _emit(self, obj) -> None:
        self._write(json.dumps(obj) + "\n")

    def _write(self, text: str) -> None:
        try:
            self.out.write(text)
            self.out.flush()
        except Exception:  # noqa: BLE001 - downstream pipe may be closed
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_render.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/agent/render.py tests/test_agent_render.py
git commit -m "feat(cli): add voice-agent renderer (human + NDJSON)"
```

---

## Task 3: `Player` (speaker playback)

A queue-backed PyAudio output stream with an injectable stream factory so tests run without PyAudio. `flush()` discards pending audio (interruption).

**Files:**
- Create: `assemblyai_cli/agent/audio.py`
- Test: `tests/test_agent_audio.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_audio.py`:

```python
from assemblyai_cli.agent.audio import Player


class FakeStream:
    def __init__(self):
        self.writes = []
        self.stopped = False
        self.closed = False

    def write(self, data):
        self.writes.append(data)

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_player_writes_enqueued_audio():
    fake = FakeStream()
    p = Player(sample_rate=24000, stream_factory=lambda rate: fake)
    p.start()
    p.enqueue(b"\x01\x02")
    p.enqueue(b"\x03\x04")
    p.close()  # drains the queue, then tears down
    assert b"\x01\x02" in fake.writes
    assert b"\x03\x04" in fake.writes
    assert fake.closed


def test_player_flush_discards_pending_audio():
    fake = FakeStream()
    p = Player(sample_rate=24000, stream_factory=lambda rate: fake)
    # Do NOT start the worker; queue items directly so flush is deterministic.
    p.enqueue(b"stale-1")
    p.enqueue(b"stale-2")
    p.flush()
    assert p.pending() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'assemblyai_cli.agent.audio'`.

- [ ] **Step 3: Write the `Player` implementation**

Create `assemblyai_cli/agent/audio.py`:

```python
from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator

from assemblyai_cli.errors import CLIError

SAMPLE_RATE = 24000  # Voice Agent native PCM16 mono rate

_MIC_MISSING_MSG = "Audio support isn't installed. Run: pip install 'assemblyai-cli[mic]'"


def _default_output_stream(rate: int):
    """Open a PyAudio PCM16 mono output stream (lazy import; needs the [mic] extra)."""
    try:
        import pyaudio
    except ImportError as exc:
        raise CLIError(_MIC_MISSING_MSG, error_type="mic_missing", exit_code=2) from exc
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True)
    stream._pa = pa  # keep a reference so it isn't GC'd before the stream
    return stream


class Player:
    """Plays queued PCM16 audio chunks through a speaker output stream."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        stream_factory: Callable[[int], object] | None = None,
    ) -> None:
        self._rate = sample_rate
        self._factory = stream_factory or _default_output_stream
        self._queue: queue.Queue = queue.Queue()
        self._stream = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stream = self._factory(self._rate)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            try:
                self._stream.write(chunk)
            except Exception:  # noqa: BLE001 - stream may be torn down mid-write
                return

    def enqueue(self, pcm: bytes) -> None:
        self._queue.put(pcm)

    def flush(self) -> None:
        """Discard pending audio (barge-in / interruption)."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def pending(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_audio.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/agent/audio.py tests/test_agent_audio.py
git commit -m "feat(cli): add voice-agent speaker Player"
```

---

## Task 4: `MicCapture` (microphone input)

Yields PCM byte chunks from the microphone via the SDK's `MicrophoneStream`, with the same `[mic]`-missing error as streaming. Injectable factory for tests.

**Files:**
- Modify: `assemblyai_cli/agent/audio.py`
- Test: `tests/test_agent_audio.py`

- [ ] **Step 1: Write the failing test (append to `tests/test_agent_audio.py`)**

```python
import pytest

from assemblyai_cli.agent.audio import MicCapture
from assemblyai_cli.errors import CLIError


def test_miccapture_yields_chunks_from_factory():
    def fake_factory(*, sample_rate, device):
        assert sample_rate == 24000
        return iter([b"aa", b"bb"])

    mic = MicCapture(sample_rate=24000, device=None, stream_factory=fake_factory)
    assert list(mic) == [b"aa", b"bb"]


def test_miccapture_missing_dependency_raises_cli_error():
    def boom(*, sample_rate, device):
        raise ImportError("no pyaudio")

    mic = MicCapture(sample_rate=24000, device=None, stream_factory=boom)
    with pytest.raises(CLIError) as excinfo:
        list(mic)
    assert excinfo.value.exit_code == 2
    assert "assemblyai-cli[mic]" in excinfo.value.message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_audio.py -k miccapture -v`
Expected: FAIL with `ImportError: cannot import name 'MicCapture'`.

- [ ] **Step 3: Add `MicCapture` to `assemblyai_cli/agent/audio.py`**

Append to `assemblyai_cli/agent/audio.py`:

```python
def _default_mic_stream(*, sample_rate: int, device: int | None) -> Iterator[bytes]:
    """SDK PyAudio-backed mic stream (lazy import so the base install stays light)."""
    from assemblyai.extras import MicrophoneStream

    return MicrophoneStream(sample_rate=sample_rate, device_index=device)


class MicCapture:
    """Iterates PCM16 chunks from the microphone (requires the [mic] extra)."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        device: int | None = None,
        stream_factory: Callable[..., Iterator[bytes]] | None = None,
    ) -> None:
        self._rate = sample_rate
        self._device = device
        self._factory = stream_factory or _default_mic_stream

    def __iter__(self) -> Iterator[bytes]:
        try:
            stream = self._factory(sample_rate=self._rate, device=self._device)
        except ImportError as exc:
            raise CLIError(_MIC_MISSING_MSG, error_type="mic_missing", exit_code=2) from exc
        except Exception as exc:  # noqa: BLE001 - surface device errors cleanly
            raise CLIError(
                f"Could not open the microphone (device {self._device}): {exc}",
                error_type="mic_error",
                exit_code=1,
            ) from exc
        close = getattr(stream, "close", None)
        try:
            yield from stream
        finally:
            if callable(close):
                close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_audio.py -v`
Expected: PASS (4 passed total in the file).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/agent/audio.py tests/test_agent_audio.py
git commit -m "feat(cli): add voice-agent MicCapture"
```

---

## Task 5: `VoiceAgentSession.dispatch` (event router + duplex state)

The pure event router: given a parsed event dict, drive the renderer, enqueue/flush the player, toggle the half-duplex mute, set the ready gate, and raise on `session.error`. No socket here — fully unit-testable.

**Files:**
- Create: `assemblyai_cli/agent/session.py`
- Test: `tests/test_agent_session.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_session.py`:

```python
import pytest

from assemblyai_cli.agent.session import VoiceAgentSession
from assemblyai_cli.errors import APIError, CLIError


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def connected(self):
        self.calls.append(("connected",))

    def user_partial(self, text):
        self.calls.append(("user_partial", text))

    def user_final(self, text):
        self.calls.append(("user_final", text))

    def reply_started(self):
        self.calls.append(("reply_started",))

    def agent_transcript(self, text, *, interrupted):
        self.calls.append(("agent_transcript", text, interrupted))

    def reply_done(self, *, interrupted):
        self.calls.append(("reply_done", interrupted))

    def error(self, message):
        self.calls.append(("error", message))


class FakePlayer:
    def __init__(self):
        self.enqueued = []
        self.flushed = 0

    def enqueue(self, pcm):
        self.enqueued.append(pcm)

    def flush(self):
        self.flushed += 1


def _session(*, full_duplex=False):
    return VoiceAgentSession(
        renderer=FakeRenderer(),
        player=FakePlayer(),
        full_duplex=full_duplex,
    )


def test_ready_opens_gate_and_announces():
    s = _session()
    assert s.ready is False
    s.dispatch({"type": "session.ready", "session_id": "sess_1"})
    assert s.ready is True
    assert ("connected",) in s.renderer.calls


def test_half_duplex_mutes_during_reply():
    s = _session(full_duplex=False)
    s.dispatch({"type": "session.ready"})
    s.dispatch({"type": "reply.started"})
    assert s.muted is True
    s.dispatch({"type": "reply.done"})
    assert s.muted is False


def test_full_duplex_never_mutes_and_flushes_on_speech_start():
    s = _session(full_duplex=True)
    s.dispatch({"type": "session.ready"})
    s.dispatch({"type": "reply.started"})
    assert s.muted is False
    s.dispatch({"type": "input.speech.started"})
    assert s.player.flushed == 1


def test_reply_audio_is_decoded_and_enqueued():
    import base64

    s = _session()
    payload = base64.b64encode(b"\x10\x20").decode()
    s.dispatch({"type": "reply.audio", "data": payload})
    assert s.player.enqueued == [b"\x10\x20"]


def test_interrupted_reply_done_flushes_playback():
    s = _session()
    s.dispatch({"type": "reply.done", "status": "interrupted"})
    assert s.player.flushed == 1
    assert ("reply_done", True) in s.renderer.calls


def test_transcripts_routed_to_renderer():
    s = _session()
    s.dispatch({"type": "transcript.user.delta", "text": "what"})
    s.dispatch({"type": "transcript.user", "text": "what time"})
    s.dispatch({"type": "transcript.agent", "text": "noon", "interrupted": False})
    assert ("user_partial", "what") in s.renderer.calls
    assert ("user_final", "what time") in s.renderer.calls
    assert ("agent_transcript", "noon", False) in s.renderer.calls


def test_unauthorized_error_raises_cli_error_exit_2():
    s = _session()
    with pytest.raises(CLIError) as excinfo:
        s.dispatch({"type": "session.error", "code": "UNAUTHORIZED", "message": "bad key"})
    assert excinfo.value.exit_code == 2


def test_other_session_error_raises_api_error():
    s = _session()
    with pytest.raises(APIError):
        s.dispatch({"type": "session.error", "code": "invalid_value", "message": "bad voice"})


def test_unknown_and_tool_events_are_ignored():
    s = _session()
    s.dispatch({"type": "tool.call", "call_id": "c1", "name": "x", "arguments": {}})
    s.dispatch({"type": "something.new"})
    assert s.renderer.calls == []  # nothing surfaced, no exception
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'assemblyai_cli.agent.session'`.

- [ ] **Step 3: Write the dispatch core**

Create `assemblyai_cli/agent/session.py`:

```python
from __future__ import annotations

import base64

from assemblyai_cli.errors import APIError, CLIError

WS_URL = "wss://agents.assemblyai.com/v1/ws"

DEFAULT_PROMPT = (
    "You are a friendly voice assistant having a casual conversation. Keep replies "
    "short and natural, usually one or two sentences. Speak the way a person would "
    "in real conversation: relaxed, low-key, no exclamation marks."
)
DEFAULT_GREETING = "Hey, what's on your mind?"

# session.error codes that mean the connection is unauthorized -> exit 2.
_AUTH_ERROR_CODES = {"UNAUTHORIZED", "FORBIDDEN"}


class VoiceAgentSession:
    """Routes Voice Agent server events to the renderer, player, and duplex state."""

    def __init__(self, *, renderer, player, full_duplex: bool = False) -> None:
        self.renderer = renderer
        self.player = player
        self.full_duplex = full_duplex
        self.ready = False
        self.muted = False

    def should_send_audio(self) -> bool:
        """True when captured mic frames should be forwarded to the server."""
        return self.ready and not self.muted

    def dispatch(self, event: dict) -> None:
        etype = event.get("type")

        if etype == "session.ready":
            self.ready = True
            self.renderer.connected()
        elif etype == "input.speech.started":
            if self.full_duplex:
                self.player.flush()
        elif etype == "input.speech.stopped":
            pass
        elif etype == "transcript.user.delta":
            self.renderer.user_partial(event.get("text", ""))
        elif etype == "transcript.user":
            self.renderer.user_final(event.get("text", ""))
        elif etype == "reply.started":
            if not self.full_duplex:
                self.muted = True
            self.renderer.reply_started()
        elif etype == "reply.audio":
            data = event.get("data")
            if data:
                self.player.enqueue(base64.b64decode(data))
        elif etype == "transcript.agent":
            self.renderer.agent_transcript(
                event.get("text", ""), interrupted=bool(event.get("interrupted", False))
            )
        elif etype == "reply.done":
            if not self.full_duplex:
                self.muted = False
            interrupted = event.get("status") == "interrupted"
            if interrupted:
                self.player.flush()
            self.renderer.reply_done(interrupted=interrupted)
        elif etype == "session.error":
            self._raise_error(event)
        # tool.call and unknown event types: intentionally ignored.

    def _raise_error(self, event: dict) -> None:
        code = event.get("code", "")
        message = event.get("message") or code or "Voice agent error."
        if code in _AUTH_ERROR_CODES:
            raise CLIError(
                f"Voice agent rejected the connection: {message}",
                error_type="unauthorized",
                exit_code=2,
            )
        raise APIError(f"Voice agent error ({code}): {message}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_session.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/agent/session.py tests/test_agent_session.py
git commit -m "feat(cli): add voice-agent event dispatch + duplex state"
```

---

## Task 6: `run_session` transport (WebSocket + threads)

The transport glue: open the raw WebSocket, send `session.update`, run capture + playback threads, loop `recv → json.loads → dispatch`. This is the patch seam for command tests (no automated socket test; covered by the manual live test). Build it so the command can drive it and Ctrl-C exits cleanly.

**Files:**
- Modify: `assemblyai_cli/agent/session.py`
- Test: covered indirectly in Task 7 (the command patches `run_session`); no new unit test here.

- [ ] **Step 1: Add the imports and `run_session` to `assemblyai_cli/agent/session.py`**

Add to the imports at the top of `assemblyai_cli/agent/session.py`:

```python
import json
import threading
```

Append `run_session` to `assemblyai_cli/agent/session.py`:

```python
def _send_audio_loop(ws, session: VoiceAgentSession, mic) -> None:
    """Forward mic PCM as input.audio while the session gate allows it."""
    for chunk in mic:
        if not session.should_send_audio():
            continue  # half-duplex: drop frames while the agent is speaking
        payload = base64.b64encode(chunk).decode("ascii")
        try:
            ws.send(json.dumps({"type": "input.audio", "audio": payload}))
        except Exception:  # noqa: BLE001 - socket closed; capture thread ends
            return


def run_session(
    api_key: str,
    *,
    renderer,
    player,
    mic,
    voice: str,
    system_prompt: str,
    greeting: str,
    full_duplex: bool = False,
    connect=None,
) -> None:
    """Open the Voice Agent WebSocket and run the bidirectional loop until close.

    `connect` defaults to websockets' synchronous client; injectable for tests.
    """
    if connect is None:
        from websockets.sync.client import connect

    session = VoiceAgentSession(renderer=renderer, player=player, full_duplex=full_duplex)

    try:
        ws = connect(WS_URL, additional_headers={"Authorization": f"Bearer {api_key}"})
    except Exception as exc:  # noqa: BLE001 - connect/auth/network failures
        raise APIError(f"Could not connect to the voice agent: {exc}") from exc

    try:
        player.start()  # opens the speaker stream; CLIError here if [mic] is missing
        capture = threading.Thread(
            target=_send_audio_loop, args=(ws, session, mic), daemon=True
        )
        capture.start()
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "system_prompt": system_prompt,
                        "greeting": greeting,
                        "output": {"voice": voice},
                    },
                }
            )
        )
        for raw in ws:
            session.dispatch(json.loads(raw))
    except (CLIError, KeyboardInterrupt):
        raise  # auth/protocol errors and user Ctrl-C handled upstream
    except Exception as exc:  # noqa: BLE001 - mid-stream socket/JSON failures
        raise APIError(f"Voice agent session failed: {exc}") from exc
    finally:
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass
        player.close()
```

- [ ] **Step 2: Verify the module imports cleanly and existing tests still pass**

Run: `python -c "import assemblyai_cli.agent.session"` then `python -m pytest tests/test_agent_session.py -v`
Expected: import succeeds; 9 passed (dispatch tests unaffected).

- [ ] **Step 3: Commit**

```bash
git add assemblyai_cli/agent/session.py
git commit -m "feat(cli): add voice-agent WebSocket transport loop"
```

---

## Task 7: `aai agent` command + registration

Thin Typer command: `--list-voices` short-circuit, prompt-file handling, key resolution, renderer/player/mic construction, `run_session` call, clean Ctrl-C. Patched in tests exactly like `commands/stream.py`.

**Files:**
- Create: `assemblyai_cli/commands/agent.py`
- Modify: `assemblyai_cli/main.py`
- Test: `tests/test_agent_command.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_command.py`:

```python
import json

from typer.testing import CliRunner

from assemblyai_cli import config
from assemblyai_cli.main import app

runner = CliRunner()


def test_agent_help_lists_command():
    result = runner.invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "voice" in result.output.lower()


def test_list_voices_prints_and_exits_without_connecting(monkeypatch):
    called = {"ran": False}

    def fake_run_session(*a, **k):
        called["ran"] = True

    monkeypatch.setattr("assemblyai_cli.commands.agent.run_session", fake_run_session)
    result = runner.invoke(app, ["agent", "--list-voices"])
    assert result.exit_code == 0
    assert "ivy" in result.output
    assert called["ran"] is False


def test_agent_unauthenticated_exits_2():
    result = runner.invoke(app, ["agent"])
    assert result.exit_code == 2


def test_agent_drives_renderer_json(monkeypatch):
    config.set_api_key("default", "sk_live")

    def fake_run_session(api_key, *, renderer, player, mic, voice, system_prompt,
                         greeting, full_duplex=False, connect=None):
        renderer.connected()
        renderer.user_final("hello agent")
        renderer.agent_transcript("hello human", interrupted=False)

    monkeypatch.setattr("assemblyai_cli.commands.agent.run_session", fake_run_session)
    result = runner.invoke(app, ["agent", "--json"])
    assert result.exit_code == 0
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"type": "transcript.user", "text": "hello agent"} in lines
    assert {"type": "transcript.agent", "text": "hello human", "interrupted": False} in lines


def test_agent_passes_voice_and_prompt_file(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_run_session(api_key, *, renderer, player, mic, voice, system_prompt,
                         greeting, full_duplex=False, connect=None):
        seen["voice"] = voice
        seen["prompt"] = system_prompt
        seen["full_duplex"] = full_duplex

    monkeypatch.setattr("assemblyai_cli.commands.agent.run_session", fake_run_session)
    prompt_file = tmp_path / "p.txt"
    prompt_file.write_text("be a pirate")
    result = runner.invoke(
        app,
        ["agent", "--voice", "james", "--prompt-file", str(prompt_file),
         "--prompt", "ignored", "--full-duplex"],
    )
    assert result.exit_code == 0
    assert seen["voice"] == "james"
    assert seen["prompt"] == "be a pirate"  # --prompt-file overrides --prompt
    assert seen["full_duplex"] is True


def test_agent_ctrl_c_exits_cleanly(monkeypatch):
    config.set_api_key("default", "sk_live")

    def raise_kbd(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("assemblyai_cli.commands.agent.run_session", raise_kbd)
    result = runner.invoke(app, ["agent"])
    assert result.exit_code == 0


def test_agent_unknown_voice_exits_2(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("assemblyai_cli.commands.agent.run_session", lambda *a, **k: None)
    result = runner.invoke(app, ["agent", "--voice", "not-a-voice"])
    assert result.exit_code == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_command.py -v`
Expected: FAIL — `agent` is not a registered command (non-zero exit / usage error).

- [ ] **Step 3: Write the command**

Create `assemblyai_cli/commands/agent.py`:

```python
from __future__ import annotations

from pathlib import Path

import typer

from assemblyai_cli import config
from assemblyai_cli.agent.audio import SAMPLE_RATE, MicCapture, Player
from assemblyai_cli.agent.render import AgentRenderer
from assemblyai_cli.agent.session import DEFAULT_GREETING, DEFAULT_PROMPT, run_session
from assemblyai_cli.agent.voices import DEFAULT_VOICE, VOICES, format_voice_list
from assemblyai_cli.context import run_command
from assemblyai_cli.errors import CLIError, UsageError

app = typer.Typer()


@app.command()
def agent(
    ctx: typer.Context,
    voice: str = typer.Option(DEFAULT_VOICE, "--voice", help="Agent voice. See --list-voices."),
    prompt: str = typer.Option(DEFAULT_PROMPT, "--prompt", help="System prompt."),
    prompt_file: Path = typer.Option(
        None, "--prompt-file", help="Read the system prompt from a file (overrides --prompt)."
    ),
    greeting: str = typer.Option(DEFAULT_GREETING, "--greeting", help="Spoken greeting."),
    full_duplex: bool = typer.Option(
        False, "--full-duplex", help="Keep the mic open while the agent speaks (needs headphones)."
    ),
    sample_rate: int = typer.Option(SAMPLE_RATE, "--sample-rate", help="Mic sample rate in Hz."),
    device: int = typer.Option(None, "--device", help="Microphone device index."),
    list_voices: bool = typer.Option(False, "--list-voices", help="Print known voices and exit."),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
) -> None:
    """Have a live two-way voice conversation with an AssemblyAI voice agent."""

    if list_voices:
        typer.echo(format_voice_list())
        raise typer.Exit(code=0)

    def body(state, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        if voice not in VOICES:
            raise UsageError(f"Unknown voice {voice!r}. Run 'aai agent --list-voices'.")
        system_prompt = prompt
        if prompt_file is not None:
            try:
                system_prompt = prompt_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise CLIError(
                    f"Could not read --prompt-file {prompt_file}: {exc}",
                    error_type="file_not_found",
                    exit_code=2,
                ) from exc

        renderer = AgentRenderer(json_mode=json_mode)
        player = Player(sample_rate=SAMPLE_RATE)
        mic = MicCapture(sample_rate=sample_rate, device=device)
        if not json_mode and not full_duplex:
            renderer.out.write(
                "Half-duplex: mic mutes while the agent talks. "
                "Use --full-duplex (with headphones) for barge-in.\n"
            )
            renderer.out.flush()
        try:
            run_session(
                api_key,
                renderer=renderer,
                player=player,
                mic=mic,
                voice=voice,
                system_prompt=system_prompt,
                greeting=greeting,
                full_duplex=full_duplex,
            )
        except KeyboardInterrupt:
            renderer.stopped()
        finally:
            renderer.close()

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Register the command in `assemblyai_cli/main.py`**

Add `agent` to the existing import line (which already imports `claude` and the others):

```python
from assemblyai_cli.commands import agent, claude, login, samples, stream, transcribe, transcripts
```

And add the registration alongside the others (e.g. after `app.add_typer(login.app)`):

```python
app.add_typer(agent.app, name="agent")
```

- [ ] **Step 5: Run the command tests to verify they pass**

Run: `python -m pytest tests/test_agent_command.py -v`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/commands/agent.py assemblyai_cli/main.py tests/test_agent_command.py
git commit -m "feat(cli): add 'aai agent' command and register it"
```

---

## Task 8: Dependency + docs

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Add the `websockets` base dependency**

In `pyproject.toml`, add `"websockets>=13"` to the `dependencies` list (after `"tomli-w>=1.0",`):

```toml
dependencies = [
    "typer>=0.12",
    "assemblyai>=0.34",
    "rich>=13.0",
    "keyring>=24.0",
    "platformdirs>=4.0",
    "tomli-w>=1.0",
    "websockets>=13",
]
```

- [ ] **Step 2: Document `aai agent` in `README.md`**

Append to `README.md`:

```markdown
## Voice agent

Have a live, two-way voice conversation with an AssemblyAI voice agent (requires the
`[mic]` extra for microphone + speaker audio):

    pip install "assemblyai-cli[mic]"
    aai agent                              # talk; the agent talks back. Ctrl-C to stop.
    aai agent --voice james --greeting "Hi there"
    aai agent --prompt-file persona.txt    # load the system prompt from a file
    aai agent --list-voices                # see available voices

By default the agent runs **half-duplex**: your mic mutes while the agent is speaking,
so it can't hear itself on your speakers. With headphones, add `--full-duplex` for
true barge-in (interrupt the agent mid-sentence). Add `--json` for newline-delimited
JSON events.
```

- [ ] **Step 3: Run the full suite and linters**

Run: `python -m pytest -q && python -m ruff check assemblyai_cli tests`
Expected: all tests pass; ruff reports no errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml README.md
git commit -m "build(cli): add websockets dep; document 'aai agent'"
```

---

## Task 9: Final verification

- [ ] **Step 1: Full suite green**

Run: `python -m pytest -q`
Expected: all tests pass (existing + new agent tests).

- [ ] **Step 2: Smoke the CLI surface**

Run: `python -m assemblyai_cli agent --help` and `python -m assemblyai_cli agent --list-voices`
Expected: help shows the agent options; `--list-voices` prints the voice list and exits 0.

- [ ] **Step 3: Lint/format clean**

Run: `python -m ruff check assemblyai_cli tests && python -m ruff format --check assemblyai_cli tests`
Expected: no errors.

- [ ] **Step 4: (Manual, optional) live conversation**

With a real key and a microphone + headphones:
Run: `aai agent --full-duplex`
Expected: prints "Connected — start talking", speaks the greeting, and holds a conversation. Confirms the WS URL, the `Authorization: Bearer <key>` header form, and the 24 kHz audio path. Not part of CI.

---

## Notes for the implementer

- **TDD discipline:** every code task writes the test first, watches it fail, then implements. Don't batch.
- **No real audio/sockets in CI:** `Player`/`MicCapture` take injectable factories; `run_session` takes an injectable `connect`; command tests patch `run_session`. Never open a real device or socket in a test.
- **Match existing style:** `from __future__ import annotations`, `# noqa: BLE001` on broad excepts that surface cleanly, the `\r\x1b[K` partial-line trick, and the shared `[mic]`-missing message.
- The agent uses the raw API key in the `Authorization: Bearer` header (the docs' Python `session.resume` example). If a live test shows the header form is wrong, that's the single place to adjust (`run_session`).
