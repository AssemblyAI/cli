# `aai stream` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `aai stream [SOURCE]` — real-time transcription from the microphone (no arg) or an audio file (path arg), via AssemblyAI's v3 streaming API.

**Architecture:** Two interchangeable audio sources (iterators of PCM byte chunks) — `MicSource` (PyAudio, optional `[mic]` extra) and `FileSource` (stdlib `wave` for 16 kHz mono WAV, else an `ffmpeg` subprocess) — both fed to `client.stream_audio(...)`, which wires a v3 `StreamingClient` and forwards `Begin`/`Turn` events to render callbacks. A `StreamRenderer` prints a live-updating turn line for humans or newline-delimited JSON for agents.

**Tech Stack:** Python ≥3.10, Typer, `assemblyai` (v3 `streaming`), stdlib `wave`/`subprocess`/`shutil`, pytest.

**Spec:** `docs/superpowers/specs/2026-06-02-streaming-design.md`

**Verified SDK facts (assemblyai 0.64.0):**
- `from assemblyai.streaming.v3 import StreamingClient, StreamingClientOptions, StreamingParameters, StreamingEvents`.
- `StreamingClient(StreamingClientOptions(api_key=..., api_host="streaming.assemblyai.com"))`; methods `.on(event, handler)`, `.connect(StreamingParameters(sample_rate=int, format_turns=True))`, `.stream(iterable_of_bytes)`, `.disconnect()`.
- Event handlers are called as `handler(client, event)`. `StreamingEvents.Begin/Turn/Termination/Error`.
- `TurnEvent` fields include `.transcript` (str) and `.end_of_turn` (bool). `BeginEvent.id`. `TerminationEvent.audio_duration_seconds`.
- `aai.extras.MicrophoneStream(sample_rate=44100, device_index=None)` is an iterator of PCM bytes; importing/instantiating it requires PyAudio.
- `StreamingClient` and the v3 module do NOT require PyAudio; only `MicrophoneStream` does.

---

## File Structure

```
assemblyai_cli/
  streaming/
    __init__.py          # empty package marker
    sources.py           # FileSource, MicSource (+ _load_microphone_stream); raise CLIError on missing deps/files
    render.py            # StreamRenderer: live human line vs NDJSON events
  client.py              # + stream_audio(...) — sole SDK boundary for v3 streaming
  commands/
    stream.py            # `aai stream` Typer command (thin)
  main.py                # register stream.app
pyproject.toml           # [mic] optional extra = pyaudio
tests/
  test_streaming_sources.py
  test_streaming_render.py
  test_stream_command.py
  test_client.py         # + stream_audio wiring tests
```

Constants used across the file source (define at top of `sources.py`): `TARGET_RATE = 16000` and `CHUNK_BYTES = 3200` (100 ms of 16-bit mono @ 16 kHz = 16000 × 2 × 0.1).

---

## Task 1: FileSource (WAV + ffmpeg)

**Files:**
- Create: `assemblyai_cli/streaming/__init__.py`
- Create: `assemblyai_cli/streaming/sources.py`
- Test: `tests/test_streaming_sources.py`

- [ ] **Step 1: Create the package marker** `assemblyai_cli/streaming/__init__.py` (empty file).

- [ ] **Step 2: Write the failing test** at `tests/test_streaming_sources.py`:

```python
import wave

import pytest

from assemblyai_cli.errors import CLIError
from assemblyai_cli.streaming import sources
from assemblyai_cli.streaming.sources import FileSource


def _write_wav(path, *, seconds=0.5, rate=16000):
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * frames)  # 2 bytes/frame, mono 16-bit


def test_filesource_streams_wav_chunks(tmp_path):
    p = tmp_path / "clip.wav"
    _write_wav(p, seconds=0.5)  # 0.5s @16k mono 16-bit = 16000 bytes
    src = FileSource(str(p), sleep=lambda _s: None)
    chunks = list(src)
    assert sum(len(c) for c in chunks) == 16000
    assert all(len(c) <= sources.CHUNK_BYTES for c in chunks)
    assert len(chunks) == 5  # 16000 / 3200 = 5


def test_filesource_missing_file_raises():
    with pytest.raises(CLIError) as exc:
        FileSource("/no/such/file.wav")
    assert exc.value.exit_code == 2


def test_filesource_non_wav_without_ffmpeg_raises(tmp_path, monkeypatch):
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"not really audio")
    monkeypatch.setattr(sources.shutil, "which", lambda _name: None)
    with pytest.raises(CLIError) as exc:
        FileSource(str(p))
    assert exc.value.error_type == "ffmpeg_missing"


def test_filesource_uses_ffmpeg_for_non_wav(tmp_path, monkeypatch):
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"not really audio")
    monkeypatch.setattr(sources.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class FakeProc:
        def __init__(self):
            self.stdout = self
            self._data = [b"\x00" * 3200, b"\x01" * 100, b""]
            self._i = 0

        def read(self, _n):
            d = self._data[self._i]
            self._i += 1
            return d

        def close(self):
            pass

        def terminate(self):
            pass

        def wait(self):
            pass

    monkeypatch.setattr(sources.subprocess, "Popen", lambda *a, **k: FakeProc())
    chunks = list(FileSource(str(p), sleep=lambda _s: None))
    assert chunks == [b"\x00" * 3200, b"\x01" * 100]
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_streaming_sources.py -v`
Expected: FAIL (`ModuleNotFoundError: assemblyai_cli.streaming.sources`).

- [ ] **Step 4: Implement `assemblyai_cli/streaming/sources.py`** (FileSource portion):

```python
from __future__ import annotations

import shutil
import subprocess
import time
import wave
from collections.abc import Iterator
from pathlib import Path

from assemblyai_cli.errors import CLIError

TARGET_RATE = 16000
CHUNK_BYTES = TARGET_RATE * 2 // 10  # 100 ms of 16-bit mono PCM


def _is_streamable_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return (
                w.getnchannels() == 1
                and w.getsampwidth() == 2
                and w.getframerate() == TARGET_RATE
            )
    except (wave.Error, EOFError, OSError):
        return False


class FileSource:
    """Yields real-time-paced 16 kHz mono PCM chunks from an audio file."""

    def __init__(self, path: str, *, sleep=time.sleep) -> None:
        self.path = Path(path)
        self._sleep = sleep
        self.sample_rate = TARGET_RATE
        if not self.path.is_file():
            raise CLIError(
                f"No such file: {self.path}", error_type="file_not_found", exit_code=2
            )
        self._wav = _is_streamable_wav(self.path)
        if not self._wav and shutil.which("ffmpeg") is None:
            raise CLIError(
                "This audio format needs ffmpeg. Install ffmpeg, or pass a "
                "16 kHz mono 16-bit WAV.",
                error_type="ffmpeg_missing",
                exit_code=2,
            )

    def __iter__(self) -> Iterator[bytes]:
        chunks = self._wav_chunks() if self._wav else self._ffmpeg_chunks()
        for chunk in chunks:
            yield chunk
            self._sleep(CHUNK_BYTES / (TARGET_RATE * 2))  # ~real-time pacing

    def _wav_chunks(self) -> Iterator[bytes]:
        frames_per_chunk = CHUNK_BYTES // 2
        with wave.open(str(self.path), "rb") as w:
            while True:
                data = w.readframes(frames_per_chunk)
                if not data:
                    return
                yield data

    def _ffmpeg_chunks(self) -> Iterator[bytes]:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(self.path),
                "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1",
                "-ar", str(TARGET_RATE), "-",
            ],
            stdout=subprocess.PIPE,
        )
        try:
            while True:
                data = proc.stdout.read(CHUNK_BYTES)
                if not data:
                    return
                yield data
        finally:
            proc.stdout.close()
            proc.terminate()
            proc.wait()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_streaming_sources.py -v`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/streaming/__init__.py assemblyai_cli/streaming/sources.py tests/test_streaming_sources.py
git commit -m "feat(stream): add FileSource (wav + ffmpeg) audio source"
```

---

## Task 2: MicSource (optional PyAudio)

**Files:**
- Modify: `assemblyai_cli/streaming/sources.py`
- Test: `tests/test_streaming_sources.py` (extend)

- [ ] **Step 1: Add the failing test** (append to `tests/test_streaming_sources.py`):

```python
def test_micsource_missing_dependency_raises(monkeypatch):
    def boom():
        raise ImportError("No module named 'pyaudio'")

    monkeypatch.setattr(sources, "_load_microphone_stream", boom)
    with pytest.raises(CLIError) as exc:
        list(sources.MicSource(sample_rate=16000))
    assert exc.value.error_type == "mic_missing"
    assert "assemblyai-cli[mic]" in exc.value.message


def test_micsource_yields_from_microphone_stream(monkeypatch):
    captured = {}

    class FakeMic:
        def __init__(self, sample_rate, device_index):
            captured["rate"] = sample_rate
            captured["device"] = device_index

        def __iter__(self):
            return iter([b"\x00\x01", b"\x02\x03"])

    monkeypatch.setattr(sources, "_load_microphone_stream", lambda: FakeMic)
    chunks = list(sources.MicSource(sample_rate=16000, device=2))
    assert chunks == [b"\x00\x01", b"\x02\x03"]
    assert captured == {"rate": 16000, "device": 2}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_streaming_sources.py -k micsource -v`
Expected: FAIL (`AttributeError: module ... has no attribute '_load_microphone_stream'` / no `MicSource`).

- [ ] **Step 3: Add to `assemblyai_cli/streaming/sources.py`**:

```python
def _load_microphone_stream():
    """Import the SDK's PyAudio-backed mic stream (isolated for testing/patching)."""
    from assemblyai.extras import MicrophoneStream

    return MicrophoneStream


class MicSource:
    """Yields PCM chunks from the default microphone (requires the [mic] extra)."""

    def __init__(self, *, sample_rate: int, device: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device

    def __iter__(self) -> Iterator[bytes]:
        try:
            microphone_stream = _load_microphone_stream()
        except (ImportError, ModuleNotFoundError) as exc:
            raise CLIError(
                "Microphone support isn't installed. "
                "Run: pip install 'assemblyai-cli[mic]'",
                error_type="mic_missing",
                exit_code=2,
            ) from exc
        return iter(microphone_stream(sample_rate=self.sample_rate, device_index=self.device))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_streaming_sources.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/streaming/sources.py tests/test_streaming_sources.py
git commit -m "feat(stream): add MicSource with graceful missing-PyAudio error"
```

---

## Task 3: StreamRenderer (human line + NDJSON)

**Files:**
- Create: `assemblyai_cli/streaming/render.py`
- Test: `tests/test_streaming_render.py`

- [ ] **Step 1: Write the failing test** at `tests/test_streaming_render.py`:

```python
import io
import json
import types

from assemblyai_cli.streaming.render import StreamRenderer


def _turn(transcript, end_of_turn):
    return types.SimpleNamespace(transcript=transcript, end_of_turn=end_of_turn)


def test_human_turn_finalizes_on_end_of_turn():
    out = io.StringIO()
    r = StreamRenderer(json_mode=False, out=out)
    r.turn(_turn("hello", False))
    r.turn(_turn("hello world", True))
    text = out.getvalue()
    assert "hello world" in text
    assert text.endswith("\n")


def test_json_mode_emits_ndjson_events():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.begin(types.SimpleNamespace(id="sess_1"))
    r.turn(_turn("hi", True))
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert lines[0] == {"type": "begin", "id": "sess_1"}
    assert lines[1] == {"type": "turn", "transcript": "hi", "end_of_turn": True}


def test_human_begin_prints_notice():
    out = io.StringIO()
    StreamRenderer(json_mode=False, out=out).begin(types.SimpleNamespace(id="x"))
    assert "Ctrl-C" in out.getvalue()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_streaming_render.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `assemblyai_cli/streaming/render.py`**:

```python
from __future__ import annotations

import json
import sys


class StreamRenderer:
    """Renders streaming events: a live-updating line for humans, NDJSON for agents."""

    def __init__(self, *, json_mode: bool, out=None) -> None:
        self.json_mode = json_mode
        self.out = out if out is not None else sys.stdout
        self._width = 0

    def begin(self, event) -> None:
        if self.json_mode:
            self._emit({"type": "begin", "id": getattr(event, "id", None)})
        else:
            self.out.write("Listening… (Ctrl-C to stop)\n")
            self.out.flush()

    def turn(self, event) -> None:
        text = getattr(event, "transcript", "") or ""
        end = bool(getattr(event, "end_of_turn", False))
        if self.json_mode:
            self._emit({"type": "turn", "transcript": text, "end_of_turn": end})
            return
        self.out.write("\r" + text.ljust(self._width))
        self._width = max(self._width, len(text))
        if end:
            self.out.write("\n")
            self._width = 0
        self.out.flush()

    def termination(self, event) -> None:
        if self.json_mode:
            self._emit(
                {
                    "type": "termination",
                    "audio_duration_seconds": getattr(event, "audio_duration_seconds", None),
                }
            )

    def _emit(self, obj) -> None:
        self.out.write(json.dumps(obj) + "\n")
        self.out.flush()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_streaming_render.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/streaming/render.py tests/test_streaming_render.py
git commit -m "feat(stream): add StreamRenderer for live text and NDJSON output"
```

---

## Task 4: client.stream_audio (v3 wiring)

**Files:**
- Modify: `assemblyai_cli/client.py`
- Test: `tests/test_client.py` (extend)

- [ ] **Step 1: Add the failing test** (append to `tests/test_client.py`):

```python
import types as _types


class _FakeStreamingClient:
    last = None

    def __init__(self, options):
        self.handlers = {}
        self.connected = False
        self.disconnected = False
        _FakeStreamingClient.last = self

    def on(self, event, handler):
        self.handlers[event] = handler

    def connect(self, params):
        self.connected = True
        self.params = params

    def stream(self, source):
        from assemblyai.streaming.v3 import StreamingEvents

        self.handlers[StreamingEvents.Turn](
            self, _types.SimpleNamespace(transcript="hi", end_of_turn=True)
        )

    def disconnect(self):
        self.disconnected = True


def test_stream_audio_wires_handlers_and_streams(monkeypatch):
    monkeypatch.setattr(client, "StreamingClient", _FakeStreamingClient)
    turns = []
    client.stream_audio(
        "sk", [b"\x00"], sample_rate=16000, on_turn=lambda e: turns.append(e.transcript)
    )
    assert turns == ["hi"]
    assert _FakeStreamingClient.last.connected
    assert _FakeStreamingClient.last.disconnected  # disconnected in finally


def test_stream_audio_raises_on_error_event(monkeypatch):
    class ErrClient(_FakeStreamingClient):
        def stream(self, source):
            from assemblyai.streaming.v3 import StreamingEvents

            self.handlers[StreamingEvents.Error](self, "boom")

    monkeypatch.setattr(client, "StreamingClient", ErrClient)
    with pytest.raises(APIError):
        client.stream_audio("sk", [b"\x00"], sample_rate=16000)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_client.py -k stream_audio -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'StreamingClient'` / no `stream_audio`).

- [ ] **Step 3: Add to `assemblyai_cli/client.py`** — module-level import near the top (after `import assemblyai as aai`):

```python
from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)
```

and the function (anywhere after `get_transcript`):

```python
def stream_audio(
    api_key: str,
    source,
    *,
    sample_rate: int,
    on_begin=None,
    on_turn=None,
) -> None:
    """Stream `source` (an iterable of PCM bytes) through the v3 realtime API.

    Forwards Begin/Turn events to the callbacks; raises APIError on a stream error.
    """
    sc = StreamingClient(
        StreamingClientOptions(api_key=api_key, api_host="streaming.assemblyai.com")
    )
    errors: list[object] = []
    if on_begin is not None:
        sc.on(StreamingEvents.Begin, lambda _client, event: on_begin(event))
    if on_turn is not None:
        sc.on(StreamingEvents.Turn, lambda _client, event: on_turn(event))
    sc.on(StreamingEvents.Error, lambda _client, error: errors.append(error))

    sc.connect(StreamingParameters(sample_rate=sample_rate, format_turns=True))
    try:
        sc.stream(source)
    finally:
        sc.disconnect()
    if errors:
        raise APIError(f"Streaming error: {errors[0]}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_client.py -v`
Expected: all client tests PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/client.py tests/test_client.py
git commit -m "feat(stream): add client.stream_audio v3 wiring"
```

---

## Task 5: `aai stream` command

**Files:**
- Modify: `assemblyai_cli/commands/stream.py` (currently does not exist — create it)
- Test: `tests/test_stream_command.py`

- [ ] **Step 1: Write the failing test** at `tests/test_stream_command.py`:

```python
import json
import types

from typer.testing import CliRunner

from assemblyai_cli import config
from assemblyai_cli.main import app

runner = CliRunner()


def _drive_turns(api_key, source, *, sample_rate, on_begin=None, on_turn=None):
    # Simulate the streaming client driving the renderer callbacks.
    if on_begin:
        on_begin(types.SimpleNamespace(id="sess"))
    if on_turn:
        on_turn(types.SimpleNamespace(transcript="hello world", end_of_turn=True))


def test_stream_help_lists_command():
    result = runner.invoke(app, ["stream", "--help"])
    assert result.exit_code == 0
    assert "microphone" in result.output.lower()


def test_stream_mic_renders_turns(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("assemblyai_cli.commands.stream.client.stream_audio", _drive_turns)
    result = runner.invoke(app, ["stream", "--json"])
    assert result.exit_code == 0
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"type": "turn", "transcript": "hello world", "end_of_turn": True} in lines


def test_stream_file_uses_filesource(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_stream_audio(api_key, source, *, sample_rate, on_begin=None, on_turn=None):
        seen["source_type"] = type(source).__name__
        seen["rate"] = sample_rate

    monkeypatch.setattr("assemblyai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    # A real WAV so FileSource constructs successfully.
    import wave

    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    result = runner.invoke(app, ["stream", str(p)])
    assert result.exit_code == 0
    assert seen["source_type"] == "FileSource"
    assert seen["rate"] == 16000


def test_stream_unauthenticated_exits_2():
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 2


def test_stream_ctrl_c_exits_cleanly(monkeypatch):
    config.set_api_key("default", "sk_live")

    def raise_kbd(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("assemblyai_cli.commands.stream.client.stream_audio", raise_kbd)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_stream_command.py -v`
Expected: FAIL (no `stream` command registered).

- [ ] **Step 3: Implement `assemblyai_cli/commands/stream.py`**:

```python
from __future__ import annotations

import typer

from assemblyai_cli import client, config
from assemblyai_cli.context import run_command
from assemblyai_cli.streaming.render import StreamRenderer
from assemblyai_cli.streaming.sources import FileSource, MicSource

app = typer.Typer()


@app.command()
def stream(
    ctx: typer.Context,
    source: str = typer.Argument(
        None, help="Audio file to stream. Omit to use the microphone."
    ),
    sample_rate: int = typer.Option(
        16000, "--sample-rate", help="Microphone sample rate in Hz."
    ),
    device: int = typer.Option(None, "--device", help="Microphone device index."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit newline-delimited JSON events."
    ),
) -> None:
    """Transcribe live audio from the microphone or a file in real time."""

    def body(state, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        if source:
            audio = FileSource(source)
            rate = audio.sample_rate
        else:
            audio = MicSource(sample_rate=sample_rate, device=device)
            rate = sample_rate
        renderer = StreamRenderer(json_mode=json_mode)
        try:
            client.stream_audio(
                api_key,
                audio,
                sample_rate=rate,
                on_begin=renderer.begin,
                on_turn=renderer.turn,
            )
        except KeyboardInterrupt:
            if not json_mode:
                renderer.out.write("\nStopped.\n")
                renderer.out.flush()

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_stream_command.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/commands/stream.py tests/test_stream_command.py
git commit -m "feat(stream): add aai stream command (mic + file)"
```

---

## Task 6: Register the command + `[mic]` extra

**Files:**
- Modify: `assemblyai_cli/main.py`
- Modify: `pyproject.toml`
- Test: `tests/test_smoke.py` (extend)

- [ ] **Step 1: Add the failing test** (append to `tests/test_smoke.py`):

```python
def test_stream_registered_top_level():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "stream" in result.output
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_smoke.py::test_stream_registered_top_level -v`
Expected: FAIL (`stream` not in top-level help).

- [ ] **Step 3: Register in `assemblyai_cli/main.py`** — add `stream` to the commands import and register it. Change the import line:

```python
from assemblyai_cli.commands import login, samples, stream, transcribe, transcripts
```

and add, alongside the other `app.add_typer(...)` calls:

```python
app.add_typer(stream.app)
```

- [ ] **Step 4: Add the `[mic]` extra to `pyproject.toml`** — change the optional-dependencies table:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.11", "pre-commit>=4.0"]
mic = ["pyaudio>=0.2.11"]
```

- [ ] **Step 5: Run the smoke + full suite**

Run: `python -m pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/main.py pyproject.toml tests/test_smoke.py
git commit -m "feat(stream): register aai stream and add [mic] optional extra"
```

---

## Task 7: Full-suite verification + docs

**Files:**
- Modify: `README.md`
- Test: none (verification)

- [ ] **Step 1: Run the entire suite**

Run: `python -m pytest -q`
Expected: all tests PASS (sources, render, client streaming, stream command, smoke, plus the pre-existing suite).

- [ ] **Step 2: Verify the help tree**

Run: `python -m assemblyai_cli stream --help` and `python -m assemblyai_cli --help`
Expected: top-level lists `stream`; `stream --help` shows the SOURCE argument and `--sample-rate`/`--device`/`--json`.

- [ ] **Step 3: Add a streaming section to `README.md`** (append under usage):

```markdown
## Streaming

Real-time transcription from a file (no extra dependency):

    aai stream path/to/audio.wav        # 16 kHz mono WAV streams directly
    aai stream path/to/audio.mp3        # other formats require ffmpeg on PATH

From the microphone (install the optional extra first):

    pip install "assemblyai-cli[mic]"
    aai stream                          # Ctrl-C to stop

Add `--json` for newline-delimited JSON events (also the default when piped or run by an agent).
```

- [ ] **Step 4: (Optional, manual) live checks** — not part of the automated suite:

```bash
# File (needs a real key; works without a mic):
ASSEMBLYAI_API_KEY=... python -m assemblyai_cli stream some.wav
# Microphone (needs the [mic] extra + a real key):
ASSEMBLYAI_API_KEY=... python -m assemblyai_cli stream
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document aai stream usage"
```

---

## Self-Review notes

- **Spec coverage:** `aai stream [SOURCE]` + flags → Tasks 5/6; mic via optional `[mic]` extra with graceful error → Tasks 2/6; file via stdlib `wave` (WAV) and `ffmpeg` (other) with real-time pacing → Task 1; v3 `StreamingClient` wiring with Begin/Turn/Error → Task 4; human live line vs NDJSON via `resolve_json`/`StreamRenderer` → Task 3 (mode supplied by `run_command`); error paths (mic-missing exit 2, ffmpeg-missing exit 2, file-not-found exit 2, unauthenticated exit 2, stream error exit 1, Ctrl-C exit 0) → Tasks 1/2/4/5; testing strategy (WAV fixture, ffmpeg mock, mic-missing, render, command wiring) → Tasks 1–5.
- **Type/signature consistency:** `FileSource(path, *, sleep=time.sleep)` exposing `.sample_rate`; `MicSource(*, sample_rate, device=None)`; both iterable of `bytes`. `client.stream_audio(api_key, source, *, sample_rate, on_begin=None, on_turn=None)` raising `APIError` on error. `StreamRenderer(json_mode, out=None)` with `.begin/.turn/.termination`. The command calls `run_command(ctx, body, json=json_out)` (matching the established per-command `--json` pattern) and selects `FileSource` when `source` is truthy else `MicSource`. `CHUNK_BYTES`/`TARGET_RATE` are defined once in `sources.py` and referenced by tests via `sources.CHUNK_BYTES`.
- **Note:** `--sample-rate`/`--device` apply to the microphone; file input is normalized to 16 kHz mono (the command reads `audio.sample_rate`, which `FileSource` fixes at `TARGET_RATE`).
```
