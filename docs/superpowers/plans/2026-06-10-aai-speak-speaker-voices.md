# `aai speak` Speaker-Aware Multi-Voice Playback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `aai speak` is fed speaker-labeled transcript text (`aai transcribe … --speaker-labels`), strip the `Speaker X:` labels and synthesize each speaker in a different voice, concatenated with short pauses; plain text keeps today's single-voice behavior.

**Architecture:** A new pure module `aai_cli/tts/dialogue.py` does all detection/parsing/voice-assignment (no I/O). `session.synthesize_dialogue` loops the existing `session.synthesize` per segment and concatenates PCM with an `audio.silence` gap. `commands/speak.py` auto-detects labeled input and dispatches to a dialogue path or the existing single-voice path. Detection is automatic; `--voice` becomes repeatable and accepts `SPEAKER=VOICE` overrides.

**Tech Stack:** Python 3.12+, Typer, the `websockets` sync client (already wired in `session.py`), pytest. Spec: `docs/superpowers/specs/2026-06-10-aai-speak-speaker-voices-design.md`.

**Conventions (read before starting):**

- Every module starts with `from __future__ import annotations`; modern typing (`X | None`).
- Run all tooling through `uv run`. The gate is `./scripts/check.sh` (mutation + 100% patch coverage tail gates — assert *behavior*, not just execution).
- The post-edit hook auto-runs `ruff --fix` + `ruff format` on saved `*.py`, so don't hand-format.
- `errors.UsageError(message, *, suggestion=...)` → exit 2. `errors.CLIError` is its base.
- Help/CLI output is pinned by syrupy `.ambr` snapshots — regenerate with `--snapshot-update`, never hand-edit.

---

## File Structure

- **Create** `aai_cli/tts/dialogue.py` — `Segment` dataclass, `looks_like_speaker_labeled`, `parse_segments`, `parse_voice_overrides`, `assign_voices`, `DEFAULT_VOICE_ROTATION`. Pure, no I/O.
- **Create** `tests/test_tts_dialogue.py` — unit tests for the above.
- **Modify** `aai_cli/tts/audio.py` — add `silence(sample_rate, seconds) -> bytes`.
- **Modify** `tests/test_tts_audio.py` — test `silence`.
- **Modify** `aai_cli/tts/session.py` — add `synthesize_dialogue(...)` + `_INTER_TURN_SILENCE_SECONDS`.
- **Modify** `tests/test_tts_session.py` — test `synthesize_dialogue`.
- **Modify** `aai_cli/commands/speak.py` — repeatable `--voice`, detection/dispatch, multi-result emit.
- **Modify** `tests/test_speak.py` — multi-voice command tests.
- **Modify** `tests/__snapshots__/test_cli_output_snapshots.ambr` — regenerate `aai speak --help`.

---

## Task 1: `dialogue.py` — detection + segment parsing

**Files:**

- Create: `aai_cli/tts/dialogue.py`
- Test: `tests/test_tts_dialogue.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tts_dialogue.py`:

```python
from __future__ import annotations

from aai_cli.tts import dialogue


def test_detects_labeled_input_on_first_nonblank_line():
    assert dialogue.looks_like_speaker_labeled("Speaker A: hi\nSpeaker B: yo") is True
    # Leading blank lines are skipped before the check.
    assert dialogue.looks_like_speaker_labeled("\n\nSpeaker A: hi") is True


def test_plain_prose_is_not_labeled():
    assert dialogue.looks_like_speaker_labeled("Hello there, friend.") is False
    # A mid-sentence "Speaker" word must not trigger detection.
    assert dialogue.looks_like_speaker_labeled("The Speaker said hello") is False
    assert dialogue.looks_like_speaker_labeled("") is False


def test_parse_single_line_turns():
    segs = dialogue.parse_segments("Speaker A: Hello.\nSpeaker B: Hi there.")
    assert [(s.speaker_id, s.text) for s in segs] == [("A", "Hello."), ("B", "Hi there.")]


def test_parse_folds_wrapped_continuation_lines():
    # An utterance wrapped across physical lines (no label on the 2nd line) folds
    # back into one segment, joined with single spaces.
    text = "Speaker A: This is a long line that wrapped\nonto a second line here\nSpeaker B: Ok."
    segs = dialogue.parse_segments(text)
    assert [(s.speaker_id, s.text) for s in segs] == [
        ("A", "This is a long line that wrapped onto a second line here"),
        ("B", "Ok."),
    ]


def test_parse_merges_consecutive_same_speaker_turns():
    segs = dialogue.parse_segments("Speaker A: One.\nSpeaker A: Two.\nSpeaker B: Three.")
    assert [(s.speaker_id, s.text) for s in segs] == [("A", "One. Two."), ("B", "Three.")]


def test_parse_skips_blank_lines_and_drops_empty_turns():
    segs = dialogue.parse_segments("Speaker A: Hi.\n\nSpeaker B: \nSpeaker A: Bye.")
    # Speaker B's empty turn is dropped; the two A turns are not merged (B is between).
    assert [(s.speaker_id, s.text) for s in segs] == [("A", "Hi."), ("A", "Bye.")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tts_dialogue.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aai_cli.tts.dialogue'`.

- [ ] **Step 3: Implement `dialogue.py` (detection + parsing portion)**

Create `aai_cli/tts/dialogue.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass

# A rendered transcript line: "Speaker A: text". The id is a non-space run; the
# colon may be followed by a single space (label-only lines can lack trailing text).
_LABEL_RE = re.compile(r"^Speaker (?P<id>\S+): ?(?P<text>.*)$")


@dataclass(frozen=True)
class Segment:
    """One speaker turn after labels are stripped and wrapped lines folded in."""

    speaker_id: str
    text: str


def looks_like_speaker_labeled(text: str) -> bool:
    """True when the first non-blank line is a ``Speaker <id>:`` label line."""
    for line in text.splitlines():
        if line.strip():
            return _LABEL_RE.match(line) is not None
    return False


def parse_segments(text: str) -> list[Segment]:
    """Parse speaker-labeled text into merged per-turn segments.

    A ``Speaker <id>:`` line starts a turn; a following line that is not itself a
    label is a wrapped continuation appended to the current turn (joined with a
    single space). Consecutive same-speaker turns merge; empty turns are dropped.
    """
    turns: list[Segment] = []
    current_id: str | None = None
    parts: list[str] = []

    def flush() -> None:
        if current_id is not None:
            turns.append(Segment(current_id, " ".join(p for p in parts if p)))

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _LABEL_RE.match(line)
        if match:
            flush()
            current_id = match.group("id")
            parts = [match.group("text").strip()]
        elif current_id is not None:
            parts.append(stripped)
    flush()

    merged: list[Segment] = []
    for turn in turns:
        if merged and merged[-1].speaker_id == turn.speaker_id:
            joined = f"{merged[-1].text} {turn.text}".strip()
            merged[-1] = Segment(turn.speaker_id, joined)
        else:
            merged.append(turn)
    return [turn for turn in merged if turn.text]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tts_dialogue.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/tts/dialogue.py tests/test_tts_dialogue.py
git commit -m "feat(speak): parse speaker-labeled transcript text into segments"
```

---

## Task 2: `dialogue.py` — voice overrides + assignment

**Files:**

- Modify: `aai_cli/tts/dialogue.py`
- Test: `tests/test_tts_dialogue.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tts_dialogue.py`:

```python
import pytest

from aai_cli.errors import UsageError


def test_parse_voice_overrides_splits_bare_and_mapped():
    bare, overrides = dialogue.parse_voice_overrides(["A=vera", "mary", "B=paul"])
    assert bare == "mary"
    assert overrides == {"a": "vera", "b": "paul"}  # ids casefolded


def test_parse_voice_overrides_bare_last_wins_and_empty_default():
    assert dialogue.parse_voice_overrides([]) == (None, {})
    assert dialogue.parse_voice_overrides(["jane", "mary"]) == ("mary", {})


def test_parse_voice_overrides_rejects_malformed_pair():
    for bad in ["=vera", "A=", "  =  "]:
        with pytest.raises(UsageError):
            dialogue.parse_voice_overrides([bad])


def test_assign_voices_rotates_in_first_appearance_order():
    segs = [dialogue.Segment(s, "x") for s in ("A", "B", "A", "C")]
    resolved, mapping = dialogue.assign_voices(segs, ["jane", "michael", "mary"], {})
    assert [v for v, _ in resolved] == ["jane", "michael", "jane", "mary"]
    assert mapping == {"A": "jane", "B": "michael", "C": "mary"}


def test_assign_voices_wraps_past_rotation_length():
    segs = [dialogue.Segment(s, "x") for s in ("A", "B", "C")]
    resolved, _ = dialogue.assign_voices(segs, ["jane", "michael"], {})
    assert [v for v, _ in resolved] == ["jane", "michael", "jane"]


def test_assign_voices_override_beats_rotation_without_consuming_a_slot():
    segs = [dialogue.Segment(s, "x") for s in ("A", "B")]
    # A is overridden, so B still gets the FIRST rotation voice, not the second.
    resolved, mapping = dialogue.assign_voices(segs, ["jane", "michael"], {"a": "vera"})
    assert [v for v, _ in resolved] == ["vera", "jane"]
    assert mapping == {"A": "vera", "B": "jane"}


def test_default_rotation_is_the_confirmed_working_voices():
    assert dialogue.DEFAULT_VOICE_ROTATION == ("jane", "michael", "mary", "paul", "eve", "george")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tts_dialogue.py -q`
Expected: FAIL — `AttributeError: module 'aai_cli.tts.dialogue' has no attribute 'parse_voice_overrides'`.

- [ ] **Step 3: Implement the assignment portion**

Add to the imports at the top of `aai_cli/tts/dialogue.py`:

```python
from collections.abc import Sequence

from aai_cli.errors import UsageError
```

Append to `aai_cli/tts/dialogue.py`:

```python
DEFAULT_VOICE_ROTATION = ("jane", "michael", "mary", "paul", "eve", "george")


def parse_voice_overrides(values: list[str]) -> tuple[str | None, dict[str, str]]:
    """Split repeatable ``--voice`` values into ``(bare_voice, {speaker_id: voice})``.

    A value containing ``=`` is a ``SPEAKER=VOICE`` mapping (ids casefolded so the
    match is case-insensitive); a bare value sets the single-voice default, last
    one wins. Raises ``UsageError`` on a malformed pair (empty side).
    """
    bare: str | None = None
    overrides: dict[str, str] = {}
    for value in values:
        if "=" in value:
            speaker, _, voice = value.partition("=")
            speaker, voice = speaker.strip(), voice.strip()
            if not speaker or not voice:
                raise UsageError(
                    f"Invalid --voice mapping {value!r}.",
                    suggestion="Use SPEAKER=VOICE, e.g. --voice A=jane.",
                )
            overrides[speaker.casefold()] = voice
        elif value.strip():
            bare = value.strip()
    return bare, overrides


def assign_voices(
    segments: list[Segment],
    rotation: Sequence[str],
    overrides: dict[str, str],
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Resolve each segment to ``(voice, text)`` and return the id→voice map.

    A speaker uses its override if present, else the next rotation voice in
    first-appearance order (wrapping). Overrides do not consume rotation slots.
    The returned map is ordered by first appearance.
    """
    id_to_voice: dict[str, str] = {}
    next_index = 0
    resolved: list[tuple[str, str]] = []
    for segment in segments:
        if segment.speaker_id not in id_to_voice:
            override = overrides.get(segment.speaker_id.casefold())
            if override is not None:
                id_to_voice[segment.speaker_id] = override
            else:
                id_to_voice[segment.speaker_id] = rotation[next_index % len(rotation)]
                next_index += 1
        resolved.append((id_to_voice[segment.speaker_id], segment.text))
    return resolved, id_to_voice
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tts_dialogue.py -q`
Expected: PASS (13 tests total).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/tts/dialogue.py tests/test_tts_dialogue.py
git commit -m "feat(speak): resolve speaker voices via rotation + overrides"
```

---

## Task 3: `audio.silence` helper

**Files:**

- Modify: `aai_cli/tts/audio.py`
- Test: `tests/test_tts_audio.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tts_audio.py`:

```python
def test_silence_returns_zeroed_pcm_of_the_right_length():
    # 16-bit mono: 100 ms at 16 kHz = 1600 frames = 3200 zero bytes.
    pcm = audio.silence(16000, 0.1)
    assert pcm == b"\x00" * 3200
    # Empty duration -> no bytes.
    assert audio.silence(24000, 0.0) == b""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tts_audio.py::test_silence_returns_zeroed_pcm_of_the_right_length -q`
Expected: FAIL — `AttributeError: module 'aai_cli.tts.audio' has no attribute 'silence'`.

- [ ] **Step 3: Implement `silence`**

Add to `aai_cli/tts/audio.py` (after `write_wav`, before `_default_output_stream`):

```python
def silence(sample_rate: int, seconds: float) -> bytes:
    """Zeroed 16-bit mono PCM of the given duration (2 bytes per frame)."""
    return b"\x00" * (int(sample_rate * seconds) * 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tts_audio.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aai_cli/tts/audio.py tests/test_tts_audio.py
git commit -m "feat(speak): add silence() PCM helper for inter-turn gaps"
```

---

## Task 4: `session.synthesize_dialogue`

**Files:**

- Modify: `aai_cli/tts/session.py`
- Test: `tests/test_tts_session.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tts_session.py` (reuses `FakeWS`, `_begin_frame`, `_audio_frame` already in that file):

```python
def test_synthesize_dialogue_concatenates_segments_with_silence():
    # One fresh fake socket per segment; record the voice each connection requested.
    sockets = [
        FakeWS([_begin_frame(sample_rate=24000), _audio_frame(b"\xaa\xbb", final=True)]),
        FakeWS([_begin_frame(sample_rate=24000), _audio_frame(b"\xcc\xdd", final=True)]),
    ]
    urls: list[str] = []

    def _connect(url: str, **_kwargs):
        urls.append(url)
        return sockets.pop(0)

    result = session.synthesize_dialogue(
        "k",
        [("jane", "Hello."), ("michael", "Hi.")],
        language="English",
        connect=_connect,
    )
    # Each segment connected with its own voice.
    assert "voice=jane" in urls[0]
    assert "voice=michael" in urls[1]
    # 0.25 s of silence (24000 * 0.25 * 2 = 12000 zero bytes) sits BETWEEN the two
    # segments' PCM, with none at the ends.
    gap = b"\x00" * 12000
    assert result.pcm == b"\xaa\xbb" + gap + b"\xcc\xdd"
    assert result.sample_rate == 24000


def test_synthesize_dialogue_single_segment_has_no_silence():
    ws = FakeWS([_begin_frame(sample_rate=24000), _audio_frame(b"\x01\x02", final=True)])
    result = session.synthesize_dialogue("k", [("jane", "Hi.")], connect=lambda *a, **k: ws)
    assert result.pcm == b"\x01\x02"  # no leading/trailing pad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tts_session.py -q -k dialogue`
Expected: FAIL — `AttributeError: module 'aai_cli.tts.session' has no attribute 'synthesize_dialogue'`.

- [ ] **Step 3: Implement `synthesize_dialogue`**

Add `from aai_cli.tts import audio` to the imports in `aai_cli/tts/session.py` (top, with the other `aai_cli` imports).

Add a constant near `_DEFAULT_SAMPLE_RATE`:

```python
# Pause inserted between speaker turns in a multi-voice dialogue, for natural pacing.
_INTER_TURN_SILENCE_SECONDS = 0.25
```

Append after `synthesize`:

```python
def synthesize_dialogue(
    api_key: str,
    segments: list[tuple[str, str]],
    *,
    language: str | None = None,
    sample_rate: int | None = None,
    connect: _Connect | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> SpeakResult:
    """Synthesize each ``(voice, text)`` segment and concatenate the PCM.

    Each segment opens its own connection (the voice is fixed at connect time).
    A short silence is inserted between turns — never at the ends. The result's
    sample rate is the rate the server reported for the segments.
    """
    pcm = bytearray()
    sample_rate_out = _DEFAULT_SAMPLE_RATE
    for index, (voice, text) in enumerate(segments):
        config = SpeakConfig(text=text, voice=voice, language=language, sample_rate=sample_rate)
        result = synthesize(api_key, config, connect=connect, on_warning=on_warning)
        if index:
            pcm.extend(audio.silence(result.sample_rate, _INTER_TURN_SILENCE_SECONDS))
        pcm.extend(result.pcm)
        sample_rate_out = result.sample_rate
    duration = len(pcm) / 2 / sample_rate_out
    return SpeakResult(bytes(pcm), sample_rate_out, duration)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tts_session.py -q`
Expected: PASS (all, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/tts/session.py tests/test_tts_session.py
git commit -m "feat(speak): synthesize_dialogue concatenates per-voice segments"
```

---

## Task 5: Wire detection + dispatch into `commands/speak.py`

**Files:**

- Modify: `aai_cli/commands/speak.py`
- Test: `tests/test_speak.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_speak.py`:

```python
from aai_cli.tts import session as tts_session


@pytest.fixture
def fake_dialogue(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    def _fake(api_key, segments, *, language=None, sample_rate=None, connect=None, on_warning=None):
        calls["segments"] = segments
        calls["language"] = language
        return tts_session.SpeakResult(
            pcm=b"\x01\x02", sample_rate=24000, audio_duration_seconds=1.5
        )

    monkeypatch.setattr(tts_session, "synthesize_dialogue", _fake)
    monkeypatch.setattr("aai_cli.commands.speak.audio.play_pcm", lambda *a, **k: None)
    return calls


def test_labeled_stdin_uses_dialogue_path_with_default_rotation(fake_dialogue):
    text = "Speaker A: Hello there.\nSpeaker B: Hi.\nSpeaker A: Bye."
    result = runner.invoke(app, ["--sandbox", "speak"], input=text)
    assert result.exit_code == 0
    # Labels stripped; consecutive A turns are NOT merged (B between); voices rotate.
    assert fake_dialogue["segments"] == [
        ("jane", "Hello there."),
        ("michael", "Hi."),
        ("jane", "Bye."),
    ]


def test_speaker_voice_override_is_applied(fake_dialogue):
    text = "Speaker A: One.\nSpeaker B: Two."
    result = runner.invoke(
        app, ["--sandbox", "speak", "--voice", "A=vera", "--voice", "B=paul"], input=text
    )
    assert result.exit_code == 0
    assert fake_dialogue["segments"] == [("vera", "One."), ("paul", "Two.")]


def test_bare_voice_in_dialogue_mode_is_ignored_with_a_note(fake_dialogue):
    text = "Speaker A: One.\nSpeaker B: Two."
    result = runner.invoke(app, ["--sandbox", "speak", "--voice", "mary"], input=text)
    assert result.exit_code == 0
    # The rotation still drives voices (bare voice ignored)...
    assert fake_dialogue["segments"] == [("jane", "One."), ("michael", "Two.")]
    # ...and the user is told why, pointed at the per-speaker form.
    assert "A=NAME" in result.stderr


def test_dialogue_json_reports_speaker_voice_map(fake_dialogue):
    text = "Speaker A: One.\nSpeaker B: Two."
    result = runner.invoke(app, ["--sandbox", "speak", "--json"], input=text)
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip())
    assert payload["mode"] == "multi"
    assert payload["speakers"] == {"A": "jane", "B": "michael"}
    assert payload["segments"] == 2
    assert payload["sample_rate"] == 24000


def test_unlabeled_text_still_uses_single_voice_path(fake_synthesize, monkeypatch):
    # A bare --voice still selects the single-voice voice for ordinary prose.
    monkeypatch.setattr("aai_cli.commands.speak.audio.play_pcm", lambda *a, **k: None)
    result = runner.invoke(app, ["--sandbox", "speak", "Just prose.", "--voice", "mary"])
    assert result.exit_code == 0
    assert fake_synthesize["cfg"].voice == "mary"
    assert fake_synthesize["cfg"].text == "Just prose."
```

Note: the existing `test_voice_and_language_flow_into_config` passes `--voice jane` (a bare value) with prose — that keeps working because a bare `--voice` still selects the single-voice voice. No change needed there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_speak.py -q`
Expected: FAIL — the new tests error (e.g. `--voice` not repeatable / dialogue path absent), single-voice tests still pass.

- [ ] **Step 3: Rewrite `commands/speak.py`**

Replace the entire file `aai_cli/commands/speak.py` with:

```python
from __future__ import annotations

import sys
from pathlib import Path

import typer

from aai_cli import config, help_panels, options, output
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError, UsageError
from aai_cli.help_text import examples_epilog
from aai_cli.tts import audio, dialogue, session

app = typer.Typer()

# The streaming-TTS reference client defaults to the PocketTTS "jane" voice and
# English, so the CLI sends the same and a bare `aai speak` works out of the box.
# Override either with --voice/--language.
DEFAULT_VOICE = "jane"
DEFAULT_LANGUAGE = "English"


def _read_text(text: str | None) -> str:
    """The text to speak: the argument if non-blank, else stdin when piped."""
    if text is not None and text.strip():
        return text
    if text is None and not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    raise UsageError(
        "No text to speak.",
        suggestion='Pass text as an argument: aai speak "Hello" — or pipe it via stdin.',
    )


def _output_audio(result: session.SpeakResult, out: Path | None) -> None:
    """Write a WAV when --out is given, else play through the speakers."""
    if out is not None:
        audio.write_wav(out, result.pcm, result.sample_rate)
    else:
        audio.play_pcm(result.pcm, result.sample_rate)


def _disposition(out: Path | None) -> str:
    return f"saved to {out}" if out is not None else "played"


def _emit_single(
    result: session.SpeakResult,
    cfg: session.SpeakConfig,
    out: Path | None,
    *,
    json_mode: bool,
) -> None:
    """Single-voice result: a JSON object on stdout, or a human note on stderr."""
    duration = round(result.audio_duration_seconds, 3)
    if json_mode:
        output.emit_ndjson(
            {
                "voice": cfg.voice,
                "language": cfg.language,
                "sample_rate": result.sample_rate,
                "audio_duration_seconds": duration,
                "bytes": len(result.pcm),
                "out": str(out) if out is not None else None,
            }
        )
        return
    output.error_console.print(
        f"[aai.muted]Spoke {duration}s of audio ({_disposition(out)}).[/aai.muted]"
    )


def _emit_multi(
    result: session.SpeakResult,
    speakers: dict[str, str],
    segment_count: int,
    out: Path | None,
    *,
    json_mode: bool,
) -> None:
    """Multi-voice result: a JSON object on stdout, or a human note on stderr."""
    duration = round(result.audio_duration_seconds, 3)
    if json_mode:
        output.emit_ndjson(
            {
                "mode": "multi",
                "speakers": speakers,
                "segments": segment_count,
                "sample_rate": result.sample_rate,
                "audio_duration_seconds": duration,
                "bytes": len(result.pcm),
                "out": str(out) if out is not None else None,
            }
        )
        return
    voices = ", ".join(f"{spk}={voice}" for spk, voice in speakers.items())
    output.error_console.print(
        f"[aai.muted]Spoke {duration}s across {len(speakers)} voices "
        f"({voices}) ({_disposition(out)}).[/aai.muted]"
    )


def _speak_single(
    api_key: str,
    text: str,
    voice: str,
    language: str,
    sample_rate: int | None,
    out: Path | None,
    *,
    json_mode: bool,
    quiet: bool,
) -> None:
    cfg = session.SpeakConfig(text=text, voice=voice, language=language, sample_rate=sample_rate)
    with output.status("Synthesizing speech…", json_mode=json_mode, quiet=quiet):
        result = session.synthesize(
            api_key, cfg, on_warning=lambda m: output.emit_warning(m, json_mode=json_mode)
        )
    _output_audio(result, out)
    _emit_single(result, cfg, out, json_mode=json_mode)


def _speak_dialogue(
    api_key: str,
    text: str,
    bare_voice: str | None,
    overrides: dict[str, str],
    language: str,
    sample_rate: int | None,
    out: Path | None,
    *,
    json_mode: bool,
    quiet: bool,
) -> None:
    segments = dialogue.parse_segments(text)
    if not segments:
        raise UsageError(
            "No text to speak.",
            suggestion="The input had speaker labels but no spoken text.",
        )
    if bare_voice is not None:
        output.emit_warning(
            "Ignoring bare --voice in multi-speaker mode; "
            "set a voice per speaker with --voice A=NAME.",
            json_mode=json_mode,
        )
    resolved, speakers = dialogue.assign_voices(
        segments, dialogue.DEFAULT_VOICE_ROTATION, overrides
    )
    with output.status("Synthesizing speech…", json_mode=json_mode, quiet=quiet):
        result = session.synthesize_dialogue(
            api_key,
            resolved,
            language=language,
            sample_rate=sample_rate,
            on_warning=lambda m: output.emit_warning(m, json_mode=json_mode),
        )
    _output_audio(result, out)
    _emit_multi(result, speakers, len(resolved), out, json_mode=json_mode)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Speak text aloud (sandbox only)", 'aai speak "Hello there, friend." --sandbox'),
            (
                "Pick a voice and language",
                'aai speak "Bonjour" --voice jane --language French --sandbox',
            ),
            (
                "Speak a diarized transcript, one voice per speaker",
                "aai transcribe meeting.mp3 --speaker-labels | aai speak --sandbox",
            ),
            (
                "Override a speaker's voice",
                "… | aai speak --voice A=vera --voice B=paul --sandbox",
            ),
            (
                "Save to a WAV instead of playing",
                'aai speak "Hello" --out /tmp/hello.wav --sandbox',
            ),
        ]
    ),
)
def speak(
    ctx: typer.Context,
    text: str | None = typer.Argument(None, help="Text to speak. Omit to read from stdin."),
    voice: list[str] = typer.Option(
        [],
        "--voice",
        help="Voice id, or SPEAKER=VOICE for diarized input (repeatable, e.g. --voice A=jane).",
    ),
    language: str = typer.Option(DEFAULT_LANGUAGE, "--language", help="Language of the text."),
    sample_rate: int | None = typer.Option(
        None, "--sample-rate", help="Output sample rate in Hz. Server default if omitted."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Write a WAV file instead of playing through the speakers."
    ),
    json_out: bool = options.json_option("Emit JSON metadata about the synthesized audio."),
) -> None:
    """Synthesize speech from text with AssemblyAI streaming TTS (sandbox only).

    Plays the audio through your speakers by default, or writes a WAV with --out.
    Speaker-labeled input (from 'aai transcribe --speaker-labels') is detected
    automatically: the labels are stripped and each speaker gets a different
    voice. This feature only exists in the sandbox today — run it with --sandbox.
    """

    def body(state: AppState, json_mode: bool) -> None:
        if not session.is_available():
            raise CLIError(
                "aai speak is only available in the sandbox.",
                error_type="unsupported_environment",
                exit_code=2,
                suggestion="Re-run with --sandbox (or --env sandbox000).",
            )
        spoken = _read_text(text)
        api_key = config.resolve_api_key(profile=state.profile)
        bare_voice, overrides = dialogue.parse_voice_overrides(voice)
        if dialogue.looks_like_speaker_labeled(spoken):
            _speak_dialogue(
                api_key,
                spoken,
                bare_voice,
                overrides,
                language,
                sample_rate,
                out,
                json_mode=json_mode,
                quiet=state.quiet,
            )
        else:
            _speak_single(
                api_key,
                spoken,
                bare_voice or DEFAULT_VOICE,
                language,
                sample_rate,
                out,
                json_mode=json_mode,
                quiet=state.quiet,
            )

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_speak.py -q`
Expected: PASS (existing + new). If `test_voice_and_language_flow_into_config` fails on `cfg.voice`, confirm it passes a bare `--voice jane` with prose text — it should still set `cfg.voice == "jane"` via the single path.

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/speak.py tests/test_speak.py
git commit -m "feat(speak): auto-detect diarized input and voice each speaker"
```

---

## Task 6: Regenerate the help snapshot, then run the full gate + live check

**Files:**

- Modify: `tests/__snapshots__/test_cli_output_snapshots.ambr`

- [ ] **Step 1: Regenerate the `aai speak --help` snapshot**

The `--voice` help text and the epilog examples changed.

Run: `uv run pytest tests/test_cli_output_snapshots.py --snapshot-update -q`
Then inspect: `git diff tests/__snapshots__/test_cli_output_snapshots.ambr`
Expected: only the `[speak]` help block changes (new `--voice` help string, new examples). No other snapshot should move.

- [ ] **Step 2: Run the full gate**

Run: `./scripts/check.sh`
Expected: ends with `All checks passed.` This is the source of truth — it covers ruff, mypy, pyright, vulture, import-linter, xenon (complexity ≤ B; if `speak.py`'s `body`/helpers trip it, the helper split above should keep each function simple — do not re-inline), 90% branch coverage, 100% patch coverage, and the diff-scoped mutation gate.

If the mutation gate reports survivors on changed lines, add a behavioral assertion that fails when that line breaks (e.g. assert the *exact* silence byte-length, the *exact* resolved voice list, or the specific note substring) rather than loosening the test.

- [ ] **Step 3: Live sandbox smoke test (manual, optional but recommended)**

A sandbox key is in the keyring (profile `default`). Verify the real pipe end-to-end, writing to `/tmp` (never the repo root) and wrapping so a blocking path can't wedge the session:

```bash
printf 'Speaker A: Hello there friend.\nSpeaker B: I am doing great, thanks for asking.\n' \
  | uv run aai --sandbox speak --out /tmp/dialogue.wav --json
file /tmp/dialogue.wav   # expect: RIFF ... WAVE audio, 16 bit, mono 24000 Hz
```

Expected JSON: `{"mode":"multi","speakers":{"A":"jane","B":"michael"},"segments":2,...}` and a valid multi-second WAV. (If the sandbox blocks the host or a chosen voice returns `RetryError`, that surfaces as a clean error — note it, don't treat it as a code defect.)

- [ ] **Step 4: Commit the snapshot**

```bash
git add tests/__snapshots__/test_cli_output_snapshots.ambr
git commit -m "test(speak): regenerate help snapshot for repeatable --voice"
```

---

## Self-Review Notes

- **Spec coverage:** detection (Task 1), parsing/wrap-folding/merge (Task 1), rotation + overrides + bare-voice rule (Tasks 2 & 5), silence gap (Tasks 3 & 4), per-segment synth/concat (Task 4), command dispatch + JSON shapes + stderr note (Task 5), help snapshot (Task 6). Error table: malformed `--voice` (Task 2), empty-after-strip (Task 5 `_speak_dialogue`), bare-voice note (Task 5).
- **Type consistency:** `Segment(speaker_id, text)`, `assign_voices -> (list[tuple[str,str]], dict[str,str])`, `parse_voice_overrides -> (str|None, dict[str,str])`, `synthesize_dialogue(api_key, segments, *, language, sample_rate, connect, on_warning)` are used identically across tasks and the command.
- **`--voice` default `[]`:** lets `_speak_single` fall back to `DEFAULT_VOICE` and lets the multi-mode note fire only on an explicit bare voice.
