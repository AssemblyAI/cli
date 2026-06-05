# Full SDK Option Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every AssemblyAI `TranscriptionConfig` and `StreamingParameters` option controllable from `aai transcribe` and `aai stream`, via curated typed flags plus a `--config KEY=VALUE` / `--config-file` escape hatch, and auto-render analysis results in human mode.

**Architecture:** A new `config_builder.py` module owns the merge of three layers (config-file < `--config` < explicit flags), coerces string values per field type, validates keys/enums, and returns ready SDK config objects. `client.py` is changed to accept prebuilt config objects. A new `transcribe_render.py` renders analysis sections (summary, chapters, sentiment, etc.) in human mode. Commands stay thin: they map typer options to SDK-named dicts and delegate to the builder.

**Tech Stack:** Python 3.11, Typer, `assemblyai` SDK (≥0.34), Rich, pytest, Hypothesis.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `assemblyai_cli/config_builder.py` (new) | Field maps, value coercion, layer merge, validation, build `TranscriptionConfig` / `StreamingParameters`; small flag-normalization helpers. |
| `assemblyai_cli/transcribe_render.py` (new) | Render transcript text + conditional analysis sections in human mode. |
| `assemblyai_cli/client.py` (modify) | `transcribe()` / `stream_audio()` accept prebuilt config objects. |
| `assemblyai_cli/commands/transcribe.py` (modify) | New curated flags + escape hatch; delegate to builder; render via `transcribe_render`. |
| `assemblyai_cli/commands/stream.py` (modify) | New curated flags + escape hatch; delegate to builder. |
| `README.md` (modify) | Document new options and the escape hatch. |
| `tests/test_config_builder.py` (new) | Coercion, precedence, validation, enum errors, per-flag mapping. |
| `tests/test_transcribe_render.py` (new) | Per-section rendering + absent-field no-ops. |
| `tests/test_transcribe.py` (modify) | Flag→config assertions for new transcribe flags. |
| `tests/test_stream_command.py` (modify) | Flag→params assertions for new stream flags. |
| `tests/test_client.py` (modify) | New `client.transcribe`/`stream_audio` signatures. |
| `tests/test_properties.py` (modify) | Property test for coercion round-trips. |
| `tests/e2e/test_cli_e2e.py` (modify) | Real-API analysis-feature run. |

---

## Task 1: config_builder — field maps, coercion, merge, validation

**Files:**
- Create: `assemblyai_cli/config_builder.py`
- Test: `tests/test_config_builder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_builder.py`:

```python
import json

import pytest

from assemblyai_cli import config_builder as cb
from assemblyai_cli.errors import UsageError


def test_coerce_bool_int_float_list():
    assert cb.coerce_value("speaker_labels", "true") is True
    assert cb.coerce_value("speaker_labels", "false") is False
    assert cb.coerce_value("speakers_expected", "2") == 2
    assert cb.coerce_value("speech_threshold", "0.5") == 0.5
    assert cb.coerce_value("redact_pii_policies", "person_name, phone_number") == [
        "person_name",
        "phone_number",
    ]


def test_coerce_str_passthrough_and_json():
    assert cb.coerce_value("language_code", "en_us") == "en_us"
    assert cb.coerce_value("custom_spelling", '{"AssemblyAI": ["assembly ai"]}') == {
        "AssemblyAI": ["assembly ai"]
    }


def test_coerce_bad_bool_and_int_raise_usageerror():
    with pytest.raises(UsageError):
        cb.coerce_value("speaker_labels", "maybe")
    with pytest.raises(UsageError):
        cb.coerce_value("speakers_expected", "two")


def test_parse_config_overrides_unknown_key_lists_valid():
    with pytest.raises(UsageError) as exc:
        cb.parse_config_overrides(cb.TRANSCRIBE_FIELDS, ["not_a_field=1"])
    assert "not_a_field" in str(exc.value)
    assert "speaker_labels" in str(exc.value)  # error lists valid fields


def test_parse_config_overrides_requires_equals():
    with pytest.raises(UsageError):
        cb.parse_config_overrides(cb.TRANSCRIBE_FIELDS, ["speaker_labels"])


def test_build_transcription_config_layer_precedence(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"speaker_labels": False, "speakers_expected": 5}))
    tc = cb.build_transcription_config(
        flags={"speaker_labels": True},  # flag beats file
        overrides=["speakers_expected=3"],  # --config beats file
        config_file=str(cfg),
    )
    assert tc.speaker_labels is True
    assert tc.raw.speakers_expected == 3


def test_build_transcription_config_ignores_unset_flags():
    tc = cb.build_transcription_config(flags={"speaker_labels": None}, overrides=[], config_file=None)
    assert tc.speaker_labels is None  # None means "not set", does not override


def test_load_config_file_rejects_non_object(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]")
    with pytest.raises(UsageError):
        cb.load_config_file(bad, cb.TRANSCRIBE_FIELDS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_config_builder.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'assemblyai_cli.config_builder'`.

- [ ] **Step 3: Write `config_builder.py`**

Create `assemblyai_cli/config_builder.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import assemblyai as aai
from assemblyai.streaming.v3 import StreamingParameters

from assemblyai_cli.errors import UsageError

# field name -> coercion kind for --config/--config-file string values.
# The KEYS are the authoritative set of valid config fields per command.
TRANSCRIBE_COERCE: dict[str, str] = {
    "language_code": "str",
    "language_codes": "list",
    "punctuate": "bool",
    "format_text": "bool",
    "dual_channel": "bool",
    "multichannel": "bool",
    "webhook_url": "str",
    "webhook_auth_header_name": "str",
    "webhook_auth_header_value": "str",
    "audio_start_from": "int",
    "audio_end_at": "int",
    "word_boost": "list",
    "boost_param": "str",
    "filter_profanity": "bool",
    "redact_pii": "bool",
    "redact_pii_audio": "bool",
    "redact_pii_audio_quality": "str",
    "redact_pii_audio_options": "json",
    "redact_pii_policies": "list",
    "redact_pii_sub": "str",
    "redact_pii_return_unredacted": "bool",
    "speaker_labels": "bool",
    "speakers_expected": "int",
    "speaker_options": "json",
    "content_safety": "bool",
    "content_safety_confidence": "int",
    "iab_categories": "bool",
    "custom_spelling": "json",
    "disfluencies": "bool",
    "sentiment_analysis": "bool",
    "auto_chapters": "bool",
    "entity_detection": "bool",
    "summarization": "bool",
    "summary_model": "str",
    "summary_type": "str",
    "auto_highlights": "bool",
    "language_detection": "bool",
    "language_confidence_threshold": "float",
    "language_detection_options": "json",
    "speech_threshold": "float",
    "speech_model": "str",
    "speech_models": "list",
    "prompt": "str",
    "temperature": "float",
    "remove_audio_tags": "str",
    "keyterms_prompt": "list",
    "keyterms_prompt_options": "json",
    "speech_understanding": "json",
    "domain": "str",
}

STREAM_COERCE: dict[str, str] = {
    "end_of_turn_confidence_threshold": "float",
    "min_end_of_turn_silence_when_confident": "int",
    "min_turn_silence": "int",
    "max_turn_silence": "int",
    "vad_threshold": "float",
    "format_turns": "bool",
    "keyterms_prompt": "list",
    "filter_profanity": "bool",
    "prompt": "str",
    "sample_rate": "int",
    "encoding": "str",
    "speech_model": "str",
    "language_detection": "bool",
    "domain": "str",
    "inactivity_timeout": "int",
    "webhook_url": "str",
    "webhook_auth_header_name": "str",
    "webhook_auth_header_value": "str",
    "llm_gateway": "json",
    "speaker_labels": "bool",
    "max_speakers": "int",
    "voice_focus": "str",
    "voice_focus_threshold": "float",
    "noise_suppression_model": "str",
    "noise_suppression_threshold": "float",
    "continuous_partials": "bool",
    "customer_support_audio_capture": "bool",
    "include_partial_turns": "bool",
    "redact_pii": "bool",
    "redact_pii_policies": "list",
    "redact_pii_sub": "str",
}

TRANSCRIBE_FIELDS = TRANSCRIBE_COERCE
STREAM_FIELDS = STREAM_COERCE

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def coerce_value(field: str, raw: str) -> object:
    """Coerce a string --config value to the type expected by `field`."""
    kind = TRANSCRIBE_COERCE.get(field) or STREAM_COERCE.get(field, "str")
    if kind == "bool":
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise UsageError(f"{field} expects a boolean (true/false), got {raw!r}.")
    if kind == "int":
        try:
            return int(raw)
        except ValueError as exc:
            raise UsageError(f"{field} expects an integer, got {raw!r}.") from exc
    if kind == "float":
        try:
            return float(raw)
        except ValueError as exc:
            raise UsageError(f"{field} expects a number, got {raw!r}.") from exc
    if kind == "list":
        return [part.strip() for part in raw.split(",") if part.strip()]
    if kind == "json":
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise UsageError(f"{field} expects a JSON value, got {raw!r}.") from exc
    return raw


def parse_config_overrides(fields: dict[str, str], pairs: list[str]) -> dict[str, object]:
    """Parse repeated KEY=VALUE strings into a coerced, validated dict."""
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise UsageError(f"--config expects KEY=VALUE, got {pair!r}.")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if key not in fields:
            valid = ", ".join(sorted(fields))
            raise UsageError(f"Unknown config field {key!r}. Valid fields: {valid}.")
        out[key] = coerce_value(key, raw)
    return out


def load_config_file(path: str | Path, fields: dict[str, str]) -> dict[str, object]:
    """Load a JSON config file and validate its keys against `fields`."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise UsageError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UsageError(f"Config file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError("Config file must contain a JSON object.")
    unknown = [k for k in data if k not in fields]
    if unknown:
        valid = ", ".join(sorted(fields))
        raise UsageError(f"Unknown config field(s) {unknown}. Valid fields: {valid}.")
    return data


def _merge(
    fields: dict[str, str],
    flags: dict[str, object],
    overrides: list[str],
    config_file: str | None,
) -> dict[str, object]:
    data: dict[str, object] = {}
    if config_file:
        data.update(load_config_file(config_file, fields))
    data.update(parse_config_overrides(fields, overrides))
    data.update({k: v for k, v in flags.items() if v is not None})
    return data


def build_transcription_config(
    *, flags: dict[str, object], overrides: list[str], config_file: str | None
) -> aai.TranscriptionConfig:
    merged = _merge(TRANSCRIBE_FIELDS, flags, overrides, config_file)
    try:
        return aai.TranscriptionConfig(**merged)
    except UsageError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface SDK validation as a usage error
        raise UsageError(f"Invalid transcription config: {exc}") from exc


def build_streaming_params(
    *, flags: dict[str, object], overrides: list[str], config_file: str | None
) -> StreamingParameters:
    merged = _merge(STREAM_FIELDS, flags, overrides, config_file)
    try:
        return StreamingParameters(**merged)
    except UsageError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise UsageError(f"Invalid streaming config: {exc}") from exc
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_config_builder.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/config_builder.py tests/test_config_builder.py
git commit -m "feat(config): add config_builder for SDK option merge/coercion"
```

---

## Task 2: config_builder — flag-normalization helpers + streaming build test

**Files:**
- Modify: `assemblyai_cli/config_builder.py`
- Test: `tests/test_config_builder.py:append`

These helpers turn shell-friendly flag shapes into SDK field values: CSV lists, `NAME:VALUE` headers, a custom-spelling JSON file, and a translation Speech-Understanding payload.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_builder.py`:

```python
def test_split_csv():
    assert cb.split_csv("a, b ,c") == ["a", "b", "c"]
    assert cb.split_csv(None) is None
    assert cb.split_csv("") is None


def test_parse_auth_header():
    assert cb.parse_auth_header("Authorization:Bearer x") == ("Authorization", "Bearer x")
    assert cb.parse_auth_header(None) is None
    with pytest.raises(UsageError):
        cb.parse_auth_header("no-colon")


def test_load_custom_spelling(tmp_path):
    p = tmp_path / "spell.json"
    p.write_text('{"AssemblyAI": ["assembly ai", "assemblyai"]}')
    assert cb.load_custom_spelling(str(p)) == {"AssemblyAI": ["assembly ai", "assemblyai"]}


def test_translation_request_shape():
    su = cb.translation_request(["es", "fr"])
    # target languages must be reachable from the payload regardless of dict/obj form.
    assert "es" in json.dumps(su, default=lambda o: getattr(o, "__dict__", str(o)))


def test_build_streaming_params_minimal():
    sp = cb.build_streaming_params(
        flags={"sample_rate": 16000, "speech_model": "universal_streaming_multilingual"},
        overrides=["max_turn_silence=400"],
        config_file=None,
    )
    assert sp.sample_rate == 16000
    assert sp.max_turn_silence == 400
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_config_builder.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'split_csv'`.

- [ ] **Step 3: Add the helpers**

Append to `assemblyai_cli/config_builder.py`:

```python
def split_csv(value: str | None) -> list[str] | None:
    """Split a comma-separated flag value into a list, or None if empty."""
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


def parse_auth_header(value: str | None) -> tuple[str, str] | None:
    """Parse a `NAME:VALUE` webhook auth header flag."""
    if value is None:
        return None
    if ":" not in value:
        raise UsageError("--webhook-auth-header expects NAME:VALUE.")
    name, header_value = value.split(":", 1)
    return name.strip(), header_value.strip()


def load_custom_spelling(path: str) -> dict[str, object]:
    """Load a custom-spelling JSON map (e.g. {"AssemblyAI": ["assembly ai"]})."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise UsageError(f"Custom spelling file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UsageError(f"Custom spelling file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise UsageError("Custom spelling file must contain a JSON object.")
    return data


def translation_request(languages: list[str]) -> dict[str, object]:
    """Build a Speech-Understanding translation payload for `speech_understanding`."""
    return {"request": {"translation": {"target_languages": list(languages)}}}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_config_builder.py -q`
Expected: PASS (13 tests).

If `test_build_streaming_params_minimal` fails because the SDK rejects a string `speech_model`, change `build_streaming_params` to coerce it: before constructing, do
`from assemblyai.streaming.v3 import SpeechModel` and `if isinstance(merged.get("speech_model"), str): merged["speech_model"] = SpeechModel(merged["speech_model"])`, then re-run. (The enum is keyed by value, e.g. `SpeechModel("universal_streaming_multilingual")`.)

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/config_builder.py tests/test_config_builder.py
git commit -m "feat(config): add flag-normalization helpers + streaming build"
```

---

## Task 3: client.py — accept prebuilt config objects

**Files:**
- Modify: `assemblyai_cli/client.py:63-78` (transcribe), `assemblyai_cli/client.py:97-139` (stream_audio)
- Test: `tests/test_client.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_client.py`:

```python
def test_transcribe_passes_prebuilt_config(monkeypatch):
    import assemblyai as aai

    from assemblyai_cli import client

    captured = {}

    class FakeTranscriber:
        def transcribe(self, audio, config=None):
            captured["audio"] = audio
            captured["config"] = config
            t = MagicMock()
            t.status = aai.TranscriptStatus.completed
            return t

    monkeypatch.setattr(aai, "Transcriber", lambda: FakeTranscriber())
    cfg = aai.TranscriptionConfig(speaker_labels=True)
    client.transcribe("sk", "audio.mp3", config=cfg)
    assert captured["audio"] == "audio.mp3"
    assert captured["config"] is cfg


def test_stream_audio_accepts_params(monkeypatch):
    from assemblyai.streaming.v3 import SpeechModel, StreamingParameters

    from assemblyai_cli import client

    captured = {}

    class FakeSC:
        def __init__(self, *a, **k):
            pass

        def on(self, *a, **k):
            pass

        def connect(self, params):
            captured["params"] = params

        def stream(self, source):
            pass

        def disconnect(self, terminate=True):
            pass

    monkeypatch.setattr("assemblyai_cli.client.StreamingClient", FakeSC)
    params = StreamingParameters(
        sample_rate=16000, speech_model=SpeechModel.universal_streaming_multilingual
    )
    client.stream_audio("sk", iter([b""]), params=params)
    assert captured["params"] is params
```

(MagicMock is already imported at the top of `tests/test_client.py`; if not, add `from unittest.mock import MagicMock`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_client.py::test_transcribe_passes_prebuilt_config tests/test_client.py::test_stream_audio_accepts_params -q`
Expected: FAIL — `TypeError: transcribe() got an unexpected keyword argument 'config'`.

- [ ] **Step 3: Update `client.transcribe` and `client.stream_audio`**

Replace `transcribe` (`assemblyai_cli/client.py:63-78`) with:

```python
def transcribe(api_key: str, audio: str, *, config: aai.TranscriptionConfig) -> aai.Transcript:
    _configure(api_key)
    try:
        transcript = aai.Transcriber().transcribe(audio, config=config)
    except APIError:
        raise
    except Exception as exc:
        if is_auth_failure(exc):
            raise auth_failure() from exc
        raise APIError(f"Transcription request failed: {exc}") from exc
    if transcript.status == aai.TranscriptStatus.error:
        raise APIError(transcript.error or "Transcription failed.", transcript_id=transcript.id)
    return transcript
```

Replace the `stream_audio` signature and the `sc.connect(...)` call (`assemblyai_cli/client.py:97-139`). Change the signature header to:

```python
def stream_audio(
    api_key: str,
    source: Iterable[bytes],
    *,
    params: StreamingParameters,
    on_begin: Callable[[Any], Any] | None = None,
    on_turn: Callable[[Any], Any] | None = None,
    on_termination: Callable[[Any], Any] | None = None,
) -> None:
    """Stream `source` (an iterable of PCM bytes) through the v3 realtime API.

    Forwards Begin/Turn/Termination events to the callbacks; raises APIError on a stream error.
    `params` is a fully-built StreamingParameters (sample_rate/speech_model/etc).
    """
```

And replace the `sc.connect(StreamingParameters(...))` block with:

```python
    try:
        sc.connect(params)
    except CLIError:
        raise
    except Exception as exc:
        if is_auth_failure(exc):
            raise auth_failure() from exc
        raise APIError(f"Could not start streaming session: {exc}") from exc
```

Remove the now-unused `SpeechModel` import only if nothing else references it; leave `StreamingParameters` imported (it's the parameter type).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_client.py -q`
Expected: PASS. (Existing client tests that call the old signature are updated in this same step — search `tests/test_client.py` for `client.transcribe(` / `client.stream_audio(` and update those calls to pass `config=`/`params=`.)

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/client.py tests/test_client.py
git commit -m "refactor(client): accept prebuilt TranscriptionConfig/StreamingParameters"
```

---

## Task 4: transcribe_render — analysis section renderers

**Files:**
- Create: `assemblyai_cli/transcribe_render.py`
- Test: `tests/test_transcribe_render.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transcribe_render.py`:

```python
from types import SimpleNamespace

from rich.console import Console

from assemblyai_cli import transcribe_render as tr


def _render(transcript) -> str:
    console = Console(width=80, force_terminal=False)
    with console.capture() as cap:
        tr.render_transcript_result(transcript, console)
    return cap.get()


def test_renders_text_only_when_no_analysis():
    out = _render(SimpleNamespace(text="hello world"))
    assert "hello world" in out
    assert "Summary" not in out


def test_renders_summary_and_chapters():
    transcript = SimpleNamespace(
        text="t",
        summary="A short summary.",
        chapters=[SimpleNamespace(start=0, end=133000, headline="Intro", gist="i", summary="s")],
    )
    out = _render(transcript)
    assert "Summary:" in out
    assert "A short summary." in out
    assert "Chapters:" in out
    assert "Intro" in out
    assert "00:00" in out and "02:13" in out  # 133000ms -> 02:13


def test_renders_sentiment_aggregate():
    transcript = SimpleNamespace(
        text="t",
        sentiment_analysis=[
            SimpleNamespace(text="a", sentiment=SimpleNamespace(value="POSITIVE")),
            SimpleNamespace(text="b", sentiment=SimpleNamespace(value="POSITIVE")),
            SimpleNamespace(text="c", sentiment=SimpleNamespace(value="NEGATIVE")),
        ],
    )
    out = _render(transcript)
    assert "Sentiment:" in out
    assert "positive" in out.lower()


def test_renders_entities_topics_content_safety_highlights():
    transcript = SimpleNamespace(
        text="t",
        entities=[SimpleNamespace(entity_type=SimpleNamespace(value="person_name"), text="Ada")],
        iab_categories=SimpleNamespace(summary={"Technology": 0.91}),
        content_safety=SimpleNamespace(summary={"profanity": 0.4}),
        auto_highlights=SimpleNamespace(
            results=[SimpleNamespace(text="key phrase", count=3, rank=0.9)]
        ),
    )
    out = _render(transcript)
    assert "Entities:" in out and "Ada" in out
    assert "Topics:" in out and "Technology" in out
    assert "Content Safety:" in out and "profanity" in out
    assert "Highlights:" in out and "key phrase" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_transcribe_render.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'assemblyai_cli.transcribe_render'`.

- [ ] **Step 3: Write `transcribe_render.py`**

Create `assemblyai_cli/transcribe_render.py`:

```python
from __future__ import annotations

from collections import Counter

from rich.console import Console


def _fmt_ms(ms: int) -> str:
    total = int(ms) // 1000
    return f"{total // 60:02d}:{total % 60:02d}"


def _enum_value(obj: object) -> str:
    return str(getattr(obj, "value", obj))


def render_transcript_result(transcript: object, console: Console) -> None:
    """Print the transcript text, then a section per analysis feature present."""
    console.print(getattr(transcript, "text", "") or "")
    _render_summary(transcript, console)
    _render_chapters(transcript, console)
    _render_highlights(transcript, console)
    _render_sentiment(transcript, console)
    _render_entities(transcript, console)
    _render_topics(transcript, console)
    _render_content_safety(transcript, console)


def _render_summary(transcript: object, console: Console) -> None:
    summary = getattr(transcript, "summary", None)
    if summary:
        console.print("\n[bold]Summary:[/bold]")
        console.print(str(summary))


def _render_chapters(transcript: object, console: Console) -> None:
    chapters = getattr(transcript, "chapters", None)
    if not chapters:
        return
    console.print("\n[bold]Chapters:[/bold]")
    for ch in chapters:
        span = f"{_fmt_ms(ch.start)}–{_fmt_ms(ch.end)}"
        console.print(f"  {span}  {ch.headline}")


def _render_highlights(transcript: object, console: Console) -> None:
    highlights = getattr(transcript, "auto_highlights", None)
    results = getattr(highlights, "results", None) if highlights else None
    if not results:
        return
    console.print("\n[bold]Highlights:[/bold]")
    for h in results:
        console.print(f"  ({h.count}×) {h.text}")


def _render_sentiment(transcript: object, console: Console) -> None:
    results = getattr(transcript, "sentiment_analysis", None)
    if not results:
        return
    counts = Counter(_enum_value(r.sentiment).lower() for r in results)
    total = sum(counts.values()) or 1
    parts = [f"{pct * 100 // total}% {label}" for label, pct in counts.items()]
    console.print("\n[bold]Sentiment:[/bold] " + ", ".join(parts))


def _render_entities(transcript: object, console: Console) -> None:
    entities = getattr(transcript, "entities", None)
    if not entities:
        return
    console.print("\n[bold]Entities:[/bold]")
    for ent in entities:
        console.print(f"  {_enum_value(ent.entity_type)}: {ent.text}")


def _render_topics(transcript: object, console: Console) -> None:
    iab = getattr(transcript, "iab_categories", None)
    summary = getattr(iab, "summary", None) if iab else None
    if not summary:
        return
    console.print("\n[bold]Topics:[/bold]")
    for label, relevance in sorted(summary.items(), key=lambda kv: kv[1], reverse=True):
        console.print(f"  {label} ({float(relevance):.2f})")


def _render_content_safety(transcript: object, console: Console) -> None:
    safety = getattr(transcript, "content_safety", None)
    summary = getattr(safety, "summary", None) if safety else None
    if not summary:
        return
    console.print("\n[bold]Content Safety:[/bold]")
    for label, confidence in sorted(summary.items(), key=lambda kv: kv[1], reverse=True):
        console.print(f"  {_enum_value(label)} ({float(confidence):.2f})")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_transcribe_render.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/transcribe_render.py tests/test_transcribe_render.py
git commit -m "feat(render): add transcribe analysis section renderers"
```

---

## Task 5: wire the `transcribe` command flags

**Files:**
- Modify: `assemblyai_cli/commands/transcribe.py`
- Test: `tests/test_transcribe.py` (modify + append)

- [ ] **Step 1: Update existing tests and add new ones**

In `tests/test_transcribe.py`, the existing `test_transcribe_passes_speaker_labels` asserts `tx.call_args.kwargs["speaker_labels"]`. The client call now passes a `config=` object. Replace that test body with:

```python
def test_transcribe_passes_speaker_labels():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        runner.invoke(app, ["transcribe", "audio.mp3", "--speaker-labels"])
    assert tx.call_args.kwargs["config"].speaker_labels is True
```

Update `test_transcribe_prompt_biases_speech_model` similarly — its final assertion becomes:

```python
    assert tx.call_args.kwargs["config"].prompt == "expect medical terms"
```

Then append new tests:

```python
def test_transcribe_maps_analysis_flags():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        runner.invoke(
            app,
            [
                "transcribe",
                "audio.mp3",
                "--summarization",
                "--summary-type",
                "bullets",
                "--sentiment-analysis",
                "--topic-detection",
            ],
        )
    cfg = tx.call_args.kwargs["config"]
    assert cfg.raw.summarization is True
    assert cfg.raw.summary_type == "bullets"
    assert cfg.raw.sentiment_analysis is True
    assert cfg.raw.iab_categories is True


def test_transcribe_redact_pii_policy_csv():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        runner.invoke(
            app,
            [
                "transcribe",
                "audio.mp3",
                "--redact-pii",
                "--redact-pii-policy",
                "person_name,phone_number",
            ],
        )
    cfg = tx.call_args.kwargs["config"]
    assert cfg.raw.redact_pii is True
    assert [_enum_or_str(p) for p in cfg.raw.redact_pii_policies] == [
        "person_name",
        "phone_number",
    ]


def _enum_or_str(value):
    return getattr(value, "value", value)


def test_transcribe_config_escape_hatch():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        runner.invoke(app, ["transcribe", "audio.mp3", "--config", "speech_threshold=0.5"])
    assert tx.call_args.kwargs["config"].raw.speech_threshold == 0.5


def test_transcribe_unknown_config_field_exits_2():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--config", "bogus=1"])
    assert result.exit_code == 2
    assert "bogus" in result.output


def test_transcribe_renders_summary_human(monkeypatch):
    _auth()
    monkeypatch.setattr("assemblyai_cli.output.resolve_json", lambda *, explicit: False)
    t = _fake_transcript()
    t.summary = "three bullet summary"
    t.chapters = []
    with patch("assemblyai_cli.commands.transcribe.client.transcribe", return_value=t):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--summarization"])
    assert result.exit_code == 0
    assert "Summary:" in result.output
    assert "three bullet summary" in result.output
```

Note: `_fake_transcript()` returns a `MagicMock`, so attributes like `.summary`/`.chapters` exist as truthy MagicMocks by default. For the render-path test above we set the ones we assert on explicitly. Add a line to `_fake_transcript()` so unanalyzed runs don't render spurious sections — set the analysis attributes to falsy by default:

```python
def _fake_transcript():
    t = MagicMock()
    t.id = "t_1"
    t.text = "hello world"
    t.status = "completed"
    t.json_response = {"id": "t_1", "text": "hello world", "status": "completed"}
    for attr in (
        "summary", "chapters", "auto_highlights", "sentiment_analysis",
        "entities", "iab_categories", "content_safety",
    ):
        setattr(t, attr, None)
    return t
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_transcribe.py -q`
Expected: FAIL — new flags are unknown (`No such option: --summarization`) and `config=` kwarg not yet passed.

- [ ] **Step 3: Rewrite `commands/transcribe.py`**

Replace the whole file with:

```python
from __future__ import annotations

import json

import typer

from assemblyai_cli import client, config, config_builder, llm, output, transcribe_render
from assemblyai_cli.context import AppState, run_command


app = typer.Typer()


@app.command()
def transcribe(
    ctx: typer.Context,
    source: str = typer.Argument(None, help="Audio file path or public URL."),
    sample: bool = typer.Option(False, "--sample", help="Use the hosted wildfires.mp3 sample."),
    # model & language
    speech_model: str = typer.Option(None, "--speech-model", help="best, nano, slam-1, universal."),
    language_code: str = typer.Option(None, "--language-code", help="Force a language (e.g. en_us)."),
    language_detection: bool = typer.Option(
        None, "--language-detection", help="Auto-detect the spoken language."
    ),
    keyterms_prompt: list[str] = typer.Option(
        None, "--keyterms-prompt", help="Boost a key term (repeatable)."
    ),
    temperature: float = typer.Option(None, "--temperature", help="Speech model temperature."),
    prompt: str = typer.Option(None, "--prompt", help="Bias the speech model (u3-pro)."),
    # formatting
    punctuate: bool = typer.Option(None, "--punctuate/--no-punctuate", help="Add punctuation."),
    format_text: bool = typer.Option(None, "--format-text/--no-format-text", help="Format text."),
    disfluencies: bool = typer.Option(None, "--disfluencies", help="Keep filler words."),
    # speakers & channels
    speaker_labels: bool = typer.Option(False, "--speaker-labels", help="Enable diarization."),
    speakers_expected: int = typer.Option(None, "--speakers-expected", help="Hint speaker count."),
    multichannel: bool = typer.Option(None, "--multichannel", help="Transcribe each channel."),
    # guardrails
    redact_pii: bool = typer.Option(None, "--redact-pii", help="Redact PII from the transcript."),
    redact_pii_policy: str = typer.Option(
        None, "--redact-pii-policy", help="Comma-separated PII policies (e.g. person_name,...)."
    ),
    redact_pii_sub: str = typer.Option(
        None, "--redact-pii-sub", help="Substitution: hash or entity_name."
    ),
    redact_pii_audio: bool = typer.Option(None, "--redact-pii-audio", help="Also redact audio."),
    filter_profanity: bool = typer.Option(None, "--filter-profanity", help="Mask profanity."),
    content_safety: bool = typer.Option(None, "--content-safety", help="Detect sensitive content."),
    content_safety_confidence: int = typer.Option(
        None, "--content-safety-confidence", help="Confidence threshold 25-100."
    ),
    speech_threshold: float = typer.Option(
        None, "--speech-threshold", help="Minimum speech proportion 0-1."
    ),
    # analysis
    summarization: bool = typer.Option(None, "--summarization", help="Summarize the transcript."),
    summary_model: str = typer.Option(None, "--summary-model", help="informative/conversational/catchy."),
    summary_type: str = typer.Option(None, "--summary-type", help="bullets/gist/headline/paragraph."),
    auto_chapters: bool = typer.Option(None, "--auto-chapters", help="Generate chapters."),
    sentiment_analysis: bool = typer.Option(None, "--sentiment-analysis", help="Analyze sentiment."),
    entity_detection: bool = typer.Option(None, "--entity-detection", help="Detect entities."),
    auto_highlights: bool = typer.Option(None, "--auto-highlights", help="Detect key phrases."),
    topic_detection: bool = typer.Option(None, "--topic-detection", help="Detect IAB topics."),
    # customization
    word_boost: list[str] = typer.Option(None, "--word-boost", help="Boost a word (repeatable)."),
    custom_spelling_file: str = typer.Option(
        None, "--custom-spelling-file", help="JSON map of custom spellings."
    ),
    audio_start: int = typer.Option(None, "--audio-start", help="Start offset in ms."),
    audio_end: int = typer.Option(None, "--audio-end", help="End offset in ms."),
    # webhooks
    webhook_url: str = typer.Option(None, "--webhook-url", help="Webhook URL for completion."),
    webhook_auth_header: str = typer.Option(
        None, "--webhook-auth-header", help="Webhook auth header as NAME:VALUE."
    ),
    # speech understanding
    translate_to: list[str] = typer.Option(
        None, "--translate-to", help="Translate transcript to a language (repeatable)."
    ),
    # escape hatch
    config_kv: list[str] = typer.Option(
        None, "--config", help="Set any TranscriptionConfig field as KEY=VALUE (repeatable)."
    ),
    config_file: str = typer.Option(None, "--config-file", help="JSON file of config fields."),
    # llm gateway transform (existing)
    llm_gateway_prompt: str = typer.Option(
        None, "--llm-gateway-prompt", help="Transform the finished transcript through LLM Gateway."
    ),
    model: str = typer.Option(llm.DEFAULT_MODEL, "--model", help="LLM Gateway model."),
    max_tokens: int = typer.Option(llm.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Transcribe an audio file or URL with the full TranscriptionConfig surface.

    Curated flags cover common features; --config KEY=VALUE and --config-file reach
    every other field. Analysis results (summary, chapters, sentiment, ...) render
    automatically in human mode.
    """

    def body(state: AppState, json_mode: bool) -> None:
        flags: dict[str, object] = {
            "speech_model": speech_model,
            "language_code": language_code,
            "language_detection": language_detection,
            "keyterms_prompt": list(keyterms_prompt) if keyterms_prompt else None,
            "temperature": temperature,
            "prompt": prompt,
            "punctuate": punctuate,
            "format_text": format_text,
            "disfluencies": disfluencies,
            "speaker_labels": speaker_labels or None,
            "speakers_expected": speakers_expected,
            "multichannel": multichannel,
            "redact_pii": redact_pii,
            "redact_pii_policies": config_builder.split_csv(redact_pii_policy),
            "redact_pii_sub": redact_pii_sub,
            "redact_pii_audio": redact_pii_audio,
            "filter_profanity": filter_profanity,
            "content_safety": content_safety,
            "content_safety_confidence": content_safety_confidence,
            "speech_threshold": speech_threshold,
            "summarization": summarization,
            "summary_model": summary_model,
            "summary_type": summary_type,
            "auto_chapters": auto_chapters,
            "sentiment_analysis": sentiment_analysis,
            "entity_detection": entity_detection,
            "auto_highlights": auto_highlights,
            "iab_categories": topic_detection,
            "word_boost": list(word_boost) if word_boost else None,
            "custom_spelling": (
                config_builder.load_custom_spelling(custom_spelling_file)
                if custom_spelling_file
                else None
            ),
            "audio_start_from": audio_start,
            "audio_end_at": audio_end,
            "webhook_url": webhook_url,
            "speech_understanding": (
                config_builder.translation_request(list(translate_to)) if translate_to else None
            ),
        }
        header = config_builder.parse_auth_header(webhook_auth_header)
        if header is not None:
            flags["webhook_auth_header_name"] = header[0]
            flags["webhook_auth_header_value"] = header[1]

        tc = config_builder.build_transcription_config(
            flags=flags, overrides=list(config_kv or []), config_file=config_file
        )

        audio = client.resolve_audio_source(source, sample=sample)
        api_key = config.resolve_api_key(profile=state.profile)
        transcript = client.transcribe(api_key, audio, config=tc)

        if llm_gateway_prompt:
            transformed = llm.transform_transcript(
                api_key,
                prompt=llm_gateway_prompt,
                model=model,
                transcript_id=transcript.id,
                max_tokens=max_tokens,
            )
            output.emit(
                {
                    "id": transcript.id,
                    "status": client.status_str(transcript),
                    "text": transcript.text,
                    "transform": {
                        "model": model,
                        "prompt": llm_gateway_prompt,
                        "output": transformed,
                    },
                },
                lambda d: str(d["transform"]["output"]),
                json_mode=json_mode,
            )
            return

        if json_mode:
            payload = getattr(transcript, "json_response", None) or {
                "id": transcript.id,
                "status": client.status_str(transcript),
                "text": transcript.text,
            }
            print(json.dumps(payload, default=str))
        else:
            transcribe_render.render_transcript_result(transcript, output.console)

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_transcribe.py tests/test_transcribe_render.py -q`
Expected: PASS.

If `--translate-to` causes a build failure (SDK rejects a dict for `speech_understanding`), construct the objects instead: in `config_builder.translation_request`, return
`from assemblyai.types import SpeechUnderstandingRequest, SpeechUnderstandingFeatureRequests, TranslationRequest` and build `SpeechUnderstandingRequest(request=SpeechUnderstandingFeatureRequests(translation=TranslationRequest(target_languages=list(languages))))`. Re-run.

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/commands/transcribe.py tests/test_transcribe.py
git commit -m "feat(transcribe): expose full TranscriptionConfig via flags + escape hatch"
```

---

## Task 6: wire the `stream` command flags

**Files:**
- Modify: `assemblyai_cli/commands/stream.py`
- Test: `tests/test_stream_command.py` (modify + append)

- [ ] **Step 1: Update existing tests and add new ones**

In `tests/test_stream_command.py`, find any call asserting `client.stream_audio` kwargs `sample_rate=`/`prompt=` and update them to read from the passed `params=`. Then append:

```python
def test_stream_maps_turn_detection_flags(monkeypatch):
    import assemblyai_cli.commands.stream as stream_cmd
    from assemblyai_cli import config

    config.set_api_key("default", "sk_live")
    captured = {}

    def fake_stream_audio(api_key, source, *, params, **kw):
        captured["params"] = params

    monkeypatch.setattr(stream_cmd.client, "stream_audio", fake_stream_audio)
    from typer.testing import CliRunner

    from assemblyai_cli.main import app

    CliRunner().invoke(
        app,
        [
            "stream",
            "--sample",
            "--max-turn-silence",
            "400",
            "--filter-profanity",
            "--speaker-labels",
        ],
    )
    params = captured["params"]
    assert params.max_turn_silence == 400
    assert params.filter_profanity is True
    assert params.speaker_labels is True


def test_stream_config_escape_hatch(monkeypatch):
    import assemblyai_cli.commands.stream as stream_cmd
    from assemblyai_cli import config

    config.set_api_key("default", "sk_live")
    captured = {}
    monkeypatch.setattr(
        stream_cmd.client,
        "stream_audio",
        lambda api_key, source, *, params, **kw: captured.update(params=params),
    )
    from typer.testing import CliRunner

    from assemblyai_cli.main import app

    CliRunner().invoke(app, ["stream", "--sample", "--config", "vad_threshold=0.7"])
    assert captured["params"].vad_threshold == 0.7
```

(If `test_stream_command.py` already imports `CliRunner`/`app`/`config` at module scope, drop the inline imports above and reuse them.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_stream_command.py -q`
Expected: FAIL — `No such option: --max-turn-silence`.

- [ ] **Step 3: Rewrite `commands/stream.py`**

Replace the whole file with:

```python
from __future__ import annotations

import typer
from assemblyai.streaming.v3 import SpeechModel

from assemblyai_cli import client, config, config_builder, llm
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import UsageError
from assemblyai_cli.microphone import MicrophoneSource
from assemblyai_cli.streaming.render import StreamRenderer
from assemblyai_cli.streaming.sources import TARGET_RATE, FileSource


app = typer.Typer()

DEFAULT_SPEECH_MODEL = SpeechModel.universal_streaming_multilingual.value


@app.command()
def stream(
    ctx: typer.Context,
    source: str = typer.Argument(
        None, help="Audio file path or URL to stream. Omit to use the microphone."
    ),
    sample: bool = typer.Option(False, "--sample", help="Stream the hosted wildfires.mp3 sample."),
    sample_rate: int = typer.Option(TARGET_RATE, "--sample-rate", help="Microphone sample rate Hz."),
    device: int | None = typer.Option(None, "--device", help="Microphone device index."),
    # model & input
    speech_model: str = typer.Option(
        DEFAULT_SPEECH_MODEL, "--speech-model", help="Streaming speech model."
    ),
    encoding: str = typer.Option(None, "--encoding", help="pcm_s16le or pcm_mulaw."),
    language_detection: bool = typer.Option(
        None, "--language-detection", help="Auto-detect the spoken language."
    ),
    domain: str = typer.Option(None, "--domain", help="Domain preset (e.g. medical)."),
    # turn detection
    end_of_turn_confidence_threshold: float = typer.Option(
        None, "--end-of-turn-confidence-threshold", help="0-1 end-of-turn confidence."
    ),
    min_turn_silence: int = typer.Option(None, "--min-turn-silence", help="Min turn silence (ms)."),
    max_turn_silence: int = typer.Option(None, "--max-turn-silence", help="Max turn silence (ms)."),
    vad_threshold: float = typer.Option(None, "--vad-threshold", help="Voice-activity threshold."),
    format_turns: bool = typer.Option(
        None, "--format-turns/--no-format-turns", help="Punctuate/format finalized turns."
    ),
    include_partial_turns: bool = typer.Option(
        None, "--include-partial-turns", help="Emit partial turns."
    ),
    # features
    keyterms_prompt: list[str] = typer.Option(
        None, "--keyterms-prompt", help="Boost a key term (repeatable)."
    ),
    filter_profanity: bool = typer.Option(None, "--filter-profanity", help="Mask profanity."),
    speaker_labels: bool = typer.Option(None, "--speaker-labels", help="Label speakers."),
    max_speakers: int = typer.Option(None, "--max-speakers", help="Max speakers."),
    voice_focus: str = typer.Option(None, "--voice-focus", help="near_field or far_field."),
    voice_focus_threshold: float = typer.Option(
        None, "--voice-focus-threshold", help="Voice-focus threshold."
    ),
    redact_pii: bool = typer.Option(None, "--redact-pii", help="Redact PII from turns."),
    redact_pii_policy: str = typer.Option(
        None, "--redact-pii-policy", help="Comma-separated PII policies."
    ),
    redact_pii_sub: str = typer.Option(None, "--redact-pii-sub", help="hash or entity_name."),
    inactivity_timeout: int = typer.Option(
        None, "--inactivity-timeout", help="Auto-close after N seconds idle."
    ),
    webhook_url: str = typer.Option(None, "--webhook-url", help="Webhook URL."),
    webhook_auth_header: str = typer.Option(
        None, "--webhook-auth-header", help="Webhook auth header as NAME:VALUE."
    ),
    # escape hatch
    config_kv: list[str] = typer.Option(
        None, "--config", help="Set any StreamingParameters field as KEY=VALUE (repeatable)."
    ),
    config_file: str = typer.Option(None, "--config-file", help="JSON file of streaming fields."),
    # existing
    prompt: str = typer.Option(None, "--prompt", help="Bias the speech model (u3-pro)."),
    llm_gateway_prompt: str = typer.Option(
        None, "--llm-gateway-prompt", help="After streaming, transform the transcript via LLM Gateway."
    ),
    model: str = typer.Option(llm.DEFAULT_MODEL, "--model", help="LLM Gateway model."),
    max_tokens: int = typer.Option(llm.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens."),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
) -> None:
    """Transcribe live audio in real time with the full StreamingParameters surface."""

    def body(state: AppState, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        from_file = bool(source) or sample
        if from_file and (sample_rate != TARGET_RATE or device is not None):
            raise UsageError("--sample-rate and --device apply only to microphone input.")
        audio: FileSource | MicrophoneSource
        if from_file:
            audio = FileSource(client.resolve_audio_source(source, sample=sample))
            rate = audio.sample_rate
        else:
            audio = MicrophoneSource(sample_rate=sample_rate, device=device)
            rate = sample_rate

        flags: dict[str, object] = {
            "sample_rate": rate,
            "speech_model": speech_model,
            "format_turns": format_turns if format_turns is not None else True,
            "encoding": encoding,
            "language_detection": language_detection,
            "domain": domain,
            "end_of_turn_confidence_threshold": end_of_turn_confidence_threshold,
            "min_turn_silence": min_turn_silence,
            "max_turn_silence": max_turn_silence,
            "vad_threshold": vad_threshold,
            "include_partial_turns": include_partial_turns,
            "keyterms_prompt": list(keyterms_prompt) if keyterms_prompt else None,
            "filter_profanity": filter_profanity,
            "speaker_labels": speaker_labels,
            "max_speakers": max_speakers,
            "voice_focus": voice_focus,
            "voice_focus_threshold": voice_focus_threshold,
            "redact_pii": redact_pii,
            "redact_pii_policies": config_builder.split_csv(redact_pii_policy),
            "redact_pii_sub": redact_pii_sub,
            "inactivity_timeout": inactivity_timeout,
            "webhook_url": webhook_url,
            "prompt": prompt,
        }
        header = config_builder.parse_auth_header(webhook_auth_header)
        if header is not None:
            flags["webhook_auth_header_name"] = header[0]
            flags["webhook_auth_header_value"] = header[1]

        params = config_builder.build_streaming_params(
            flags=flags, overrides=list(config_kv or []), config_file=config_file
        )

        renderer = StreamRenderer(json_mode=json_mode)
        turns: list[str] = []

        def on_turn(event: object) -> None:
            renderer.turn(event)
            if llm_gateway_prompt and getattr(event, "end_of_turn", False):
                text = getattr(event, "transcript", "") or ""
                if text:
                    turns.append(text)

        try:
            client.stream_audio(
                api_key,
                audio,
                params=params,
                on_begin=renderer.begin,
                on_turn=on_turn,
                on_termination=renderer.termination,
            )
        except KeyboardInterrupt:
            renderer.close()
            renderer.stopped()
        except BrokenPipeError:
            raise typer.Exit(code=0) from None
        finally:
            renderer.close()

        if llm_gateway_prompt and turns:
            transformed = llm.transform_transcript(
                api_key,
                prompt=llm_gateway_prompt,
                model=model,
                transcript_text=" ".join(turns),
                max_tokens=max_tokens,
            )
            renderer.llm(transformed)

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_stream_command.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/commands/stream.py tests/test_stream_command.py
git commit -m "feat(stream): expose full StreamingParameters via flags + escape hatch"
```

---

## Task 7: docs + samples

**Files:**
- Modify: `README.md`
- Modify: `assemblyai_cli/templates/` (the transcribe/stream sample templates, if present)

- [ ] **Step 1: Document the new options**

In `README.md`, under the `transcribe` and `stream` sections, add a subsection that lists the curated flag groups (Model & language / Formatting / Speakers / Guardrails / Analysis / Customization / Webhooks) and documents the escape hatch with an example:

````markdown
#### Advanced options

Every `TranscriptionConfig` / `StreamingParameters` field has a curated flag or is
reachable through the escape hatch:

```bash
aai transcribe call.mp3 \
  --speaker-labels --speakers-expected 2 \
  --redact-pii --redact-pii-policy person_name,phone_number \
  --summarization --summary-type bullets \
  --sentiment-analysis --auto-chapters \
  --config speech_threshold=0.5 \
  --config-file extra.json
```

`--config KEY=VALUE` (repeatable) and `--config-file FILE` (JSON object) accept any
SDK field by its exact name. Precedence: config file < `--config` < explicit flags.
````

- [ ] **Step 2: Verify the README examples are accurate**

Run: `aai transcribe --help` and confirm every flag named in the README example appears in the help output.
Expected: all referenced flags are present.

- [ ] **Step 3: Refresh sample templates (if they hard-code config)**

If `assemblyai_cli/templates/` contains a transcribe/stream starter that builds a
`TranscriptionConfig`, add one or two of the new options (e.g. `summarization=True`)
as a comment-documented example. If templates don't build config, skip.

- [ ] **Step 4: Commit**

```bash
git add README.md assemblyai_cli/templates
git commit -m "docs: document full SDK option flags and the --config escape hatch"
```

---

## Task 8: exhaustive tests — property coercion + per-flag coverage + e2e

**Files:**
- Modify: `tests/test_properties.py`
- Modify: `tests/test_config_builder.py`
- Modify: `tests/e2e/test_cli_e2e.py`

- [ ] **Step 1: Add a property test for coercion round-trips**

Append to `tests/test_properties.py`:

```python
from hypothesis import given
from hypothesis import strategies as st

from assemblyai_cli import config_builder as cb


@given(value=st.integers(min_value=0, max_value=10_000_000))
def test_int_coercion_roundtrips(value):
    assert cb.coerce_value("speakers_expected", str(value)) == value


@given(value=st.lists(st.text(alphabet="abcdefghijklmnop", min_size=1, max_size=6), max_size=5))
def test_list_coercion_roundtrips(value):
    raw = ",".join(value)
    assert cb.coerce_value("word_boost", raw) == [v for v in value if v]


@given(value=st.booleans())
def test_bool_coercion_roundtrips(value):
    assert cb.coerce_value("speaker_labels", str(value).lower()) is value
```

- [ ] **Step 2: Add a parametrized per-flag mapping test**

Append to `tests/test_config_builder.py`:

```python
import pytest as _pytest


@_pytest.mark.parametrize(
    "field,raw,expected",
    [
        ("punctuate", "false", False),
        ("multichannel", "true", True),
        ("audio_start_from", "1500", 1500),
        ("temperature", "0.2", 0.2),
        ("summary_type", "bullets", "bullets"),
        ("keyterms_prompt", "a,b", ["a", "b"]),
    ],
)
def test_transcribe_field_coercion_matrix(field, raw, expected):
    tc = cb.build_transcription_config(
        flags={}, overrides=[f"{field}={raw}"], config_file=None
    )
    assert getattr(tc.raw, field) == expected


@_pytest.mark.parametrize("field", sorted(cb.STREAM_FIELDS))
def test_every_stream_field_is_a_valid_param(field):
    # Each declared field must be a real StreamingParameters attribute.
    from assemblyai.streaming.v3 import StreamingParameters

    assert field in StreamingParameters.model_fields
```

- [ ] **Step 3: Run the new unit/property tests**

Run: `pytest tests/test_config_builder.py tests/test_properties.py -q`
Expected: PASS.

- [ ] **Step 4: Add a real-API e2e analysis run**

Open `tests/e2e/test_cli_e2e.py`, mirror the existing subprocess-invocation pattern used by the current transcribe e2e test, and add:

```python
def test_e2e_transcribe_analysis(real_api_key):
    # Mirrors the existing transcribe e2e: run the CLI as a subprocess with --json,
    # using --sample so no local audio is required.
    result = run_cli(  # use the same helper the other e2e tests use
        ["transcribe", "--sample", "--summarization", "--auto-chapters", "--json"],
        api_key=real_api_key,
    )
    assert result.returncode == 0
    import json as _json

    payload = _json.loads(result.stdout)
    # The full transcript object is returned; analysis fields are present.
    assert payload.get("summary") or payload.get("chapters")
```

If the existing e2e file uses a different helper name than `run_cli`, match it exactly (read the top of the file first). Keep the test guarded by the `real_api_key` fixture so it skips without a key.

- [ ] **Step 5: Run the full suite**

Run: `pytest -q` (e2e skips without `ASSEMBLYAI_API_KEY`).
Expected: PASS / SKIPPED for e2e.

- [ ] **Step 6: Run linters**

Run: `ruff check . && ruff format --check .`
Expected: clean. Fix any issues and re-run.

- [ ] **Step 7: Commit**

```bash
git add tests/test_properties.py tests/test_config_builder.py tests/e2e/test_cli_e2e.py
git commit -m "test: exhaustive coercion property tests + analysis e2e"
```

---

## Self-Review Notes

- **Spec coverage:** hybrid mapping (Tasks 1–2, 5–6), JSON-only config file (Task 1 `load_config_file`), precedence file<config<flags (Task 1 `_merge`, asserted), auto-rendered analysis (Task 4 + Task 5 human path), LeMUR excluded (not present), Speech-Understanding via `--config` with `--translate-to` curated (Task 5), exhaustive tests (Task 8). All covered.
- **Type consistency:** `build_transcription_config`/`build_streaming_params` keyword-only `(flags, overrides, config_file)`; `client.transcribe(..., config=)`, `client.stream_audio(..., params=)`; `render_transcript_result(transcript, console)` — used consistently across tasks.
- **Known follow-up:** `--translate-to` (`speech_understanding` dict) and streaming `speech_model` string coercion each have an inline fallback (Task 2 Step 4, Task 5 Step 4) in case the SDK requires constructed objects/enums rather than raw values — the failing-test loop surfaces this deterministically.
