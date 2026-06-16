"""Command + wiring tests for `assembly agent-cascade`.

Covers the argv -> options seam, the validation guards, _open_audio source
selection, and CascadeDeps.real's three live legs (all driven against fakes).
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import typer
from typer.testing import CliRunner

from aai_cli.agent.render import AgentRenderer
from aai_cli.agent_cascade import engine
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.agent_cascade.engine import CascadeDeps
from aai_cli.app.context import AppState
from aai_cli.commands.agent_cascade import _exec
from aai_cli.commands.agent_cascade._exec import AgentCascadeOptions, run_agent_cascade
from aai_cli.core import config
from aai_cli.core.errors import CLIError, UsageError
from aai_cli.main import app

runner = CliRunner()


_DEFAULTS = AgentCascadeOptions(
    source=None,
    sample=False,
    voice="jane",
    model="claude-haiku-4-5-20251001",
    system_prompt="be nice",
    system_prompt_file=None,
    greeting="hello",
    device=None,
    output_field=None,
)


def _opts(**overrides) -> AgentCascadeOptions:
    return dataclasses.replace(_DEFAULTS, **overrides)


# --- help / list-voices ------------------------------------------------------


def test_list_voices_human_lists_catalog():
    result = runner.invoke(app, ["agent-cascade", "--list-voices"])
    assert result.exit_code == 0
    assert "jane" in result.output
    assert "English:" in result.output


def test_list_voices_json_emits_array():
    result = runner.invoke(app, ["agent-cascade", "--list-voices", "--json"])
    assert result.exit_code == 0
    assert result.output.lstrip().startswith("[")
    assert '"jane"' in result.output


# --- validation guards -------------------------------------------------------


def test_options_are_frozen():
    attr = "voice"  # not a literal, so ruff's B010 leaves the setattr in place
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(_DEFAULTS, attr, "other")


def test_unknown_voice_is_a_usage_error():
    with pytest.raises(UsageError, match="Unknown voice"):
        run_agent_cascade(_opts(voice="nope"), AppState(), json_mode=False)


def test_missing_system_prompt_file_is_rejected_by_typer():
    # exists=True on the option makes Typer reject a nonexistent path before the body,
    # so the sandbox guard (the other exit-2 path) never runs. Asserting the guard's
    # message is absent kills the exists=True mutant without depending on the Rich error
    # text, which CI renders with ANSI + width ellipsis.
    result = runner.invoke(app, ["agent-cascade", "--system-prompt-file", "/no/such/file"])
    assert result.exit_code == 2
    assert "sandbox" not in result.output.lower()


def test_production_env_is_rejected_with_sandbox_hint():
    # Default env is production, which has no streaming-TTS host.
    result = runner.invoke(app, ["agent-cascade", "--voice", "jane"])
    assert result.exit_code == 2
    assert "only available in the sandbox" in result.output


def test_device_with_file_source_is_rejected(monkeypatch):
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    with pytest.raises(UsageError, match="--device applies only to microphone"):
        run_agent_cascade(_opts(source="clip.wav", device=2), AppState(), json_mode=False)


# --- system prompt resolution ------------------------------------------------


def test_resolve_system_prompt_prefers_file(tmp_path):
    path = tmp_path / "persona.txt"
    path.write_text("you are a pirate", encoding="utf-8")
    assert _exec._resolve_system_prompt("ignored", path) == "you are a pirate"


def test_resolve_system_prompt_without_file_passes_through():
    assert _exec._resolve_system_prompt("default persona", None) == "default persona"


def test_resolve_system_prompt_unreadable_file_errors(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(CLIError, match="Could not read --system-prompt-file") as exc:
        _exec._resolve_system_prompt("x", missing)
    assert exc.value.exit_code == 2
    assert exc.value.error_type == "file_not_found"
    assert "readable" in (exc.value.suggestion or "")


# --- _open_audio -------------------------------------------------------------


def _renderer() -> AgentRenderer:
    return AgentRenderer(json_mode=False, text_mode=False)


def test_open_audio_file_uses_nullplayer_and_source_rate(monkeypatch):
    fake_source = types.SimpleNamespace(sample_rate=16000)
    monkeypatch.setattr(_exec, "FileSource", lambda src: fake_source)
    monkeypatch.setattr(_exec.client, "resolve_audio_source", lambda source, sample: "clip.wav")
    audio, player, rate = _exec._open_audio(
        _renderer(), source="clip.wav", sample=False, device=None, from_file=True
    )
    assert audio is fake_source
    assert isinstance(player, _exec.NullPlayer)
    assert rate == 16000


def test_open_audio_mic_warns_and_uses_duplex_rate(monkeypatch):
    fake_duplex = types.SimpleNamespace(mic=object(), player=object())
    monkeypatch.setattr(_exec, "DuplexAudio", lambda **kwargs: fake_duplex)
    renderer = _renderer()
    notices: list[str] = []
    monkeypatch.setattr(renderer, "notice", notices.append)
    audio, player, rate = _exec._open_audio(
        renderer, source=None, sample=False, device=None, from_file=False
    )
    assert audio is fake_duplex.mic
    assert player is fake_duplex.player
    assert rate == _exec.SAMPLE_RATE
    assert any("headphones" in note for note in notices)


# --- run_agent_cascade wiring ----------------------------------------------


def test_run_wires_deps_and_invokes_cascade(monkeypatch):
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "test-key")
    fake_source = types.SimpleNamespace(sample_rate=16000)
    monkeypatch.setattr(_exec, "FileSource", lambda src: fake_source)
    monkeypatch.setattr(_exec.client, "resolve_audio_source", lambda source, sample: "clip.wav")
    captured = {}

    def fake_run_cascade(*, renderer, player, config, deps):
        captured["config"] = config
        captured["deps"] = deps

    monkeypatch.setattr(_exec.engine, "run_cascade", fake_run_cascade)
    run_agent_cascade(
        _opts(source="clip.wav", voice="michael", greeting="hi there"), AppState(), json_mode=False
    )
    # File-driven runs drop the greeting and carry the chosen voice into the config.
    assert captured["config"].greeting == ""
    assert captured["config"].voice == "michael"
    assert isinstance(captured["deps"], CascadeDeps)


class _RecordingRenderer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stopped_called = False
        self.closed = False

    def notice(self, text):
        pass

    def stopped(self):
        self.stopped_called = True

    def close(self):
        self.closed = True


def _wire_run(monkeypatch, run_cascade):
    """Stub out auth/audio/cascade so run_agent_cascade reaches the run_cascade call."""
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "k")
    monkeypatch.setattr(_exec, "FileSource", lambda src: types.SimpleNamespace(sample_rate=16000))
    monkeypatch.setattr(_exec.client, "resolve_audio_source", lambda source, sample: "clip.wav")
    monkeypatch.setattr(_exec.engine, "run_cascade", run_cascade)
    rendered = {}
    monkeypatch.setattr(
        _exec, "AgentRenderer", lambda **kw: rendered.setdefault("r", _RecordingRenderer(**kw))
    )
    return rendered


def test_keyboard_interrupt_stops_cleanly(monkeypatch):
    def boom(**kwargs):
        raise KeyboardInterrupt

    rendered = _wire_run(monkeypatch, boom)
    run_agent_cascade(_opts(source="clip.wav"), AppState(), json_mode=False)
    assert rendered["r"].stopped_called is True
    assert rendered["r"].closed is True


def test_broken_pipe_exits_zero(monkeypatch):
    def boom(**kwargs):
        raise BrokenPipeError

    rendered = _wire_run(monkeypatch, boom)
    with pytest.raises(typer.Exit) as exc:
        run_agent_cascade(_opts(source="clip.wav"), AppState(), json_mode=False)
    assert exc.value.exit_code == 0
    assert rendered["r"].closed is True


# --- CascadeDeps.real (the three live legs) ----------------------------------


def test_deps_real_run_stt_passes_formatted_params(monkeypatch):
    captured = {}

    def fake_stream_audio(api_key, source, *, params, on_turn):
        captured["api_key"] = api_key
        captured["source"] = source
        captured["params"] = params

    monkeypatch.setattr(engine.client, "stream_audio", fake_stream_audio)
    audio: list[bytes] = []
    deps = CascadeDeps.real("k", CascadeConfig(), audio=audio, sample_rate=16000)
    deps.run_stt(lambda event: None)
    assert captured["api_key"] == "k"
    assert captured["source"] is audio
    assert captured["params"].sample_rate == 16000
    assert captured["params"].format_turns is True


def test_deps_real_complete_reply_returns_content(monkeypatch):
    monkeypatch.setattr(engine.llm, "complete", lambda api_key, **kwargs: "raw-response")
    monkeypatch.setattr(engine.llm, "content_of", lambda response: response.upper())
    deps = CascadeDeps.real("k", CascadeConfig(model="m"), audio=[], sample_rate=16000)
    assert deps.complete_reply([{"role": "user", "content": "hi"}]) == "RAW-RESPONSE"


def test_deps_real_synthesize_returns_pcm(monkeypatch):
    captured = {}

    def fake_synth(api_key, spec):
        captured["voice"] = spec.voice
        captured["text"] = spec.text
        captured["sample_rate"] = spec.sample_rate
        return types.SimpleNamespace(pcm=b"AUDIO")

    monkeypatch.setattr(engine.tts_session, "synthesize", fake_synth)
    deps = CascadeDeps.real("k", CascadeConfig(voice="vera"), audio=[], sample_rate=16000)
    assert deps.synthesize("say this") == b"AUDIO"
    assert captured["voice"] == "vera"
    assert captured["text"] == "say this"
    # TTS always synthesizes at the 24 kHz the live player is opened at.
    assert captured["sample_rate"] == engine.TTS_SAMPLE_RATE == 24000
