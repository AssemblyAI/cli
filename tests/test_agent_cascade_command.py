"""Command + wiring tests for `assembly live`.

Covers the argv -> options seam, the validation guards, _open_audio source
selection, and CascadeDeps.real's three live legs (all driven against fakes).
"""

from __future__ import annotations

import dataclasses
import signal
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
from aai_cli.core import config, config_builder
from aai_cli.core.errors import CLIError, UsageError
from aai_cli.main import app
from aai_cli.streaming import turn_presets

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
    speech_model="u3-rt-pro",
    format_turns=True,
    turn_detection=None,
    stt_config=(),
    stt_config_file=None,
    max_tokens=1000,
    llm_config=(),
    language=None,
    tts_config=(),
    mcp_config=(),
    show_code=False,
)


def _opts(**overrides) -> AgentCascadeOptions:
    return dataclasses.replace(_DEFAULTS, **overrides)


# --- help / list-voices ------------------------------------------------------


def test_list_voices_human_lists_catalog():
    result = runner.invoke(app, ["live", "--list-voices"])
    assert result.exit_code == 0
    assert "jane" in result.output
    assert "English:" in result.output


def test_list_voices_json_emits_array():
    result = runner.invoke(app, ["live", "--list-voices", "--json"])
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
    result = runner.invoke(app, ["live", "--system-prompt-file", "/no/such/file"])
    assert result.exit_code == 2
    assert "sandbox" not in result.output.lower()


def test_production_env_is_rejected_with_sandbox_hint():
    # Default env is production, which has no streaming-TTS host.
    result = runner.invoke(app, ["live", "--voice", "jane"])
    assert result.exit_code == 2
    assert "only available in the sandbox" in result.output


def test_device_with_file_source_is_rejected(monkeypatch):
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    with pytest.raises(UsageError, match="--device applies only to microphone"):
        run_agent_cascade(_opts(source="clip.wav", device=2), AppState(), json_mode=False)


# --- argv -> options seam ----------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], True), (["--no-format-turns"], False), (["--format-turns"], True)],
)
def test_format_turns_flag_resolves_into_options(monkeypatch, argv, expected):
    # Pin the Typer default (omitted -> True) and both explicit forms, captured at the
    # argv->options seam so the run body never executes.
    captured = {}

    def fake_run(opts, state, *, json_mode):
        captured["opts"] = opts

    monkeypatch.setattr(_exec, "run_agent_cascade", fake_run)
    result = runner.invoke(app, ["live", *argv])
    assert result.exit_code == 0
    assert captured["opts"].format_turns is expected


def test_stt_config_file_must_exist():
    # --stt-config-file is existence-checked at parse time (exists=True), so a missing
    # path fails as a Typer usage error before the body runs — not later on open. Wide
    # terminal so the "does not exist" message isn't wrapped by the 80-col error box.
    result = runner.invoke(
        app,
        ["live", "--stt-config-file", "/no/such/file.json"],
        env={"COLUMNS": "300"},
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_mcp_config_file_must_exist():
    # --mcp-config is existence-checked at parse time (exists=True), so a missing path
    # fails as a Typer usage error before the body runs. Wide terminal so the "does not
    # exist" message isn't wrapped by the 80-col error box.
    result = runner.invoke(
        app,
        ["live", "--mcp-config", "/no/such/servers.json"],
        env={"COLUMNS": "300"},
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


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


# --- MCP servers (resolution unit-tested in test_agent_cascade_mcp.py) -------
def test_default_mcp_servers_flow_into_cascade_config(monkeypatch):
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "k")
    monkeypatch.setattr(_exec, "FileSource", lambda src: types.SimpleNamespace(sample_rate=16000))
    monkeypatch.setattr(_exec.client, "resolve_audio_source", lambda source, sample: "clip.wav")
    captured = {}

    # Capture config at the deps seam so the graph (and its npx/uvx servers) never builds.
    def fake_real(api_key, config, *, audio, stt_params):
        captured["config"] = config
        return "deps"

    monkeypatch.setattr(_exec.engine.CascadeDeps, "real", fake_real)
    monkeypatch.setattr(_exec.engine, "run_cascade", lambda **kwargs: None)
    # With no flags, the default servers (e.g. weather) ride into the config the brain reads.
    run_agent_cascade(_opts(source="clip.wav"), AppState(), json_mode=False)
    assert "weather" in captured["config"].mcp_servers


# --- run_agent_cascade wiring ----------------------------------------------


def test_run_wires_deps_and_invokes_cascade(monkeypatch):
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "test-key")
    fake_source = types.SimpleNamespace(sample_rate=16000)
    monkeypatch.setattr(_exec, "FileSource", lambda src: fake_source)
    monkeypatch.setattr(_exec.client, "resolve_audio_source", lambda source, sample: "clip.wav")
    # CascadeDeps.real builds the brain graph (which would launch the default MCP servers);
    # stub the completer so deps still wire up without spawning any npx/uvx subprocess.
    monkeypatch.setattr(_exec.engine.brain, "build_completer", lambda api_key, config: lambda m: "")
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
    # Stub the brain completer so CascadeDeps.real never launches the default MCP servers.
    monkeypatch.setattr(_exec.engine.brain, "build_completer", lambda api_key, config: lambda m: "")
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
    # Ctrl-C ends the cascade cleanly (Stopped., renderer closed) but exits 130 (cancel),
    # not success, so a caller can tell an interrupt from a normal finish.
    with pytest.raises(typer.Exit) as exc:
        run_agent_cascade(_opts(source="clip.wav"), AppState(), json_mode=False)
    assert exc.value.exit_code == 130
    assert rendered["r"].stopped_called is True
    assert rendered["r"].closed is True


def test_installs_sigterm_handler_around_run(monkeypatch):
    captured: dict[str, object] = {}

    def capture(**kwargs):
        captured["handler"] = signal.getsignal(signal.SIGTERM)

    _wire_run(monkeypatch, capture)
    run_agent_cascade(_opts(source="clip.wav"), AppState(), json_mode=False)
    handler = captured["handler"]
    # While the cascade runs, SIGTERM maps to KeyboardInterrupt (the same clean stop
    # as Ctrl-C); without the wrapper this would be the default, non-callable SIG_DFL.
    assert callable(handler)
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)


def test_broken_pipe_exits_zero(monkeypatch):
    def boom(**kwargs):
        raise BrokenPipeError

    rendered = _wire_run(monkeypatch, boom)
    with pytest.raises(typer.Exit) as exc:
        run_agent_cascade(_opts(source="clip.wav"), AppState(), json_mode=False)
    assert exc.value.exit_code == 0
    assert rendered["r"].closed is True


# --- STT param + TTS config builders -----------------------------------------


def test_build_stt_params_threads_named_flags():
    params = _exec._build_stt_params(_opts(speech_model="u3-rt-pro", format_turns=False), 8000)
    assert params.sample_rate == 8000  # fixed by the audio source, not a flag
    assert params.format_turns is False
    assert params.speech_model.value == "u3-rt-pro"


def test_build_stt_params_expands_turn_detection_preset():
    params = _exec._build_stt_params(
        _opts(turn_detection=turn_presets.TurnDetectionPreset.conservative), 16000
    )
    # The conservative preset's published end-of-turn confidence threshold.
    assert params.end_of_turn_confidence_threshold == 0.7


def test_build_stt_params_stt_config_overrides_any_field():
    params = _exec._build_stt_params(
        _opts(stt_config=("end_of_turn_confidence_threshold=0.9",)), 16000
    )
    assert params.end_of_turn_confidence_threshold == 0.9


def test_build_stt_params_reads_config_file(tmp_path):
    cfg = tmp_path / "stt.json"
    cfg.write_text('{"min_turn_silence": 123}', encoding="utf-8")
    params = _exec._build_stt_params(_opts(stt_config_file=cfg), 16000)
    assert params.min_turn_silence == 123


def test_parse_tts_config_parses_pairs():
    assert _exec._parse_tts_config(("chunk_size_ms=100", "foo=bar")) == {
        "chunk_size_ms": "100",
        "foo": "bar",
    }


def test_parse_tts_config_rejects_malformed_pair():
    with pytest.raises(UsageError, match="expects KEY=VALUE"):
        _exec._parse_tts_config(("no-equals",))


@pytest.mark.parametrize(
    ("key", "hint"),
    [("voice", "--voice"), ("language", "--language"), ("sample_rate", "fixed")],
)
def test_parse_tts_config_rejects_reserved_keys(key, hint):
    with pytest.raises(UsageError, match=hint):
        _exec._parse_tts_config((f"{key}=x",))


def test_run_threads_all_leg_options_into_config_and_params(monkeypatch):
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "k")
    monkeypatch.setattr(_exec, "FileSource", lambda src: types.SimpleNamespace(sample_rate=16000))
    monkeypatch.setattr(_exec.client, "resolve_audio_source", lambda source, sample: "clip.wav")
    monkeypatch.setattr(_exec.engine, "run_cascade", lambda **kw: None)
    captured = {}

    def fake_real(api_key, config, *, audio, stt_params):
        captured["config"] = config
        captured["stt_params"] = stt_params
        return CascadeDeps(
            run_stt=lambda _o: None, complete_reply=lambda _m: "", synthesize=lambda _t: b""
        )

    monkeypatch.setattr(_exec.engine.CascadeDeps, "real", fake_real)
    run_agent_cascade(
        _opts(
            source="clip.wav",
            language="en",
            max_tokens=321,
            format_turns=False,
            llm_config=("temperature=0.3",),
            tts_config=("chunk_size_ms=100",),
            speech_model="u3-rt-pro",
        ),
        AppState(),
        json_mode=False,
    )
    cfg = captured["config"]
    assert cfg.language == "en"
    assert cfg.max_tokens == 321
    assert cfg.format_turns is False
    assert cfg.llm_extra == {"temperature": 0.3}
    assert cfg.tts_extra == {"chunk_size_ms": "100"}
    # The STT flags are realized into the params the cascade will stream with.
    assert captured["stt_params"].format_turns is False
    assert captured["stt_params"].sample_rate == 16000


# --- CascadeDeps.real (the three live legs) ----------------------------------


def _stt_params(**flags: object):
    merged = config_builder.merge_streaming_params(
        flags={"sample_rate": 16000, "format_turns": True, "speech_model": "u3-rt-pro", **flags}
    )
    return config_builder.construct_streaming_params(merged)


def test_deps_real_run_stt_passes_prebuilt_params_through(monkeypatch):
    captured = {}

    def fake_stream_audio(api_key, source, *, params, on_turn):
        captured["api_key"] = api_key
        captured["source"] = source
        captured["params"] = params

    monkeypatch.setattr(engine.client, "stream_audio", fake_stream_audio)
    audio: list[bytes] = []
    params = _stt_params()
    deps = CascadeDeps.real("k", CascadeConfig(), audio=audio, stt_params=params)
    deps.run_stt(lambda event: None)
    assert captured["api_key"] == "k"
    assert captured["source"] is audio
    # The cascade streams exactly the params it was handed — no re-derivation.
    assert captured["params"] is params


def test_deps_real_complete_reply_is_built_by_the_deepagents_brain(monkeypatch):
    # The LLM leg is now a deepagents graph: .real delegates to brain.build_completer,
    # passing the api key + config, and uses whatever completer it returns. We assert the
    # exact wiring so the brain swap (not a plain llm.complete) can't silently regress.
    captured = {}

    def fake_build_completer(api_key, config):
        captured["api_key"] = api_key
        captured["config"] = config
        return lambda messages: f"reply to {messages[-1]['content']}"

    monkeypatch.setattr(engine.brain, "build_completer", fake_build_completer)
    cfg = CascadeConfig(model="m", max_tokens=222, llm_extra={"temperature": 0.5})
    deps = CascadeDeps.real("k", cfg, audio=[], stt_params=_stt_params())
    assert deps.complete_reply([{"role": "user", "content": "hi"}]) == "reply to hi"
    assert captured["api_key"] == "k"
    assert captured["config"] is cfg


def test_deps_real_synthesize_threads_voice_language_and_extra(monkeypatch):
    captured = {}

    def fake_synth(api_key, spec):
        captured["voice"] = spec.voice
        captured["language"] = spec.language
        captured["text"] = spec.text
        captured["sample_rate"] = spec.sample_rate
        captured["params"] = spec.query_params()
        return types.SimpleNamespace(pcm=b"AUDIO")

    monkeypatch.setattr(engine.tts_session, "synthesize", fake_synth)
    cfg = CascadeConfig(voice="vera", language="en", tts_extra={"chunk_size_ms": "100"})
    deps = CascadeDeps.real("k", cfg, audio=[], stt_params=_stt_params())
    assert deps.synthesize("say this") == b"AUDIO"
    assert captured["voice"] == "vera"
    assert captured["language"] == "en"
    assert captured["text"] == "say this"
    # TTS always synthesizes at the 24 kHz the live player is opened at.
    assert captured["sample_rate"] == engine.TTS_SAMPLE_RATE == 24000
    # The --tts-config escape hatch rides along as an extra query param.
    assert captured["params"]["chunk_size_ms"] == "100"
