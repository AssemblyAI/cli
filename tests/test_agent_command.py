import json

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


def test_agent_help_lists_command():
    result = runner.invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "voice" in result.output.lower()


def test_list_voices_prints_and_exits_without_connecting(monkeypatch):
    called = {"ran": False}

    def fake_run_session(*a, **k):
        called["ran"] = True

    monkeypatch.setattr("aai_cli.commands.agent.run_session", fake_run_session)
    result = runner.invoke(app, ["agent", "--list-voices"])
    assert result.exit_code == 0
    assert "ivy" in result.output
    assert called["ran"] is False


def test_agent_unauthenticated_exits_2():
    result = runner.invoke(app, ["agent"])
    assert result.exit_code == 2


def test_agent_drives_renderer_json(monkeypatch):
    config.set_api_key("default", "sk_live")

    def fake_run_session(
        api_key,
        *,
        renderer,
        player,
        mic,
        voice,
        system_prompt,
        greeting,
        full_duplex=False,
        exit_after_reply=False,
    ):
        renderer.connected()
        renderer.user_final("hello agent")
        renderer.agent_transcript("hello human", interrupted=False)

    monkeypatch.setattr("aai_cli.commands.agent.run_session", fake_run_session)
    result = runner.invoke(app, ["agent", "--json"])
    assert result.exit_code == 0
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"type": "transcript.user", "text": "hello agent"} in lines
    assert {"type": "transcript.agent", "text": "hello human", "interrupted": False} in lines


def test_agent_passes_voice_and_prompt_file(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_run_session(
        api_key,
        *,
        renderer,
        player,
        mic,
        voice,
        system_prompt,
        greeting,
        full_duplex=False,
        exit_after_reply=False,
    ):
        seen["voice"] = voice
        seen["prompt"] = system_prompt
        seen["full_duplex"] = full_duplex

    monkeypatch.setattr("aai_cli.commands.agent.run_session", fake_run_session)
    prompt_file = tmp_path / "p.txt"
    prompt_file.write_text("be a pirate")
    result = runner.invoke(
        app,
        [
            "agent",
            "--voice",
            "james",
            "--system-prompt-file",
            str(prompt_file),
            "--system-prompt",
            "ignored",
        ],
    )
    assert result.exit_code == 0
    assert seen["voice"] == "james"
    assert seen["prompt"] == "be a pirate"  # --system-prompt-file overrides --system-prompt
    assert seen["full_duplex"] is True  # always full duplex now (one stream)


def test_agent_headphones_notice_in_human_mode(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    monkeypatch.setattr("aai_cli.commands.agent.run_session", lambda *a, **k: None)
    result = runner.invoke(app, ["agent"])
    assert result.exit_code == 0
    assert "headphones" in result.output.lower()  # mic stays open -> warn to use headphones


def test_agent_ctrl_c_exits_cleanly(monkeypatch):
    config.set_api_key("default", "sk_live")

    def raise_kbd(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("aai_cli.commands.agent.run_session", raise_kbd)
    result = runner.invoke(app, ["agent"])
    assert result.exit_code == 0


def test_agent_unknown_voice_exits_2(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.commands.agent.run_session", lambda *a, **k: None)
    result = runner.invoke(app, ["agent", "--voice", "not-a-voice"])
    assert result.exit_code == 2


def test_agent_prompt_file_not_found_exits_2(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.commands.agent.run_session", lambda *a, **k: None)
    result = runner.invoke(
        app, ["agent", "--system-prompt-file", "/tmp/no_such_file_xyz_voiceagent.txt"]
    )
    assert result.exit_code == 2


def _capture_run_session(monkeypatch):
    """Patch run_session to record its kwargs and return the dict it fills in."""
    seen = {}

    def fake_run_session(api_key, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr("aai_cli.commands.agent.run_session", fake_run_session)
    return seen


def test_agent_file_source_streams_clip_and_exits_after_reply(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    wav = tmp_path / "say.wav"
    wav.write_bytes(b"RIFF")  # FileSource is faked below; contents don't matter

    monkeypatch.setattr("aai_cli.commands.agent.FileSource", lambda src: f"filesrc:{src}")
    seen = _capture_run_session(monkeypatch)

    result = runner.invoke(app, ["agent", str(wav)])
    assert result.exit_code == 0
    # File input drives a deterministic, headless, self-terminating session.
    assert seen["mic"] == f"filesrc:{wav}"
    assert seen["exit_after_reply"] is True
    assert seen["full_duplex"] is True
    assert seen["greeting"] == ""
    from aai_cli.agent.audio import NullPlayer

    assert isinstance(seen["player"], NullPlayer)


def test_agent_sample_uses_hosted_clip(monkeypatch):
    config.set_api_key("default", "sk_live")
    captured = {}

    def fake_file_source(src):
        captured["src"] = src
        return "filesrc"

    monkeypatch.setattr("aai_cli.commands.agent.FileSource", fake_file_source)
    seen = _capture_run_session(monkeypatch)

    result = runner.invoke(app, ["agent", "--sample"])
    assert result.exit_code == 0
    assert captured["src"].endswith("wildfires.mp3")
    assert seen["exit_after_reply"] is True


def test_agent_file_source_with_device_exits_2(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.commands.agent.run_session", lambda *a, **k: None)
    wav = tmp_path / "say.wav"
    wav.write_bytes(b"RIFF")
    result = runner.invoke(app, ["agent", str(wav), "--device", "1"])
    assert result.exit_code == 2  # --device is microphone-only


def test_agent_file_source_no_headphones_notice(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    monkeypatch.setattr("aai_cli.commands.agent.FileSource", lambda src: "filesrc")
    monkeypatch.setattr("aai_cli.commands.agent.run_session", lambda *a, **k: None)
    wav = tmp_path / "say.wav"
    wav.write_bytes(b"RIFF")
    result = runner.invoke(app, ["agent", str(wav)])
    assert result.exit_code == 0
    assert "headphones" not in result.output.lower()  # mic-only note; file mode is silent


def test_agent_file_source_no_start_talking_notice(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    monkeypatch.setattr("aai_cli.commands.agent.FileSource", lambda src: "filesrc")

    def fake_run_session(api_key, *, renderer, **kwargs):
        renderer.connected()  # session.ready arrives even for a file-driven run

    monkeypatch.setattr("aai_cli.commands.agent.run_session", fake_run_session)
    wav = tmp_path / "say.wav"
    wav.write_bytes(b"RIFF")
    result = runner.invoke(app, ["agent", str(wav)])
    assert result.exit_code == 0
    # No mic on a file-driven run -> no "start talking" prompt.
    assert "start talking" not in result.output.lower()


def test_agent_mic_shows_start_talking_notice(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)

    # Avoid opening real audio hardware; the renderer is what we're testing.
    class FakeDuplex:
        def __init__(self, **kwargs):
            self.mic = iter([])
            self.player = self

        def start(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("aai_cli.commands.agent.DuplexAudio", FakeDuplex)

    def fake_run_session(api_key, *, renderer, **kwargs):
        renderer.connected()

    monkeypatch.setattr("aai_cli.commands.agent.run_session", fake_run_session)
    result = runner.invoke(app, ["agent"])
    assert result.exit_code == 0
    assert "start talking" in result.output.lower()  # live mic -> prompt the user to speak


def test_agent_show_code_prints_without_session(monkeypatch):
    # Print-only: emits the agent script, never starts a session or opens audio, no auth.
    called = []
    monkeypatch.setattr("aai_cli.commands.agent.run_session", lambda *a, **k: called.append(True))
    result = runner.invoke(app, ["agent", "--voice", "ivy", "--show-code"])
    assert result.exit_code == 0
    assert called == []  # never ran a session
    assert "agents.assemblyai.com" in result.output
    assert '"voice": "ivy"' in result.output
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in result.output


def test_agent_show_code_ignores_json_flag(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not run a session")

    monkeypatch.setattr(
        "aai_cli.commands.agent.run_session",
        _boom,
    )
    result = runner.invoke(app, ["agent", "--voice", "ivy", "--show-code", "--json"])
    assert result.exit_code == 0
    assert "agents.assemblyai.com" in result.output


def test_agent_output_text_emits_plain_transcript(monkeypatch):
    # `-o text` -> plain you:/agent: lines on stdout (pipe into aai llm).
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.commands.agent.FileSource", lambda src: "filesrc")

    def fake_run_session(api_key, *, renderer, **kwargs):
        renderer.user_final("hello there")
        renderer.agent_transcript("hi, how can I help?", interrupted=False)

    monkeypatch.setattr("aai_cli.commands.agent.run_session", fake_run_session)
    result = runner.invoke(app, ["agent", "--sample", "-o", "text"])
    assert result.exit_code == 0
    assert "you: hello there" in result.output
    assert "agent: hi, how can I help?" in result.output
    assert '"type"' not in result.output  # not NDJSON


def test_agent_help_has_examples():
    from typer.testing import CliRunner

    from aai_cli.main import app

    result = CliRunner().invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
