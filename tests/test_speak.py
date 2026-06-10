from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app
from aai_cli.tts import session

runner = CliRunner()


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "test-key")


@pytest.fixture
def fake_synthesize(monkeypatch: pytest.MonkeyPatch):
    calls: dict[str, object] = {}

    def _fake(api_key, cfg, *, connect=None, on_warning=None):
        calls["api_key"] = api_key
        calls["cfg"] = cfg
        return session.SpeakResult(
            pcm=b"\x01\x02\x03\x04", sample_rate=24000, audio_duration_seconds=0.123456
        )

    monkeypatch.setattr(session, "synthesize", _fake)
    return calls


def test_production_env_is_rejected_with_sandbox_hint():
    result = runner.invoke(app, ["speak", "Hello"])  # default = production
    assert result.exit_code == 2
    assert "only available in the sandbox" in result.output
    assert "--sandbox" in result.output


def test_plays_audio_by_default(monkeypatch, fake_synthesize):
    played: dict = {}
    monkeypatch.setattr(
        "aai_cli.commands.speak.audio.play_pcm",
        lambda pcm, rate, **_: played.update(pcm=pcm, rate=rate),
    )
    result = runner.invoke(app, ["--sandbox", "speak", "Hello there"])
    assert result.exit_code == 0
    assert played == {"pcm": b"\x01\x02\x03\x04", "rate": 24000}
    assert fake_synthesize["cfg"].text == "Hello there"
    # Human summary (stderr) reports the default "played" disposition.
    assert "played" in result.stderr
    assert "saved to" not in result.stderr


def test_out_writes_wav_and_does_not_play(monkeypatch, tmp_path, fake_synthesize):
    monkeypatch.setattr(
        "aai_cli.commands.speak.audio.play_pcm",
        lambda *a, **k: pytest.fail("should not play when --out is given"),
    )
    written: dict = {}
    monkeypatch.setattr(
        "aai_cli.commands.speak.audio.write_wav",
        lambda path, pcm, rate: written.update(path=path, pcm=pcm, rate=rate),
    )
    out = tmp_path / "x.wav"
    result = runner.invoke(app, ["--sandbox", "speak", "Hi", "--out", str(out)])
    assert result.exit_code == 0
    assert written["pcm"] == b"\x01\x02\x03\x04"
    assert str(written["path"]) == str(out)
    # Human summary (stderr) reports the file disposition, not "played".
    assert "saved to" in result.stderr
    assert "played" not in result.stderr


def test_reads_text_from_stdin_when_arg_omitted(monkeypatch, fake_synthesize):
    monkeypatch.setattr("aai_cli.commands.speak.audio.play_pcm", lambda *a, **k: None)
    result = runner.invoke(app, ["--sandbox", "speak"], input="piped text\n")
    assert result.exit_code == 0
    assert fake_synthesize["cfg"].text == "piped text"


def test_empty_text_is_a_usage_error(monkeypatch):
    # No arg and empty stdin -> usage error, before any synthesis.
    result = runner.invoke(app, ["--sandbox", "speak"], input="")
    assert result.exit_code == 2
    assert "No text to speak" in result.output


def test_blank_arg_does_not_fall_back_to_stdin(monkeypatch):
    # A blank argument is a usage error; stdin is only read when the arg is omitted
    # entirely, so an explicit empty arg must NOT silently pull from the pipe.
    result = runner.invoke(app, ["--sandbox", "speak", "   "], input="from stdin")
    assert result.exit_code == 2
    assert "No text to speak" in result.output


def test_voice_and_language_flow_into_config(monkeypatch, fake_synthesize):
    monkeypatch.setattr("aai_cli.commands.speak.audio.play_pcm", lambda *a, **k: None)
    result = runner.invoke(
        app, ["--sandbox", "speak", "Hi", "--voice", "jane", "--language", "English"]
    )
    assert result.exit_code == 0
    cfg = fake_synthesize["cfg"]
    assert cfg.voice == "jane"
    assert cfg.language == "English"
    assert cfg.query_params() == {"voice": "jane", "language": "English"}


def test_json_mode_emits_metadata_object_on_stdout(monkeypatch, fake_synthesize):
    monkeypatch.setattr("aai_cli.commands.speak.audio.play_pcm", lambda *a, **k: None)
    result = runner.invoke(app, ["--sandbox", "speak", "Hi", "--voice", "jane", "--json"])
    assert result.exit_code == 0
    # The behavioral split: --json yields a parseable object, not human prose.
    payload = json.loads(result.stdout.strip())
    assert payload["voice"] == "jane"
    assert payload["sample_rate"] == 24000
    assert payload["bytes"] == 4
    # Duration is rounded to 3 decimals (0.123456 -> 0.123, not 0.1235).
    assert payload["audio_duration_seconds"] == 0.123
    # No --out -> the reported path is null, not the string "None".
    assert payload["out"] is None


def test_human_mode_keeps_stdout_clean(monkeypatch, fake_synthesize):
    monkeypatch.setattr("aai_cli.commands.speak.audio.play_pcm", lambda *a, **k: None)
    result = runner.invoke(app, ["--sandbox", "speak", "Hi"])
    assert result.exit_code == 0
    # Human summary goes to stderr; stdout stays empty (audio went to the speaker).
    assert result.stdout.strip() == ""
