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
