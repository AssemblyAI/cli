"""`assembly transcribe` flag handling: flag-to-config mapping and flag validation.

Output rendering, LLM transforms, sources, and --show-code tests live in
test_transcribe.py.
"""

import pytest
from typer.testing import CliRunner

from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def audio_file(tmp_path, monkeypatch):
    # The command checks the local path exists before resolving credentials, so the
    # "audio.mp3" the tests pass must be a real file; run each test in its own cwd.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "audio.mp3").write_bytes(b"fake-audio")


def _auth():
    config.set_api_key("default", "sk_live")


def _fake_transcript(mocker):
    t = mocker.MagicMock()
    t.id = "t_1"
    t.text = "hello world"
    t.status = "completed"
    t.json_response = {"id": "t_1", "text": "hello world", "status": "completed"}
    for attr in (
        "summary",
        "chapters",
        "auto_highlights",
        "sentiment_analysis",
        "entities",
        "iab_categories",
        "content_safety",
    ):
        setattr(t, attr, None)
    t.utterances = None
    return t


def _enum_or_str(value):
    return getattr(value, "value", value)


def test_transcribe_passes_speaker_labels(mocker):
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    runner.invoke(app, ["transcribe", "audio.mp3", "--speaker-labels"])
    assert tx.call_args.kwargs["config"].speaker_labels is True


def test_transcribe_prompt_biases_speech_model(mocker):
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--prompt", "expect medical terms"])
    assert result.exit_code == 0
    # --prompt is the speech-model prompt, forwarded to the transcription call.
    assert tx.call_args.kwargs["config"].prompt == "expect medical terms"


def test_transcribe_maps_analysis_flags(mocker):
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
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


def test_transcribe_redact_pii_policy_csv(mocker):
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
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


def test_transcribe_config_escape_hatch(mocker):
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    runner.invoke(app, ["transcribe", "audio.mp3", "--config", "speech_threshold=0.5"])
    assert tx.call_args.kwargs["config"].raw.speech_threshold == 0.5


def test_transcribe_unknown_config_field_exits_2(mocker):
    _auth()
    mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--config", "bogus=1"])
    assert result.exit_code == 2
    assert "bogus" in result.output


def test_transcribe_webhook_auth_header(mocker):
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    runner.invoke(
        app,
        [
            "transcribe",
            "audio.mp3",
            "--webhook-url",
            "https://example.com/hook",
            "--webhook-auth-header",
            "X-Token:secret",
        ],
    )
    cfg = tx.call_args.kwargs["config"]
    assert cfg.raw.webhook_url == "https://example.com/hook"
    assert cfg.raw.webhook_auth_header_name == "X-Token"
    assert cfg.raw.webhook_auth_header_value == "secret"


def test_transcribe_negative_audio_start_exits_2(mocker):
    _auth()
    tx = mocker.patch("aai_cli.app.transcribe.run.client.transcribe", autospec=True)
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--audio-start", "-100"])
    assert result.exit_code == 2
    tx.assert_not_called()


def test_transcribe_language_code_with_detection_exits_2(mocker):
    _auth()
    tx = mocker.patch("aai_cli.app.transcribe.run.client.transcribe", autospec=True)
    result = runner.invoke(
        app,
        ["transcribe", "audio.mp3", "--language-code", "en_us", "--language-detection"],
    )
    assert result.exit_code == 2
    assert "--language-code and --language-detection can't be combined." in result.output
    assert "Force a language or auto-detect it, not both." in result.output
    tx.assert_not_called()


def test_transcribe_language_flags_alone_are_accepted(mocker):
    # Only the combination is contradictory; each flag works on its own.
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--language-code", "en_us"])
    assert result.exit_code == 0
    assert tx.call_args.kwargs["config"].language_code == "en_us"
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--language-detection"])
    assert result.exit_code == 0
    assert tx.call_args.kwargs["config"].language_detection is True


def test_transcribe_speakers_expected_without_labels_exits_2(mocker):
    _auth()
    tx = mocker.patch("aai_cli.app.transcribe.run.client.transcribe", autospec=True)
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--speakers-expected", "2"])
    assert result.exit_code == 2
    assert "--speakers-expected only applies when diarization is enabled." in result.output
    assert "Add --speaker-labels." in result.output
    tx.assert_not_called()


def test_transcribe_download_sections_on_local_file_exits_2(mocker):
    # --download-sections only slices a downloadable-URL fetch; on a local file it would
    # be dropped silently, so it's rejected up front (matching `clip`/`dub`) rather than
    # billing a full-file transcription. The guard fires before any API call.
    _auth()
    tx = mocker.patch("aai_cli.app.transcribe.run.client.transcribe", autospec=True)
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--download-sections", "*0:00-5:00"])
    assert result.exit_code == 2
    assert "--download-sections only applies to a downloadable URL source" in result.output
    tx.assert_not_called()


def test_transcribe_speakers_expected_with_labels_is_accepted(mocker):
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(
        app, ["transcribe", "audio.mp3", "--speaker-labels", "--speakers-expected", "2"]
    )
    assert result.exit_code == 0
    assert tx.call_args.kwargs["config"].speakers_expected == 2


def test_transcribe_speakers_expected_with_config_speaker_labels_is_accepted(mocker):
    # Diarization enabled through the --config escape hatch counts too: the check
    # runs on the merged config, not just the curated flag.
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(
        app,
        ["transcribe", "audio.mp3", "--config", "speaker_labels=true", "--speakers-expected", "2"],
    )
    assert result.exit_code == 0
    assert tx.call_args.kwargs["config"].speakers_expected == 2


@pytest.mark.parametrize("value", ["-0.1", "1.5", "99"])
def test_transcribe_temperature_out_of_range_exits_2(mocker, value):
    # The API documents temperature as 0 (most deterministic) to 1 (least); reject
    # out-of-range values client-side instead of letting them flow to the request.
    _auth()
    tx = mocker.patch("aai_cli.app.transcribe.run.client.transcribe", autospec=True)
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--temperature", value])
    assert result.exit_code == 2
    tx.assert_not_called()


@pytest.mark.parametrize("value", ["0", "1"])
def test_transcribe_temperature_bounds_are_inclusive(mocker, value):
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--temperature", value])
    assert result.exit_code == 0
    assert tx.call_args.kwargs["config"].temperature == float(value)


def test_transcribe_negative_audio_end_exits_2(mocker):
    _auth()
    tx = mocker.patch("aai_cli.app.transcribe.run.client.transcribe", autospec=True)
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--audio-end", "-100"])
    assert result.exit_code == 2
    tx.assert_not_called()


def test_transcribe_audio_end_zero_is_accepted(mocker):
    _auth()
    tx = mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--audio-end", "0"])
    assert result.exit_code == 0
    tx.assert_called_once()


def test_transcribe_json_with_non_json_output_field_exits_2(mocker):
    # --json means "the full JSON payload" (same as -o json); -o text contradicts it
    # and must not silently win.
    _auth()
    tx = mocker.patch("aai_cli.app.transcribe.run.client.transcribe", autospec=True)
    result = runner.invoke(app, ["transcribe", "audio.mp3", "-o", "text", "--json"])
    assert result.exit_code == 2
    assert "--json and -o text can't be combined." in result.output
    assert "use -o json for the full JSON payload" in result.output
    tx.assert_not_called()


def test_transcribe_json_with_o_json_is_accepted(mocker):
    import json

    _auth()
    mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "-o", "json", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output.strip())["id"] == "t_1"


def test_transcribe_warns_on_non_audio_extension(mocker, tmp_path):
    _auth()
    (tmp_path / "notes.txt").write_bytes(b"fake")
    mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "notes.txt"])
    assert result.exit_code == 0  # a warning, never an error — the server decides
    assert "doesn't look like audio" in result.output
    assert "'.txt'" in result.output


def test_transcribe_non_audio_warning_suppressed_by_quiet(mocker, tmp_path):
    _auth()
    (tmp_path / "notes.txt").write_bytes(b"fake")
    mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["-q", "transcribe", "notes.txt"])
    assert result.exit_code == 0
    assert "doesn't look like audio" not in result.output


def test_transcribe_non_audio_warning_is_structured_under_json(mocker, tmp_path):
    import json

    _auth()
    (tmp_path / "notes.txt").write_bytes(b"fake")
    mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "notes.txt", "--json"])
    assert result.exit_code == 0
    warning = next(
        json.loads(line) for line in result.output.splitlines() if line.startswith('{"warning"')
    )
    assert "doesn't look like audio" in warning["warning"]


@pytest.mark.parametrize("name", ["audio.mp3", "noext"])
def test_transcribe_no_warning_for_audio_or_extensionless_files(mocker, tmp_path, name):
    # Known audio extensions and extensionless files (we can't judge those) stay quiet.
    _auth()
    (tmp_path / name).write_bytes(b"fake")
    mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", name])
    assert result.exit_code == 0
    assert "doesn't look like audio" not in result.output


@pytest.mark.parametrize("argv", [["https://example.com/notes.txt"], ["--sample"]])
def test_transcribe_no_warning_for_urls_or_sample(mocker, argv):
    # Remote sources aren't local files; the extension heuristic doesn't apply.
    _auth()
    mocker.patch(
        "aai_cli.app.transcribe.run.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", *argv])
    assert result.exit_code == 0
    assert "doesn't look like audio" not in result.output


def test_transcribe_unknown_pii_policy_exits_2_and_lists_valid(mocker):
    _auth()
    tx = mocker.patch("aai_cli.app.transcribe.run.client.transcribe", autospec=True)
    result = runner.invoke(
        app,
        ["transcribe", "audio.mp3", "--redact-pii", "--redact-pii-policy", "not_a_policy"],
    )
    assert result.exit_code == 2
    assert "Unknown PII policy(s) ['not_a_policy']" in result.output
    assert "person_name" in result.output  # the valid values are listed
    tx.assert_not_called()
