import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from assemblyai_cli import config
from assemblyai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_api_key("default", "sk_live")


def _fake_transcript():
    t = MagicMock()
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


def test_transcribe_sample_prints_text():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        result = runner.invoke(app, ["transcribe", "--sample"])
    assert result.exit_code == 0
    assert "hello world" in result.output
    audio_arg = tx.call_args.args[1]
    assert audio_arg.endswith("wildfires.mp3")


def test_transcribe_requires_source():
    _auth()
    result = runner.invoke(app, ["transcribe"])
    assert result.exit_code == 2


def test_transcribe_passes_speaker_labels():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        runner.invoke(app, ["transcribe", "audio.mp3", "--speaker-labels"])
    assert tx.call_args.kwargs["config"].speaker_labels is True


def test_transcribe_json_output():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--json"])
    assert '"id": "t_1"' in result.output


def test_transcribe_unauthenticated_exits_2():
    result = runner.invoke(app, ["transcribe", "--sample"])
    assert result.exit_code == 2


def test_transcribe_status_renders_enum_value():
    import assemblyai as aai

    _auth()
    t = _fake_transcript()
    t.status = aai.TranscriptStatus.completed
    t.json_response = None
    with patch("assemblyai_cli.commands.transcribe.client.transcribe", return_value=t):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--json"])
    assert result.exit_code == 0
    assert '"status": "completed"' in result.output


def test_transcribe_prompt_transforms_json(monkeypatch):
    _auth()
    seen = {}

    def fake_transform(api_key, *, prompt, model, transcript_id, max_tokens):
        seen["prompt"] = prompt
        seen["model"] = model
        seen["transcript_id"] = transcript_id
        return "a short summary"

    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ):
        monkeypatch.setattr(
            "assemblyai_cli.commands.transcribe.llm.transform_transcript", fake_transform
        )
        result = runner.invoke(
            app, ["transcribe", "audio.mp3", "--llm-gateway-prompt", "summarize", "--json"]
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["text"] == "hello world"  # raw transcript still present in JSON
    assert data["transform"]["output"] == "a short summary"
    assert data["transform"]["prompt"] == "summarize"
    # The transform is injected server-side via the transcript id.
    assert seen["transcript_id"] == "t_1"
    assert seen["model"] == "claude-sonnet-4-6"


def test_transcribe_prompt_human_shows_only_transform(monkeypatch):
    _auth()
    monkeypatch.setattr("assemblyai_cli.output.resolve_json", lambda *, explicit: False)
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ):
        monkeypatch.setattr(
            "assemblyai_cli.commands.transcribe.llm.transform_transcript",
            lambda *a, **k: "TRANSFORMED",
        )
        result = runner.invoke(
            app, ["transcribe", "audio.mp3", "--llm-gateway-prompt", "summarize"]
        )
    assert result.exit_code == 0
    assert "TRANSFORMED" in result.output
    assert "hello world" not in result.output  # human mode shows the transform only


def test_transcribe_prompt_biases_speech_model():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--prompt", "expect medical terms"])
    assert result.exit_code == 0
    # --prompt is the speech-model prompt, forwarded to the transcription call.
    assert tx.call_args.kwargs["config"].prompt == "expect medical terms"


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


def test_transcribe_webhook_auth_header():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
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


def test_transcribe_youtube_url_downloads_then_transcribes(monkeypatch, tmp_path):
    _auth()
    fake = tmp_path / "vid.m4a"
    fake.write_bytes(b"x")
    monkeypatch.setattr(
        "assemblyai_cli.commands.transcribe.youtube.download_audio", lambda url, d: fake
    )
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        result = runner.invoke(app, ["transcribe", "https://youtu.be/abc", "--json"])
    assert result.exit_code == 0
    assert tx.call_args.args[1] == str(fake)  # transcribed the downloaded local file


def test_transcribe_show_code_prints_without_transcribing(monkeypatch):
    # Print-only: emits code, never calls the API, needs no auth.
    called = []
    monkeypatch.setattr(
        "assemblyai_cli.commands.transcribe.client.transcribe",
        lambda *a, **k: called.append(True),
    )
    result = runner.invoke(app, ["transcribe", "--sample", "--speaker-labels", "--show-code"])
    assert result.exit_code == 0
    assert called == []  # never transcribed
    assert "import assemblyai as aai" in result.output
    assert "TranscriptionConfig(" in result.output
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in result.output


def test_transcribe_show_code_ignores_json_flag(monkeypatch):
    # --show-code is print-only; --json does not suppress or wrap it.
    def _boom(*a, **k):
        raise AssertionError("must not transcribe")

    monkeypatch.setattr(
        "assemblyai_cli.commands.transcribe.client.transcribe",
        _boom,
    )
    result = runner.invoke(app, ["transcribe", "--sample", "--show-code", "--json"])
    assert result.exit_code == 0
    assert "import assemblyai as aai" in result.output


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
