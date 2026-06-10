"""Input validation that must run before credentials or any request.

A typo'd local path has to read as "file not found" (never trigger a login or an
upload attempt), and a transcript "id" is interpolated into the API path so anything
non-token-shaped is rejected before an authenticated GET can be steered elsewhere.
"""

import pytest
from typer.testing import CliRunner

from aai_cli import client, config
from aai_cli.errors import CLIError, UsageError
from aai_cli.main import app

runner = CliRunner()


def test_resolve_audio_source_sample_explicit_and_missing(tmp_path):
    assert client.resolve_audio_source(None, sample=True) == client.SAMPLE_AUDIO_URL
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"fake")
    assert client.resolve_audio_source(str(clip), sample=False) == str(clip)
    with pytest.raises(UsageError) as exc:
        client.resolve_audio_source(None, sample=False)
    assert exc.value.message == "Provide an audio path or URL."
    assert "--sample" in (exc.value.suggestion or "")


def test_resolve_audio_source_rejects_explicit_source_plus_sample(tmp_path):
    # Both an explicit source and --sample is a contradiction: neither may silently win.
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"fake")
    with pytest.raises(UsageError) as exc:
        client.resolve_audio_source(str(clip), sample=True)
    assert exc.value.exit_code == 2
    assert exc.value.message == "An audio source and --sample cannot be combined."
    assert exc.value.suggestion == "Pass the file/URL or --sample, not both."


def test_resolve_audio_source_source_plus_sample_rejected_even_without_checks():
    # The conflict fires before any existence check, including --show-code paths.
    with pytest.raises(UsageError) as exc:
        client.resolve_audio_source("missing.mp3", sample=True, check_local=False)
    assert exc.value.message == "An audio source and --sample cannot be combined."


def test_transcribe_source_plus_sample_exits_2(mocker, tmp_path):
    # No key configured: the conflict must fail before credential resolution.
    tx = mocker.patch("aai_cli.commands.transcribe.client.transcribe", autospec=True)
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"fake")
    result = runner.invoke(app, ["transcribe", str(clip), "--sample"])
    assert result.exit_code == 2
    assert "An audio source and --sample cannot be combined." in result.output
    assert "starting browser login" not in result.output
    tx.assert_not_called()


def test_resolve_audio_source_rejects_directory(tmp_path):
    # Path(...).exists() is true for a directory; it must still be rejected up front.
    with pytest.raises(CLIError) as exc:
        client.resolve_audio_source(str(tmp_path), sample=False)
    assert exc.value.error_type == "not_a_file"
    assert exc.value.exit_code == 2
    assert exc.value.message == f"Not a file: {tmp_path}"
    assert exc.value.suggestion == "Pass an audio file, not a directory."


def test_transcribe_directory_source_fails_before_credentials(mocker, tmp_path):
    # No key configured: a directory must read as "not a file", never trigger a login
    # (or an upload attempt).
    tx = mocker.patch("aai_cli.commands.transcribe.client.transcribe", autospec=True)
    result = runner.invoke(app, ["transcribe", str(tmp_path)])
    assert result.exit_code == 2
    # Rich may wrap the long tmp path mid-message; compare on unwrapped text.
    unwrapped = " ".join(result.output.split())
    assert f"Not a file: {tmp_path}" in unwrapped
    assert "starting browser login" not in result.output
    tx.assert_not_called()


def test_resolve_audio_source_missing_local_file_fails_cleanly():
    with pytest.raises(CLIError) as exc:
        client.resolve_audio_source("no-such-clip.mp3", sample=False)
    assert exc.value.error_type == "file_not_found"
    assert exc.value.exit_code == 2
    assert exc.value.message == "File not found: no-such-clip.mp3"


def test_resolve_audio_source_skips_existence_check_for_urls_and_show_code():
    # URLs are not local paths; --show-code legitimately generates code for a file
    # the user does not have yet.
    url = "https://example.com/a.mp3"
    assert client.resolve_audio_source(url, sample=False) == url
    assert (
        client.resolve_audio_source("future.mp3", sample=False, check_local=False) == "future.mp3"
    )


def test_validate_transcript_id_rejects_path_segments():
    assert client.validate_transcript_id("t_42-abc") == "t_42-abc"
    for bad in ("../../etc/passwd", "", "a/b", "id?x=1"):
        with pytest.raises(UsageError):
            client.validate_transcript_id(bad)


def test_get_transcript_validates_id_before_request(mocker):
    get_by_id = mocker.patch.object(client.aai.Transcript, "get_by_id", autospec=True)
    with pytest.raises(UsageError):
        client.get_transcript("sk", "../../etc/passwd")
    get_by_id.assert_not_called()


def test_transcripts_get_rejects_path_traversal_id():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["transcripts", "get", "../../etc/passwd"])
    assert result.exit_code == 2
    assert "doesn't look like a transcript id" in result.output


def test_transcribe_missing_file_fails_before_credentials(mocker):
    # No key is configured: the path check must fire first, so the user sees
    # "file not found" instead of a login prompt (or a keyring error).
    tx = mocker.patch("aai_cli.commands.transcribe.client.transcribe", autospec=True)
    result = runner.invoke(app, ["transcribe", "missing.wav"])
    assert result.exit_code == 2
    assert "File not found: missing.wav" in result.output
    assert "starting browser login" not in result.output
    tx.assert_not_called()


def test_transcribe_requires_source():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["transcribe"])
    assert result.exit_code == 2


def test_transcribe_empty_stdin_exits_2():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["transcribe", "-"], input=b"")
    assert result.exit_code == 2  # nothing piped -> usage error


def test_stream_missing_file_fails_before_credentials(monkeypatch):
    called = {"stream": False}
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda *a, **k: called.__setitem__("stream", True),
    )
    result = runner.invoke(app, ["stream", "missing.wav"])
    assert result.exit_code == 2
    assert "File not found: missing.wav" in result.output
    assert "starting browser login" not in result.output
    assert called["stream"] is False


def test_agent_missing_file_fails_before_credentials():
    result = runner.invoke(app, ["agent", "missing.wav"])
    assert result.exit_code == 2
    assert "File not found: missing.wav" in result.output
    assert "starting browser login" not in result.output


def test_show_code_does_not_require_local_file_to_exist():
    # Generating code for audio you don't have yet is legitimate (check_local=False).
    result = runner.invoke(app, ["transcribe", "missing.wav", "--show-code"])
    assert result.exit_code == 0
    assert "missing.wav" in result.output
    assert "import assemblyai" in result.output
