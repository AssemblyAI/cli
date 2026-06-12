"""`assembly stream` source/streaming behavior.

Flag-to-params mapping and conflicting-flag validation live in
test_stream_command_flags.py; --show-code print-only behavior lives in
test_stream_show_code.py.
"""

import json
import re
import time
import types

import pytest
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.errors import APIError
from aai_cli.main import app

runner = CliRunner()


def _drive_turns(api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None):
    # Simulate the streaming client driving the renderer callbacks.
    if on_begin:
        on_begin(types.SimpleNamespace(id="sess"))
    if on_turn:
        on_turn(types.SimpleNamespace(transcript="hello world", end_of_turn=True))


def _login_result(*, json_mode=False):
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=7
    )


def test_stream_help_lists_command():
    result = runner.invoke(app, ["stream", "--help"])
    assert result.exit_code == 0
    assert "microphone" in result.output.lower()


def test_stream_mic_renders_turns(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", _drive_turns)
    result = runner.invoke(app, ["stream", "--json"])
    assert result.exit_code == 0
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"type": "turn", "transcript": "hello world", "end_of_turn": True} in lines


def test_stream_file_uses_filesource(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
        seen["source_type"] = type(source).__name__
        seen["rate"] = params.sample_rate

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake_stream_audio)
    import wave

    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    result = runner.invoke(app, ["stream", str(p)])
    assert result.exit_code == 0
    assert seen["source_type"] == "FileSource"
    assert seen["rate"] == 16000


def test_stream_mic_listening_notice_waits_for_mic_open(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)

    captured = {}

    class FakeMic:
        def __init__(self, *, device=None, capture_rate=None, on_open=None):
            captured["on_open"] = on_open
            self.sample_rate = 16000

        def __iter__(self):
            captured["on_open"]()  # the SDK iterating us == the mic is now live
            return iter([b"\x00\x00"])

    monkeypatch.setattr("aai_cli.stream_exec.MicrophoneSource", FakeMic)

    order = []

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
        if on_begin:
            on_begin(types.SimpleNamespace(id="x"))  # Begin must NOT print "Listening…"
        order.append("begin")
        list(source)  # consume the mic -> on_open fires -> "Listening…" prints
        order.append("consumed")

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0
    assert "Listening" in result.output  # shown once the mic opened
    assert callable(captured["on_open"])  # wired to the renderer's listening notice


def test_stream_file_shows_no_listening_notice(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)

    def fake(api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None):
        if on_begin:
            on_begin(types.SimpleNamespace(id="x"))

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake)
    import wave

    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    result = runner.invoke(app, ["stream", str(p)])
    assert result.exit_code == 0
    assert "Listening" not in result.output  # no mic -> no listening notice


def test_stream_unauthenticated_runs_login(monkeypatch):
    monkeypatch.setattr("aai_cli.context._interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.context.run_login_flow", _login_result)

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
        raise AssertionError(f"streaming should not start after auto-login: {api_key}")

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "--json"])
    assert result.exit_code == 4
    assert config.get_api_key("default") == "sk_from_oauth"
    assert "Run the same command again" in result.output


def _capture_source(seen):
    def fake(api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None):
        seen["source"] = source
        seen["rate"] = params.sample_rate

    return fake


def test_stream_sample_uses_hosted_clip(monkeypatch):
    from aai_cli import client

    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.streaming.sources.shutil.which", lambda _n: "/usr/bin/ffmpeg")
    seen = {}
    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", _capture_source(seen))
    result = runner.invoke(app, ["stream", "--sample"])
    assert result.exit_code == 0
    assert type(seen["source"]).__name__ == "FileSource"
    assert seen["source"].source == client.SAMPLE_AUDIO_URL  # same clip as `transcribe --sample`
    assert seen["rate"] == 16000


def test_stream_url_source_uses_filesource(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.streaming.sources.shutil.which", lambda _n: "/usr/bin/ffmpeg")
    seen = {}
    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", _capture_source(seen))
    result = runner.invoke(app, ["stream", "https://example.com/clip.mp3"])
    assert result.exit_code == 0
    assert type(seen["source"]).__name__ == "FileSource"
    assert seen["source"].source == "https://example.com/clip.mp3"


def test_stream_ctrl_c_exits_cleanly(monkeypatch):
    config.set_api_key("default", "sk_live")

    def raise_kbd(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", raise_kbd)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0


def test_stream_ctrl_c_human_mode_prints_stopped(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)

    def raise_kbd(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", raise_kbd)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0
    assert "Stopped." in result.output


def test_stream_broken_pipe_exits_zero(monkeypatch):
    config.set_api_key("default", "sk_live")

    def raise_broken_pipe(*a, **k):
        raise BrokenPipeError

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", raise_broken_pipe)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0


def test_stream_file_json_output(monkeypatch, tmp_path):
    import json as _json
    import wave

    config.set_api_key("default", "sk_live")

    def fake(api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None):
        # In non-follow mode begin/turn/termination must all be wired through to the
        # renderer (pins the `follow is not None` None-vs-handler choices).
        if on_begin:
            on_begin(types.SimpleNamespace(id="sess_1"))
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="from file", end_of_turn=True))
        if on_termination:
            on_termination(types.SimpleNamespace(audio_duration_seconds=2.0))

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake)
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    result = runner.invoke(app, ["stream", str(p), "--json"])
    assert result.exit_code == 0
    lines = [_json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"type": "begin", "id": "sess_1"} in lines
    assert {"type": "turn", "transcript": "from file", "end_of_turn": True} in lines
    assert {"type": "termination", "audio_duration_seconds": 2.0} in lines


def test_stream_prompt_biases_speech_model(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake(api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None):
        seen["prompt"] = params.prompt

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake)
    result = runner.invoke(app, ["stream", "--prompt", "expect crypto jargon", "--json"])
    assert result.exit_code == 0
    # --prompt is the speech-model prompt, forwarded to the streaming session.
    assert seen["prompt"] == "expect crypto jargon"


def test_stream_youtube_url_downloads_then_streams(monkeypatch, tmp_path):
    import wave

    config.set_api_key("default", "sk_live")
    fake = tmp_path / "vid.wav"
    with wave.open(str(fake), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    monkeypatch.setattr("aai_cli.stream_exec.youtube.download_audio", lambda url, d: fake)
    seen = {}

    def fake_stream(api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None):
        seen["source_type"] = type(source).__name__
        seen["src"] = getattr(source, "source", None)

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake_stream)
    result = runner.invoke(app, ["stream", "https://youtu.be/abc"])
    assert result.exit_code == 0
    assert seen["source_type"] == "FileSource"  # streamed the downloaded local file
    assert seen["src"] == str(fake)


def test_stream_podcast_page_url_downloads_then_streams(monkeypatch, tmp_path):
    import wave

    config.set_api_key("default", "sk_live")
    fake = tmp_path / "episode.wav"
    with wave.open(str(fake), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    monkeypatch.setattr("aai_cli.stream_exec.youtube.download_audio", lambda url, d: fake)
    seen = {}

    def fake_stream(api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None):
        seen["source_type"] = type(source).__name__
        seen["src"] = getattr(source, "source", None)

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake_stream)
    result = runner.invoke(app, ["stream", "https://www.spreaker.com/episode/12345"])
    assert result.exit_code == 0
    assert seen["source_type"] == "FileSource"  # streamed the downloaded local file
    assert seen["src"] == str(fake)


def test_stream_downloadable_url_resolves_credentials_before_downloading(monkeypatch):
    # Regression guard for ordering: with no usable credential the command must fail
    # authentication *before* yt-dlp runs, so a signed-out user never downloads a
    # whole video only to be told to log in (mirrors transcribe's source -> auth ->
    # work ordering).
    monkeypatch.setattr("aai_cli.context._interactive_session", lambda: False)
    downloads = []
    monkeypatch.setattr(
        "aai_cli.stream_exec.youtube.download_audio",
        lambda url, dest: downloads.append(url),
    )
    monkeypatch.setattr(
        "aai_cli.stream_exec.client.stream_audio",
        lambda *a, **k: pytest.fail("must not stream without credentials"),
    )
    result = runner.invoke(app, ["stream", "https://youtu.be/abc"])
    assert result.exit_code == 4  # not authenticated
    assert downloads == []  # nothing was fetched before the credential check


def test_stream_sample_rate_must_be_positive():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["stream", "--sample-rate", "0"])
    assert result.exit_code == 2
    # CI forces color on (Rich under GITHUB_ACTIONS), interleaving style codes
    # mid-message, so assert on the color-free render (see test_help_rendering.py).
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--sample-rate" in plain


def test_stream_sample_rate_floor_accepts_one_for_stdin(monkeypatch):
    # min=1 exactly — and --sample-rate also declares the rate of raw PCM piped on
    # stdin (it is not mic-only), so the declared value must reach the session params.
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
        seen["rate"] = params.sample_rate
        b"".join(source)  # drain the StdinSource

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "-", "--sample-rate", "1"], input=b"\x00\x00")
    assert result.exit_code == 0
    assert seen["rate"] == 1


def test_stream_reads_raw_pcm_from_stdin(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
        seen["rate"] = params.sample_rate
        seen["audio"] = b"".join(source)  # consume the StdinSource

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "-"], input=b"\x01\x02" * 100)
    assert result.exit_code == 0
    assert seen["rate"] == 16000  # default raw-PCM rate
    assert seen["audio"] == b"\x01\x02" * 100


def test_stream_system_audio_parallel_worker_error_surfaces(monkeypatch):
    config.set_api_key("default", "sk_live")

    class FakeSystemAudio:
        def __init__(self, *, on_open=None):
            self.sample_rate = 16000

        def __iter__(self):
            return iter([b"system"])

    class FakeMic:
        def __init__(self, *, target_rate=None, device=None, capture_rate=None, on_open=None):
            self.sample_rate = target_rate

        def __iter__(self):
            return iter([b"mic"])

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
        if type(source).__name__ == "FakeMic":
            raise APIError("mic failed")
        time.sleep(0.2)

    monkeypatch.setattr("aai_cli.stream_exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.stream_exec.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "--system-audio", "--json"])
    assert result.exit_code == 1
    assert "mic failed" in result.output


def test_stream_output_text_emits_plain_finalized_turns(monkeypatch):
    # `-o text` -> only finalized transcripts as plain stdout lines (pipe into assembly llm).
    config.set_api_key("default", "sk_live")

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="partial", end_of_turn=False))
            on_turn(types.SimpleNamespace(transcript="hello world", end_of_turn=True))

    monkeypatch.setattr("aai_cli.stream_exec.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "-", "-o", "text"], input=b"\x00\x00")
    assert result.exit_code == 0
    # Final turn only, plain text; partials and JSON envelopes are not on stdout.
    assert result.output.strip() == "hello world"
    assert '"type"' not in result.output
