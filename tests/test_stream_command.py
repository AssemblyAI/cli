import json
import time
import types
from collections.abc import Callable

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.errors import APIError
from aai_cli.main import app

runner = CliRunner()


def _drive_turns(
    api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None, **_kwargs
):
    # Simulate the streaming client driving the renderer callbacks.
    if on_begin:
        on_begin(types.SimpleNamespace(id="sess"))
    if on_turn:
        on_turn(types.SimpleNamespace(transcript="hello world", end_of_turn=True))


def _login_result():
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=7
    )


def test_stream_help_lists_command():
    result = runner.invoke(app, ["stream", "--help"])
    assert result.exit_code == 0
    assert "microphone" in result.output.lower()


def test_stream_mic_renders_turns(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", _drive_turns)
    result = runner.invoke(app, ["stream", "--json"])
    assert result.exit_code == 0
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"type": "turn", "transcript": "hello world", "end_of_turn": True} in lines


def test_stream_file_uses_filesource(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None, **_kwargs
    ):
        seen["source_type"] = type(source).__name__
        seen["rate"] = params.sample_rate

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
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

    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", FakeMic)

    order = []

    def fake_stream_audio(api_key, source, *, params, on_begin=None, **_kwargs):
        if on_begin:
            on_begin(types.SimpleNamespace(id="x"))  # Begin must NOT print "Listening…"
        order.append("begin")
        list(source)  # consume the mic -> on_open fires -> "Listening…" prints
        order.append("consumed")

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0
    assert "Listening" in result.output  # shown once the mic opened
    assert callable(captured["on_open"])  # wired to the renderer's listening notice


def test_stream_file_shows_no_listening_notice(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)

    def fake(api_key, source, *, params, on_begin=None, **_kwargs):
        if on_begin:
            on_begin(types.SimpleNamespace(id="x"))

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
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
    monkeypatch.setattr("aai_cli.context.run_login_flow", _login_result)

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        raise AssertionError(f"streaming should not start after auto-login: {api_key}")

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "--json"])
    assert result.exit_code == 2
    assert config.get_api_key("default") == "sk_from_oauth"
    assert "Run the same command again" in result.output


def _capture_source(seen):
    def fake(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None, **_kwargs
    ):
        seen["source"] = source
        seen["rate"] = params.sample_rate

    return fake


def test_stream_sample_uses_hosted_clip(monkeypatch):
    from aai_cli import client

    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.streaming.sources.shutil.which", lambda _n: "/usr/bin/ffmpeg")
    seen = {}
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", _capture_source(seen))
    result = runner.invoke(app, ["stream", "--sample"])
    assert result.exit_code == 0
    assert type(seen["source"]).__name__ == "FileSource"
    assert seen["source"].source == client.SAMPLE_AUDIO_URL  # same clip as `transcribe --sample`
    assert seen["rate"] == 16000


def test_stream_url_source_uses_filesource(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.streaming.sources.shutil.which", lambda _n: "/usr/bin/ffmpeg")
    seen = {}
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", _capture_source(seen))
    result = runner.invoke(app, ["stream", "https://example.com/clip.mp3"])
    assert result.exit_code == 0
    assert type(seen["source"]).__name__ == "FileSource"
    assert seen["source"].source == "https://example.com/clip.mp3"


def test_stream_sample_with_sample_rate_rejected():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["stream", "--sample", "--sample-rate", "44100"])
    assert result.exit_code == 2  # mic-only flags don't apply to a file/sample source


def test_stream_ctrl_c_exits_cleanly(monkeypatch):
    config.set_api_key("default", "sk_live")

    def raise_kbd(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", raise_kbd)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0


def test_stream_ctrl_c_human_mode_prints_stopped(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)

    def raise_kbd(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", raise_kbd)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0
    assert "Stopped." in result.output


def test_stream_file_with_sample_rate_flag_rejected(tmp_path):
    config.set_api_key("default", "sk_live")
    import wave

    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    result = runner.invoke(app, ["stream", str(p), "--sample-rate", "44100"])
    assert result.exit_code == 2


def test_stream_broken_pipe_exits_zero(monkeypatch):
    config.set_api_key("default", "sk_live")

    def raise_broken_pipe(*a, **k):
        raise BrokenPipeError

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", raise_broken_pipe)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0


def test_stream_file_json_output(monkeypatch, tmp_path):
    import json as _json
    import wave

    config.set_api_key("default", "sk_live")

    def fake(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None, **_kwargs
    ):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="from file", end_of_turn=True))

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    result = runner.invoke(app, ["stream", str(p), "--json"])
    assert result.exit_code == 0
    lines = [_json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"type": "turn", "transcript": "from file", "end_of_turn": True} in lines


def test_stream_llm_refreshes_live_over_growing_transcript(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {"texts": []}

    def fake(api_key, source, *, params, on_turn=None, **kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="hola", end_of_turn=True))
            on_turn(types.SimpleNamespace(transcript="mundo", end_of_turn=True))
            on_turn(types.SimpleNamespace(transcript="partial", end_of_turn=False))  # ignored

    def fake_run_chain(api_key, prompts, *, transcript_text, model, max_tokens):
        seen["texts"].append(transcript_text)
        seen["prompts"] = prompts
        seen["model"] = model
        seen["max_tokens"] = max_tokens
        return f"answer:{transcript_text}"

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
    monkeypatch.setattr("aai_cli.commands.stream.llm.run_chain", fake_run_chain)
    result = runner.invoke(
        app,
        [
            "stream",
            "--llm",
            "translate to english",
            "--model",
            "gpt-4.1",
            "--max-tokens",
            "50",
            "--json",
        ],
    )
    assert result.exit_code == 0
    # One refresh per finalized turn, over the growing transcript (partials ignored).
    assert seen["texts"] == ["hola", "hola mundo"]
    assert seen["prompts"] == ["translate to english"]
    assert seen["model"] == "gpt-4.1"
    assert seen["max_tokens"] == 50
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"turns": 1, "output": "answer:hola"} in lines
    assert {"turns": 2, "output": "answer:hola mundo"} in lines
    # Live mode replaces the raw turn envelopes; only follow refreshes reach stdout.
    assert '"type"' not in result.output


def test_stream_llm_chains_multiple_prompts(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake(api_key, source, *, params, on_turn=None, **kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="hi", end_of_turn=True))

    def fake_run_chain(api_key, prompts, *, transcript_text, model, max_tokens):
        seen["prompts"] = prompts
        return "done"

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
    monkeypatch.setattr("aai_cli.commands.stream.llm.run_chain", fake_run_chain)
    result = runner.invoke(
        app, ["stream", "--llm", "summarize", "--llm", "translate to french", "--json"]
    )
    assert result.exit_code == 0
    assert seen["prompts"] == ["summarize", "translate to french"]


def test_stream_llm_rejects_output_text(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not stream")),
    )
    result = runner.invoke(app, ["stream", "--llm", "summarize", "-o", "text"])
    assert result.exit_code == 2  # --llm renders a panel/NDJSON; -o text is contradictory


def test_stream_without_prompt_does_not_transform(monkeypatch):
    config.set_api_key("default", "sk_live")
    called = {"ran": False}

    def fake(api_key, source, *, params, on_turn=None, **kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="hi", end_of_turn=True))

    def fake_run_chain(*a, **k):
        called["ran"] = True
        return "x"

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
    monkeypatch.setattr("aai_cli.commands.stream.llm.run_chain", fake_run_chain)
    result = runner.invoke(app, ["stream", "--json"])
    assert result.exit_code == 0
    assert called["ran"] is False  # no --llm -> no gateway call


def test_stream_prompt_biases_speech_model(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake(api_key, source, *, params, **kwargs):
        seen["prompt"] = params.prompt

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake)
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
    monkeypatch.setattr("aai_cli.commands.stream.youtube.download_audio", lambda url, d: fake)
    seen = {}

    def fake_stream(api_key, source, *, params, **kwargs):
        seen["source_type"] = type(source).__name__
        seen["src"] = getattr(source, "source", None)

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream)
    result = runner.invoke(app, ["stream", "https://youtu.be/abc"])
    assert result.exit_code == 0
    assert seen["source_type"] == "FileSource"  # streamed the downloaded local file
    assert seen["src"] == str(fake)


def test_stream_maps_turn_detection_flags(monkeypatch):
    config.set_api_key("default", "sk_live")
    captured = {}

    def fake_stream_audio(api_key, source, *, params, **kw):
        captured["params"] = params

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)

    runner.invoke(
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
    config.set_api_key("default", "sk_live")
    captured = {}
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda api_key, source, *, params, **kw: captured.update(params=params),
    )

    runner.invoke(app, ["stream", "--sample", "--config", "vad_threshold=0.7"])
    assert captured["params"].vad_threshold == 0.7


def test_stream_maps_webhook_auth_header(monkeypatch):
    config.set_api_key("default", "sk_live")
    captured = {}
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda api_key, source, *, params, **kw: captured.update(params=params),
    )

    runner.invoke(
        app,
        [
            "stream",
            "--sample",
            "--webhook-url",
            "https://example.com/hook",
            "--webhook-auth-header",
            "Authorization:Bearer xyz",
        ],
    )
    params = captured["params"]
    assert params.webhook_auth_header_name == "Authorization"
    assert params.webhook_auth_header_value == "Bearer xyz"


def test_stream_format_turns_tristate(monkeypatch):
    config.set_api_key("default", "sk_live")
    captured = {}
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda api_key, source, *, params, **kw: captured.update(params=params),
    )

    runner.invoke(app, ["stream", "--sample"])
    assert captured["params"].format_turns is True  # unset defaults to True

    runner.invoke(app, ["stream", "--sample", "--no-format-turns"])
    assert captured["params"].format_turns is False


def test_stream_show_code_prints_without_streaming(monkeypatch):
    # Print-only: emits the mic-streaming script, never opens audio or streams, no auth.
    called = []
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda *a, **k: called.append(True),
    )
    result = runner.invoke(app, ["stream", "--show-code"])
    assert result.exit_code == 0
    assert called == []  # never streamed
    assert "StreamingClient(" in result.output
    assert "MicrophoneStream(sample_rate=16000)" in result.output
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in result.output


def test_stream_show_code_ignores_json_flag(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not stream")

    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        _boom,
    )
    result = runner.invoke(app, ["stream", "--show-code", "--json"])
    assert result.exit_code == 0
    assert "StreamingClient(" in result.output


def test_stream_reads_raw_pcm_from_stdin(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_stream_audio(api_key, source, *, params, on_begin=None, **_kwargs):
        seen["rate"] = params.sample_rate
        seen["audio"] = b"".join(source)  # consume the StdinSource

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "-"], input=b"\x01\x02" * 100)
    assert result.exit_code == 0
    assert seen["rate"] == 16000  # default raw-PCM rate
    assert seen["audio"] == b"\x01\x02" * 100


def test_stream_stdin_rejects_device(monkeypatch):
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["stream", "-", "--device", "2"], input=b"\x00\x00")
    assert result.exit_code == 2  # --device applies only to the microphone


def test_stream_system_audio_uses_macos_source(monkeypatch):
    config.set_api_key("default", "sk_live")
    source_types: list[str] = []
    rates: list[int] = []
    mic_target_rate: list[int | None] = [None]
    system_on_open: list[Callable[[], None] | None] = [None]
    mic_on_open: list[Callable[[], None] | None] = [None]

    class FakeSystemAudio:
        def __init__(self, *, on_open=None):
            system_on_open[0] = on_open
            self.sample_rate = 16000

        def __iter__(self):
            if system_on_open[0] is not None:
                system_on_open[0]()
            return iter([b"system"])

    class FakeMic:
        def __init__(self, *, target_rate=None, device=None, capture_rate=None, on_open=None):
            mic_target_rate[0] = target_rate
            mic_on_open[0] = on_open
            self.sample_rate = 16000

        def __iter__(self):
            if mic_on_open[0] is not None:
                mic_on_open[0]()
            return iter([b"mic"])

    def fake_stream_audio(api_key, source, *, params, on_begin=None, on_turn=None, **_kwargs):
        source_type = type(source).__name__
        source_types.append(source_type)
        rates.append(params.sample_rate)
        if on_begin:
            on_begin(types.SimpleNamespace(id=source_type))
        list(source)
        if on_turn:
            on_turn(types.SimpleNamespace(transcript=source_type, end_of_turn=True))

    monkeypatch.setattr("aai_cli.commands.stream.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "--system-audio", "--json"])
    assert result.exit_code == 0
    assert set(source_types) == {"FakeSystemAudio", "FakeMic"}
    assert rates == [16000, 16000]
    assert mic_target_rate[0] == 16000
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {
        "type": "turn",
        "transcript": "FakeSystemAudio",
        "end_of_turn": True,
        "source": "system",
    } in lines
    assert {"type": "turn", "transcript": "FakeMic", "end_of_turn": True, "source": "you"} in lines


def test_stream_system_audio_only_disables_mic(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    class FakeSystemAudio:
        def __init__(self, *, on_open=None):
            self.sample_rate = 16000

        def __iter__(self):
            return iter([b"\x00\x00"])

    def fail_mic(**_kwargs):
        raise AssertionError("system-audio-only must not open the microphone")

    monkeypatch.setattr("aai_cli.commands.stream.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", fail_mic)
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", _capture_source(seen))
    result = runner.invoke(app, ["stream", "--system-audio-only", "--json"])
    assert result.exit_code == 0
    assert type(seen["source"]).__name__ == "FakeSystemAudio"


def test_stream_system_audio_rejects_other_sources():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["stream", "--system-audio", "--sample"])
    assert result.exit_code == 2
    assert "cannot be combined" in result.output


def test_stream_system_audio_forwards_mic_device_flags(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    class FakeSystemAudio:
        def __init__(self, *, on_open=None):
            self.sample_rate = 16000

        def __iter__(self):
            return iter([b"system"])

    class FakeMic:
        def __init__(self, *, target_rate=None, device=None, capture_rate=None, on_open=None):
            seen["target_rate"] = target_rate
            seen["device"] = device
            seen["capture_rate"] = capture_rate
            self.sample_rate = target_rate

        def __iter__(self):
            return iter([b"mic"])

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        list(source)

    monkeypatch.setattr("aai_cli.commands.stream.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    result = runner.invoke(
        app,
        ["stream", "--system-audio", "--device", "2", "--sample-rate", "44100", "--json"],
    )
    assert result.exit_code == 0
    assert seen == {"target_rate": 16000, "device": 2, "capture_rate": 44100}


def test_stream_system_audio_llm_prefixes_sources(monkeypatch):
    config.set_api_key("default", "sk_live")
    transcript_inputs = []

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

    def fake_stream_audio(api_key, source, *, params, on_turn=None, **_kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="", end_of_turn=True))
            on_turn(types.SimpleNamespace(transcript=type(source).__name__, end_of_turn=True))

    def fake_run_chain(api_key, prompts, *, transcript_text, model, max_tokens):
        transcript_inputs.append(transcript_text)
        return "summary"

    monkeypatch.setattr("aai_cli.commands.stream.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    monkeypatch.setattr("aai_cli.commands.stream.llm.run_chain", fake_run_chain)
    result = runner.invoke(app, ["stream", "--system-audio", "--llm", "summarize", "--json"])
    assert result.exit_code == 0
    assert any("System: FakeSystemAudio" in value for value in transcript_inputs)
    assert any("You: FakeMic" in value for value in transcript_inputs)


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

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        if type(source).__name__ == "FakeMic":
            raise APIError("mic failed")
        time.sleep(0.2)

    monkeypatch.setattr("aai_cli.commands.stream.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "--system-audio", "--json"])
    assert result.exit_code == 1
    assert "mic failed" in result.output


def test_stream_system_audio_parallel_final_worker_error_surfaces(monkeypatch):
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

    class ImmediateThread:
        def __init__(self, *, target, args, daemon):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        raise APIError(f"{type(source).__name__} failed")

    monkeypatch.setattr("aai_cli.commands.stream.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    monkeypatch.setattr("aai_cli.commands.stream.threading.Thread", ImmediateThread)
    result = runner.invoke(app, ["stream", "--system-audio", "--json"])
    assert result.exit_code == 1
    assert "failed" in result.output


def test_stream_system_audio_parallel_keyboard_interrupt_exits_cleanly(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)

    class FakeSystemAudio:
        def __init__(self, *, on_open=None):
            self.sample_rate = 16000

    class FakeMic:
        def __init__(self, *, target_rate=None, device=None, capture_rate=None, on_open=None):
            self.sample_rate = target_rate

    class InterruptingThread:
        def __init__(self, *, target, args, daemon):
            pass

        def start(self):
            raise KeyboardInterrupt

    monkeypatch.setattr("aai_cli.commands.stream.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream.threading.Thread", InterruptingThread)
    result = runner.invoke(app, ["stream", "--system-audio"])
    assert result.exit_code == 0
    assert "Stopped." in result.output


def test_stream_system_audio_parallel_broken_pipe_exits_zero(monkeypatch):
    config.set_api_key("default", "sk_live")

    class FakeSystemAudio:
        def __init__(self, *, on_open=None):
            self.sample_rate = 16000

    class FakeMic:
        def __init__(self, *, target_rate=None, device=None, capture_rate=None, on_open=None):
            self.sample_rate = target_rate

    class BrokenPipeThread:
        def __init__(self, *, target, args, daemon):
            pass

        def start(self):
            raise BrokenPipeError

    monkeypatch.setattr("aai_cli.commands.stream.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream.threading.Thread", BrokenPipeThread)
    result = runner.invoke(app, ["stream", "--system-audio"])
    assert result.exit_code == 0


def test_stream_system_audio_only_rejects_mic_device_flags():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["stream", "--system-audio-only", "--device", "2"])
    assert result.exit_code == 2
    assert "--device" in result.output

    result = runner.invoke(app, ["stream", "--system-audio-only", "--sample-rate", "44100"])
    assert result.exit_code == 2
    assert "--sample-rate" in result.output


def test_stream_system_audio_rejects_both_modes():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["stream", "--system-audio", "--system-audio-only"])
    assert result.exit_code == 2
    assert "either --system-audio" in result.output


def test_stream_show_code_rejects_system_audio():
    result = runner.invoke(app, ["stream", "--system-audio", "--show-code"])
    assert result.exit_code == 2
    assert "--show-code" in result.output


def test_stream_output_text_emits_plain_finalized_turns(monkeypatch):
    # `-o text` -> only finalized transcripts as plain stdout lines (pipe into aai llm).
    config.set_api_key("default", "sk_live")

    def fake_stream_audio(api_key, source, *, params, on_begin=None, on_turn=None, **_kwargs):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="partial", end_of_turn=False))
            on_turn(types.SimpleNamespace(transcript="hello world", end_of_turn=True))

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "-", "-o", "text"], input=b"\x00\x00")
    assert result.exit_code == 0
    # Final turn only, plain text; partials and JSON envelopes are not on stdout.
    assert result.output.strip() == "hello world"
    assert '"type"' not in result.output


def test_stream_show_code_with_llm_emits_follow_loop(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not stream")

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", _boom)
    result = runner.invoke(app, ["stream", "--llm", "summarize", "--show-code"])
    assert result.exit_code == 0
    assert "from openai import OpenAI" in result.output
    assert "summarize" in result.output
    assert "run_chain" in result.output  # the live transcribe->LLM-per-turn loop
