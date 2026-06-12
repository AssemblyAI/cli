import json
import types
from collections.abc import Callable

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.errors import APIError
from aai_cli.main import app

runner = CliRunner()


def _capture_source(seen):
    def fake(api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None):
        seen["source"] = source
        seen["rate"] = params.sample_rate

    return fake


def test_stream_session_listening_notice_latches(monkeypatch):
    # _listening_once must announce "Listening…" exactly once even if the first-audio
    # callback fires repeatedly (pins the `self._listening_started = True` latch).
    import io

    from aai_cli.commands.stream import StreamSession
    from aai_cli.streaming.render import StreamRenderer

    renderer = StreamRenderer(json_mode=False, out=io.StringIO())
    calls = {"n": 0}
    monkeypatch.setattr(renderer, "listening", lambda: calls.__setitem__("n", calls["n"] + 1))
    session = StreamSession(
        api_key="sk",
        base_flags={},
        overrides=None,
        config_file=None,
        renderer=renderer,
        follow=None,
        llm_prompts=[],
        model="m",
        max_tokens=1,
    )
    session._listening_once()
    session._listening_once()
    assert calls["n"] == 1


def test_stream_session_closes_renderer_on_error(monkeypatch):
    # When streaming raises mid-run, the live region must still be torn down (pins the
    # `if self.follow is None: self.renderer.close()` in the finally block).
    import io

    import pytest

    from aai_cli.commands.stream import StreamSession
    from aai_cli.errors import CLIError
    from aai_cli.streaming.render import StreamRenderer

    renderer = StreamRenderer(json_mode=False, out=io.StringIO())
    closed = {"n": 0}
    monkeypatch.setattr(renderer, "close", lambda: closed.__setitem__("n", closed["n"] + 1))

    def boom(*_args, **_kwargs):
        raise CLIError("stream blew up")

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", boom)
    session = StreamSession(
        api_key="sk",
        base_flags={},
        overrides=None,
        config_file=None,
        renderer=renderer,
        follow=None,
        llm_prompts=[],
        model="m",
        max_tokens=1,
    )
    with pytest.raises(CLIError):
        session.run([b"\x00"], 16000)
    assert closed["n"] >= 1


def test_stream_system_audio_uses_macos_source(monkeypatch) -> None:
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

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
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

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
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

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
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


def test_stream_system_audio_speaker_labels_only_diarizes_system(monkeypatch):
    # --speaker-labels diarizes the system audio but never the mic: the "you" session
    # is forced to speaker_labels=False so the mic stays a single "You".
    config.set_api_key("default", "sk_live")
    speaker_labels_by_chunk = {}

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
        chunk = next(iter(source))
        speaker_labels_by_chunk[chunk] = params.speaker_labels

    monkeypatch.setattr("aai_cli.commands.stream.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "--system-audio", "--speaker-labels", "--json"])
    assert result.exit_code == 0
    assert speaker_labels_by_chunk[b"system"] is True
    assert speaker_labels_by_chunk[b"mic"] is False


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

    daemons = []

    class ImmediateThread:
        def __init__(self, *, target, args, daemon):
            self._target = target
            self._args = args
            daemons.append(daemon)

        def start(self):
            self._target(*self._args)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    def fake_stream_audio(
        api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None
    ):
        raise APIError(f"{type(source).__name__} failed")

    monkeypatch.setattr("aai_cli.commands.stream.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    monkeypatch.setattr("aai_cli.streaming.session.threading.Thread", ImmediateThread)
    result = runner.invoke(app, ["stream", "--system-audio", "--json"])
    assert result.exit_code == 1
    assert "failed" in result.output
    # Both source workers run as daemons so a wedged stream can't block process exit.
    assert daemons and all(d is True for d in daemons)


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
    monkeypatch.setattr("aai_cli.streaming.session.threading.Thread", InterruptingThread)
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
    monkeypatch.setattr("aai_cli.streaming.session.threading.Thread", BrokenPipeThread)
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
