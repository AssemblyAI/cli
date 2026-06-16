"""`assembly stream` system-audio capture tests (macOS parallel mic + system sessions).

Split out of test_stream_session.py: the --system-audio family is a cohesive block
(parallel sessions, worker-error propagation, the mic/system source wiring) large
enough to live on its own under the 500-line file gate.
"""

import json
import types
import wave
from collections.abc import Callable
from datetime import datetime

from typer.testing import CliRunner

from aai_cli.core import config
from aai_cli.core.errors import APIError
from aai_cli.main import app

runner = CliRunner()


class _FixedDatetime:
    """Freezes datetime.now() so an auto-assembled --save-dir filename is deterministic."""

    @staticmethod
    def now(*_args, **_kwargs):
        # Naive local wall-clock; _exec's .astimezone() keeps the same 14:30:05.
        return datetime(2026, 6, 16, 14, 30, 5)


def _wav_frames(path):
    """The raw PCM frames written to a tee'd WAV, for asserting per-channel contents."""
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes())


def _capture_source(seen):
    def fake(api_key, source, *, params, on_begin=None, on_turn=None, on_termination=None):
        seen["source"] = source
        seen["rate"] = params.sample_rate

    return fake


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

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", fake_stream_audio)
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

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", fail_mic)
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", _capture_source(seen))
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

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", fake_stream_audio)
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

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", fake_stream_audio)
    monkeypatch.setattr("aai_cli.core.llm.run_chain", fake_run_chain)
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

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", fake_stream_audio)
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

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", fake_stream_audio)
    monkeypatch.setattr("aai_cli.streaming.session.threading.Thread", ImmediateThread)
    result = runner.invoke(app, ["stream", "--system-audio", "--json"])
    assert result.exit_code == 1
    assert "failed" in result.output
    # Both source workers run as daemons so a wedged stream can't block process exit.
    assert daemons and all(d is True for d in daemons)


def test_stream_system_audio_parallel_unexpected_worker_error_fails_the_run(monkeypatch):
    # A non-CLIError bug inside a worker must still fail the run with a clean error:
    # uncaught, it would die with the daemon thread and the command would exit 0
    # for a stream that actually failed.
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
        raise RuntimeError("event parsing blew up")

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", fake_stream_audio)
    monkeypatch.setattr("aai_cli.streaming.session.threading.Thread", ImmediateThread)
    result = runner.invoke(app, ["stream", "--system-audio", "--json"])
    assert result.exit_code == 1
    # Normalized to a clean worker error that names the source and the cause.
    assert "Streaming worker" in result.output
    assert "event parsing blew up" in result.output
    assert "Traceback" not in result.output


def test_stream_system_audio_parallel_keyboard_interrupt_exits_cleanly(monkeypatch):
    config.set_api_key("default", "sk_live")

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

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", FakeMic)
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

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", FakeMic)
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
    assert "--system-audio and --system-audio-only can't be combined." in result.output


def test_stream_show_code_rejects_system_audio():
    result = runner.invoke(app, ["stream", "--system-audio", "--show-code"])
    assert result.exit_code == 2
    assert "--show-code" in result.output


def test_stream_system_audio_save_dir_writes_one_wav_per_channel(monkeypatch, tmp_path):
    # --save-dir + --system-audio can't tee two streams into one WAV, so each channel
    # gets its own <stem>-{you,system}.wav beside the single shared transcript.
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.commands.stream._exec.datetime", _FixedDatetime)

    class FakeSystemAudio:
        def __init__(self, *, on_open=None):
            self.sample_rate = 16000

        def __iter__(self):
            return iter([b"\x10\x11\x12\x13"])

    class FakeMic:
        def __init__(self, *, target_rate=None, device=None, capture_rate=None, on_open=None):
            self.sample_rate = 16000

        def __iter__(self):
            return iter([b"\x20\x21\x22\x23"])

    def fake_stream_audio(api_key, source, *, params, on_turn=None, **_kwargs):
        b"".join(source)  # draining the tee'd generator is what writes the channel WAV
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="hi", end_of_turn=True, speaker_label=None))

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", FakeMic)
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", fake_stream_audio)

    result = runner.invoke(
        app,
        ["stream", "--system-audio", "--save-dir", str(tmp_path / "rec"), "--name", "Irma", "-j"],
    )
    assert result.exit_code == 0
    bucket = tmp_path / "rec" / "2026-06-16"
    assert _wav_frames(bucket / "2026-06-16-143005-irma-you.wav") == b"\x20\x21\x22\x23"
    assert _wav_frames(bucket / "2026-06-16-143005-irma-system.wav") == b"\x10\x11\x12\x13"
    # One shared transcript carries both channels' turns, each with its source prefix.
    transcript = (bucket / "2026-06-16-143005-irma.txt").read_text(encoding="utf-8")
    assert "You: hi" in transcript
    assert "System: hi" in transcript


def test_stream_system_audio_only_save_dir_writes_one_labeled_wav(monkeypatch, tmp_path):
    # A lone --system-audio-only stream saves to a single channel-labeled WAV (never the
    # bare <stem>.wav a mic recording uses) and still never opens the microphone.
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.commands.stream._exec.datetime", _FixedDatetime)

    class FakeSystemAudio:
        def __init__(self, *, on_open=None):
            self.sample_rate = 16000

        def __iter__(self):
            return iter([b"\x30\x31\x32\x33"])

    def fail_mic(**_kwargs):
        raise AssertionError("system-audio-only must not open the microphone")

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        b"".join(source)

    monkeypatch.setattr("aai_cli.commands.stream._exec.MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr("aai_cli.commands.stream._exec.MicrophoneSource", fail_mic)
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", fake_stream_audio)

    result = runner.invoke(
        app, ["stream", "--system-audio-only", "--save-dir", str(tmp_path / "rec"), "-j"]
    )
    assert result.exit_code == 0
    bucket = tmp_path / "rec" / "2026-06-16"
    assert _wav_frames(bucket / "2026-06-16-143005-system.wav") == b"\x30\x31\x32\x33"
    assert not (bucket / "2026-06-16-143005.wav").exists()
