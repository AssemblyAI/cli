"""Batch streaming: `assembly stream --from-stdin` reads a list of audio files/URLs
on stdin and streams each as its own realtime session, in turn.

These drive the whole command through CliRunner with the file-resolution/streaming
boundary faked, so no real audio (or network) is needed — the focus is the
sequencing, the per-source failure handling, and the Ctrl-C/broken-pipe lifecycle
that distinguish the batch driver from a single stream.
"""

import json
import types

from typer.testing import CliRunner

from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()


class _FakeFile:
    """A FileSource stand-in: yields one chunk; raises CLIError for a named source."""

    def __init__(self, source):
        from aai_cli.core.errors import CLIError

        if source == "missing.wav":
            raise CLIError(f"File not found: {source}", error_type="file_not_found", exit_code=2)
        self.source = source
        self.sample_rate = 16000

    def __iter__(self):
        return iter([b"\x00\x00"])


def _patch_batch_inputs(monkeypatch, fake_stream_audio):
    """Wire the file-resolution/streaming boundary so --from-stdin needs no real files."""
    monkeypatch.setattr("aai_cli.commands.stream._exec.FileSource", _FakeFile)
    monkeypatch.setattr(
        "aai_cli.commands.stream._exec.client.resolve_audio_source",
        lambda source, *, sample=False: source,
    )
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", fake_stream_audio)


def test_stream_from_stdin_streams_each_source_in_turn(monkeypatch):
    config.set_api_key("default", "sk_live")
    streamed = []

    def fake_stream_audio(api_key, source, *, params, on_turn=None, **_kwargs):
        streamed.append(source.source)
        if on_turn:
            on_turn(types.SimpleNamespace(transcript=source.source, end_of_turn=True))

    _patch_batch_inputs(monkeypatch, fake_stream_audio)
    result = runner.invoke(app, ["stream", "--from-stdin", "--json"], input="a.wav\nb.wav\n")
    assert result.exit_code == 0
    # Sequential, in stdin order — the realtime API is one session at a time.
    assert streamed == ["a.wav", "b.wav"]
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"type": "source", "source": "a.wav", "index": 1, "total": 2} in lines
    assert {"type": "source", "source": "b.wav", "index": 2, "total": 2} in lines
    assert {"type": "turn", "transcript": "a.wav", "end_of_turn": True} in lines


def test_stream_from_stdin_resolves_each_source_not_the_hosted_sample(monkeypatch):
    # --from-stdin sources are real files/URLs, so each is resolved with sample=False
    # — never coerced to the hosted --sample clip.
    config.set_api_key("default", "sk_live")
    sample_flags = []

    def recording_resolve(source, *, sample=False):
        sample_flags.append(sample)
        return source

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        pass

    monkeypatch.setattr("aai_cli.commands.stream._exec.FileSource", _FakeFile)
    monkeypatch.setattr(
        "aai_cli.commands.stream._exec.client.resolve_audio_source", recording_resolve
    )
    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", fake_stream_audio)
    result = runner.invoke(app, ["stream", "--from-stdin"], input="a.wav\nb.wav\n")
    assert result.exit_code == 0
    assert sample_flags == [False, False]


def test_stream_from_stdin_failed_source_is_recorded_and_batch_continues(monkeypatch):
    config.set_api_key("default", "sk_live")
    streamed = []

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        streamed.append(source.source)

    _patch_batch_inputs(monkeypatch, fake_stream_audio)
    result = runner.invoke(app, ["stream", "--from-stdin"], input="missing.wav\nb.wav\n")
    # The good source still streamed; the batch fails (exit 1) because one source did.
    assert streamed == ["b.wav"]
    assert result.exit_code == 1
    assert "1 of 2 sources failed" in result.output
    assert "missing.wav" in result.output


def test_stream_from_stdin_not_authenticated_aborts_the_whole_batch(monkeypatch):
    config.set_api_key("default", "sk_live")
    from aai_cli.core.errors import NotAuthenticated

    streamed = []

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        streamed.append(source.source)
        raise NotAuthenticated("rejected")

    _patch_batch_inputs(monkeypatch, fake_stream_audio)
    result = runner.invoke(app, ["stream", "--from-stdin"], input="a.wav\nb.wav\n")
    # One rejected key fails every source identically, so the batch aborts at the first.
    assert streamed == ["a.wav"]
    assert result.exit_code != 0


def test_stream_from_stdin_keyboard_interrupt_stops_the_batch(monkeypatch):
    config.set_api_key("default", "sk_live")
    streamed = []

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        streamed.append(source.source)
        raise KeyboardInterrupt

    _patch_batch_inputs(monkeypatch, fake_stream_audio)
    result = runner.invoke(app, ["stream", "--from-stdin"], input="a.wav\nb.wav\n")
    # One Ctrl-C stops the whole batch (exit 0), not just the current source.
    assert streamed == ["a.wav"]
    assert result.exit_code == 0
    assert "Stopped." in result.output


def test_stream_from_stdin_broken_pipe_exits_zero(monkeypatch):
    config.set_api_key("default", "sk_live")
    streamed = []

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        streamed.append(source.source)
        raise BrokenPipeError

    _patch_batch_inputs(monkeypatch, fake_stream_audio)
    result = runner.invoke(app, ["stream", "--from-stdin"], input="a.wav\nb.wav\n")
    assert streamed == ["a.wav"]
    assert result.exit_code == 0


def test_stream_from_stdin_empty_pipe_is_a_usage_error(monkeypatch):
    config.set_api_key("default", "sk_live")

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        raise AssertionError("nothing should stream from an empty pipe")

    _patch_batch_inputs(monkeypatch, fake_stream_audio)
    result = runner.invoke(app, ["stream", "--from-stdin"], input="")
    assert result.exit_code == 2
    assert "No sources received on stdin" in result.output


def test_stream_batch_sources_reports_exact_failure_count():
    # Every source failing raises a CLIError naming the exact count (pins the `failures +=`
    # accumulator: a sign/operator slip would report a wrong total like "-2 of 2").
    import io

    import pytest

    from aai_cli.core.errors import CLIError
    from aai_cli.streaming.batch import stream_batch_sources
    from aai_cli.streaming.render import StreamRenderer

    def open_source(source):
        raise CLIError(f"nope: {source}", error_type="file_not_found", exit_code=2)

    def make_session():
        raise AssertionError("a failed open must short-circuit before a session opens")

    with pytest.raises(CLIError) as excinfo:
        stream_batch_sources(
            ["a.wav", "b.wav"],
            make_session=make_session,
            open_source=open_source,
            renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
            json_mode=True,
        )
    assert excinfo.value.message == "2 of 2 sources failed."


def test_stream_source_strips_newlines_from_failure_warning(capsys):
    # A crafted source/error with embedded CR/LF must not inject extra log lines: the
    # emitted warning is flattened to a single line (pins the newline sanitization).
    import io
    import json as _json

    from aai_cli.core.errors import CLIError
    from aai_cli.streaming import batch
    from aai_cli.streaming.render import StreamRenderer

    def open_source(source):
        raise CLIError("bad\ninjected line")

    failed = batch._stream_source(
        "a\nb.wav",
        index=1,
        total=1,
        make_session=lambda: (_ for _ in ()).throw(AssertionError("never")),
        open_source=open_source,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        json_mode=True,
    )
    assert failed is True
    warning = _json.loads(capsys.readouterr().err.strip())["warning"]
    assert warning == "a b.wav: bad injected line"
    assert "\n" not in warning
