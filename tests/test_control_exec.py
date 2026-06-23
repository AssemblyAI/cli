"""Tests for `assembly control` wiring: the native-helper transport and build,
the mic listener, and the command/`_run_control` seam.

All external legs are faked (see tests/_control_helpers.py); the pure loop,
actions, bridge, and rendering are covered by test_control.py.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from assemblyai.streaming.v3 import StreamingParameters
from typer.testing import CliRunner

from aai_cli.app.context import AppState
from aai_cli.commands.control import _exec as control_exec
from aai_cli.control import engine, helper, listen
from aai_cli.control.actions import Action
from aai_cli.core import config
from aai_cli.core.errors import APIError, CLIError
from aai_cli.main import app
from tests._control_helpers import (
    OPTS,
    BrokenStdin,
    FakeMic,
    FakeProc,
    RecordingHelper,
    deps_for,
    last_json,
    scripted,
)

# --- helper (native UI process transport) -------------------------------------


def test_build_helper_refuses_non_macos(monkeypatch):
    monkeypatch.setattr(helper, "_is_macos", lambda: False)
    with pytest.raises(CLIError, match="only available on macOS") as exc:
        helper.build_helper()
    assert exc.value.exit_code == 2


def test_build_helper_needs_swiftc(monkeypatch):
    monkeypatch.setattr(helper, "_is_macos", lambda: True)
    monkeypatch.setattr(helper.shutil, "which", lambda _name: None)
    with pytest.raises(CLIError, match="Swift compiler") as exc:
        helper.build_helper()
    assert exc.value.exit_code == 2


def test_execute_round_trips_one_action():
    proc = FakeProc(json.dumps({"ok": True, "elements": []}) + "\n")
    hands = helper.UiHelper(helper=Path("/fake/bin"), popen=lambda command: proc)
    result = hands.execute(Action("get_ui_tree", {}))
    assert result == {"ok": True, "elements": []}
    assert isinstance(proc.stdin, io.StringIO)
    assert json.loads(proc.stdin.getvalue()) == {"action": "get_ui_tree"}


def test_execute_raises_when_helper_closes_silently():
    hands = helper.UiHelper(helper=Path("/fake/bin"), popen=lambda command: FakeProc(""))
    with pytest.raises(APIError, match="closed without responding"):
        hands.execute(Action("get_ui_tree", {}))


def test_execute_raises_on_non_json_line():
    hands = helper.UiHelper(helper=Path("/fake/bin"), popen=lambda command: FakeProc("not-json\n"))
    with pytest.raises(APIError, match="non-JSON"):
        hands.execute(Action("get_ui_tree", {}))


def test_execute_treats_non_object_result_as_failure():
    hands = helper.UiHelper(helper=Path("/fake/bin"), popen=lambda command: FakeProc("[1, 2]\n"))
    assert hands.execute(Action("get_ui_tree", {})) == {
        "ok": False,
        "error": "helper returned a non-object result",
    }


def test_execute_raises_when_write_fails():
    proc = FakeProc("", stdin=BrokenStdin())
    hands = helper.UiHelper(helper=Path("/fake/bin"), popen=lambda command: proc)
    with pytest.raises(APIError, match="stopped accepting input"):
        hands.execute(Action("get_ui_tree", {}))


def test_execute_raises_when_streams_missing():
    proc = FakeProc("", stdin=None)
    proc.stdin = None
    hands = helper.UiHelper(helper=Path("/fake/bin"), popen=lambda command: proc)
    with pytest.raises(APIError, match="did not expose"):
        hands.execute(Action("get_ui_tree", {}))


def test_close_terminates_a_running_helper():
    proc = FakeProc(json.dumps({"ok": True}) + "\n")
    hands = helper.UiHelper(helper=Path("/fake/bin"), popen=lambda command: proc)
    hands.execute(Action("screenshot", {}))
    hands.close()
    assert proc.terminated is True
    hands.close()  # idempotent: nothing to do the second time


# --- helper build + spawn (macOS-only paths, mocked) --------------------------


def test_platform_and_resource_probes():
    # Compares against the live platform so the == (not !=) is pinned on any OS.
    assert helper._is_macos() == (sys.platform == "darwin")
    assert helper._resource_bytes().startswith(b"import AppKit")


def test_build_helper_compiles_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(helper, "_is_macos", lambda: True)
    monkeypatch.setattr(helper.shutil, "which", lambda _tool: "/usr/bin/swiftc")
    monkeypatch.setattr(helper, "_resource_bytes", lambda: b"swift source")
    monkeypatch.setattr(helper, "user_cache_path", lambda _app: tmp_path)
    captured_cmd: list[str] = []
    seen_kwargs: dict[str, object] = {}

    def fake_run(cmd, *, capture_output, text, check):
        captured_cmd[:] = cmd
        seen_kwargs.update(capture_output=capture_output, text=text, check=check)
        Path(cmd[-1]).write_bytes(b"binary")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    built = helper.build_helper()
    assert built.read_bytes() == b"binary"
    assert "-parse-as-library" in captured_cmd
    assert "AppKit" in captured_cmd
    # stderr/stdout captured as text; a non-zero compile is inspected, not raised.
    assert seen_kwargs["capture_output"] is True
    assert seen_kwargs["text"] is True
    assert seen_kwargs["check"] is False


def _compile_ok(cmd, **_kwargs):
    Path(cmd[-1]).write_bytes(b"bin")
    return SimpleNamespace(returncode=0, stderr="", stdout="")


def test_build_helper_creates_missing_cache_parents(monkeypatch, tmp_path):
    # The cache dir's parents may not exist; build_helper must create the whole chain.
    nested = tmp_path / "missing1" / "missing2"
    monkeypatch.setattr(helper, "_is_macos", lambda: True)
    monkeypatch.setattr(helper.shutil, "which", lambda _tool: "/usr/bin/swiftc")
    monkeypatch.setattr(helper, "_resource_bytes", lambda: b"swift source")
    monkeypatch.setattr(helper, "user_cache_path", lambda _app: nested)
    monkeypatch.setattr(helper.subprocess, "run", _compile_ok)
    assert helper.build_helper().read_bytes() == b"bin"


def test_build_helper_tolerates_existing_cache_dir(monkeypatch, tmp_path):
    # A rebuild runs with the cache dir already present, so its mkdir must tolerate it.
    monkeypatch.setattr(helper, "_is_macos", lambda: True)
    monkeypatch.setattr(helper.shutil, "which", lambda _tool: "/usr/bin/swiftc")
    monkeypatch.setattr(helper, "_resource_bytes", lambda: b"swift source")
    monkeypatch.setattr(helper, "user_cache_path", lambda _app: tmp_path)
    (tmp_path / "macos-ui-control").mkdir(parents=True)  # pre-exists
    monkeypatch.setattr(helper.subprocess, "run", _compile_ok)
    assert helper.build_helper().read_bytes() == b"bin"  # must not raise FileExistsError


def test_build_helper_reuses_cached_binary(monkeypatch, tmp_path):
    source = b"swift source"
    digest = helper.hashlib.sha256(source).hexdigest()[:16]
    cached = tmp_path / "macos-ui-control" / f"aai-macos-ui-control-{digest}"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")
    monkeypatch.setattr(helper, "_is_macos", lambda: True)
    monkeypatch.setattr(helper.shutil, "which", lambda _tool: "/usr/bin/swiftc")
    monkeypatch.setattr(helper, "_resource_bytes", lambda: source)
    monkeypatch.setattr(helper, "user_cache_path", lambda _app: tmp_path)

    def must_not_compile(*_a, **_k):
        raise AssertionError("a cached binary must not be recompiled")

    monkeypatch.setattr(helper.subprocess, "run", must_not_compile)
    assert helper.build_helper() == cached


def test_build_helper_compile_failure_surfaces_detail(monkeypatch, tmp_path):
    monkeypatch.setattr(helper, "_is_macos", lambda: True)
    monkeypatch.setattr(helper.shutil, "which", lambda _tool: "/usr/bin/swiftc")
    monkeypatch.setattr(helper, "_resource_bytes", lambda: b"swift source")
    monkeypatch.setattr(helper, "user_cache_path", lambda _app: tmp_path)
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="compile broke", stdout=""),
    )
    with pytest.raises(CLIError) as exc:
        helper.build_helper()
    assert exc.value.exit_code == 2
    assert exc.value.suggestion == "compile broke"


def test_open_process_wires_json_line_pipes(monkeypatch):
    captured_command: list[str] = []
    captured_kwargs: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured_command[:] = command
        captured_kwargs.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(helper.subprocess, "Popen", fake_popen)
    helper._open_process(["/bin/helper"])
    assert captured_command == ["/bin/helper"]
    assert captured_kwargs["text"] is True
    assert captured_kwargs["bufsize"] == 1
    assert captured_kwargs["stdin"] == helper.subprocess.PIPE
    assert captured_kwargs["stdout"] == helper.subprocess.PIPE


# --- listen (mic -> finalized utterances) -------------------------------------


def test_finalized_text_only_returns_finished_nonempty_turns():
    assert listen._finalized_text(SimpleNamespace(end_of_turn=True, transcript="hi")) == "hi"
    assert listen._finalized_text(SimpleNamespace(end_of_turn=False, transcript="partial")) is None
    assert listen._finalized_text(SimpleNamespace(end_of_turn=True, transcript="")) is None
    # No end_of_turn attribute defaults to "not finalized" -> None (not treated as done).
    assert listen._finalized_text(SimpleNamespace(transcript="hi")) is None


class _BareMic:
    """A mic with no sample_rate attribute, to exercise the rate fallback."""

    def __iter__(self) -> Iterator[bytes]:
        return iter(())


def test_listen_yields_finalized_utterances_with_mic_rate():
    seen_params: list[StreamingParameters] = []

    def fake_stream(api_key, source, *, params, on_turn):
        seen_params.append(params)
        on_turn(SimpleNamespace(end_of_turn=True, transcript="open safari"))
        on_turn(SimpleNamespace(end_of_turn=False, transcript="ignored partial"))
        on_turn(SimpleNamespace(end_of_turn=True, transcript="click go"))

    heard = list(listen.listen("k", stream=fake_stream, mic_factory=lambda **_k: FakeMic()))
    assert heard == ["open safari", "click go"]
    # Turn formatting is requested, and the mic's own rate is declared to the API.
    assert seen_params[0].format_turns is True
    assert seen_params[0].sample_rate == 16000


def test_listen_falls_back_to_explicit_rate_when_mic_lacks_one():
    seen_params: list[StreamingParameters] = []

    def fake_stream(api_key, source, *, params, on_turn):
        seen_params.append(params)

    list(
        listen.listen(
            "k", sample_rate=24000, stream=fake_stream, mic_factory=lambda **_k: _BareMic()
        )
    )
    assert seen_params[0].sample_rate == 24000


def test_listen_reraises_a_streaming_failure():
    def boom(api_key, source, *, params, on_turn):
        raise APIError("stream failed")

    with pytest.raises(APIError, match="stream failed"):
        list(listen.listen("k", stream=boom, mic_factory=lambda **_k: FakeMic()))


# --- _exec wiring -------------------------------------------------------------


def test_run_control_drives_a_session_and_closes_the_helper(capsys):
    config.set_api_key("default", "sk_live")
    hands = RecordingHelper()
    deps = deps_for(hands, transcripts=["say hi"], respond=scripted([engine.Reply("done", ())]))
    control_exec._run_control(OPTS, AppState(), json_mode=True, deps=deps)
    assert hands.closed is True
    assert last_json(capsys.readouterr().out) == {"type": "reply", "text": "done"}


def test_run_control_dry_run_refuses_mutation(capsys):
    config.set_api_key("default", "sk_live")
    hands = RecordingHelper()
    # The model keeps trying to type; --dry-run must refuse it every step.
    forever = engine.Reply("", (engine.ToolCall("c", "type_text", {"text": "x"}),))
    deps = deps_for(hands, transcripts=["type x"], respond=scripted([forever, forever]))
    opts = dataclasses.replace(OPTS, dry_run=True, max_steps=2)
    control_exec._run_control(opts, AppState(), json_mode=True, deps=deps)
    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert any(e["type"] == "refused" for e in events)


def test_run_control_closes_helper_even_if_a_leg_raises():
    config.set_api_key("default", "sk_live")
    hands = RecordingHelper()

    def explode(api_key: str, opts: control_exec.ControlOptions) -> list[str]:
        raise APIError("listen failed")

    deps = control_exec.ControlDeps(
        transcripts=explode,
        responder=lambda api_key, opts: scripted([engine.Reply("x", ())]),
        helper=lambda: hands,
    )
    with pytest.raises(APIError, match="listen failed"):
        control_exec._run_control(OPTS, AppState(), json_mode=False, deps=deps)
    assert hands.closed is True


def test_run_control_delegates_to_run_with_default_deps(monkeypatch):
    config.set_api_key("default", "sk_live")
    hands = RecordingHelper()
    deps = deps_for(hands, transcripts=[], respond=scripted([]))
    monkeypatch.setattr(control_exec, "_DEFAULT_DEPS", deps)
    control_exec.run_control(OPTS, AppState(), json_mode=True)
    assert hands.closed is True


def test_default_builders_construct_the_real_legs(monkeypatch):
    sentinel = helper.UiHelper.__new__(helper.UiHelper)
    monkeypatch.setattr(control_exec, "UiHelper", lambda: sentinel)
    assert control_exec._default_helper() is sentinel

    captured: dict[str, object] = {}

    def fake_build_responder(api_key: str, *, model: str, max_tokens: int) -> engine.Responder:
        captured["model"] = model
        captured["max_tokens"] = max_tokens
        return lambda messages: engine.Reply("", ())

    monkeypatch.setattr(control_exec.bridge, "build_responder", fake_build_responder)
    control_exec._default_responder("k", OPTS)
    assert captured == {"model": "m", "max_tokens": 8}

    monkeypatch.setattr(
        control_exec.listen_mod, "listen", lambda api_key, *, device, sample_rate: iter(["hi"])
    )
    assert list(control_exec._default_transcripts("k", OPTS)) == ["hi"]


# --- command body (Typer seam) ------------------------------------------------


def test_control_command_builds_options_and_runs(monkeypatch):
    config.set_api_key("default", "sk_live")
    hands = RecordingHelper()
    deps = deps_for(hands, transcripts=[], respond=scripted([]))
    monkeypatch.setattr(control_exec, "_DEFAULT_DEPS", deps)
    result = CliRunner().invoke(app, ["control", "--dry-run", "--max-steps", "3"])
    assert result.exit_code == 0
    assert hands.closed is True
