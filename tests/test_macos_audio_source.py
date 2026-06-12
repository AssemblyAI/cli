import io
import types
from pathlib import Path

import pytest

from aai_cli.errors import APIError, CLIError
from aai_cli.streaming import macos
from aai_cli.streaming.sources import CHUNK_BYTES


class _FakeProc:
    def __init__(
        self,
        *,
        stdout: bytes | None = b"",
        stderr: bytes | None = b"",
        returncode: int | None = None,
    ):
        self.stdout = io.BytesIO(stdout) if stdout is not None else None
        self.stderr = io.BytesIO(stderr) if stderr is not None else None
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode


def test_build_helper_rejects_non_macos(monkeypatch):
    monkeypatch.setattr(macos.sys, "platform", "linux")
    with pytest.raises(CLIError) as exc:
        macos.build_helper()
    assert exc.value.error_type == "mac_system_audio_unavailable"
    assert exc.value.exit_code == 2


def test_build_helper_requires_swiftc(monkeypatch):
    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.setattr(macos.shutil, "which", lambda _tool: None)
    with pytest.raises(CLIError) as exc:
        macos.build_helper()
    assert "xcode-select" in (exc.value.suggestion or "")
    assert exc.value.exit_code == 2


def test_build_helper_compiles_to_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.setattr(macos.shutil, "which", lambda _tool: "/usr/bin/swiftc")
    monkeypatch.setattr(macos, "_resource_bytes", lambda: b"swift source")
    monkeypatch.setattr(macos, "user_cache_path", lambda _app: tmp_path)
    seen = {}

    def fake_run(cmd, *, capture_output, text, check):
        seen["cmd"] = cmd
        seen["kwargs"] = {"capture_output": capture_output, "text": text, "check": check}
        Path(cmd[-1]).write_bytes(b"binary")
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(macos.subprocess, "run", fake_run)
    helper = macos.build_helper()
    assert helper.read_bytes() == b"binary"
    assert "-parse-as-library" in seen["cmd"]
    assert "ScreenCaptureKit" in seen["cmd"]
    # stderr/stdout are captured as text, and a non-zero compile is inspected (not
    # raised): check must stay False so build_helper surfaces its own error.
    assert seen["kwargs"]["capture_output"] is True
    assert seen["kwargs"]["text"] is True
    assert seen["kwargs"]["check"] is False


def test_build_helper_creates_missing_cache_parents(monkeypatch, tmp_path):
    # The cache dir's parents may not exist yet; build_helper must create the whole
    # chain (cache_dir.mkdir parents=True), not just the leaf.
    nested = tmp_path / "missing1" / "missing2"  # parents do not exist
    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.setattr(macos.shutil, "which", lambda _tool: "/usr/bin/swiftc")
    monkeypatch.setattr(macos, "_resource_bytes", lambda: b"swift source")
    monkeypatch.setattr(macos, "user_cache_path", lambda _app: nested)
    monkeypatch.setattr(
        macos.subprocess,
        "run",
        lambda cmd, **k: (
            Path(cmd[-1]).write_bytes(b"bin"),
            types.SimpleNamespace(returncode=0, stderr="", stdout=""),
        )[1],
    )
    helper = macos.build_helper()
    assert helper.read_bytes() == b"bin"


def test_build_helper_tolerates_existing_cache_dirs(monkeypatch, tmp_path):
    # A rebuild (new source digest) runs with the cache dir and module cache already
    # present, so their mkdirs must tolerate existing dirs (exist_ok=True).
    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.setattr(macos.shutil, "which", lambda _tool: "/usr/bin/swiftc")
    monkeypatch.setattr(macos, "_resource_bytes", lambda: b"swift source")
    monkeypatch.setattr(macos, "user_cache_path", lambda _app: tmp_path)
    (tmp_path / "macos-system-audio" / "swift-module-cache").mkdir(parents=True)  # pre-exist
    monkeypatch.setattr(
        macos.subprocess,
        "run",
        lambda cmd, **k: (
            Path(cmd[-1]).write_bytes(b"bin"),
            types.SimpleNamespace(returncode=0, stderr="", stdout=""),
        )[1],
    )
    helper = macos.build_helper()  # must not raise FileExistsError on the mkdirs
    assert helper.read_bytes() == b"bin"


def test_build_helper_reuses_cached_binary(monkeypatch, tmp_path):
    source = b"swift source"
    digest = macos.hashlib.sha256(source).hexdigest()[:16]
    helper = tmp_path / "macos-system-audio" / f"aai-macos-audio-{digest}"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"cached")
    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.setattr(macos.shutil, "which", lambda _tool: "/usr/bin/swiftc")
    monkeypatch.setattr(macos, "_resource_bytes", lambda: source)
    monkeypatch.setattr(macos, "user_cache_path", lambda _app: tmp_path)
    monkeypatch.setattr(
        macos.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not compile")),
    )
    assert macos.build_helper() == helper


def test_build_helper_compile_failure_surfaces_stderr(monkeypatch, tmp_path):
    monkeypatch.setattr(macos.sys, "platform", "darwin")
    monkeypatch.setattr(macos.shutil, "which", lambda _tool: "/usr/bin/swiftc")
    monkeypatch.setattr(macos, "_resource_bytes", lambda: b"swift source")
    monkeypatch.setattr(macos, "user_cache_path", lambda _app: tmp_path)
    monkeypatch.setattr(
        macos.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stderr="compile broke", stdout=""),
    )
    with pytest.raises(CLIError) as exc:
        macos.build_helper()
    assert exc.value.error_type == "mac_system_audio_unavailable"
    assert exc.value.exit_code == 2
    assert exc.value.suggestion == "compile broke"


def test_resource_bytes_reads_bundled_swift_source():
    assert b"ScreenCaptureKit" in macos._resource_bytes()


def test_read_stderr_none_is_empty():
    assert macos._read_stderr(None) == ""


def test_open_process_exposes_stdout():
    proc = macos._open_process(["/bin/echo", "ok"])
    assert proc.stdout is not None
    try:
        assert proc.stdout.read().strip() == b"ok"
        proc.wait(timeout=2.0)
    finally:
        proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


def test_require_stdout_rejects_missing_pipe():
    proc = _FakeProc(stdout=None)
    with pytest.raises(APIError):
        macos._require_stdout(proc)


def test_cleanup_process_kills_after_wait_timeout():
    class TimeoutProc(_FakeProc):
        def __init__(self):
            super().__init__(stdout=b"")
            self.waits = 0
            self.wait_timeouts = []

        def wait(self, timeout=None):
            self.waits += 1
            self.wait_timeouts.append(timeout)
            if self.waits == 1:
                raise macos.subprocess.TimeoutExpired("helper", timeout or 0.0)
            return self.returncode

    proc = TimeoutProc()
    assert proc.stdout is not None
    macos._cleanup_process(proc, proc.stdout, completed=True)
    assert proc.killed is True
    assert proc.waits == 2
    assert proc.terminated is False  # completed=True -> the `and` guard skips terminate()
    assert proc.wait_timeouts == [2.0, 2.0]  # both waits use the 2s backstop


def test_raise_helper_exit_handles_clean_eof():
    proc = _FakeProc(stdout=b"", stderr=b"permission hint", returncode=0)
    with pytest.raises(CLIError) as exc:
        macos._raise_helper_exit(proc)
    assert "ended unexpectedly" in exc.value.message
    assert exc.value.suggestion == "permission hint"


def test_returncode_detail_names_signals():
    assert macos._returncode_detail(-5) == "SIGTRAP (-5)"
    assert macos._returncode_detail(-99999) == "signal 99999 (-99999)"
    assert macos._returncode_detail(2) == "exit 2"
    assert macos._returncode_detail(0) == "exit 0"  # 0 is a clean exit (pins `>= 0`)
    assert macos._returncode_detail(None) == "unknown exit"


def test_raise_helper_exit_names_signal_without_stderr():
    proc = _FakeProc(stdout=b"", stderr=b"", returncode=-5)
    with pytest.raises(CLIError) as exc:
        macos._raise_helper_exit(proc)
    assert "SIGTRAP" in exc.value.message


def test_source_starts_helper_and_yields_pcm(tmp_path):
    helper = tmp_path / "helper"
    helper.write_text("")
    events = []
    procs = []
    commands = []

    def fake_popen(cmd):
        commands.append(cmd)
        proc = _FakeProc(stdout=b"\x01" * CHUNK_BYTES + b"\x02" * CHUNK_BYTES)
        procs.append(proc)
        return proc

    src = macos.MacSystemAudioSource(
        helper=helper,
        on_open=lambda: events.append("open"),
        popen=fake_popen,
    )
    gen = src.__iter__()
    assert next(gen) == b"\x01" * CHUNK_BYTES
    assert next(gen) == b"\x02" * CHUNK_BYTES
    gen.close()
    assert events == ["open"]
    assert "--system-only" in commands[0]
    assert procs[0].terminated is True
    # On a non-completed teardown the helper's stderr pipe is closed too (pins the
    # `proc.stderr is not None` guard against an `is None` flip that would leak it).
    assert procs[0].stderr is not None and procs[0].stderr.closed is True
    # chunk-frames is ~100 ms of frames at the target rate (sample_rate // 10).
    cmd = commands[0]
    assert cmd[cmd.index("--chunk-frames") + 1] == str(src.sample_rate // 10)


def test_source_start_failure_is_cli_error(tmp_path):
    helper = tmp_path / "helper"
    helper.write_text("")

    def fail(_cmd):
        raise OSError("not executable")

    src = macos.MacSystemAudioSource(helper=helper, popen=fail)
    with pytest.raises(CLIError) as exc:
        list(src)
    assert "not executable" in exc.value.message


def test_source_system_audio_only_flag(tmp_path):
    helper = tmp_path / "helper"
    helper.write_text("")
    commands = []

    def fake_popen(cmd):
        commands.append(cmd)
        return _FakeProc(stdout=b"\x00" * CHUNK_BYTES)

    src = macos.MacSystemAudioSource(helper=helper, popen=fake_popen)
    gen = src.__iter__()
    next(gen)
    gen.close()
    assert "--system-only" in commands[0]


def test_source_surfaces_helper_failure(tmp_path):
    helper = tmp_path / "helper"
    helper.write_text("")

    def fake_popen(cmd):
        return _FakeProc(stdout=b"", stderr=b"permission denied", returncode=1)

    src = macos.MacSystemAudioSource(helper=helper, popen=fake_popen)
    with pytest.raises(CLIError) as exc:
        list(src)
    assert exc.value.error_type == "mac_system_audio_error"
    assert "permission denied" in exc.value.message
