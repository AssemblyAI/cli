from __future__ import annotations

import contextlib
import hashlib
import io
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Generator, Sequence
from importlib import resources
from pathlib import Path
from typing import IO, Protocol

from platformdirs import user_cache_path

from aai_cli.errors import APIError, CLIError
from aai_cli.streaming.sources import CHUNK_BYTES, TARGET_RATE

_HELPER_RESOURCE = "macos_system_audio.swift"
_CACHE_DIR = "macos-system-audio"
_HELPER_PREFIX = "aai-macos-audio"
type _Pipe = IO[bytes] | io.BytesIO


class _CaptureProcess(Protocol):
    @property
    def stdout(self) -> _Pipe | None:
        """The helper's PCM output pipe."""

    @property
    def stderr(self) -> _Pipe | None:
        """The helper's diagnostic pipe."""

    @property
    def returncode(self) -> int | None:
        """Exit code once the helper has exited."""

    def poll(self) -> int | None:
        """Non-blocking exit-code check."""

    def terminate(self) -> None:
        """Ask the helper to exit."""

    def kill(self) -> None:
        """Force the helper to exit."""

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the helper exits."""


def _unsupported_platform() -> CLIError:
    return CLIError(
        "macOS system audio capture is only available on macOS.",
        error_type="mac_system_audio_unavailable",
        exit_code=2,
    )


def _missing_swiftc() -> CLIError:
    return CLIError(
        "macOS system audio capture needs Apple's Swift compiler.",
        error_type="mac_system_audio_unavailable",
        exit_code=2,
        suggestion="Install Xcode Command Line Tools: xcode-select --install",
    )


def _resource_bytes() -> bytes:
    return resources.files("aai_cli.streaming").joinpath(_HELPER_RESOURCE).read_bytes()


def _is_macos() -> bool:
    return sys.platform == "darwin"


def build_helper() -> Path:
    """Compile the bundled ScreenCaptureKit helper once and return its executable path."""
    if not _is_macos():
        raise _unsupported_platform()
    swiftc = shutil.which("swiftc")
    if swiftc is None:
        raise _missing_swiftc()

    source = _resource_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    cache_dir = user_cache_path("aai-cli") / _CACHE_DIR
    helper = cache_dir / f"{_HELPER_PREFIX}-{digest}"
    if helper.exists():
        return helper

    cache_dir.mkdir(parents=True, exist_ok=True)
    module_cache = cache_dir / "swift-module-cache"
    # parents=True is an equivalent mutant here: cache_dir was just created above, so
    # module_cache's parent always exists. exist_ok is covered by the rebuild test.
    module_cache.mkdir(parents=True, exist_ok=True)  # pragma: no mutate
    source_path = cache_dir / f"{_HELPER_PREFIX}-{digest}.swift"
    source_path.write_bytes(source)
    tmp_helper = helper.with_suffix(".tmp")
    result = subprocess.run(
        [
            swiftc,
            "-parse-as-library",
            "-module-cache-path",
            str(module_cache),
            str(source_path),
            "-O",
            "-framework",
            "ScreenCaptureKit",
            "-framework",
            "AVFoundation",
            "-framework",
            "CoreMedia",
            "-framework",
            "CoreGraphics",
            "-o",
            str(tmp_helper),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CLIError(
            "Could not build the macOS system audio helper.",
            error_type="mac_system_audio_unavailable",
            exit_code=2,
            suggestion=detail or "Install Xcode Command Line Tools: xcode-select --install",
        )
    tmp_helper.replace(helper)
    return helper


def _read_stderr(stderr: _Pipe | None) -> str:
    if stderr is None:
        return ""
    try:
        data = stderr.read()
        return data.decode("utf-8", "replace").strip()
    finally:
        with contextlib.suppress(Exception):
            stderr.close()


def _open_process(command: Sequence[str]) -> _CaptureProcess:
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _require_stdout(proc: _CaptureProcess) -> _Pipe:
    stdout = proc.stdout
    if stdout is None:
        raise APIError("macOS system audio helper did not expose an output stream.")
    return stdout


def _cleanup_process(proc: _CaptureProcess, stdout: _Pipe, *, completed: bool) -> None:
    if not completed and proc.poll() is None:
        proc.terminate()
    with contextlib.suppress(Exception):
        stdout.close()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2.0)
    if not completed and proc.stderr is not None:
        with contextlib.suppress(Exception):
            proc.stderr.close()


def _returncode_detail(returncode: int | None) -> str:
    if returncode is None:
        return "unknown exit"
    if returncode >= 0:
        return f"exit {returncode}"
    signum = -returncode
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = f"signal {signum}"
    return f"{name} ({returncode})"


def _raise_helper_exit(proc: _CaptureProcess) -> None:
    detail = _read_stderr(proc.stderr)
    if proc.returncode:
        raise CLIError(
            f"Could not capture macOS system audio: {detail or _returncode_detail(proc.returncode)}",
            error_type="mac_system_audio_error",
            suggestion=(
                "Grant Screen & System Audio Recording and Microphone permissions "
                "to your terminal, then run the command again."
            ),
        )
    raise CLIError(
        "macOS system audio capture ended unexpectedly.",
        error_type="mac_system_audio_error",
        suggestion=detail or "Run the command again and check macOS privacy permissions.",
    )


class MacSystemAudioSource:
    """Streams macOS system/app audio as PCM16 mono.

    A tiny bundled Swift helper owns the native macOS ScreenCaptureKit APIs.
    Python treats the helper's stdout exactly like any other raw PCM stream.
    """

    def __init__(
        self,
        *,
        on_open: Callable[[], None] | None = None,
        helper: Path | None = None,
        popen: Callable[[Sequence[str]], _CaptureProcess] = _open_process,
    ) -> None:
        self.source = "macos-system-audio"
        self.sample_rate = TARGET_RATE
        self._on_open = on_open
        self._helper = helper or build_helper()
        self._popen = popen

    def _command(self) -> list[str]:
        command = [
            str(self._helper),
            "--sample-rate",
            str(self.sample_rate),
            "--chunk-frames",
            str(self.sample_rate // 10),
        ]
        command.append("--system-only")
        return command

    def _start(self) -> _CaptureProcess:
        try:
            return self._popen(self._command())
        except OSError as exc:
            raise CLIError(
                f"Could not start macOS system audio capture: {exc}",
                error_type="mac_system_audio_error",
            ) from exc

    def __iter__(self) -> Generator[bytes, None, None]:
        proc = self._start()
        stdout = _require_stdout(proc)
        opened = False
        completed = False
        try:
            while True:
                data = stdout.read(CHUNK_BYTES)
                if not data:
                    break
                if not opened:
                    opened = True
                    if self._on_open is not None:
                        self._on_open()
                yield bytes(data)
            completed = True
        finally:
            _cleanup_process(proc, stdout, completed=completed)

        _raise_helper_exit(proc)
