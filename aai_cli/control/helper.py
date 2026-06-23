"""Talk to the native macOS UI-control helper over a JSON-lines pipe.

The "hands" of the agent are a tiny bundled Swift program
(``macos_ui_control.swift``) that owns the native APIs Python can't reach —
``CGEvent`` for synthetic keystrokes/clicks, the Accessibility API for reading
the focused app's element tree, ``NSWorkspace`` for launching/activating apps.
:class:`UiHelper` compiles it once, runs it as a long-lived child, and exchanges
one JSON request/response line per :class:`~aai_cli.control.actions.Action` — the
same stdout-pipe pattern as the streaming system-audio helper.

The process factory is injected, so the request/response logic is unit-tested
with in-memory pipes and never spawns anything.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from importlib import resources
from pathlib import Path
from typing import IO, Protocol

from platformdirs import user_cache_path

from aai_cli.control.actions import Action
from aai_cli.core import jsonshape
from aai_cli.core.errors import APIError, CLIError

_HELPER_RESOURCE = "macos_ui_control.swift"
_CACHE_DIR = "macos-ui-control"
_HELPER_PREFIX = "aai-macos-ui-control"
# Frameworks the helper links: synthetic input + window list (CoreGraphics),
# app launch/activation (AppKit), the Accessibility element tree (ApplicationServices).
_FRAMEWORKS = ("AppKit", "CoreGraphics", "ApplicationServices")


class _HelperProcess(Protocol):
    @property
    def stdin(self) -> IO[str] | None:
        """The helper's JSON request pipe."""

    @property
    def stdout(self) -> IO[str] | None:
        """The helper's JSON response pipe."""

    def poll(self) -> int | None:
        """Non-blocking exit-code check."""

    def terminate(self) -> None:
        """Ask the helper to exit."""

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the helper exits."""


def _unsupported_platform() -> CLIError:
    return CLIError(
        "Voice computer-control is only available on macOS.",
        error_type="control_unavailable",
        exit_code=2,
    )


def _missing_swiftc() -> CLIError:
    return CLIError(
        "Voice computer-control needs Apple's Swift compiler.",
        error_type="control_unavailable",
        exit_code=2,
        suggestion="Install Xcode Command Line Tools: xcode-select --install",
    )


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _resource_bytes() -> bytes:
    return resources.files("aai_cli.control").joinpath(_HELPER_RESOURCE).read_bytes()


def build_helper() -> Path:
    """Compile the bundled UI-control helper once and return its executable path."""
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
    source_path = cache_dir / f"{_HELPER_PREFIX}-{digest}.swift"
    source_path.write_bytes(source)
    tmp_helper = helper.with_suffix(".tmp")
    frameworks = [arg for framework in _FRAMEWORKS for arg in ("-framework", framework)]
    result = subprocess.run(
        [swiftc, "-parse-as-library", str(source_path), "-O", *frameworks, "-o", str(tmp_helper)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise CLIError(
            "Could not build the macOS UI-control helper.",
            error_type="control_unavailable",
            exit_code=2,
            suggestion=detail or "Install Xcode Command Line Tools: xcode-select --install",
        )
    tmp_helper.replace(helper)
    return helper


def _open_process(command: Sequence[str]) -> _HelperProcess:
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )


class UiHelper:
    """A long-lived UI-control helper process, addressed one action at a time."""

    def __init__(
        self,
        *,
        helper: Path | None = None,
        popen: Callable[[Sequence[str]], _HelperProcess] = _open_process,
    ) -> None:
        self._helper = helper or build_helper()
        self._popen = popen
        self._proc: _HelperProcess | None = None

    def _streams(self) -> tuple[IO[str], IO[str]]:
        """Spawn the helper on first use and return its (stdin, stdout) pipes."""
        if self._proc is None:
            self._proc = self._popen([str(self._helper)])
        stdin, stdout = self._proc.stdin, self._proc.stdout
        if stdin is None or stdout is None:
            raise APIError("The UI-control helper did not expose its IO streams.")
        return stdin, stdout

    def execute(self, action: Action) -> dict[str, object]:
        """Send one action and return the helper's JSON result.

        Matches the engine's ``Executor`` seam: a closed pipe or a non-JSON line
        becomes an :class:`APIError` so the session fails cleanly rather than
        hanging or dumping a traceback.
        """
        stdin, stdout = self._streams()
        try:
            stdin.write(json.dumps(action.request()) + "\n")
            stdin.flush()
        except OSError as exc:
            raise APIError(f"The UI-control helper stopped accepting input: {exc}") from exc
        line = stdout.readline()
        if not line:
            raise APIError("The UI-control helper closed without responding.")
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise APIError("The UI-control helper returned a non-JSON line.") from exc
        mapping = jsonshape.as_mapping(parsed)
        if mapping is None:
            return {"ok": False, "error": "helper returned a non-object result"}
        return mapping

    def close(self) -> None:
        """Terminate the helper if it is running."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
        with contextlib.suppress(Exception):
            # The 2s grace before giving up is not observable from a test.
            self._proc.wait(timeout=2.0)  # pragma: no mutate
        self._proc = None
