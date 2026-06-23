"""OS-sandboxed shell execution for ``assembly live --files``.

deepagents binds a functional ``execute`` tool only when the backend implements
``SandboxBackendProtocol``. :class:`SandboxedShellBackend` does — but its ``execute`` never
runs an unconfined host shell: it wraps the command in an OS sandbox (``sandbox-exec`` SBPL on
macOS, ``bwrap`` on Linux) that confines writes to cwd, denies the network, and read-denies
credential stores. On any other platform (or with the sandbox binary missing) it refuses and
runs nothing — never a fallback to unconfined execution. The policy renderers are pure and the
subprocess/capability boundaries are injected, so the suite asserts *what we would run* with no
real sandbox.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

from aai_cli.core.env import child_env

# Credential dirs/files under $HOME, read-denied precisely on both platforms.
HOME_SECRETS: tuple[str, ...] = (".ssh", ".aws", ".gnupg", ".netrc", ".npmrc")
# Project-local secrets denied for reads even though cwd is otherwise readable.
CWD_READ_DENY: tuple[str, ...] = (".env", ".claude")
# Persistence paths denied for writes even inside cwd.
CWD_WRITE_DENY: tuple[str, ...] = (".git/hooks",)
# Shell rc files denied for writes (only inside the write region when cwd == $HOME).
SHELL_RC: tuple[str, ...] = (".bashrc", ".zshrc", ".profile", ".bash_profile")
# Which read-denied secret names are directories (the rest are plain files). bwrap masks a
# directory with an empty tmpfs but a file with a /dev/null bind — a tmpfs mountpoint must be a
# directory and a /dev/null bind must be a file — so the two kinds need different masks.
_SECRET_DIRS: frozenset[str] = frozenset({".ssh", ".aws", ".gnupg", ".claude"})


def _sbpl_str(path: str) -> str:
    """Escape a path for embedding in an SBPL string literal (``"…"``): backslash and quote.

    A launch directory can contain either, and an unescaped ``"`` would terminate the literal
    early, producing a profile ``sandbox-exec`` rejects (so every ``execute`` would then fail).
    """
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _sbpl_regex(path: str) -> str:
    """Escape a path for use as a literal inside an SBPL ``#"…"`` regex literal.

    ``re.escape`` neutralizes regex metacharacters (``(``/``)``/``+``/``[``…) a path may contain —
    without it ``/Users/me/My (Project)`` emits an invalid regex that breaks the whole profile —
    and the lone ``"`` escape keeps the surrounding string literal intact.
    """
    return re.escape(path).replace('"', '\\"')


def _mask_secret(target: str, name: str) -> list[str]:
    """bwrap args to hide a secret ``target``: an empty tmpfs over a directory secret, a
    ``/dev/null`` bind over a file secret (the two kinds need different bwrap directives)."""
    if name in _SECRET_DIRS:
        return ["--tmpfs", target]
    return ["--ro-bind", "/dev/null", target]


def render_seatbelt_profile(
    cwd: str,
    tmp: str,
    home: str,
    *,
    home_secrets: Sequence[str] = HOME_SECRETS,
    cwd_read_deny: Sequence[str] = CWD_READ_DENY,
    cwd_write_deny: Sequence[str] = CWD_WRITE_DENY,
    shell_rc: Sequence[str] = SHELL_RC,
) -> str:
    """Render an Apple Seatbelt (SBPL) profile: default-allow reads, deny secrets, writes only
    in cwd + tmp, no network. Last-match-wins, so the denies override the broad allows."""
    # Escape the interpolated paths once: a launch dir / tmp / home with regex or quote
    # metacharacters would otherwise emit a profile sandbox-exec can't parse.
    c, t, h = _sbpl_str(cwd), _sbpl_str(tmp), _sbpl_str(home)
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec*)",
        "(allow process-fork)",
        "(allow file-read*)",
    ]
    lines.extend(f'(deny file-read* (subpath "{h}/{name}"))' for name in home_secrets)
    # .env and .env.* under cwd, denied via regex; .claude/ via subpath.
    lines.append(f'(deny file-read* (regex #"^{_sbpl_regex(cwd)}/\\.env($|\\.)"))')
    lines.extend(
        f'(deny file-read* (subpath "{c}/{name}"))' for name in cwd_read_deny if name != ".env"
    )
    lines.append(f'(allow file-write* (subpath "{c}") (subpath "{t}"))')
    lines.extend(f'(deny file-write* (subpath "{c}/{name}"))' for name in cwd_write_deny)
    lines.extend(f'(deny file-write* (subpath "{h}/{name}"))' for name in shell_rc)
    return "\n".join(lines) + "\n"


def build_bwrap_argv(
    cwd: str,
    tmp: str,
    command: str,
    home: str,
    *,
    home_secrets: Sequence[str] = HOME_SECRETS,
    cwd_read_deny: Sequence[str] = CWD_READ_DENY,
    cwd_write_deny: Sequence[str] = CWD_WRITE_DENY,
) -> list[str]:
    """Build a bubblewrap argv: whole FS read-only (default-allow reads), cwd + tmp read-write,
    secret stores masked, ``.git/hooks`` read-only, network unshared. Path-based, so in-cwd
    secret-file protection is coarser than Seatbelt's globbing (a documented asymmetry); the
    directory-level credential stores are masked precisely on both."""
    argv = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        cwd,
        cwd,
        "--bind",
        tmp,
        tmp,
    ]
    # Mask credential stores under $HOME and the project-local secrets (best-effort; coarser than
    # Seatbelt). Each path is masked by kind — an empty tmpfs over a directory, a /dev/null bind
    # over a file — since bwrap can't tmpfs a file mountpoint or bind /dev/null onto a directory.
    for name in home_secrets:
        argv += _mask_secret(f"{home}/{name}", name)
    for name in cwd_read_deny:
        argv += _mask_secret(f"{cwd}/{name}", name)
    # Block writes to persistence paths inside cwd by re-binding them read-only.
    for name in cwd_write_deny:
        argv += ["--ro-bind", f"{cwd}/{name}", f"{cwd}/{name}"]
    argv += ["--chdir", cwd, "/bin/sh", "-c", command]
    return argv


Capability = Literal["seatbelt", "bwrap", "none"]

DEFAULT_TIMEOUT_SECONDS = 120  # pragma: no mutate
MAX_TIMEOUT_SECONDS = 600  # pragma: no mutate
CPU_LIMIT_SECONDS = 60  # pragma: no mutate
ADDRESS_LIMIT_KB = 4_000_000  # pragma: no mutate
_TIMEOUT_EXIT = 124  # conventional timeout exit code


def detect_capability(
    *,
    system: Callable[[], str] = platform.system,
    which: Callable[[str], str | None] = shutil.which,
) -> Capability:
    """Resolve the sandbox mechanism for this host: ``seatbelt`` (macOS + ``sandbox-exec``),
    ``bwrap`` (Linux + ``bwrap``), else ``none`` — the refuse-don't-fall-back signal."""
    name = system()
    if name == "Darwin" and which("sandbox-exec"):
        return "seatbelt"
    if name == "Linux" and which("bwrap"):
        return "bwrap"
    return "none"


class CompletedProcessLike(Protocol):
    """The slice of a finished process the backend reads: combined output + exit code."""

    output: str
    returncode: int | None


class _Result:
    """Concrete :class:`CompletedProcessLike` the default runner returns."""

    def __init__(self, output: str, returncode: int | None) -> None:
        self.output = output
        self.returncode = returncode


Runner = Callable[[list[str], str, int], CompletedProcessLike]

# The only environment variables the sandboxed command inherits. The OS sandbox blocks the
# network and read-denies credential *files*, but secrets such as ASSEMBLYAI_API_KEY ride in the
# *environment* too — an unrestricted env would hand the agent-run command every key to print into
# output the model reads or write into a cwd file. So the child env is a minimal non-secret
# allowlist, never a copy of the parent environment.
_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "TZ",
)


def _sandbox_env() -> dict[str, str]:
    """A minimal child environment for the sandboxed command: only the non-secret basics in
    :data:`_ENV_ALLOWLIST`, so no inherited API key or token leaks to agent-run code."""
    parent = child_env()
    return {name: parent[name] for name in _ENV_ALLOWLIST if name in parent}


def default_runner(argv: list[str], cwd: str, timeout: int) -> CompletedProcessLike:
    """Run ``argv`` with combined output, in ``cwd``, time-bounded, with a minimal child env.

    A timeout returns the partial output + a sentinel exit code (information, not a crash); a
    launch failure is left to raise so the caller turns it into an apology string."""
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_sandbox_env(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.output or ""
        text = out.decode() if isinstance(out, bytes) else out
        return _Result(text + f"\n[timed out after {timeout}s]", _TIMEOUT_EXIT)
    return _Result(proc.stdout or "", proc.returncode)


NO_SANDBOX_MESSAGE = "I can't run code on this system."
LAUNCH_FAILURE_MESSAGE = "I couldn't start a sandbox to run that."


def _ulimit_wrap(command: str) -> str:
    """Cap CPU + address space so a runaway can't peg the box inside the timeout."""
    return f"ulimit -t {CPU_LIMIT_SECONDS}; ulimit -v {ADDRESS_LIMIT_KB}; {command}"  # pragma: no mutate


class SandboxedShellBackend(LocalShellBackend):
    """A ``LocalShellBackend`` whose ``execute`` runs through an OS sandbox, never the host shell.

    Inherits the cwd-rooted file operations (``read_file``/``write_file``/``edit_file``/``ls``/
    ``glob``/``grep``) unchanged; implementing ``SandboxBackendProtocol`` (via the base) is what
    makes deepagents auto-add the ``execute`` tool. The override confines every run to cwd, denies
    the network, and refuses outright when no sandbox is available."""

    def __init__(
        self,
        *,
        root_dir: str,
        virtual_mode: bool = True,
        runner: Runner | None = None,
        capability: Capability | None = None,
        tmp: str | None = None,
        home: str | None = None,
    ) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=virtual_mode)
        self._runner: Runner = runner or default_runner
        self._capability: Capability = capability if capability is not None else detect_capability()
        self._tmp = tmp if tmp is not None else tempfile.gettempdir()
        self._home = home if home is not None else str(Path("~").expanduser())

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run ``command`` confined to cwd via the OS sandbox; refuse when none is available."""
        if self._capability == "none":
            return ExecuteResponse(output=NO_SANDBOX_MESSAGE, exit_code=None)
        cwd = str(self.cwd)
        wrapped = _ulimit_wrap(command)
        if self._capability == "seatbelt":
            profile = render_seatbelt_profile(cwd, self._tmp, self._home)
            argv = ["sandbox-exec", "-p", profile, "/bin/sh", "-c", wrapped]
        else:
            argv = build_bwrap_argv(cwd, self._tmp, wrapped, self._home)
        bounded = min(timeout or DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS)
        try:
            result = self._runner(argv, cwd, bounded)
        except (OSError, ValueError, subprocess.SubprocessError):
            return ExecuteResponse(output=LAUNCH_FAILURE_MESSAGE, exit_code=None)
        return ExecuteResponse(output=result.output, exit_code=result.returncode)
