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

from collections.abc import Sequence

# Credential dirs/files under $HOME, read-denied precisely on both platforms.
HOME_SECRETS: tuple[str, ...] = (".ssh", ".aws", ".gnupg", ".netrc", ".npmrc")
# Project-local secrets denied for reads even though cwd is otherwise readable.
CWD_READ_DENY: tuple[str, ...] = (".env", ".claude")
# Persistence paths denied for writes even inside cwd.
CWD_WRITE_DENY: tuple[str, ...] = (".git/hooks",)
# Shell rc files denied for writes (only inside the write region when cwd == $HOME).
SHELL_RC: tuple[str, ...] = (".bashrc", ".zshrc", ".profile", ".bash_profile")


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
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec*)",
        "(allow process-fork)",
        "(allow file-read*)",
    ]
    lines.extend(f'(deny file-read* (subpath "{home}/{name}"))' for name in home_secrets)
    # .env and .env.* under cwd, denied via regex; .claude/ via subpath.
    lines.append(f'(deny file-read* (regex #"^{cwd}/\\.env($|\\.)"))')
    lines.extend(
        f'(deny file-read* (subpath "{cwd}/{name}"))' for name in cwd_read_deny if name != ".env"
    )
    lines.append(f'(allow file-write* (subpath "{cwd}") (subpath "{tmp}"))')
    lines.extend(f'(deny file-write* (subpath "{cwd}/{name}"))' for name in cwd_write_deny)
    lines.extend(f'(deny file-write* (subpath "{home}/{name}"))' for name in shell_rc)
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
    # Mask credential stores under $HOME (tmpfs hides their contents).
    for name in home_secrets:
        argv += ["--tmpfs", f"{home}/{name}"]
    # Project-local secrets: mask each path (best-effort; coarser than Seatbelt).
    for name in cwd_read_deny:
        argv += ["--ro-bind", "/dev/null", f"{cwd}/{name}"]
    # Block writes to persistence paths inside cwd by re-binding them read-only.
    for name in cwd_write_deny:
        argv += ["--ro-bind", f"{cwd}/{name}", f"{cwd}/{name}"]
    argv += ["--chdir", cwd, "/bin/sh", "-c", command]
    return argv
