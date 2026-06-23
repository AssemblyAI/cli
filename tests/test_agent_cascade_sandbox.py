from __future__ import annotations

from deepagents.backends.protocol import ExecuteResponse

from aai_cli.agent_cascade import sandbox


def test_seatbelt_profile_is_default_allow_reads_deny_default():
    profile = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    assert "(version 1)" in profile
    assert "(deny default)" in profile
    assert "(allow process-exec*)" in profile
    assert "(allow file-read*)" in profile  # default-allow reads
    # No network allow anywhere — network stays denied by (deny default).
    assert "network" not in profile


def test_seatbelt_profile_denies_each_home_secret_for_reads():
    profile = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    for name in sandbox.HOME_SECRETS:
        assert f'(deny file-read* (subpath "/home/u/{name}"))' in profile


def test_seatbelt_profile_denies_project_secrets_for_reads():
    profile = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    # .env (and .env.*) under cwd are read-denied via a regex; .claude/ via subpath.
    assert "file-read*" in profile and "/work/proj" in profile
    assert any(".env" in line and "deny file-read*" in line for line in profile.splitlines())
    assert '(deny file-read* (subpath "/work/proj/.claude"))' in profile


def test_seatbelt_profile_writes_confined_to_cwd_and_tmp():
    profile = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    assert '(allow file-write* (subpath "/work/proj") (subpath "/tmp"))' in profile


def test_seatbelt_profile_denies_persistence_writes_inside_cwd():
    profile = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    assert '(deny file-write* (subpath "/work/proj/.git/hooks"))' in profile
    # Shell rc files denied for writes (covers the cwd == $HOME case).
    for name in sandbox.SHELL_RC:
        assert f'(deny file-write* (subpath "/home/u/{name}"))' in profile


def test_bwrap_argv_confines_to_cwd_with_network_unshared():
    argv = sandbox.build_bwrap_argv("/work/proj", "/tmp", "echo hi", "/home/u")
    assert argv[0] == "bwrap"
    assert "--unshare-all" in argv  # includes network namespace
    assert "--die-with-parent" in argv
    # Whole FS read-only = default-allow reads.
    assert _has_pair(argv, "--ro-bind", "/", "/")
    # cwd + tmp are read-write bound; chdir into cwd.
    assert _has_pair(argv, "--bind", "/work/proj", "/work/proj")
    assert _has_pair(argv, "--bind", "/tmp", "/tmp")
    assert _adjacent(argv, "--chdir", "/work/proj")
    # The command lands at the tail via a shell.
    assert argv[-1] == "echo hi" or "echo hi" in argv[-1]


def test_bwrap_argv_masks_home_secrets_and_git_hooks():
    argv = sandbox.build_bwrap_argv("/work/proj", "/tmp", "echo hi", "/home/u")
    joined = " ".join(argv)
    for name in sandbox.HOME_SECRETS:
        assert f"/home/u/{name}" in joined  # masked (tmpfs / ro-bind /dev/null)
    assert "/work/proj/.git/hooks" in joined  # write blocked via ro-bind


def test_bwrap_masks_directories_with_tmpfs_and_files_with_dev_null():
    # bwrap can't tmpfs a file mountpoint nor bind /dev/null onto a directory, so each secret is
    # masked by kind: directory secrets get --tmpfs, file secrets get --ro-bind /dev/null.
    argv = sandbox.build_bwrap_argv("/work/proj", "/tmp", "echo hi", "/home/u")
    # .claude and .ssh are directories -> tmpfs (the old code wrongly bound /dev/null over .claude).
    assert _adjacent(argv, "--tmpfs", "/work/proj/.claude")
    assert _adjacent(argv, "--tmpfs", "/home/u/.ssh")
    # .env and .netrc/.npmrc are files -> /dev/null bind (the old code wrongly tmpfs'd the files).
    assert _has_pair(argv, "--ro-bind", "/dev/null", "/work/proj/.env")
    assert _has_pair(argv, "--ro-bind", "/dev/null", "/home/u/.netrc")
    # And never the wrong directive for either kind.
    assert not _has_pair(argv, "--ro-bind", "/dev/null", "/work/proj/.claude")
    assert not _adjacent(argv, "--tmpfs", "/home/u/.netrc")


def _has_pair(argv, flag, a, b):
    return any(
        argv[i] == flag and argv[i + 1] == a and argv[i + 2] == b for i in range(len(argv) - 2)
    )


def _adjacent(argv, flag, value):
    return any(argv[i] == flag and argv[i + 1] == value for i in range(len(argv) - 1))


def test_renderers_cover_the_same_denylists():
    # Parity: both platform renderers must reference every denylist constant, so a path added
    # to one platform can't silently be left unprotected on the other.
    seatbelt = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    bwrap = " ".join(sandbox.build_bwrap_argv("/work/proj", "/tmp", "x", "/home/u"))
    for name in sandbox.HOME_SECRETS:
        assert f"/home/u/{name}" in seatbelt
        assert f"/home/u/{name}" in bwrap
    assert "/work/proj/.git/hooks" in seatbelt
    assert "/work/proj/.git/hooks" in bwrap


def test_seatbelt_profile_escapes_regex_metacharacters_in_cwd():
    # A launch dir with regex metacharacters must not corrupt the .env-deny regex literal:
    # parens are escaped so sandbox-exec gets a valid profile (else every execute would fail).
    profile = sandbox.render_seatbelt_profile("/work/My (Proj)", "/tmp", "/home/u")
    assert r"\(Proj\)/\.env" in profile  # parens neutralized inside the regex
    assert "My (Proj)/\\.env" not in profile  # never the raw, unescaped paren form


def test_seatbelt_profile_escapes_quotes_in_subpath():
    # A double-quote in a path would terminate the SBPL string literal early; it must be escaped.
    profile = sandbox.render_seatbelt_profile('/work/a"b', "/tmp", "/home/u")
    assert '/work/a\\"b' in profile


def test_detect_capability_seatbelt_on_macos_with_binary():
    cap = sandbox.detect_capability(
        system=lambda: "Darwin", which=lambda _n: "/usr/bin/sandbox-exec"
    )
    assert cap == "seatbelt"


def test_detect_capability_bwrap_on_linux_with_binary():
    cap = sandbox.detect_capability(system=lambda: "Linux", which=lambda _n: "/usr/bin/bwrap")
    assert cap == "bwrap"


def test_detect_capability_none_when_binary_missing():
    cap = sandbox.detect_capability(system=lambda: "Darwin", which=lambda _n: None)
    assert cap == "none"


def test_detect_capability_none_on_unsupported_platform():
    cap = sandbox.detect_capability(system=lambda: "Windows", which=lambda _n: "anything")
    assert cap == "none"


# ---------------------------------------------------------------------------
# default_runner tests
# ---------------------------------------------------------------------------


def test_default_runner_runs_and_shapes_result(monkeypatch):
    import subprocess

    captured: dict[str, object] = {}

    class _Proc:
        stdout = "the output"
        returncode = 0

    def fake_run(argv: list[str], **kwargs: object) -> _Proc:
        captured["argv"] = argv
        captured.update(kwargs)
        return _Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = sandbox.default_runner(["echo", "hi"], "/work", 30)
    assert result.output == "the output"
    assert result.returncode == 0
    assert captured["argv"] == ["echo", "hi"]
    assert captured["cwd"] == "/work"
    assert captured["timeout"] == 30
    assert captured["check"] is False
    assert captured["text"] is True
    assert captured["stdout"] == subprocess.PIPE
    assert captured["stderr"] == subprocess.STDOUT


def test_sandbox_env_allowlists_basics_and_drops_secrets(monkeypatch):
    # The sandboxed command must not inherit secrets via the environment, even though the OS
    # sandbox blocks the network and credential files.
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-secret")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-secret")
    monkeypatch.setenv("SOME_TOKEN", "tok")
    env = sandbox._sandbox_env()
    assert env["PATH"] == "/usr/bin"  # a non-secret basic is kept
    assert "ASSEMBLYAI_API_KEY" not in env
    assert "FIRECRAWL_API_KEY" not in env
    assert "SOME_TOKEN" not in env


def test_default_runner_passes_scrubbed_env(monkeypatch):
    import subprocess

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-secret")
    captured: dict[str, object] = {}

    class _Proc:
        stdout = "out"
        returncode = 0

    def fake_run(argv: list[str], **kwargs: object) -> _Proc:
        captured.update(kwargs)
        return _Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    sandbox.default_runner(["echo", "hi"], "/work", 5)
    env = captured["env"]
    assert isinstance(env, dict)
    assert "ASSEMBLYAI_API_KEY" not in env  # the secret never reaches the sandboxed command
    assert env.get("PATH") == "/usr/bin"


def test_default_runner_handles_none_stdout(monkeypatch):
    import subprocess

    class _Proc:
        stdout = None
        returncode = 2

    monkeypatch.setattr(subprocess, "run", lambda argv, **k: _Proc())
    result = sandbox.default_runner(["x"], "/w", 1)
    assert result.output == "" and result.returncode == 2


def test_default_runner_timeout_returns_partial_text_output(monkeypatch):
    import subprocess

    def fake_run(argv: list[str], **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=5, output="partial")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = sandbox.default_runner(["sleep", "99"], "/w", 5)
    assert "partial" in result.output
    assert "timed out after 5s" in result.output
    assert result.returncode == 124  # conventional timeout exit code (literal pins the value)


def test_default_runner_timeout_decodes_bytes_output(monkeypatch):
    import subprocess

    def fake_run(argv: list[str], **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=1, output=b"raw bytes")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert "raw bytes" in sandbox.default_runner(["x"], "/w", 1).output


def test_default_runner_timeout_with_no_output(monkeypatch):
    import subprocess

    def fake_run(argv: list[str], **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=3, output=None)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert "timed out after 3s" in sandbox.default_runner(["x"], "/w", 3).output


# ---------------------------------------------------------------------------
# SandboxedShellBackend tests
# ---------------------------------------------------------------------------


def _backend(
    tmp_path: object,
    cap: sandbox.Capability,
    runner: sandbox.Runner,
) -> sandbox.SandboxedShellBackend:
    return sandbox.SandboxedShellBackend(
        root_dir=str(tmp_path),
        capability=cap,
        runner=runner,
        tmp="/tmp",
        home="/home/u",
    )


def test_execute_seatbelt_wraps_command_in_sandbox_exec(tmp_path):
    calls: list[tuple[list[str], str, int]] = []

    def runner(argv: list[str], cwd: str, timeout: int) -> sandbox._Result:
        calls.append((argv, cwd, timeout))
        return sandbox._Result("done", 0)

    backend = _backend(tmp_path, "seatbelt", runner)
    resp = backend.execute("pytest -q", timeout=30)

    argv, cwd, timeout = calls[0]
    assert argv[0] == "sandbox-exec" and argv[1] == "-p"
    assert "(deny default)" in argv[2]  # the rendered profile
    assert "pytest -q" in argv[-1]  # command at the tail (ulimit-wrapped)
    assert cwd == str(tmp_path.resolve())
    assert timeout == 30
    assert isinstance(resp, ExecuteResponse)
    assert resp.output == "done" and resp.exit_code == 0


def test_execute_bwrap_uses_bwrap_argv(tmp_path):
    seen: dict[str, list[str]] = {}

    def runner(argv: list[str], cwd: str, timeout: int) -> sandbox._Result:
        seen["argv"] = argv
        return sandbox._Result("ok", 0)

    _backend(tmp_path, "bwrap", runner).execute("ls")
    assert seen["argv"][0] == "bwrap"


def test_execute_capability_none_refuses_and_never_runs(tmp_path):
    # Record-and-assert-not-called: the runner stays coverable rather than being marked
    # uncoverable with a gated escape-hatch pragma.
    calls: list[list[str]] = []

    def runner(argv: list[str], cwd: str, timeout: int) -> sandbox._Result:
        calls.append(argv)
        return sandbox._Result("", 0)

    resp = _backend(tmp_path, "none", runner).execute("rm -rf /")
    assert resp.output == sandbox.NO_SANDBOX_MESSAGE
    assert resp.exit_code is None
    assert calls == []  # the killer assertion: refusal must run nothing


def test_execute_never_calls_super_execute(tmp_path, monkeypatch):
    # The unconfined host shell must never run, even on the happy path. A one-line lambda
    # records the call so there's no never-executed function body to leave uncovered.
    from deepagents.backends.local_shell import LocalShellBackend

    super_calls: list[str] = []
    monkeypatch.setattr(
        LocalShellBackend,
        "execute",
        lambda self, command, *, timeout=None: super_calls.append(command),
    )
    backend = _backend(tmp_path, "seatbelt", lambda a, c, t: sandbox._Result("x", 0))
    assert backend.execute("echo hi").output == "x"
    assert super_calls == []  # host shell never invoked


def test_execute_runner_failure_returns_apology(tmp_path):
    def runner(argv: list[str], cwd: str, timeout: int) -> sandbox._Result:
        raise OSError("sandbox-exec missing")

    resp = _backend(tmp_path, "seatbelt", runner).execute("echo hi")
    assert resp.output == sandbox.LAUNCH_FAILURE_MESSAGE
    assert resp.exit_code is None


def test_execute_nonzero_exit_passes_output_and_code_through(tmp_path):
    def runner(argv: list[str], cwd: str, timeout: int) -> sandbox._Result:
        return sandbox._Result("boom\n", 1)

    resp = _backend(tmp_path, "seatbelt", runner).execute("false")
    assert resp.output == "boom\n" and resp.exit_code == 1


def test_execute_clamps_timeout_to_max(tmp_path):
    seen: dict[str, int] = {}

    def runner(argv: list[str], cwd: str, timeout: int) -> sandbox._Result:
        seen["timeout"] = timeout
        return sandbox._Result("", 0)

    _backend(tmp_path, "seatbelt", runner).execute("x", timeout=10_000)
    assert seen["timeout"] == sandbox.MAX_TIMEOUT_SECONDS


def test_execute_defaults_timeout_when_unset(tmp_path):
    seen: dict[str, int] = {}
    _backend(
        tmp_path, "seatbelt", lambda a, c, t: seen.update(t=t) or sandbox._Result("", 0)
    ).execute("x")
    assert seen["t"] == sandbox.DEFAULT_TIMEOUT_SECONDS


def test_execute_value_error_runner_failure_returns_apology(tmp_path):
    # The narrowed except must catch each arm of (OSError, ValueError, SubprocessError) -> apology.
    def runner(argv: list[str], cwd: str, timeout: int) -> sandbox._Result:
        raise ValueError("bad argv")

    resp = _backend(tmp_path, "seatbelt", runner).execute("echo hi")
    assert resp.output == sandbox.LAUNCH_FAILURE_MESSAGE
    assert resp.exit_code is None


def test_execute_subprocess_error_runner_failure_returns_apology(tmp_path):
    import subprocess

    def runner(argv: list[str], cwd: str, timeout: int) -> sandbox._Result:
        raise subprocess.SubprocessError("spawn failed")

    resp = _backend(tmp_path, "seatbelt", runner).execute("echo hi")
    assert resp.output == sandbox.LAUNCH_FAILURE_MESSAGE
    assert resp.exit_code is None


def test_backend_defaults_runner_capability_tmp_and_home(tmp_path):
    # No runner/capability/tmp/home given: each falls back to its real default. Asserting the
    # fallbacks took effect kills the mutants that drop the `or default_runner` / `is not None` arms.
    import tempfile
    from pathlib import Path

    backend = sandbox.SandboxedShellBackend(root_dir=str(tmp_path))

    assert backend._runner is sandbox.default_runner
    assert backend._capability in ("seatbelt", "bwrap", "none")  # the real detector ran
    assert backend._tmp == tempfile.gettempdir()
    assert backend._home == str(Path("~").expanduser())
    assert backend.virtual_mode is True  # defaults to traversal-blocked virtual mode
