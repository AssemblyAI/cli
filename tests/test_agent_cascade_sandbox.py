from __future__ import annotations

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
