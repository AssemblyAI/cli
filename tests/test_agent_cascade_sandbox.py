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
