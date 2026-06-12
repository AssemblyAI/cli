"""Tests for the opt-in diagnostic logging behind the root -v/--verbose flag
(aai_cli/debuglog.py): the stderr handler, secret redaction, and the places
that register secrets (config.resolve_api_key, AppState.resolve_session).
"""

from __future__ import annotations

import logging

import pytest
from typer.testing import CliRunner

from aai_cli import config, debuglog
from aai_cli import ws as wsutil
from aai_cli.context import AppState
from aai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_debuglog(monkeypatch):
    # enable() mutates process-global state (the root logger and the module's
    # verbosity/secret registries); snapshot and restore so pytest-randomly
    # ordering can't leak a verbose run into unrelated tests.
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    # Logger levels are process-global too: any earlier test that exercised the
    # realtime silencers left the websockets loggers clamped at CRITICAL, which
    # would swallow the wire-level records asserted here. Reset them so these
    # tests are order-independent under pytest-randomly, then restore.
    wire_loggers = [logging.getLogger(name) for name in wsutil.WEBSOCKETS_LOGGERS]
    previous_wire_levels = [logger.level for logger in wire_loggers]
    for logger in wire_loggers:
        logger.setLevel(logging.NOTSET)
    monkeypatch.setattr(debuglog, "_verbosity", 0)
    monkeypatch.setattr(debuglog, "_secrets", set())
    yield
    root.handlers[:] = previous_handlers
    root.setLevel(previous_level)
    for logger, level in zip(wire_loggers, previous_wire_levels, strict=True):
        logger.setLevel(level)


def test_enable_zero_is_the_everyday_no_op():
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    debuglog.enable(0)
    assert not debuglog.active()
    assert root.handlers == handlers_before
    assert root.level == level_before


def test_enable_one_logs_info_but_not_debug_to_stderr(capsys):
    debuglog.enable(1)
    assert debuglog.active()
    assert logging.getLogger().level == logging.INFO
    probe = logging.getLogger("aai_cli.test_probe")
    probe.info("hello request log")
    probe.debug("wire frame detail")
    err = capsys.readouterr().err
    assert "[aai_cli.test_probe] hello request log" in err
    assert "wire frame detail" not in err


def test_enable_two_logs_wire_level_debug(capsys):
    debuglog.enable(2)
    assert logging.getLogger().level == logging.DEBUG
    logging.getLogger("websockets.client").debug("> TEXT frame")
    assert "[websockets.client] > TEXT frame" in capsys.readouterr().err


def test_registered_secrets_are_masked_in_every_record(capsys):
    debuglog.enable(2)
    debuglog.register_secret("sk_super_secret_value")
    logging.getLogger("websockets.client").debug("authorization: sk_super_secret_value")
    err = capsys.readouterr().err
    assert "sk_super_secret_value" not in err
    assert "authorization: [redacted]" in err


def test_register_secret_ignores_empty_values():
    debuglog.register_secret(None)
    debuglog.register_secret("")
    assert debuglog._secrets == set()


def test_resolve_api_key_registers_the_key_for_redaction(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_env_key_for_redaction")
    assert config.resolve_api_key() == "sk_env_key_for_redaction"
    assert "sk_env_key_for_redaction" in debuglog._secrets


def test_resolve_session_registers_the_jwt_for_redaction(monkeypatch):
    monkeypatch.setattr(config, "get_session", lambda profile: {"jwt": "jwt_secret", "token": "t"})
    monkeypatch.setattr(config, "get_account_id", lambda profile: 42)
    assert AppState().resolve_session() == (42, "jwt_secret")
    assert "jwt_secret" in debuglog._secrets


def test_root_vv_flag_enables_wire_level_diagnostics():
    result = runner.invoke(app, ["-vv"])  # bare invocation: prints help and exits 0
    assert result.exit_code == 0
    assert debuglog.active()
    assert logging.getLogger().level == logging.DEBUG


def test_root_v_flag_sets_info_and_logs_the_environment(caplog):
    caplog.set_level(logging.DEBUG, logger="aai_cli")
    result = runner.invoke(app, ["-v"])
    assert result.exit_code == 0
    assert logging.getLogger().level == logging.INFO
    assert "environment" in caplog.text
    assert "production" in caplog.text
