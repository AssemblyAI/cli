"""Tests for the opt-in diagnostic logging behind the root -v/--verbose flag
(aai_cli/debuglog.py): the stderr handler, secret redaction, and the places
that register secrets (config.resolve_api_key, AppState.resolve_session).
"""

from __future__ import annotations

import logging

import pytest
from typer.testing import CliRunner

from aai_cli import config, debuglog
from aai_cli.context import AppState
from aai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_debuglog(preserve_logging_state, monkeypatch):
    # The shared conftest fixture snapshots the process-global logging state
    # (root handlers/level, websockets wire-logger levels); enable() also
    # mutates the module's own verbosity/secret registries, reset here.
    monkeypatch.setattr(debuglog, "_verbosity", 0)
    monkeypatch.setattr(debuglog, "_secrets", set())


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


def test_binary_frame_hex_is_collapsed_to_byte_count(capsys):
    debuglog.enable(2)
    logging.getLogger("websockets.client").debug(
        "> BINARY fb ff fe ff fb ff f5 ff ef ff 00 00 fd ff fd ff "
        "... 0d 00 f7 ff 01 00 0c 00 [9600 bytes]"
    )
    err = capsys.readouterr().err
    assert "[websockets.client] > BINARY [9600 bytes]" in err
    assert "fb ff" not in err


def test_real_websockets_binary_frame_rendering_is_collapsed(capsys):
    # Pin the regex to the library's actual frame format: a Frame's __str__ is
    # what websockets interpolates into its DEBUG records, elision and all.
    from websockets.frames import OP_BINARY, OP_CONT, Frame

    debuglog.enable(2)
    log = logging.getLogger("websockets.client")
    log.debug("> %s", Frame(OP_BINARY, bytes(9600), fin=False))
    log.debug("> %s", Frame(OP_CONT, b"\xfb\xff\x0d\x00"))
    err = capsys.readouterr().err
    assert "> BINARY [9600 bytes, continued]" in err
    assert "> CONT [binary, 4 bytes]" in err
    assert "00 00" not in err


def test_text_frames_and_other_loggers_keep_their_payload(capsys):
    debuglog.enable(2)
    logging.getLogger("websockets.client").debug("> TEXT '{\"audio\": true}' [16 bytes]")
    logging.getLogger("httpx").debug("> BINARY fb ff 0d 00 [4 bytes]")
    err = capsys.readouterr().err
    assert "> TEXT '{\"audio\": true}' [16 bytes]" in err
    assert "[httpx] > BINARY fb ff 0d 00 [4 bytes]" in err


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
