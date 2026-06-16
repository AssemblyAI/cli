"""`assembly stream` wires SIGTERM into the clean-stop path (see core.signals).

These assert the *wiring*: that ``run_stream`` installs the SIGTERM->KeyboardInterrupt
handler around the streaming body for both the single-source and ``--from-stdin`` batch
paths. The handler's own behavior is covered in test_signals.py; the graceful stop the
KeyboardInterrupt then triggers is covered in test_stream_session.py / test_stream_batch.py.
"""

from __future__ import annotations

import signal

import pytest
from typer.testing import CliRunner

from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()


def test_stream_installs_sigterm_handler_around_dispatch(monkeypatch):
    config.set_api_key("default", "sk_live")
    captured: dict[str, object] = {}

    def fake_dispatch(session, sources):
        captured["handler"] = signal.getsignal(signal.SIGTERM)

    monkeypatch.setattr("aai_cli.commands.stream._exec._dispatch", fake_dispatch)
    result = runner.invoke(app, ["stream"])

    assert result.exit_code == 0
    handler = captured["handler"]
    # While streaming, SIGTERM raises KeyboardInterrupt — without the wrapper this
    # would be the default disposition (SIG_DFL), which is not callable.
    assert callable(handler)
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)


def test_stream_batch_installs_sigterm_handler_around_run(monkeypatch):
    config.set_api_key("default", "sk_live")
    captured: dict[str, object] = {}

    def fake_run_batch(opts, state, *, json_mode, text_mode):
        captured["handler"] = signal.getsignal(signal.SIGTERM)

    monkeypatch.setattr("aai_cli.commands.stream._exec._run_batch", fake_run_batch)
    result = runner.invoke(app, ["stream", "--from-stdin"])

    assert result.exit_code == 0
    handler = captured["handler"]
    assert callable(handler)
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)
