from __future__ import annotations

from aai_cli.tts import session

from aai_cli import environments


def _use_env(name: str) -> None:
    environments.set_active(environments.get(name))


def test_is_available_true_in_sandbox():
    _use_env("sandbox000")
    assert session.is_available() is True


def test_is_available_false_in_production():
    _use_env("production")
    assert session.is_available() is False


def test_ws_url_includes_set_params_only():
    _use_env("sandbox000")
    cfg = session.SpeakConfig(text="hi", voice="jane", language="English")
    url = session.ws_url(cfg.query_params())
    assert url.startswith("wss://streaming-tts.sandbox000.assemblyai-labs.com/v1/ws/?")
    assert "voice=jane" in url
    assert "language=English" in url
    assert "sample_rate" not in url


def test_ws_url_no_params_has_no_query_string():
    _use_env("sandbox000")
    url = session.ws_url(session.SpeakConfig(text="hi").query_params())
    assert url == "wss://streaming-tts.sandbox000.assemblyai-labs.com/v1/ws/"


def test_query_params_serializes_sample_rate_as_string():
    cfg = session.SpeakConfig(text="hi", sample_rate=16000)
    assert cfg.query_params() == {"sample_rate": "16000"}
