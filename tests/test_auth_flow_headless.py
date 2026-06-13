"""The browser-fallback messaging in the login flow (headless/SSH boxes)."""

import json

from aai_cli.auth import flow


def _fail_browser(monkeypatch):
    monkeypatch.setattr(flow.webbrowser, "open", lambda _url: False)


def _unwrapped(err: str) -> str:
    """Collapse the console's soft line wrapping so substrings match reliably."""
    return " ".join(err.split())


def test_fallback_names_the_port_forward_and_stdin_recipe(monkeypatch, capsys):
    _fail_browser(monkeypatch)
    flow._open_browser("https://login.example", json_mode=False)
    err = _unwrapped(capsys.readouterr().err)
    assert "Could not open a browser" in err
    # The callback lands on *this* machine's loopback, so the SSH case needs the
    # exact forward command (default port 8585) before the URL can work remotely.
    assert "ssh -L 8585:127.0.0.1:8585" in err
    assert "printenv ASSEMBLYAI_API_KEY" in err  # the no-browser-at-all escape


def test_fallback_honors_auth_port_override(monkeypatch, capsys):
    monkeypatch.setenv("AAI_AUTH_PORT", "9000")
    _fail_browser(monkeypatch)
    flow._open_browser("https://login.example", json_mode=False)
    assert "ssh -L 9000:127.0.0.1:9000" in _unwrapped(capsys.readouterr().err)


def test_fallback_json_mode_ships_hint_objects(monkeypatch, capsys):
    _fail_browser(monkeypatch)
    flow._open_browser("https://login.example", json_mode=True)
    lines = [json.loads(line) for line in capsys.readouterr().err.strip().splitlines()]
    fallback = next(obj for obj in lines if "Could not open a browser" in obj["hint"])
    assert "ssh -L 8585:127.0.0.1:8585" in fallback["hint"]
    assert fallback["url"] == "https://login.example"
