import pytest

from aai_cli.auth import endpoints
from aai_cli.errors import CLIError


def test_redirect_uri_is_fixed_loopback():
    assert endpoints.redirect_uri() == "http://127.0.0.1:8585/callback"


def test_env_specific_values_come_from_active_environment():
    # With no active env set, helpers fall back to the default (production).
    assert endpoints.ams_base() == "https://ams.internal.assemblyai-labs.com"
    assert endpoints.stytch_domain() == "https://api.stytch.com"
    assert endpoints.stytch_public_token().startswith("public-token-")
    assert endpoints.signup_url().startswith("https://")


def test_constants_are_environment_independent():
    assert endpoints.STYTCH_OAUTH_PROVIDER == "google"
    assert endpoints.CLI_TOKEN_NAME == "AssemblyAI CLI"


def test_env_override_changes_redirect_uri(monkeypatch):
    monkeypatch.setenv("AAI_AUTH_PORT", "9999")
    assert endpoints.redirect_uri() == "http://127.0.0.1:9999/callback"


def test_loopback_port_rejects_non_integer(monkeypatch):
    # A typo'd AAI_AUTH_PORT must surface as a clean CLIError on the login path,
    # not a raw ValueError that would crash every command at import time.
    monkeypatch.setenv("AAI_AUTH_PORT", "abc")
    with pytest.raises(CLIError) as excinfo:
        endpoints.loopback_port()
    assert excinfo.value.exit_code == 2
    assert "AAI_AUTH_PORT" in str(excinfo.value)


def test_loopback_port_rejects_above_max(monkeypatch):
    # 65535 is the highest valid TCP port; one past it must be rejected.
    monkeypatch.setenv("AAI_AUTH_PORT", "65536")
    with pytest.raises(CLIError):
        endpoints.loopback_port()


def test_loopback_port_accepts_boundary_values(monkeypatch):
    # The valid range is exactly 1..65535 inclusive.
    monkeypatch.setenv("AAI_AUTH_PORT", "1")
    assert endpoints.loopback_port() == 1
    monkeypatch.setenv("AAI_AUTH_PORT", "65535")
    assert endpoints.loopback_port() == 65535


def test_loopback_port_rejects_zero(monkeypatch):
    # Port 0 (OS-assign) is meaningless for a fixed, pre-registered redirect URI.
    monkeypatch.setenv("AAI_AUTH_PORT", "0")
    with pytest.raises(CLIError):
        endpoints.loopback_port()
