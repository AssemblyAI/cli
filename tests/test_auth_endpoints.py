from aai_cli.auth import endpoints


def test_redirect_uri_is_fixed_loopback():
    assert endpoints.redirect_uri() == "http://127.0.0.1:8585/callback"


def test_env_specific_values_come_from_active_environment():
    # With no active env set, helpers fall back to the default (sandbox000).
    assert endpoints.ams_base() == "https://ams.sandbox000.assemblyai-labs.com"
    assert endpoints.stytch_domain() == "https://test.stytch.com"
    assert endpoints.stytch_public_token().startswith("public-token-")
    assert endpoints.signup_url().startswith("https://")


def test_constants_are_environment_independent():
    assert endpoints.STYTCH_OAUTH_PROVIDER == "google"
    assert endpoints.CLI_TOKEN_NAME == "AssemblyAI CLI"


def test_env_override_changes_redirect_uri(monkeypatch):
    monkeypatch.setattr(endpoints, "LOOPBACK_PORT", 9999)
    assert endpoints.redirect_uri() == "http://127.0.0.1:9999/callback"
