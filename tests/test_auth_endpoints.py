from aai_cli.auth import endpoints


def test_redirect_uri_is_fixed_loopback():
    assert endpoints.redirect_uri() == "http://127.0.0.1:8585/callback"


def test_defaults_point_at_sandbox():
    assert endpoints.AMS_BASE_URL == "https://ams.sandbox000.assemblyai-labs.com"
    assert endpoints.STYTCH_OAUTH_PROVIDER == "google"
    assert endpoints.CLI_TOKEN_NAME == "AssemblyAI CLI"
    assert endpoints.STYTCH_PUBLIC_TOKEN.startswith("public-token-")
    assert endpoints.SIGNUP_URL.startswith("https://")


def test_env_override_changes_redirect_uri(monkeypatch):
    monkeypatch.setattr(endpoints, "LOOPBACK_PORT", 9999)
    assert endpoints.redirect_uri() == "http://127.0.0.1:9999/callback"
