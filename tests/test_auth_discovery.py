from urllib.parse import parse_qs, urlparse

from assemblyai_cli.auth import discovery, endpoints


def test_build_start_url_targets_b2b_discovery_for_provider():
    url = discovery.build_start_url()
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.path == "/v1/b2b/public/oauth/google/discovery/start"
    assert url.startswith(endpoints.STYTCH_PROJECT_DOMAIN)


def test_build_start_url_includes_public_token_and_redirect():
    url = discovery.build_start_url()
    qs = parse_qs(urlparse(url).query)
    assert qs["public_token"] == [endpoints.STYTCH_PUBLIC_TOKEN]
    assert qs["discovery_redirect_url"] == ["http://127.0.0.1:8585/callback"]
