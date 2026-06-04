from urllib.parse import parse_qs, urlparse

from aai_cli.auth import discovery, endpoints


def test_build_start_url_targets_b2b_discovery_for_provider():
    url = discovery.build_start_url()
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.path == "/v1/b2b/public/oauth/google/discovery/start"
    assert url.startswith(endpoints.stytch_domain())


def test_build_start_url_includes_public_token_and_redirect():
    url = discovery.build_start_url()
    qs = parse_qs(urlparse(url).query)
    assert qs["public_token"] == [endpoints.stytch_public_token()]
    assert qs["discovery_redirect_url"] == ["http://127.0.0.1:8585/callback"]
