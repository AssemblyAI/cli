from urllib.parse import parse_qs, urlparse

from aai_cli.auth import discovery, endpoints


def test_build_start_url_targets_b2b_discovery_for_provider():
    url = discovery.build_start_url("state-xyz")
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.path == "/v1/b2b/public/oauth/google/discovery/start"
    assert url.startswith(endpoints.stytch_domain())


def test_build_start_url_includes_public_token_and_redirect():
    url = discovery.build_start_url("state-xyz")
    qs = parse_qs(urlparse(url).query)
    assert qs["public_token"] == [endpoints.stytch_public_token()]
    # The state nonce rides as a query param on the (still path-exact) redirect URL.
    assert qs["discovery_redirect_url"] == ["http://127.0.0.1:8585/callback?state=state-xyz"]


def test_build_start_url_carries_state_into_redirect():
    url = discovery.build_start_url("nonce-123")
    redirect = parse_qs(urlparse(url).query)["discovery_redirect_url"][0]
    redirect_parsed = urlparse(redirect)
    # The redirect path stays exactly /callback (what Stytch validates); only the
    # query string gains the nonce, so redirect-URL matching is unaffected.
    assert redirect_parsed.path == "/callback"
    assert parse_qs(redirect_parsed.query)["state"] == ["nonce-123"]
